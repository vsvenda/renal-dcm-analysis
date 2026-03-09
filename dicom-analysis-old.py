import os
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pydicom


def _looks_like_dicom(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False


def _read_ct_header(path: str):
    try:
        if not _looks_like_dicom(path):
            return None

        ds = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            force=False,
            specific_tags=[
                "Modality",
                "SeriesInstanceUID", "SeriesNumber", "SeriesDescription", "ProtocolName",
                "InstanceNumber",
                "SliceThickness", "SpacingBetweenSlices",
                "ImagePositionPatient", "ImageOrientationPatient",
                "FrameOfReferenceUID",
                "PixelSpacing", "Rows", "Columns",
            ],
        )

        if getattr(ds, "Modality", "") != "CT":
            return None

        return ds
    except Exception:
        return None


@dataclass
class FolderPhaseSummary:
    folder_name: str
    n_files: int
    n_slices: int

    series_instance_uids: list[str]
    series_numbers: list[int]
    series_descriptions: list[str]
    protocol_names: list[str]

    slice_thickness_values: list[float]
    spacing_between_slices_values: list[float]
    computed_inter_slice_spacing: float | None

    pixel_spacing_values: list[tuple[float, float]]
    matrix_sizes: list[tuple[int, int]]

    frame_of_reference_uids: list[str]


def _collect_files(root: str) -> list[str]:
    file_paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.upper() == "DICOMDIR":
                continue
            file_paths.append(os.path.join(dirpath, fn))
    return file_paths


def _report_path_for_root(root: str, out_dir: str | None = None) -> str:
    root = os.path.abspath(root)
    folder_name = os.path.basename(os.path.normpath(root))
    if out_dir is None:
        out_dir = root
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"{folder_name}_phase_folders_report_{ts}.txt")


def _summarize_headers_for_folder(folder_name: str, headers: list) -> FolderPhaseSummary:
    n_files = len(headers)

    series_instance_uids = sorted({
        str(getattr(h, "SeriesInstanceUID"))
        for h in headers
        if getattr(h, "SeriesInstanceUID", None) is not None
    })

    series_numbers = sorted({
        int(getattr(h, "SeriesNumber"))
        for h in headers
        if getattr(h, "SeriesNumber", None) is not None
    })

    series_descriptions = sorted({
        str(getattr(h, "SeriesDescription")).strip()
        for h in headers
        if getattr(h, "SeriesDescription", None) not in (None, "")
    })

    protocol_names = sorted({
        str(getattr(h, "ProtocolName")).strip()
        for h in headers
        if getattr(h, "ProtocolName", None) not in (None, "")
    })

    slice_thickness_values = sorted({
        float(getattr(h, "SliceThickness"))
        for h in headers
        if getattr(h, "SliceThickness", None) is not None
    })

    spacing_between_slices_values = sorted({
        float(getattr(h, "SpacingBetweenSlices"))
        for h in headers
        if getattr(h, "SpacingBetweenSlices", None) is not None
    })

    pixel_spacing_values = sorted({
        (round(float(h.PixelSpacing[0]), 6), round(float(h.PixelSpacing[1]), 6))
        for h in headers
        if getattr(h, "PixelSpacing", None) is not None and len(h.PixelSpacing) >= 2
    })

    matrix_sizes = sorted({
        (int(getattr(h, "Rows")), int(getattr(h, "Columns")))
        for h in headers
        if getattr(h, "Rows", None) is not None and getattr(h, "Columns", None) is not None
    })

    frame_of_reference_uids = sorted({
        str(getattr(h, "FrameOfReferenceUID")).strip()
        for h in headers
        if getattr(h, "FrameOfReferenceUID", None) not in (None, "")
    })

    # compute slice count / dz from geometry if possible
    n_slices = n_files
    dz = None

    rep = headers[0] if headers else None
    if rep is not None:
        iop = getattr(rep, "ImageOrientationPatient", None)
        ipp = [getattr(h, "ImagePositionPatient", None) for h in headers]
        ipp = [tuple(map(float, x)) for x in ipp if x is not None]

        if iop is not None and len(ipp) >= 2:
            iop_arr = np.array(list(map(float, iop)))
            row, col = iop_arr[:3], iop_arr[3:]
            normal = np.cross(row, col)
            normal = normal / (np.linalg.norm(normal) + 1e-12)

            proj = np.dot(np.array(ipp), normal)
            proj = np.sort(proj)

            diffs = np.diff(proj)
            diffs = diffs[np.abs(diffs) > 1e-6]
            if diffs.size:
                dz = float(np.median(np.abs(diffs)))

            n_slices = int(np.unique(np.round(proj, 3)).size)

    return FolderPhaseSummary(
        folder_name=folder_name,
        n_files=n_files,
        n_slices=n_slices,
        series_instance_uids=series_instance_uids,
        series_numbers=series_numbers,
        series_descriptions=series_descriptions,
        protocol_names=protocol_names,
        slice_thickness_values=[round(x, 6) for x in slice_thickness_values],
        spacing_between_slices_values=[round(x, 6) for x in spacing_between_slices_values],
        computed_inter_slice_spacing=None if dz is None else round(dz, 6),
        pixel_spacing_values=pixel_spacing_values,
        matrix_sizes=matrix_sizes,
        frame_of_reference_uids=frame_of_reference_uids,
    )


def analyze_phase_folders(
    root: str,
    max_workers: int = 8,
    progress_every: int = 500,
    write_txt: bool = True,
    out_dir: str | None = None,
):
    """
    Analyze datasets where each phase is already split into a separate subfolder.
    Returns:
        summaries, report_path
    """
    root = os.path.abspath(root)

    # only immediate subfolders are treated as phases
    phase_dirs = []
    for entry in os.scandir(root):
        if entry.is_dir():
            phase_dirs.append(entry.path)

    phase_dirs = sorted(phase_dirs)

    if not phase_dirs:
        raise ValueError(f"No subfolders found under: {root}")

    summaries: list[FolderPhaseSummary] = []

    for phase_dir in phase_dirs:
        folder_name = os.path.basename(os.path.normpath(phase_dir))
        file_paths = _collect_files(phase_dir)

        print(f"Scanning folder '{folder_name}' ({len(file_paths)} files)")

        headers = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_read_ct_header, p): p for p in file_paths}
            done = 0
            for fut in as_completed(futures):
                ds = fut.result()
                done += 1

                if progress_every and done % progress_every == 0:
                    print(f"  ...processed {done}/{len(file_paths)} in {folder_name}")

                if ds is None:
                    continue
                headers.append(ds)

        if not headers:
            print(f"  No CT DICOM files found in '{folder_name}'")
            print()
            continue

        summary = _summarize_headers_for_folder(folder_name, headers)
        summaries.append(summary)

        th = summary.slice_thickness_values
        sbs = summary.spacing_between_slices_values
        dz = summary.computed_inter_slice_spacing

        th_str = "unknown" if not th else (f"{th[0]} mm" if len(th) == 1 else f"varies {th} mm")
        dz_str = "" if dz is None else f", computed dz~{dz:.4f} mm"
        sbs_str = "" if not sbs else (
            f", SpacingBetweenSlices={sbs[0]} mm"
            if len(sbs) == 1 else f", SpacingBetweenSlices varies {sbs}"
        )

        if not summary.matrix_sizes:
            matrix_str = "unknown"
        elif len(summary.matrix_sizes) == 1:
            rows, cols = summary.matrix_sizes[0]
            matrix_str = f"{rows} x {cols} px"
        else:
            matrix_str = f"varies {summary.matrix_sizes}"

        if not summary.pixel_spacing_values:
            px_str = "unknown"
        elif len(summary.pixel_spacing_values) == 1:
            r, c = summary.pixel_spacing_values[0]
            px_str = f"{r} x {c} mm"
        else:
            px_str = f"varies {summary.pixel_spacing_values}"

        print(f"Folder: {summary.folder_name}")
        print(f"  Slices: {summary.n_slices}")
        print(f"  SliceThickness: {th_str}{sbs_str}{dz_str}")
        print(f"  Resolution: {matrix_str}")
        print(f"  PixelSpacing: {px_str}")
        print(f"  FrameOfReferenceUID(s): {summary.frame_of_reference_uids or ['(missing)']}")
        if summary.series_descriptions:
            print(f"  SeriesDescription(s): {summary.series_descriptions}")
        print()

    report_path = None
    if write_txt:
        report_path = write_phase_folders_report_txt(root, summaries, out_dir=out_dir)
        print(f"Wrote TXT report: {report_path}")

    return summaries, report_path


def write_phase_folders_report_txt(root: str, summaries: list[FolderPhaseSummary], out_dir: str | None = None) -> str:
    report_path = _report_path_for_root(root, out_dir=out_dir)

    root = os.path.abspath(root)
    root_name = os.path.basename(os.path.normpath(root))

    all_for = sorted({uid for s in summaries for uid in s.frame_of_reference_uids if uid})

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Root folder name: {root_name}\n")
        f.write(f"Root path: {root}\n\n")
        f.write(f"Detected phase folders: {len(summaries)}\n\n")

        f.write("Alignment hint (metadata-level):\n")
        if all_for:
            if len(all_for) == 1:
                f.write("  - All folders share the same FrameOfReferenceUID.\n")
            else:
                f.write(f"  - Multiple FrameOfReferenceUID values found across folders ({len(all_for)}).\n")
        else:
            f.write("  - FrameOfReferenceUID missing in all folders.\n")
        f.write("\n")

        for s in summaries:
            th = s.slice_thickness_values
            sbs = s.spacing_between_slices_values
            dz = s.computed_inter_slice_spacing

            th_str = "unknown" if not th else (f"{th[0]} mm" if len(th) == 1 else f"varies {th} mm")
            dz_str = "" if dz is None else f", computed dz~{dz:.4f} mm"
            sbs_str = "" if not sbs else (
                f", SpacingBetweenSlices={sbs[0]} mm"
                if len(sbs) == 1 else f", SpacingBetweenSlices varies {sbs}"
            )

            if not s.matrix_sizes:
                matrix_str = "unknown"
            elif len(s.matrix_sizes) == 1:
                rows, cols = s.matrix_sizes[0]
                matrix_str = f"{rows} x {cols} px"
            else:
                matrix_str = f"varies {s.matrix_sizes}"

            if not s.pixel_spacing_values:
                px_str = "unknown"
            elif len(s.pixel_spacing_values) == 1:
                r, c = s.pixel_spacing_values[0]
                px_str = f"{r} x {c} mm"
            else:
                px_str = f"varies {s.pixel_spacing_values}"

            f.write(f"=== Folder: {s.folder_name} ===\n")
            f.write(f"Files: {s.n_files}\n")
            f.write(f"Slices: {s.n_slices}\n")
            f.write(f"SliceThickness: {th_str}{sbs_str}{dz_str}\n")
            f.write(f"Resolution: {matrix_str}\n")
            f.write(f"PixelSpacing: {px_str}\n")
            f.write(f"FrameOfReferenceUID(s): {s.frame_of_reference_uids or ['(missing)']}\n")

            if s.series_numbers:
                f.write(f"SeriesNumber(s): {s.series_numbers}\n")
            if s.series_descriptions:
                f.write(f"SeriesDescription(s): {s.series_descriptions}\n")
            if s.protocol_names:
                f.write(f"ProtocolName(s): {s.protocol_names}\n")
            if s.series_instance_uids:
                f.write(f"SeriesInstanceUID(s): {s.series_instance_uids}\n")

            f.write("\n")

    return report_path