import os
import re
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pydicom


# ============================================================
# CONFIG
# ============================================================

JUNK_PATTERNS = [
    r"\bscout\b", r"\btopogram\b", r"\blocalizer\b", r"\btopo\b",
    r"\bdose\b", r"\bdlp\b", r"\bctdi\b",
    r"\bprotocol\b",
    r"\bresults?\b", r"\breport\b", r"\breading\b",
    r"\bbolus\b", r"\btracker\b", r"\bmonitor\b",
    r"\btest\b",
]


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class SeriesSummary:
    series_instance_uid: str
    series_number: int | None
    series_description: str
    protocol_name: str

    folder_names: list[str]

    n_files: int
    n_slices: int

    slice_thickness_values: list[float]
    spacing_between_slices_values: list[float]
    computed_inter_slice_spacing: float | None

    pixel_spacing_values: list[tuple[float, float]]
    matrix_sizes: list[tuple[int, int]]

    frame_of_reference_uid: str


# ============================================================
# FILTERING
# ============================================================

def is_obvious_non_phase(series_description: str, protocol_name: str) -> bool:
    text = f"{series_description} {protocol_name}".lower()
    return any(re.search(p, text) for p in JUNK_PATTERNS)


# ============================================================
# DICOM READING
# ============================================================

def _read_ct_header(path: str):
    """
    Read minimal CT DICOM header.
    force=True is important because many PACS/SECTRA exports do not have DICM preamble.
    """
    try:
        ds = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            force=True,
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
        if getattr(ds, "SeriesInstanceUID", None) is None:
            return None

        return ds
    except Exception:
        return None


# ============================================================
# REPORT HELPERS
# ============================================================

def _report_path_for_root(root: str, out_dir: str | None = None) -> str:
    root = os.path.abspath(root)
    folder_name = os.path.basename(os.path.normpath(root))
    if out_dir is None:
        out_dir = root
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"{folder_name}_dicom_report_{ts}.txt")


def write_series_report_txt(
    root: str,
    summaries: list[SeriesSummary],
    out_dir: str | None = None,
    min_slices: int = 10,
) -> str:
    """
    Writes a TXT report.
    Keeps candidate phases by removing only obvious junk and tiny series.
    """
    report_path = _report_path_for_root(root, out_dir=out_dir)

    kept: list[SeriesSummary] = []
    removed: list[tuple[SeriesSummary, str]] = []

    for s in summaries:
        if is_obvious_non_phase(s.series_description, s.protocol_name):
            removed.append((s, "junk keyword"))
            continue
        if s.n_slices < min_slices:
            removed.append((s, f"n_slices<{min_slices}"))
            continue
        kept.append(s)

    for_uids = [s.frame_of_reference_uid for s in kept if s.frame_of_reference_uid]
    distinct_for = sorted(set(for_uids))

    root = os.path.abspath(root)
    folder_name = os.path.basename(os.path.normpath(root))

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Root folder name: {folder_name}\n")
        f.write(f"Root path: {root}\n\n")

        f.write(f"Kept series (candidate phases): {len(kept)} (min_slices={min_slices})\n")
        f.write(f"Removed series: {len(removed)}\n\n")

        f.write("Alignment hint (metadata-level):\n")
        if distinct_for:
            if len(distinct_for) == 1:
                f.write("  - All kept series share the same FrameOfReferenceUID (same patient coordinate frame).\n")
            else:
                f.write(f"  - Multiple FrameOfReferenceUID values among kept series ({len(distinct_for)}).\n")
        else:
            f.write("  - FrameOfReferenceUID missing for kept series.\n")
        f.write("\n")

        f.write("=== KEPT (candidate phases) ===\n\n")
        for s in sorted(kept, key=lambda x: (x.series_number is None, x.series_number or 10**9)):
            folders = ", ".join(s.folder_names) if s.folder_names else "(unknown folder)"

            th = s.slice_thickness_values
            dz = s.computed_inter_slice_spacing
            px = s.pixel_spacing_values
            mx = s.matrix_sizes

            if not th:
                th_str = "unknown"
            elif len(th) == 1:
                th_str = f"{th[0]} mm"
            else:
                th_str = f"varies {th} mm"

            dz_str = "" if dz is None else f", computed dz~{dz:.4f} mm"

            if not mx:
                matrix_str = "unknown"
            elif len(mx) == 1:
                rows, cols = mx[0]
                matrix_str = f"{rows} x {cols} px"
            else:
                matrix_str = f"varies {mx}"

            if not px:
                px_str = "unknown"
            elif len(px) == 1:
                r, c = px[0]
                px_str = f"{r} x {c} mm"
            else:
                px_str = f"varies {px}"

            f.write(f"- Series {s.series_number if s.series_number is not None else '?'} | {s.series_description}\n")
            f.write(f"  Folder(s): {folders}\n")
            if s.protocol_name:
                f.write(f"  Protocol: {s.protocol_name}\n")
            f.write(f"  UID: {s.series_instance_uid}\n")
            f.write(f"  Files: {s.n_files}\n")
            f.write(f"  Slices: {s.n_slices}\n")
            f.write(f"  SliceThickness: {th_str}{dz_str}\n")
            f.write(f"  Resolution: {matrix_str}\n")
            f.write(f"  PixelSpacing: {px_str}\n")
            f.write(f"  FrameOfReferenceUID: {s.frame_of_reference_uid or '(missing)'}\n\n")

        f.write("\n=== REMOVED (obvious non-phase / too small) ===\n\n")
        for s, reason in removed:
            folders = ", ".join(s.folder_names) if s.folder_names else "(unknown folder)"
            f.write(f"- Series {s.series_number if s.series_number is not None else '?'} | {s.series_description}\n")
            f.write(f"  Folder(s): {folders}\n")
            if s.protocol_name:
                f.write(f"  Protocol: {s.protocol_name}\n")
            f.write(f"  UID: {s.series_instance_uid}\n")
            f.write(f"  Reason: {reason}\n\n")

    return report_path


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_dicom_folder(
    root: str,
    max_workers: int = 8,
    progress_every: int = 500,
    min_slices_for_report: int = 10,
    write_txt: bool = True,
    out_dir: str | None = None,
):
    """
    Analyze a CT DICOM folder recursively.

    Important:
    - For PACS/SECTRA exports, point this at the DICOM folder.
      Example:
          ...\\case_folder\\DICOM

    Returns:
      series_headers, summaries, report_path
    """
    root = os.path.abspath(root)

    file_paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.upper() == "DICOMDIR":
                continue
            file_paths.append(os.path.join(dirpath, fn))

    print(f"Scanning {len(file_paths)} files under: {root}")

    series_headers = defaultdict(list)   # uid -> list[pydicom.Dataset]
    series_folders = defaultdict(set)    # uid -> set[relative folder paths]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_read_ct_header, p): p for p in file_paths}
        done = 0

        for fut in as_completed(futures):
            p = futures[fut]
            ds = fut.result()
            done += 1

            if progress_every and done % progress_every == 0:
                print(f"  ...processed {done}/{len(file_paths)}")

            if ds is None:
                continue

            series_uid = str(ds.SeriesInstanceUID)
            series_headers[series_uid].append(ds)

            rel_folder = os.path.relpath(os.path.dirname(p), root)
            series_folders[series_uid].add(rel_folder)

    print(f"Found {len(series_headers)} CT series.")
    print()

    summaries: list[SeriesSummary] = []

    for series_uid, headers in series_headers.items():
        rep = headers[0]
        desc = (getattr(rep, "SeriesDescription", "") or "").strip()
        proto = (getattr(rep, "ProtocolName", "") or "").strip()
        series_no = getattr(rep, "SeriesNumber", None)

        th = sorted({
            float(x) for x in [getattr(h, "SliceThickness", None) for h in headers]
            if x is not None
        })

        sbs = sorted({
            float(x) for x in [getattr(h, "SpacingBetweenSlices", None) for h in headers]
            if x is not None
        })

        pixel_spacing_values = set()
        for h in headers:
            ps = getattr(h, "PixelSpacing", None)
            if ps is not None and len(ps) >= 2:
                try:
                    pixel_spacing_values.add((round(float(ps[0]), 6), round(float(ps[1]), 6)))
                except Exception:
                    pass
        pixel_spacing_values = sorted(pixel_spacing_values)

        matrix_sizes = set()
        for h in headers:
            rows = getattr(h, "Rows", None)
            cols = getattr(h, "Columns", None)
            if rows is not None and cols is not None:
                try:
                    matrix_sizes.add((int(rows), int(cols)))
                except Exception:
                    pass
        matrix_sizes = sorted(matrix_sizes)

        iop = getattr(rep, "ImageOrientationPatient", None)
        ipp = [getattr(h, "ImagePositionPatient", None) for h in headers]
        ipp = [tuple(map(float, x)) for x in ipp if x is not None]

        n_files = len(headers)
        n_slices = n_files
        dz = None

        if iop is not None and len(ipp) >= 2:
            try:
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
            except Exception:
                pass

        for_uid = (getattr(rep, "FrameOfReferenceUID", "") or "").strip()
        folders_sorted = sorted(series_folders.get(series_uid, set()))

        th_str = "unknown" if not th else (f"{th[0]} mm" if len(th) == 1 else f"varies {th} mm")
        dz_str = "" if dz is None else f", computed dz~{dz:.4f} mm"
        sbs_str = "" if not sbs else (
            f", SpacingBetweenSlices={sbs[0]} mm"
            if len(sbs) == 1 else f", SpacingBetweenSlices varies {sbs}"
        )

        if not pixel_spacing_values:
            px_str = "unknown"
        elif len(pixel_spacing_values) == 1:
            r, c = pixel_spacing_values[0]
            px_str = f"{r} x {c} mm"
        else:
            px_str = f"varies {pixel_spacing_values}"

        if not matrix_sizes:
            matrix_str = "unknown"
        elif len(matrix_sizes) == 1:
            rows, cols = matrix_sizes[0]
            matrix_str = f"{rows} x {cols} px"
        else:
            matrix_str = f"varies {matrix_sizes}"

        print(f"Series {series_no if series_no is not None else '?'} | {desc} | {proto}")
        print(f"  Folder(s): {', '.join(folders_sorted) if folders_sorted else '(unknown folder)'}")
        print(f"  UID: {series_uid}")
        print(f"  Files: {n_files}")
        print(f"  Slices: {n_slices}")
        print(f"  SliceThickness: {th_str}{sbs_str}{dz_str}")
        print(f"  Resolution: {matrix_str}")
        print(f"  PixelSpacing: {px_str}")
        print(f"  FrameOfReferenceUID: {for_uid or '(missing)'}")
        print()

        summaries.append(
            SeriesSummary(
                series_instance_uid=series_uid,
                series_number=int(series_no) if series_no is not None else None,
                series_description=desc,
                protocol_name=proto,
                folder_names=folders_sorted,
                n_files=n_files,
                n_slices=n_slices,
                slice_thickness_values=[round(x, 6) for x in th],
                spacing_between_slices_values=[round(x, 6) for x in sbs],
                computed_inter_slice_spacing=None if dz is None else round(dz, 6),
                pixel_spacing_values=pixel_spacing_values,
                matrix_sizes=matrix_sizes,
                frame_of_reference_uid=for_uid,
            )
        )

    report_path = None
    if write_txt:
        report_path = write_series_report_txt(
            root=root,
            summaries=summaries,
            out_dir=out_dir,
            min_slices=min_slices_for_report,
        )
        print(f"Wrote TXT report: {report_path}")

    return series_headers, summaries, report_path


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    folder = r"C:\path\to\your\case\DICOM"
    series_headers, summaries, report_path = analyze_dicom_folder(
        folder,
        max_workers=8,
        progress_every=500,
        min_slices_for_report=10,
        write_txt=True,
    )
    print(report_path)