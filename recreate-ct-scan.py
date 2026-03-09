import os
from collections import defaultdict

import numpy as np
import pydicom
import matplotlib.pyplot as plt


def get_dicomdir_image_paths(export_folder: str) -> list[str]:
    """
    Read DICOMDIR and return all referenced image file paths.
    """
    dicomdir_path = os.path.join(export_folder, "DICOMDIR")
    if not os.path.exists(dicomdir_path):
        raise ValueError(f"DICOMDIR not found in: {export_folder}")

    ds = pydicom.dcmread(dicomdir_path, force=True)

    seq = getattr(ds, "DirectoryRecordSequence", None)
    if seq is None:
        raise ValueError("DirectoryRecordSequence not found in DICOMDIR.")

    image_paths = []

    for rec in seq:
        if getattr(rec, "DirectoryRecordType", None) != "IMAGE":
            continue

        file_id = getattr(rec, "ReferencedFileID", None)
        if file_id is None:
            continue

        if isinstance(file_id, str):
            parts = [file_id]
        else:
            parts = list(file_id)

        p1 = os.path.join(export_folder, *parts)
        p2 = os.path.join(export_folder, "DICOM", *parts)

        if os.path.exists(p1):
            image_paths.append(p1)
        elif os.path.exists(p2):
            image_paths.append(p2)

    return image_paths


def read_ct_headers_grouped_by_series(export_folder: str):
    """
    Reads all DICOM files referenced by DICOMDIR and groups CT slices by SeriesInstanceUID.
    Returns:
      series_dict: uid -> list[Dataset]
    """
    image_paths = get_dicomdir_image_paths(export_folder)

    if not image_paths:
        raise ValueError("No IMAGE paths found in DICOMDIR.")

    series_dict = defaultdict(list)

    for path in image_paths:
        try:
            ds = pydicom.dcmread(path, force=True, stop_before_pixels=False)

            if getattr(ds, "Modality", "") != "CT":
                continue

            if not hasattr(ds, "PixelData"):
                continue

            series_uid = getattr(ds, "SeriesInstanceUID", None)
            if series_uid is None:
                continue

            series_dict[str(series_uid)].append(ds)

        except Exception:
            continue

    return series_dict


def sort_slices(datasets):
    """
    Sort slices by ImagePositionPatient along slice normal if possible,
    otherwise by z-component, otherwise InstanceNumber.
    """
    if not datasets:
        return datasets

    rep = datasets[0]
    iop = getattr(rep, "ImageOrientationPatient", None)

    if iop is not None:
        try:
            iop = np.array(list(map(float, iop)), dtype=float)
            row = iop[:3]
            col = iop[3:]
            normal = np.cross(row, col)
            normal = normal / (np.linalg.norm(normal) + 1e-12)

            def sort_key(ds):
                ipp = getattr(ds, "ImagePositionPatient", None)
                if ipp is not None:
                    try:
                        ipp = np.array(list(map(float, ipp)), dtype=float)
                        return float(np.dot(ipp, normal))
                    except Exception:
                        pass

                if hasattr(ds, "InstanceNumber"):
                    try:
                        return float(ds.InstanceNumber)
                    except Exception:
                        pass

                return 0.0

            return sorted(datasets, key=sort_key)

        except Exception:
            pass

    def fallback_sort_key(ds):
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None:
            try:
                return float(ipp[2])
            except Exception:
                pass

        if hasattr(ds, "InstanceNumber"):
            try:
                return float(ds.InstanceNumber)
            except Exception:
                pass

        return 0.0

    return sorted(datasets, key=fallback_sort_key)


def build_volume(datasets):
    """
    Build 3D HU volume from sorted CT slices.
    Returns:
      volume: [z, y, x]
      z_positions_along_normal: array of slice coordinates along stack direction
      datasets_sorted
    """
    datasets = sort_slices(datasets)

    volume = np.stack([ds.pixel_array for ds in datasets], axis=0).astype(np.float32)

    slope = float(getattr(datasets[0], "RescaleSlope", 1.0))
    intercept = float(getattr(datasets[0], "RescaleIntercept", 0.0))
    volume = volume * slope + intercept

    rep = datasets[0]
    iop = getattr(rep, "ImageOrientationPatient", None)

    z_positions = []
    if iop is not None:
        try:
            iop = np.array(list(map(float, iop)), dtype=float)
            row = iop[:3]
            col = iop[3:]
            normal = np.cross(row, col)
            normal = normal / (np.linalg.norm(normal) + 1e-12)

            for ds in datasets:
                ipp = getattr(ds, "ImagePositionPatient", None)
                if ipp is not None:
                    try:
                        ipp = np.array(list(map(float, ipp)), dtype=float)
                        z_positions.append(float(np.dot(ipp, normal)))
                    except Exception:
                        pass
        except Exception:
            z_positions = []

    if not z_positions:
        for ds in datasets:
            if hasattr(ds, "ImagePositionPatient"):
                try:
                    z_positions.append(float(ds.ImagePositionPatient[2]))
                except Exception:
                    pass

    z_positions = np.array(z_positions, dtype=float) if z_positions else None

    return volume, z_positions, datasets


def infer_phase_name(desc: str) -> str:
    text = (desc or "").lower()

    if "nativ" in text or "plain" in text or ("non" in text and "contrast" in text):
        return "Non-contrast"
    if "art" in text or "arter" in text:
        return "Arterial"
    if "ven" in text or "portal" in text:
        return "Venous"
    if "delay" in text or "late" in text or "4 min" in text or "5 min" in text:
        return "Delayed"

    return desc if desc else "Unknown"


def coronal_view_aligned(volume: np.ndarray, z_positions: np.ndarray, common_z: np.ndarray) -> np.ndarray:
    """
    Build a coronal view [z, x] from the middle y-plane and resample it onto a common z-axis.
    Missing regions outside the native z-range are set to NaN.
    """
    y_mid = volume.shape[1] // 2
    cor = volume[:, y_mid, :]  # [z, x]

    if z_positions is None or len(z_positions) != cor.shape[0]:
        raise ValueError("Invalid z_positions for coronal alignment.")

    order = np.argsort(z_positions)
    z = z_positions[order]
    cor = cor[order, :]

    aligned = np.full((len(common_z), cor.shape[1]), np.nan, dtype=np.float32)

    for j in range(cor.shape[1]):
        aligned[:, j] = np.interp(
            common_z,
            z,
            cor[:, j],
            left=np.nan,
            right=np.nan,
        )

    # put superior at top
    aligned = np.flipud(aligned)
    return aligned


def sagittal_view_aligned(volume: np.ndarray, z_positions: np.ndarray, common_z: np.ndarray) -> np.ndarray:
    """
    Build a sagittal view [z, y] from the middle x-plane and resample it onto a common z-axis.
    Missing regions outside the native z-range are set to NaN.
    """
    x_mid = volume.shape[2] // 2
    sag = volume[:, :, x_mid]  # [z, y]

    if z_positions is None or len(z_positions) != sag.shape[0]:
        raise ValueError("Invalid z_positions for sagittal alignment.")

    order = np.argsort(z_positions)
    z = z_positions[order]
    sag = sag[order, :]

    aligned = np.full((len(common_z), sag.shape[1]), np.nan, dtype=np.float32)

    for j in range(sag.shape[1]):
        aligned[:, j] = np.interp(
            common_z,
            z,
            sag[:, j],
            left=np.nan,
            right=np.nan,
        )

    # put superior at top
    aligned = np.flipud(aligned)
    return aligned


def summarize_series(series_uid: str, datasets):
    rep = datasets[0]

    desc = str(getattr(rep, "SeriesDescription", "") or "").strip()
    protocol = str(getattr(rep, "ProtocolName", "") or "").strip()
    series_no = getattr(rep, "SeriesNumber", None)
    for_uid = str(getattr(rep, "FrameOfReferenceUID", "") or "").strip()
    iop = getattr(rep, "ImageOrientationPatient", None)

    thicknesses = []
    for ds in datasets:
        val = getattr(ds, "SliceThickness", None)
        if val is not None:
            try:
                thicknesses.append(float(val))
            except Exception:
                pass
    thicknesses = sorted(set(thicknesses))

    volume, z_positions, datasets = build_volume(datasets)

    coverage = None
    if z_positions is not None and len(z_positions) >= 2:
        coverage = float(np.max(z_positions) - np.min(z_positions))

    return {
        "series_uid": series_uid,
        "series_number": series_no,
        "description": desc,
        "protocol": protocol,
        "phase_name": infer_phase_name(desc),
        "frame_of_reference_uid": for_uid,
        "image_orientation_patient": list(iop) if iop is not None else None,
        "slice_thicknesses": thicknesses,
        "n_slices": volume.shape[0],
        "coverage_mm": coverage,
        "volume": volume,
        "z_positions": z_positions,
        "datasets_sorted": datasets,
    }


def compute_common_z_grid(summaries):
    """
    Compute a shared z-axis for all phases using the union of coverage ranges
    and a representative dz from the available series.
    """
    all_z = []
    dz_candidates = []

    for s in summaries:
        z = s["z_positions"]
        if z is None or len(z) < 2:
            continue

        z = np.sort(z)
        all_z.extend(list(z))

        diffs = np.diff(z)
        diffs = diffs[np.abs(diffs) > 1e-6]
        if len(diffs) > 0:
            dz_candidates.append(float(np.median(np.abs(diffs))))

    if not all_z:
        raise ValueError("No valid z positions found to compute common grid.")

    z_min = float(np.min(all_z))
    z_max = float(np.max(all_z))
    common_dz = float(np.median(dz_candidates)) if dz_candidates else 1.0

    common_z = np.arange(z_min, z_max + common_dz, common_dz, dtype=float)
    return common_z, common_dz, z_min, z_max


def visualize_phases(export_folder: str, min_slices: int = 20):
    """
    Main function:
    - reads DICOMDIR
    - loads all CT series
    - groups by SeriesInstanceUID
    - filters out tiny/non-diagnostic series
    - prints summary
    - shows z-aligned coronal views
    - shows z-aligned sagittal views
    - shows z-coverage plot
    """
    series_dict = read_ct_headers_grouped_by_series(export_folder)

    if not series_dict:
        print("No CT series found")
        return

    summaries = []
    for series_uid, datasets in series_dict.items():
        try:
            summary = summarize_series(series_uid, datasets)
            summaries.append(summary)
        except Exception as e:
            print(f"Skipping series {series_uid}: {e}")

    summaries = [s for s in summaries if s["n_slices"] >= min_slices]

    if not summaries:
        print("No CT series found after filtering")
        return

    def sort_key(s):
        sn = s["series_number"]
        return (sn is None, sn if sn is not None else 10**9)

    summaries = sorted(summaries, key=sort_key)

    print("\nDetected CT series:\n")
    for s in summaries:
        th = s["slice_thicknesses"]
        th_str = "unknown" if not th else (f"{th[0]} mm" if len(th) == 1 else f"varies {th} mm")

        cov = s["coverage_mm"]
        cov_str = "unknown" if cov is None else f"{cov:.1f} mm"

        print(f"Series {s['series_number'] if s['series_number'] is not None else '?':>4} | {s['description'] or 'Unknown'}")
        print(f"  Interpreted phase: {s['phase_name']}")
        print(f"  Slices: {s['n_slices']}")
        print(f"  Slice thickness: {th_str}")
        print(f"  Z coverage: {cov_str}")
        print(f"  FrameOfReferenceUID: {s['frame_of_reference_uid'] or '(missing)'}")
        print(f"  ImageOrientationPatient: {s['image_orientation_patient']}")
        print()

    common_z, common_dz, z_min, z_max = compute_common_z_grid(summaries)
    print(f"Common z-grid: {z_min:.2f} to {z_max:.2f} mm, step ~{common_dz:.3f} mm")
    print()

    n = len(summaries)

    # Coronal views aligned to the same z-axis
    fig1, axes1 = plt.subplots(1, n, figsize=(5 * n, 8))
    if n == 1:
        axes1 = [axes1]

    for ax, s in zip(axes1, summaries):
        cor = coronal_view_aligned(
            s["volume"],
            s["z_positions"],
            common_z=common_z,
        )
        ax.imshow(cor, cmap="gray", aspect="auto", vmin=-200, vmax=300)
        ax.set_title(
            f"{s['phase_name']}\n"
            f"{s['description']}\n"
            f"{s['n_slices']} slices"
        )
        ax.axis("off")

    fig1.suptitle("Coronal views of CT phases aligned to common z-axis", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Sagittal views aligned to the same z-axis
    fig2, axes2 = plt.subplots(1, n, figsize=(5 * n, 8))
    if n == 1:
        axes2 = [axes2]

    for ax, s in zip(axes2, summaries):
        sag = sagittal_view_aligned(
            s["volume"],
            s["z_positions"],
            common_z=common_z,
        )
        ax.imshow(sag, cmap="gray", aspect="auto", vmin=-200, vmax=300)
        ax.set_title(
            f"{s['phase_name']}\n"
            f"{s['description']}\n"
            f"{s['n_slices']} slices"
        )
        ax.axis("off")

    fig2.suptitle("Sagittal views of CT phases aligned to common z-axis", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Z coverage plot
    plt.figure(figsize=(10, max(4, 0.8 * len(summaries))))
    y_labels = []
    y_positions = []

    for i, s in enumerate(summaries):
        z = s["z_positions"]
        if z is None or len(z) < 2:
            continue

        zmin = float(np.min(z))
        zmax = float(np.max(z))
        plt.plot([zmin, zmax], [i, i], linewidth=6)

        y_labels.append(f"{s['phase_name']} | {s['description']}")
        y_positions.append(i)

    plt.yticks(y_positions, y_labels)
    plt.xlabel("Z position (mm)")
    plt.title("Z-coverage of phases")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    export_folder = r"C:\Users\vanja.svenda\Documents\GitHub\bias-correction\16-komplet-export"
    visualize_phases(export_folder, min_slices=20)