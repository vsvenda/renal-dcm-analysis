from pathlib import Path
import pydicom
import numpy as np


SERIES_FOLDERS = [
    "0000674D",  # Venous
    "00002C0A",  # Arterial
    "000076FE",  # Delayed
    "0000B43E",  # Non-contrast
]

BASE = Path("DICOM/00007F1D/AA0D871C/AA9E2850")


def analyze_series(folder: Path):
    files = sorted(folder.glob("*"))
    if len(files) < 2:
        return None

    slice_thicknesses = set()
    pixel_spacings = set()
    image_types = set()
    positions = []

    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=True)

        slice_thicknesses.add(ds.get("SliceThickness", None))
        ps = ds.get("PixelSpacing", None)
        if ps is not None:
            pixel_spacings.add(tuple(ps))

        image_types.add(tuple(ds.get("ImageType", [])))

        ipp = ds.get("ImagePositionPatient", None)
        if ipp is not None:
            positions.append(ipp[2])

    positions = sorted(positions)
    z_spacings = np.diff(positions)
    mean_z = float(np.mean(np.abs(z_spacings))) if len(z_spacings) > 0 else None
    std_z = float(np.std(np.abs(z_spacings))) if len(z_spacings) > 0 else None

    return {
        "num_slices": len(files),
        "slice_thickness": slice_thicknesses,
        "pixel_spacing": pixel_spacings,
        "image_types": image_types,
        "mean_z_spacing": mean_z,
        "std_z_spacing": std_z,
    }


def main():
    print("\n=== PROVERA CT SERIJA ===\n")

    for name in SERIES_FOLDERS:
        folder = BASE / name
        if not folder.exists():
            print(f"[SKIP] Folder ne postoji: {name}")
            continue

        result = analyze_series(folder)
        if result is None:
            print(f"[SKIP] Premalo slice-ova u {name}")
            continue

        print(f"SERIES FOLDER: {name}")
        print(f"  #Slices: {result['num_slices']}")
        print(f"  SliceThickness values: {result['slice_thickness']}")
        print(f"  PixelSpacing values: {result['pixel_spacing']}")
        print(f"  ImageType values: {result['image_types']}")
        print(f"  Mean Z spacing: {result['mean_z_spacing']:.4f}")
        print(f"  Std Z spacing: {result['std_z_spacing']:.6f}")

        # Jednostavne validacije
        if len(result["slice_thickness"]) > 1:
            print("  ⚠ WARNING: SliceThickness nije konzistentan!")
        if result["std_z_spacing"] and result["std_z_spacing"] > 1e-3:
            print("  ⚠ WARNING: Z spacing varira!")
        print()

    print("=== KRAJ ===")


if __name__ == "__main__":
    main()