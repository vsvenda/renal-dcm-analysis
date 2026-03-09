import os
import pydicom


def debug_export(export_folder: str, max_print: int = 20):
    dicomdir_path = os.path.join(export_folder, "DICOMDIR")
    dicom_root = os.path.join(export_folder, "DICOM")

    print("Export folder:", export_folder)
    print("DICOMDIR exists:", os.path.exists(dicomdir_path))
    print("DICOM folder exists:", os.path.exists(dicom_root))
    print()

    if not os.path.exists(dicomdir_path):
        print("ERROR: DICOMDIR not found.")
        return

    ds = pydicom.dcmread(dicomdir_path, force=True)

    seq = getattr(ds, "DirectoryRecordSequence", None)
    if seq is None:
        print("ERROR: DirectoryRecordSequence not found in DICOMDIR.")
        return

    print("Directory records:", len(seq))
    print()

    n_series = 0
    n_images = 0
    existing_files = 0
    nonzero_files = 0
    readable_dicom = 0
    ct_files = 0
    pixel_files = 0

    current_series_uid = None
    current_series_desc = "Unknown"

    printed = 0

    for rec in seq:
        rec_type = getattr(rec, "DirectoryRecordType", None)

        if rec_type == "SERIES":
            n_series += 1
            current_series_uid = getattr(rec, "SeriesInstanceUID", None)
            current_series_desc = getattr(rec, "SeriesDescription", "Unknown")

        elif rec_type == "IMAGE":
            n_images += 1

            file_id = getattr(rec, "ReferencedFileID", None)
            if file_id is None:
                continue

            if isinstance(file_id, str):
                parts = [file_id]
            else:
                parts = list(file_id)

            # Try both with and without "DICOM" prefix
            p1 = os.path.join(export_folder, *parts)
            p2 = os.path.join(export_folder, "DICOM", *parts)

            path = None
            if os.path.exists(p1):
                path = p1
            elif os.path.exists(p2):
                path = p2

            if path is None:
                if printed < max_print:
                    print("MISSING FILE")
                    print("  Series:", current_series_desc)
                    print("  ReferencedFileID:", parts)
                    print("  Tried:", p1)
                    print("  Tried:", p2)
                    print()
                    printed += 1
                continue

            existing_files += 1

            size = os.path.getsize(path)
            if size > 0:
                nonzero_files += 1

            if printed < max_print:
                print("FOUND FILE")
                print("  Series:", current_series_desc)
                print("  Path:", path)
                print("  Size:", size, "bytes")
                print()
                printed += 1

            try:
                img = pydicom.dcmread(path, force=True, stop_before_pixels=False)
                readable_dicom += 1

                modality = getattr(img, "Modality", None)
                if modality == "CT":
                    ct_files += 1

                if hasattr(img, "PixelData"):
                    pixel_files += 1

            except Exception as e:
                if printed < max_print:
                    print("READ ERROR:", path)
                    print(" ", repr(e))
                    print()
                    printed += 1

    print("========== SUMMARY ==========")
    print("SERIES records:", n_series)
    print("IMAGE records:", n_images)
    print("Existing referenced files:", existing_files)
    print("Non-zero referenced files:", nonzero_files)
    print("Readable DICOM files:", readable_dicom)
    print("CT files:", ct_files)
    print("Files with PixelData:", pixel_files)


if __name__ == "__main__":
    export_folder = r"C:\Users\vanja.svenda\Documents\GitHub\bias-correction\16-komplet-export"
    debug_export(export_folder)