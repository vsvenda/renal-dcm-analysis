from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import pydicom


ROOT = Path(__file__).resolve().parent
DICOMDIR_PATH = ROOT / "DICOMDIR"

# Ako ti DICOM fajlovi nisu u ROOT/DICOM, promeni ovde:
DICOM_FOLDER_NAME = "DICOM"


def safe_get(obj, name: str, default=""):
    return getattr(obj, name, default)


def main():
    ds = pydicom.dcmread(str(DICOMDIR_PATH))

    if not hasattr(ds, "DirectoryRecordSequence"):
        raise RuntimeError("Ovaj fajl nema DirectoryRecordSequence (0004,1220). Da li je ovo stvarno DICOMDIR?")

    # Po seriji ćemo skupljati: opis, protokol, broj, modalitet, i listu fajlova
    series_info = {}  # key: series_key -> dict
    series_files = defaultdict(list)  # key -> list[path_parts]

    current_patient = None
    current_study = None
    current_series_key = None

    for rec in ds.DirectoryRecordSequence:
        rtype = safe_get(rec, "DirectoryRecordType", "")

        if rtype == "PATIENT":
            current_patient = {
                "PatientName": str(safe_get(rec, "PatientName", "")),
                "PatientID": str(safe_get(rec, "PatientID", "")),
            }

        elif rtype == "STUDY":
            current_study = {
                "StudyDate": str(safe_get(rec, "StudyDate", "")),
                "StudyDescription": str(safe_get(rec, "StudyDescription", "")),
                "StudyID": str(safe_get(rec, "StudyID", "")),
            }

        elif rtype == "SERIES":
            # Napravi “ključ serije” stabilan čak i ako neki tagovi fale
            series_number = safe_get(rec, "SeriesNumber", None)
            series_uid = str(safe_get(rec, "SeriesInstanceUID", ""))

            # Ako nema UID (nekad), koristi kombinaciju Study+SeriesNumber kao fallback
            current_series_key = series_uid if series_uid else f"STUDY:{safe_get(current_study,'StudyID','')}-SER:{series_number}"

            series_info[current_series_key] = {
                "PatientName": safe_get(current_patient, "PatientName", "") if current_patient else "",
                "PatientID": safe_get(current_patient, "PatientID", "") if current_patient else "",
                "StudyDate": safe_get(current_study, "StudyDate", "") if current_study else "",
                "StudyDescription": safe_get(current_study, "StudyDescription", "") if current_study else "",
                "Modality": str(safe_get(rec, "Modality", "")),
                "SeriesNumber": str(series_number if series_number is not None else ""),
                "SeriesDescription": str(safe_get(rec, "SeriesDescription", "")),
                "ProtocolName": str(safe_get(rec, "ProtocolName", "")),
                "BodyPartExamined": str(safe_get(rec, "BodyPartExamined", "")),
            }

        elif rtype == "IMAGE":
            ref = safe_get(rec, "ReferencedFileID", None)
            if current_series_key and ref:
                # ref je lista delova putanje, npr ["DICOM","00007F1D","...","00001D33","EE12F778"]
                series_files[current_series_key].append(list(ref))

    # Napravi mapiranje “folder (npr 00001D33) -> opis”
    # Folder serije uzimamo kao "pretposlednji element" referencirane putanje
    folder_map = defaultdict(lambda: {
        "SeriesKeys": set(),
        "SeriesDescription": set(),
        "ProtocolName": set(),
        "Modality": set(),
        "SeriesNumber": set(),
        "ExampleFile": None,
        "NumFiles": 0,
    })

    for skey, refs in series_files.items():
        for ref_parts in refs:
            # na kraju je filename, pre toga je folder serije
            if len(ref_parts) < 2:
                continue
            series_folder = ref_parts[-2]  # npr "00001D33"
            filename = ref_parts[-1]

            entry = folder_map[series_folder]
            entry["SeriesKeys"].add(skey)
            info = series_info.get(skey, {})
            entry["SeriesDescription"].add(info.get("SeriesDescription", ""))
            entry["ProtocolName"].add(info.get("ProtocolName", ""))
            entry["Modality"].add(info.get("Modality", ""))
            entry["SeriesNumber"].add(info.get("SeriesNumber", ""))
            entry["NumFiles"] += 1

            if entry["ExampleFile"] is None:
                entry["ExampleFile"] = str(ROOT / DICOM_FOLDER_NAME / Path(*ref_parts[1:])) \
                    if ref_parts[0].upper() == "DICOM" else str(ROOT / Path(*ref_parts))

    # Ispis po folderu, sortiran po imenu foldera
    print("\n=== MAPIRANJE: folder -> (SeriesDescription / ProtocolName / SeriesNumber) ===\n")
    for folder in sorted(folder_map.keys()):
        e = folder_map[folder]

        # očisti prazne stringove u setovima
        def clean(s): return sorted([x for x in s if x and x.strip()])

        sd = clean(e["SeriesDescription"])
        pn = clean(e["ProtocolName"])
        sn = clean(e["SeriesNumber"])
        md = clean(e["Modality"])

        print(f"FOLDER: {folder}")
        print(f"  Modality: {', '.join(md) if md else '(n/a)'}")
        print(f"  SeriesNumber: {', '.join(sn) if sn else '(n/a)'}")
        print(f"  SeriesDescription: {sd[0] if sd else '(n/a)'}" + (f"  [+{len(sd)-1} varijanti]" if len(sd) > 1 else ""))
        print(f"  ProtocolName: {pn[0] if pn else '(n/a)'}" + (f"  [+{len(pn)-1} varijanti]" if len(pn) > 1 else ""))
        print(f"  #Files (prema DICOMDIR): {e['NumFiles']}")
        print(f"  Example referenced file: {e['ExampleFile']}")
        print()

    # Bonus: folderi koji fizički postoje u DICOM tree-u, a nisu u DICOMDIR (ili obrnuto)
    # (ovo pomaže da uhvatiš nepodudarnosti)
    dicom_root = ROOT / DICOM_FOLDER_NAME
    physical_series_folders = set()
    if dicom_root.exists():
        # heuristika: folder serije je obično 4-8 heks cifara (npr 00001D33) i sadrži više fajlova bez ekstenzije
        for p in dicom_root.rglob("*"):
            if p.is_dir() and len(p.name) == 8 and all(c in "0123456789ABCDEFabcdef" for c in p.name):
                physical_series_folders.add(p.name)

    missing_in_dir = sorted(list(physical_series_folders - set(folder_map.keys())))
    missing_on_disk = sorted(list(set(folder_map.keys()) - physical_series_folders))

    if missing_in_dir:
        print("Folderi koji postoje na disku, ali nisu viđeni u DICOMDIR (moguće MPR/extra):")
        print("  ", ", ".join(missing_in_dir))
    if missing_on_disk:
        print("Folderi koje DICOMDIR referencira, ali ih nema na disku (nepotpun download/export):")
        print("  ", ", ".join(missing_on_disk))


if __name__ == "__main__":
    main()