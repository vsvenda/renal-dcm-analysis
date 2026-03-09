# renal-dcm-analysis

DICOM CT Analysis Toolkit

This repository contains a collection of Python scripts for exploring, validating, and analyzing CT DICOM datasets, particularly multi-phase abdominal CT studies  (e.g., non-contrast, arterial, venous, delayed phases). The tools focus on inspecting DICOM exports, understanding the structure of CT series and phases, generating metadata reports for datasets, visualizing CT volumes and phase coverage, etc.


# dicom-analysis.py
Main tool for analyzing CT DICOM folders and producing structured reports.
It recursively scans a directory containing DICOM files, reads relevant metadata, groups slices by SeriesInstanceUID, and summarizes each CT series.

# dicom-analysis-old.py
Used for earlier DICOM exports. Unlike the newer script, this version assumes that each CT phase is already separated into its own folder. It scans each folder independently and summarizes the DICOM metadata for that phase.

# dicom-report.py
Aggregates multiple DICOM analysis reports into a dataset-level summary.

# dicom-gif.py
Creates a quick animated preview of a CT series.
This allows quick visual inspection of a CT series without a full DICOM viewer

# read-dicomdir.py
Parses a DICOMDIR file and reconstructs the dataset hierarchy.
This is especially useful when working with PACS exports that rely on DICOMDIR.

# recreate-ct-scan.py
Loads CT slices from a DICOM export and reconstructs 3D volumes for each series.
This script is useful for visual sanity checks of CT datasets.

# slice-thickness.py
Small utility for checking slice spacing consistency in CT series.

# debug-dicom.py
Diagnostic tool for debugging DICOM exports containing a DICOMDIR.
