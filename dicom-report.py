import os
import re
import json
from dataclasses import dataclass, asdict
from typing import Optional

import matplotlib.pyplot as plt


# ============================================================
# CONFIG / HEURISTICS
# ============================================================

TXT_EXTENSIONS = (".txt",)

# tolerance for saying slice thickness is "the same"
THICKNESS_TOL_MM = 0.05

# Some reports use "same patient coordinate frame" wording;
# we interpret that as registered/aligned at metadata level.
REGISTERED_PATTERNS = [
    r"all kept series share the same frameofreferenceuid",
    r"all folders share the same frameofreferenceuid",
]

# Conservative phase-name heuristics.
# We use BOTH folder name and series description text.
PHASE_PATTERNS = {
    "NC": [
        r"\bnc\b",
        r"\bnativn",        # nativno / nativna / nativni
        r"\bnative\b",
        r"\bnon[\s\-]?contrast\b",
        r"\bplain\b",
        r"\bunenhanced\b",
        r"\bwithout contrast\b",
        r"\bprecontrast\b",
        r"\bpre[\s\-]?contrast\b",
    ],
    "ART": [
        r"\bart\b",
        r"\barter",
        r"\barterial\b",
        r"\ba\.phase\b",
        r"\bap\b",
        r"\bcorticomedullary\b",
    ],
    "VEN": [
        r"\bven\b",
        r"\bvensk",         # venska / venski
        r"\bvenous\b",
        r"\bportal\b",
        r"\bportal venous\b",
        r"\bpv\b",
        r"\bpvp\b",
        r"\bnephrographic\b",
    ],
    "DELAY": [
        r"\bdelay",
        r"\bdelayed\b",
        r"\blate\b",
        r"\bexcret",
        r"\burographic\b",
        r"\b4\s*min\b",
        r"\b5\s*min\b",
        r"\b6\s*min\b",
        r"\b7\s*min\b",
        r"\b8\s*min\b",
        r"\b10\s*min\b",
    ],
}

# things that should not count as phases if they somehow appear in the text
NON_PHASE_PATTERNS = [
    r"\bscout\b",
    r"\bsurview\b",
    r"\btopogram\b",
    r"\blocalizer\b",
    r"\bprotocol\b",
    r"\bresults?\b",
    r"\breport\b",
    r"\breading\b",
    r"\bdose\b",
    r"\bctdi\b",
    r"\bdlp\b",
    r"\bbolus\b",
    r"\btracker\b",
    r"\bmonitor\b",
]


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class PhaseEntry:
    source_file: str
    case_name: str
    section_name: str          # folder name or series heading
    label_text: str            # text used for classification
    phase_label: str           # NC / ART / VEN / DELAY / OTHER
    slice_thickness_mm: Optional[float]
    frame_of_reference_uid: Optional[str]


@dataclass
class CaseSummary:
    source_file: str
    case_name: str
    phase_count: int
    has_nc: bool
    has_art: bool
    has_ven: bool
    has_delay: bool
    has_nc_art_ven: bool
    registered: bool
    same_slice_width: bool
    nc_art_ven_registered_same_width: bool
    phase_labels: list[str]
    raw_phase_sections: list[str]


# ============================================================
# PARSING HELPERS
# ============================================================

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _extract_case_name(text: str, fallback_path: str) -> str:
    m = re.search(r"^Root folder name:\s*(.+)$", text, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    return os.path.splitext(os.path.basename(fallback_path))[0]


def _extract_registered(text: str) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in REGISTERED_PATTERNS)


def _extract_float_after(label: str, block: str) -> Optional[float]:
    """
    Example lines:
      SliceThickness: 1.5 mm, computed dz~1.0000 mm
      SliceThickness: varies [1.0, 1.5] mm
      SliceThickness: unknown
    We use the first numeric thickness if available.
    """
    m = re.search(rf"{re.escape(label)}\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*mm", block, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _extract_frame_uid(block: str) -> Optional[str]:
    m = re.search(r"FrameOfReferenceUID(?:\(s\))?:\s*(.+)", block)
    if not m:
        return None
    val = m.group(1).strip()
    if "(missing)" in val.lower():
        return None

    # Handle formats like:
    # ['1.2.3...']
    # 1.2.3...
    q = re.search(r"([0-9]+(?:\.[0-9]+)+)", val)
    return q.group(1) if q else None


def _looks_like_non_phase(text: str) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in NON_PHASE_PATTERNS)


def _infer_phase_label(text: str) -> str:
    lower = text.lower()

    if _looks_like_non_phase(lower):
        return "OTHER"

    matches = []
    for label, patterns in PHASE_PATTERNS.items():
        for p in patterns:
            if re.search(p, lower):
                matches.append(label)
                break

    if not matches:
        return "OTHER"

    # Priority if multiple match accidentally
    for preferred in ["NC", "ART", "VEN", "DELAY"]:
        if preferred in matches:
            return preferred
    return matches[0]


# ============================================================
# REPORT PARSERS
# ============================================================

def parse_kept_series_report(path: str, text: str) -> tuple[str, bool, list[PhaseEntry]]:
    """
    Parses reports of the form:

    === KEPT (candidate phases) ===
    - Series ...
      Folder(s): ...
      ...
      SliceThickness: 1.5 mm
      FrameOfReferenceUID: ...
    """
    case_name = _extract_case_name(text, path)
    registered = _extract_registered(text)

    m = re.search(
        r"=== KEPT \(candidate phases\) ===(.*?)(?:=== REMOVED|\Z)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return case_name, registered, []

    kept_block = m.group(1)
    chunks = re.split(r"\n(?=- Series )", kept_block)
    phases = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk.startswith("- Series "):
            continue

        first_line = chunk.splitlines()[0].strip()
        # Example: - Series 501 | ARTERIAL 1.5, iDose (4)
        parts = [x.strip() for x in first_line.split("|")]
        section_name = parts[1] if len(parts) >= 2 else first_line

        thickness = _extract_float_after("SliceThickness", chunk)
        frame_uid = _extract_frame_uid(chunk)

        label_text = section_name
        label = _infer_phase_label(label_text)

        phases.append(
            PhaseEntry(
                source_file=path,
                case_name=case_name,
                section_name=section_name,
                label_text=label_text,
                phase_label=label,
                slice_thickness_mm=thickness,
                frame_of_reference_uid=frame_uid,
            )
        )

    return case_name, registered, phases


def parse_folder_report(path: str, text: str) -> tuple[str, bool, list[PhaseEntry]]:
    """
    Parses reports of the form:

    === Folder: arterijska ... ===
    Files: ...
    SliceThickness: ...
    FrameOfReferenceUID(s): ...
    SeriesDescription(s): [...]
    """
    case_name = _extract_case_name(text, path)
    registered = _extract_registered(text)

    folder_blocks = re.findall(
        r"(=== Folder:\s*.*?===\n.*?)(?=\n=== Folder:|\Z)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    phases = []

    for block in folder_blocks:
        header = re.search(r"=== Folder:\s*(.+?)\s*===", block)
        folder_name = header.group(1).strip() if header else "UNKNOWN_FOLDER"

        sd = re.search(r"SeriesDescription\(s\):\s*(.+)", block)
        series_desc_text = sd.group(1).strip() if sd else ""

        label_text = f"{folder_name} {series_desc_text}".strip()
        label = _infer_phase_label(label_text)

        thickness = _extract_float_after("SliceThickness", block)
        frame_uid = _extract_frame_uid(block)

        phases.append(
            PhaseEntry(
                source_file=path,
                case_name=case_name,
                section_name=folder_name,
                label_text=label_text,
                phase_label=label,
                slice_thickness_mm=thickness,
                frame_of_reference_uid=frame_uid,
            )
        )

    return case_name, registered, phases


def parse_report_file(path: str) -> tuple[str, bool, list[PhaseEntry]]:
    text = _read_text(path)

    if "=== KEPT (candidate phases) ===" in text:
        return parse_kept_series_report(path, text)

    if "=== Folder:" in text:
        return parse_folder_report(path, text)

    case_name = _extract_case_name(text, path)
    return case_name, False, []


# ============================================================
# CASE AGGREGATION
# ============================================================

def _unique_phase_entries(entries: list[PhaseEntry]) -> list[PhaseEntry]:
    """
    Deduplicate obvious duplicates inside a case.
    Uses section_name + phase_label + thickness + FOR UID.
    """
    seen = set()
    out = []
    for e in entries:
        key = (
            e.section_name.strip().lower(),
            e.phase_label,
            None if e.slice_thickness_mm is None else round(e.slice_thickness_mm, 3),
            e.frame_of_reference_uid or "",
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _same_slice_width(entries: list[PhaseEntry]) -> bool:
    vals = [e.slice_thickness_mm for e in entries if e.slice_thickness_mm is not None]
    if len(vals) <= 1:
        return False if len(entries) > 1 else False
    ref = vals[0]
    return all(abs(v - ref) <= THICKNESS_TOL_MM for v in vals)


def summarize_case(source_file: str, case_name: str, registered: bool, entries: list[PhaseEntry]) -> CaseSummary:
    entries = [e for e in entries if e.phase_label != "OTHER"]
    entries = _unique_phase_entries(entries)

    labels = [e.phase_label for e in entries]
    unique_labels = sorted(set(labels), key=lambda x: ["NC", "ART", "VEN", "DELAY", "OTHER"].index(x) if x in ["NC", "ART", "VEN", "DELAY", "OTHER"] else 99)

    has_nc = "NC" in unique_labels
    has_art = "ART" in unique_labels
    has_ven = "VEN" in unique_labels
    has_delay = "DELAY" in unique_labels

    phase_count = len(unique_labels)
    same_slice_width = _same_slice_width(entries)
    has_nc_art_ven = has_nc and has_art and has_ven

    return CaseSummary(
        source_file=source_file,
        case_name=case_name,
        phase_count=phase_count,
        has_nc=has_nc,
        has_art=has_art,
        has_ven=has_ven,
        has_delay=has_delay,
        has_nc_art_ven=has_nc_art_ven,
        registered=registered,
        same_slice_width=same_slice_width,
        nc_art_ven_registered_same_width=(has_nc_art_ven and registered and same_slice_width),
        phase_labels=unique_labels,
        raw_phase_sections=[e.section_name for e in entries],
    )


# ============================================================
# PLOTTING
# ============================================================

def make_bar_chart(labels, values, title, ylabel, out_path):
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def aggregate_dicom_reports(input_dir: str, output_dir: Optional[str] = None):
    input_dir = os.path.abspath(input_dir)
    if output_dir is None:
        output_dir = os.path.join(input_dir, "aggregate_report")
    os.makedirs(output_dir, exist_ok=True)

    txt_files = [
        os.path.join(input_dir, fn)
        for fn in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, fn)) and fn.lower().endswith(TXT_EXTENSIONS)
    ]

    if not txt_files:
        raise ValueError(f"No .txt report files found in: {input_dir}")

    all_phase_entries = []
    case_summaries = []

    for path in sorted(txt_files):
        case_name, registered, entries = parse_report_file(path)
        all_phase_entries.extend(entries)

        case_summary = summarize_case(path, case_name, registered, entries)
        case_summaries.append(case_summary)

    # Counts
    total_cases = len(case_summaries)

    phase_count_bins = {1: 0, 2: 0, 3: 0, 4: 0, "5+": 0}
    for cs in case_summaries:
        if cs.phase_count in [1, 2, 3, 4]:
            phase_count_bins[cs.phase_count] += 1
        elif cs.phase_count >= 5:
            phase_count_bins["5+"] += 1

    n_three_phase_nc_art_ven = sum(cs.has_nc_art_ven for cs in case_summaries)
    n_registered = sum(cs.registered for cs in case_summaries)
    n_same_slice_width = sum(cs.same_slice_width for cs in case_summaries)
    n_three_phase_nc_art_ven_registered_same_width = sum(
        cs.nc_art_ven_registered_same_width for cs in case_summaries
    )

    final_summary = {
        "total_cases": total_cases,
        "phase_count_distribution": phase_count_bins,
        "three_phase_nc_art_ven": n_three_phase_nc_art_ven,
        "registered": n_registered,
        "same_slice_width": n_same_slice_width,
        "three_phase_nc_art_ven_registered_same_width": n_three_phase_nc_art_ven_registered_same_width,
        "cases": [asdict(cs) for cs in case_summaries],
    }

    # Save JSON
    json_path = os.path.join(output_dir, "final_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    # Save TXT
    txt_path = os.path.join(output_dir, "final_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Total cases: {total_cases}\n\n")

        f.write("How many examples have 1 / 2 / 3 / 4 phases:\n")
        f.write(f"  1 phase: {phase_count_bins[1]}\n")
        f.write(f"  2 phases: {phase_count_bins[2]}\n")
        f.write(f"  3 phases: {phase_count_bins[3]}\n")
        f.write(f"  4 phases: {phase_count_bins[4]}\n")
        f.write(f"  5+ phases: {phase_count_bins['5+']}\n\n")

        f.write(f"How many examples with three phases specifically NC / ART / VEN: {n_three_phase_nc_art_ven}\n")
        f.write(f"How many examples where phases are aligned (registered): {n_registered}\n")
        f.write(f"How many examples with same slice width: {n_same_slice_width}\n")
        f.write(
            "How many examples with three phases (NC / ART / VEN) "
            f"that are registered and with same slice width: {n_three_phase_nc_art_ven_registered_same_width}\n\n"
        )

        f.write("Per-case details:\n")
        for cs in case_summaries:
            f.write(
                f"- {cs.case_name}: phases={cs.phase_labels}, "
                f"phase_count={cs.phase_count}, "
                f"registered={cs.registered}, "
                f"same_slice_width={cs.same_slice_width}, "
                f"nc_art_ven={cs.has_nc_art_ven}\n"
            )

    # Charts
    chart1 = os.path.join(output_dir, "phase_count_distribution.png")
    make_bar_chart(
        labels=["1", "2", "3", "4", "5+"],
        values=[
            phase_count_bins[1],
            phase_count_bins[2],
            phase_count_bins[3],
            phase_count_bins[4],
            phase_count_bins["5+"],
        ],
        title="Number of cases by number of phases",
        ylabel="Cases",
        out_path=chart1,
    )

    chart2 = os.path.join(output_dir, "summary_metrics.png")
    make_bar_chart(
        labels=[
            "NC+ART+VEN",
            "Registered",
            "Same slice width",
            "NC+ART+VEN\n+ registered\n+ same width",
        ],
        values=[
            n_three_phase_nc_art_ven,
            n_registered,
            n_same_slice_width,
            n_three_phase_nc_art_ven_registered_same_width,
        ],
        title="Summary metrics across cases",
        ylabel="Cases",
        out_path=chart2,
    )

    # Optional per-phase counts
    phase_presence = {
        "NC": sum(cs.has_nc for cs in case_summaries),
        "ART": sum(cs.has_art for cs in case_summaries),
        "VEN": sum(cs.has_ven for cs in case_summaries),
        "DELAY": sum(cs.has_delay for cs in case_summaries),
    }
    chart3 = os.path.join(output_dir, "phase_presence.png")
    make_bar_chart(
        labels=list(phase_presence.keys()),
        values=list(phase_presence.values()),
        title="How often each phase appears",
        ylabel="Cases",
        out_path=chart3,
    )

    print("Done.")
    print(f"Input folder: {input_dir}")
    print(f"Cases processed: {total_cases}")
    print(f"Summary TXT: {txt_path}")
    print(f"Summary JSON: {json_path}")
    print(f"Charts:")
    print(f"  {chart1}")
    print(f"  {chart2}")
    print(f"  {chart3}")

    return final_summary


if __name__ == "__main__":
    # Put all your DICOM-analysis TXT files into one folder, then point here.
    input_dir = r"C:\Users\vanja.svenda\Documents\GitHub\bias-correction\dicom-analysis"
    aggregate_dicom_reports(input_dir)