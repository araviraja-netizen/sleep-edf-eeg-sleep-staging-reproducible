"""Strict official Sleep-EDF filename parsing and study-qualified identities."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re

PATTERN = re.compile(
    r"(?P<prefix>SC4|ST7)(?P<subject>\d{2})(?P<night>[12])"
    r"(?P<setup>[EFGJ])(?P<marker>[0A-Z])-(?P<kind>PSG|Hypnogram)\.edf"
)


@dataclass(frozen=True)
class Identity:
    study: str
    subject_id: str
    night: int
    recording_id: str
    kind: str
    marker: str


def parse_filename(name: str) -> Identity:
    match = PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Not an official Sleep-EDF filename: {name}")
    prefix, setup, kind, marker = (match[k] for k in ("prefix", "setup", "kind", "marker"))
    if setup not in {"SC4": "EFG", "ST7": "J"}[prefix]:
        raise ValueError(f"Incorrect study/acquisition code: {name}")
    if (kind == "PSG" and marker != "0") or (kind == "Hypnogram" and marker == "0"):
        raise ValueError(f"Incorrect PSG marker or hypnogram scorer: {name}")
    return Identity(prefix[:2], prefix + match["subject"], int(match["night"]),
                    name[:7], kind, marker)


@dataclass(frozen=True)
class RecordingPair:
    identity: Identity
    psg: tuple[Path, ...]
    hypnogram: tuple[Path, ...]

    @property
    def paired(self) -> bool:
        return len(self.psg) == len(self.hypnogram) == 1


def discover(root: Path) -> tuple[list[RecordingPair], list[dict]]:
    """Include unpaired and duplicate files as explicit issues, never choose silently."""
    groups = defaultdict(lambda: {"PSG": [], "Hypnogram": []})
    identities = {}
    issues = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".edf":
            continue
        relative = str(path.relative_to(root))
        try:
            identity = parse_filename(path.name)
        except ValueError as exc:
            issues.append({"file": relative, "issue": str(exc)})
            continue
        expected_folder = {"SC": "sleep-cassette", "ST": "sleep-telemetry"}[identity.study]
        if path.relative_to(root).parts[0] != expected_folder:
            issues.append({"file": relative, "issue": "unexpected study directory"})
        groups[identity.recording_id][identity.kind].append(path)
        identities[identity.recording_id] = identity
    pairs = []
    for key, group in sorted(groups.items()):
        pair = RecordingPair(identities[key], tuple(group["PSG"]), tuple(group["Hypnogram"]))
        pairs.append(pair)
        if not pair.paired:
            issues.append({"recording_id": key, "issue": "missing or ambiguous pairing",
                           "psg_count": len(pair.psg), "hypnogram_count": len(pair.hypnogram)})
    return pairs, issues


def subject_groups(pairs: list[RecordingPair]) -> dict[str, list[str]]:
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair.identity.subject_id].append(pair.identity.recording_id)
    return {key: sorted(values) for key, values in sorted(groups.items())}
