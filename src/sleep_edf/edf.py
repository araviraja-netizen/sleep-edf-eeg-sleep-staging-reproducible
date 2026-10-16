"""Small read-only EDF header reader using the published fixed-field layout.

No patient/demographic fields are exported. Timestamps are local EDF wall times,
not UTC. Native sample rates are calculated with exact rational arithmetic.
"""

from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path


@dataclass(frozen=True)
class EDFHeader:
    start: datetime
    records: int
    record_seconds: Fraction
    labels: tuple[str, ...]
    units: tuple[str, ...]
    samples_per_record: tuple[int, ...]
    physical_min: tuple[float, ...]
    physical_max: tuple[float, ...]
    prefilters: tuple[str, ...]
    reserved: str
    header_bytes: int
    file_bytes: int

    @property
    def duration(self) -> Fraction:
        return self.records * self.record_seconds

    @property
    def size_valid(self) -> bool:
        return self.file_bytes == self.header_bytes + self.records * sum(self.samples_per_record) * 2

    def rate(self, channel: str) -> Fraction:
        if self.record_seconds <= 0:
            raise ValueError("Annotation-only EDF has no signal sampling rate")
        return Fraction(self.samples_per_record[self.labels.index(channel)]) / self.record_seconds


def read_header(path: Path) -> EDFHeader:
    with path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) != 256 or fixed[:8].strip() != b"0":
            raise ValueError(f"Unsupported/truncated EDF header: {path.name}")
        ns = int(fixed[252:256])
        header_bytes = int(fixed[184:192])
        if ns <= 0 or header_bytes != 256 * (ns + 1):
            raise ValueError(f"Invalid signal/header size: {path.name}")
        block = handle.read(ns * 256)
        if len(block) != ns * 256:
            raise ValueError(f"Truncated signal header: {path.name}")
    date = fixed[168:176].decode("ascii")
    time = fixed[176:184].decode("ascii")
    day, month, year = map(int, date.split("."))
    hour, minute, second = map(int, time.split("."))
    start = datetime(year + (1900 if year >= 85 else 2000), month, day, hour, minute, second)
    records = int(fixed[236:244])
    seconds = Fraction(fixed[244:252].decode("ascii").strip())
    if records < 0 or seconds < 0:
        raise ValueError(f"Unknown/negative record count or duration: {path.name}")
    offset = 0

    def fields(width):
        nonlocal offset
        values = tuple(block[offset + i * width:offset + (i + 1) * width].decode("ascii").strip()
                       for i in range(ns))
        offset += width * ns
        return values

    labels = fields(16)
    fields(80)  # Transducer descriptions.
    units = fields(8)
    minimum = tuple(map(float, fields(8)))
    maximum = tuple(map(float, fields(8)))
    fields(8)  # Digital min/max not required for this metadata inspection.
    fields(8)
    prefilters = fields(80)
    samples = tuple(map(int, fields(8)))
    if len(set(labels)) != ns or any(n <= 0 for n in samples):
        raise ValueError(f"Duplicate labels or invalid sample counts: {path.name}")
    return EDFHeader(start, records, seconds, labels, units, samples, minimum, maximum,
                     prefilters, fixed[192:236].decode("ascii").strip(), header_bytes, path.stat().st_size)


def verify_common_origin(psg: EDFHeader, hypnogram: EDFHeader) -> None:
    if psg.start != hypnogram.start:
        raise ValueError("PSG/hypnogram header start mismatch; timestamp shifts are forbidden")
    if psg.reserved.startswith("EDF+"):
        # Phase 1 supports the continuous, ordinary EDF PSGs in Sleep-EDF.
        # EDF+ PSGs require a separate TAL timekeeping validation before use.
        raise ValueError("EDF+ PSG timekeeping requires explicit validation")
    if hypnogram.reserved != "EDF+C" or hypnogram.labels != ("EDF Annotations",):
        raise ValueError("Unsupported hypnogram format/timekeeping")
