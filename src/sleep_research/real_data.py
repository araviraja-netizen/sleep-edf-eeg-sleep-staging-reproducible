"""Phase 2 loader adapter preserving quality flags; no data regeneration."""

from sleep_edf.dataset import EpochDataset


class ResearchEpochDataset(EpochDataset):
    def __init__(self, project, split, max_open_recordings=128):
        if split not in ("train", "validation"):
            raise PermissionError("Phase 3 permits training/validation EEG only; test remains locked")
        self.split = split
        super().__init__(project, protocol="A", split=split, normalize=True,
                         max_open_recordings=max_open_recordings)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        row = self.index.iloc[index]
        _, metadata, _ = self._recording(sample["recording_id"])
        saved = metadata[int(row.array_row)]
        sample["raw_rail_contact_samples"] = int(saved["raw_rail_contact_samples"])
        sample["filter_boundary_flag"] = int(saved["filter_boundary_flag"])
        sample["quality_flagged"] = int(sample["raw_rail_contact_samples"] > 0 or sample["filter_boundary_flag"] > 0)
        return sample
