"""ECTSum dataset loader.

Loads the upstream ECTSum dataset from HuggingFace and returns a `DatasetDict`
with `train` / `validation` / `test` splits whose rows have a normalised
`transcript` + `summary` schema.

Defensive about upstream column naming: ECTSum mirrors on the Hub vary in
whether they use `text`/`summary` vs `transcript`/`summary` vs other aliases.
`_normalize_columns` probes a small set of known names and renames into our
canonical pair, raising a clear error if neither side can be found.

Implemented in Weekend 1.
"""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

from earningslora.config import get_settings
from earningslora.data.format import build_record

logger = logging.getLogger(__name__)

_TRANSCRIPT_ALIASES = ("transcript", "text", "article", "document", "input", "source")
_SUMMARY_ALIASES = ("summary", "highlights", "target", "output", "bullet_summary", "summaries")


def _normalize_columns(dataset: Dataset) -> Dataset:
    """Rename upstream columns into the canonical (transcript, summary) pair."""
    cols = dataset.column_names
    transcript_col = next((c for c in _TRANSCRIPT_ALIASES if c in cols), None)
    summary_col = next((c for c in _SUMMARY_ALIASES if c in cols), None)
    if transcript_col is None or summary_col is None:
        raise ValueError(
            f"Could not find transcript/summary columns in {cols}. "
            f"Expected one of {_TRANSCRIPT_ALIASES} and one of {_SUMMARY_ALIASES}."
        )

    rename_map = {}
    if transcript_col != "transcript":
        rename_map[transcript_col] = "transcript"
    if summary_col != "summary":
        rename_map[summary_col] = "summary"
    if rename_map:
        dataset = dataset.rename_columns(rename_map)

    keep = {"transcript", "summary"}
    drop = [c for c in dataset.column_names if c not in keep]
    if drop:
        dataset = dataset.remove_columns(drop)
    return dataset


def _filter_invalid(dataset: Dataset) -> Dataset:
    """Drop rows with empty transcripts or summaries."""
    return dataset.filter(
        lambda row: bool(row["transcript"] and row["transcript"].strip())
        and bool(row["summary"] and row["summary"].strip())
    )


def load_ectsum(
    dataset_id: str | None = None,
    revision: str | None = None,
) -> DatasetDict:
    """Load ECTSum from the Hub, normalised to a (transcript, summary) schema.

    If no `validation` split exists upstream, one is carved from `train`
    deterministically (5% slice with `seed=42`).
    """
    settings = get_settings()
    dataset_id = dataset_id or settings.upstream_dataset

    raw = load_dataset(dataset_id, revision=revision)
    splits = list(raw.keys())
    logger.info("Loaded %s — splits: %s", dataset_id, splits)

    out: dict[str, Dataset] = {}
    for split in splits:
        ds = _normalize_columns(raw[split])
        ds = _filter_invalid(ds)
        out[split] = ds

    if "validation" not in out and "train" in out:
        carved = out["train"].train_test_split(test_size=0.05, seed=42)
        out["train"] = carved["train"]
        out["validation"] = carved["test"]
        logger.info("Carved validation split from train (5%%, seed=42)")

    return DatasetDict(out)


def make_holdout(
    dataset_dict: DatasetDict,
    seed: int | None = None,
    size: int | None = None,
) -> Dataset:
    """Deterministically carve a frozen N-row hold-out from the `test` split.

    Defaults pull from `Settings.eval_seed` and `Settings.eval_holdout_size`.
    """
    settings = get_settings()
    seed = seed if seed is not None else settings.eval_seed
    size = size if size is not None else settings.eval_holdout_size

    if "test" not in dataset_dict:
        raise ValueError(
            f"Expected a 'test' split for hold-out. Got: {list(dataset_dict.keys())}"
        )
    test = dataset_dict["test"]
    if len(test) <= size:
        logger.warning(
            "Test split has %d rows ≤ requested hold-out size %d; using full test split.",
            len(test),
            size,
        )
        return test
    return test.shuffle(seed=seed).select(range(size))


def to_chat_format(dataset_dict: DatasetDict) -> DatasetDict:
    """Apply `build_record` across all rows.

    Result schema: `{"messages": [{role, content}, ...]}` per row, all original
    columns dropped.
    """

    def _format(row):
        return build_record(row["transcript"], row["summary"]).to_dict()

    out: dict[str, Dataset] = {}
    for split, ds in dataset_dict.items():
        out[split] = ds.map(_format, remove_columns=ds.column_names)
    return DatasetDict(out)


def materialise_local(dataset_dict: DatasetDict, output_dir: Path | str) -> Path:
    """Save the dataset to disk via `save_to_disk`. Returns the output path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))
    return output_dir
