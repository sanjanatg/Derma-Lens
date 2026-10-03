"""Rebuild leakage-aware train/validation/test manifests for FairDermNet.

The master dataset intentionally contains the original HAM10000 and ISIC2019
rows. This script joins their source IDs, collapses their shared image IDs,
groups dermoscopic images by lesion and PAD-UFES images by patient, then writes
a deterministic approximately 70/15/15 grouped-stratified split manifest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


SOURCE_ORDER = ("HAM10000", "ISIC2019", "PAD_UFES_20")
SPLIT_FOLDS = {
    **{fold: "train" for fold in range(14)},
    **{fold: "val" for fold in range(14, 17)},
    **{fold: "test" for fold in range(17, 20)},
}


def _stem(value: object) -> str:
    return Path(str(value)).stem.strip().lower()


def _valid_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    result = str(value).strip()
    return result if result and result.lower() not in {"nan", "none", "null"} else None


def _unique_id_map(frame: pd.DataFrame, key: str, value: str, label: str) -> dict[str, str]:
    pairs = frame[[key, value]].copy()
    pairs[key] = pairs[key].map(_stem)
    pairs[value] = pairs[value].map(_valid_id)
    pairs = pairs.dropna(subset=[key, value]).drop_duplicates()
    conflicts = pairs.groupby(key)[value].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise ValueError(f"Conflicting {label} values for {len(conflicts)} source image IDs")
    return pairs.drop_duplicates(key).set_index(key)[value].to_dict()


def build_manifest(model_root: Path, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    datasets = model_root / "datasets"
    master_path = datasets / "master_dataset.csv"
    if not master_path.is_file():
        raise FileNotFoundError(f"Missing source master CSV: {master_path}")

    master = pd.read_csv(master_path)
    required = {"image_path", "class_idx", "skin_tone", "dataset", "dx_raw"}
    missing = required - set(master.columns)
    if missing:
        raise ValueError(f"master_dataset.csv is missing columns: {sorted(missing)}")

    master["image_id"] = master["image_path"].map(_stem)
    master["class_idx"] = pd.to_numeric(master["class_idx"], errors="raise").astype(int)
    master["skin_tone"] = pd.to_numeric(master["skin_tone"], errors="coerce").fillna(-1).astype(int)

    ham_meta = pd.read_csv(datasets / "HAM10000" / "HAM10000_metadata.csv")
    isic_meta = pd.read_csv(datasets / "ISIC2019" / "ISIC_2019_Training_Metadata.csv")
    pad_meta = pd.read_csv(datasets / "PAD_UFES_20" / "metadata.csv")

    ham_lesions = _unique_id_map(ham_meta, "image_id", "lesion_id", "HAM10000 lesion ID")
    isic_lesions = _unique_id_map(isic_meta, "image", "lesion_id", "ISIC2019 lesion ID")
    pad_patients = _unique_id_map(pad_meta, "img_id", "patient_id", "PAD-UFES patient ID")
    pad_lesions = _unique_id_map(pad_meta, "img_id", "lesion_id", "PAD-UFES lesion ID")

    master["lesion_id"] = None
    master["patient_id"] = None
    for source, lesion_map in (("HAM10000", ham_lesions), ("ISIC2019", isic_lesions)):
        mask = master["dataset"].eq(source)
        master.loc[mask, "lesion_id"] = master.loc[mask, "image_id"].map(lesion_map)

    pad_mask = master["dataset"].eq("PAD_UFES_20")
    master.loc[pad_mask, "patient_id"] = master.loc[pad_mask, "image_id"].map(pad_patients)
    master.loc[pad_mask, "lesion_id"] = master.loc[pad_mask, "image_id"].map(pad_lesions)

    derm_mask = master["dataset"].isin(("HAM10000", "ISIC2019"))
    derm_rows = master.loc[derm_mask]
    # The same official ISIC image ID may appear in both source datasets. A
    # shared source image ID is the strongest identity signal, including when
    # the ISIC lesion ID is absent; HAM metadata then supplies the lesion ID.
    lesion_by_image = (
        derm_rows.dropna(subset=["lesion_id"])
        .groupby("image_id")["lesion_id"]
        .agg(lambda values: sorted(set(map(str, values))))
    )
    bad = lesion_by_image[lesion_by_image.map(len) > 1]
    if not bad.empty:
        raise ValueError(f"Conflicting HAM10000/ISIC2019 lesion IDs for {len(bad)} shared images")
    shared_lesion = lesion_by_image.map(lambda values: values[0]).to_dict()
    shared_lesion_ids = master.loc[derm_mask, "image_id"].map(shared_lesion)
    master.loc[derm_mask, "lesion_id"] = shared_lesion_ids.combine_first(
        master.loc[derm_mask, "lesion_id"]
    )

    unexpected = set(master["dataset"].dropna()) - set(SOURCE_ORDER)
    if unexpected:
        raise ValueError(f"Unexpected dataset labels in master CSV: {sorted(unexpected)}")
    lesion_ids = master["lesion_id"].map(_valid_id)
    patient_ids = master["patient_id"].map(_valid_id)
    master["group_id"] = np.where(
        derm_mask,
        np.where(lesion_ids.notna(), "dermo_lesion_" + lesion_ids.fillna(""),
                 "dermo_image_" + master["image_id"]),
        np.where(patient_ids.notna(), "pad_patient_" + patient_ids.fillna(""),
                 "pad_fallback_" + lesion_ids.fillna(master["image_id"])),
    )

    # Conflicting source labels need adjudication; never silently choose one.
    # HAM/ISIC share image IDs; PAD IDs use their own namespace.
    master["_dedup_key"] = master["image_id"]
    pad_mask = master["dataset"].eq("PAD_UFES_20")
    master.loc[pad_mask, "_dedup_key"] = (
        "PAD_UFES_20:" + master.loc[pad_mask, "image_id"]
    )
    conflicts = master.groupby("_dedup_key")["class_idx"].nunique()
    conflicts = conflicts[conflicts > 1]
    conflict_rows = master[master["_dedup_key"].isin(conflicts.index)].copy()
    conflict_report = conflict_rows[
        ["image_id", "image_path", "class_idx", "dx_raw", "dataset", "lesion_id", "patient_id", "group_id"]
    ].sort_values(["image_id", "dataset"]).reset_index(drop=True)
    master = master[~master["_dedup_key"].isin(conflicts.index)].copy()

    # Collapse HAM/ISIC representations of the same image. PAD image IDs live
    # in a separate namespace. Retain HAM10000's original path when available
    # and record every contributing source in dataset_origins.
    source_rank = {name: rank for rank, name in enumerate(SOURCE_ORDER)}
    origin_map = master.groupby("_dedup_key")["dataset"].agg(
        lambda values: ",".join(sorted(set(values.astype(str)), key=source_rank.__getitem__))
    )
    master["_source_rank"] = master["dataset"].map(source_rank)
    manifest = master.sort_values(["_source_rank", "image_id"]).drop_duplicates(
        "_dedup_key", keep="first"
    ).copy()
    manifest["dataset_origins"] = manifest["_dedup_key"].map(origin_map)
    manifest = manifest.drop(columns=["_dedup_key", "_source_rank"]).reset_index(drop=True)

    # Preserve class and provenance strata while keeping lesion/patient groups
    # indivisible. Twenty folds compose approximately 70/15/15 partitions.
    manifest["strat_key"] = manifest["class_idx"].astype(str) + "_" + manifest["dataset_origins"]
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold_for_row = pd.Series(index=manifest.index, dtype="int64")
    for fold, (_, held_out_idx) in enumerate(
        splitter.split(manifest.index, manifest["strat_key"], manifest["group_id"])
    ):
        fold_for_row.iloc[held_out_idx] = fold
    if fold_for_row.isna().any():
        raise RuntimeError("Grouped stratification failed to assign every row to one fold")
    manifest["split"] = fold_for_row.astype(int).map(SPLIT_FOLDS)

    group_splits = manifest.groupby("group_id")["split"].nunique()
    if group_splits.max() != 1:
        raise RuntimeError("Leakage detected: a lesion/patient group crosses splits")
    image_splits = manifest.groupby("image_id")["split"].nunique()
    if image_splits.max() != 1 or manifest["image_id"].duplicated().any():
        raise RuntimeError("Leakage detected: a duplicate image ID crosses splits")

    manifest = manifest.sort_values(["split", "class_idx", "image_id"]).reset_index(drop=True)
    return manifest, conflict_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_root = args.model_root.resolve()
    output = args.output or (model_root / "datasets" / "fairdermnet_rebuilt_splits.csv")
    manifest, conflict_report = build_manifest(model_root, seed=args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    conflict_output = output.with_name("fairdermnet_label_conflicts.csv")
    conflict_report.to_csv(conflict_output, index=False)

    counts = manifest["split"].value_counts(normalize=True).reindex(["train", "val", "test"])
    print(f"Saved {len(manifest):,} rows to {output}")
    print(f"Excluded {conflict_report['image_id'].nunique():,} conflicting image IDs; report: {conflict_output}")
    print("Split proportions:")
    print((counts * 100).round(2).to_string())
    print("\nClass-index × source provenance:")
    print(pd.crosstab(manifest["class_idx"], manifest["dataset_origins"]).to_string())
    print(f"\nUnique groups: {manifest['group_id'].nunique():,}")
    print(f"Cross-split groups: {int((manifest.groupby('group_id')['split'].nunique() > 1).sum())}")


if __name__ == "__main__":
    main()
