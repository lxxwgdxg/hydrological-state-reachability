"""Recompute the retained CEE statistics from compact derived tables."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs" / "reproduced_statistics.json"
SEEDS = (11, 29, 47)
CONDITIONS = ("zero_L365", "zero_L730", "zero_L1095", "smax_L365")
N_BOOTSTRAP = 10_000


def interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def indexed_arrays(window: pd.DataFrame, retrain: pd.DataFrame, basins: list[str]) -> dict[str, Any]:
    basin_index = {basin: index for index, basin in enumerate(basins)}
    arrays: dict[str, Any] = {"window": {}, "retrain": {}}
    for condition in CONDITIONS:
        for seed in SEEDS:
            subset = window[(window.condition == condition) & (window.seed == seed)].copy()
            subset["order"] = subset.basin.map(basin_index)
            subset = subset.sort_values("order")
            if subset.basin.tolist() != basins:
                raise RuntimeError(f"Window basin mismatch: {condition}, seed {seed}")
            arrays["window"][(condition, seed)] = {
                "negative": subset.negative_day_fraction.to_numpy(float),
                "deficit": subset.maximum_deficit_mm.to_numpy(float),
                "decline": subset.nse_decline.to_numpy(float),
            }
    for seed in SEEDS:
        subset = retrain[retrain.seed == seed].copy()
        subset["basin"] = subset.basin.astype(str).str.zfill(8)
        subset["order"] = subset.basin.map(basin_index)
        subset = subset.sort_values("order")
        if subset.basin.tolist() != basins:
            raise RuntimeError(f"Retraining basin mismatch: seed {seed}")
        arrays["retrain"][seed] = {
            "eligible": subset.eligible_recovery_case.astype(bool).to_numpy(),
            "recovery": subset.recovery_fraction.to_numpy(float),
            "gap": subset.paired_nse_gap_unconstrained_minus_feasible.to_numpy(float),
        }
    return arrays


def summarize_core(arrays: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    zero_seed = []
    retrain_seed = []
    for seed in SEEDS:
        zero = arrays["window"][("zero_L365", seed)]
        zero_seed.append(
            {
                "prevalence": float(np.mean(zero["negative"][indices] >= 0.01)),
                "deficit": float(np.median(zero["deficit"][indices])),
                "decline": float(np.median(zero["decline"][indices])),
            }
        )
        retrain = arrays["retrain"][seed]
        eligible = retrain["eligible"][indices]
        recovery_values = retrain["recovery"][indices][eligible]
        recovery = float(np.median(recovery_values)) if recovery_values.size else math.nan
        gap = float(np.median(retrain["gap"][indices]))
        retrain_seed.append(
            {
                "recovery": recovery,
                "gap": gap,
                "avoidable": bool(math.isfinite(recovery) and recovery >= 0.80 and gap <= 0.05),
            }
        )

    extended_counts = {}
    for condition in ("zero_L730", "zero_L1095"):
        passes = 0
        for seed in SEEDS:
            data = arrays["window"][(condition, seed)]
            gates = (
                np.mean(data["negative"][indices] >= 0.01) >= 0.75,
                np.median(data["negative"][indices]) >= 0.50,
                np.median(data["deficit"][indices]) >= 1.0,
                np.median(data["decline"][indices]) >= 0.05,
            )
            passes += int(all(gates))
        extended_counts[condition] = passes

    smax_passes = 0
    for seed in SEEDS:
        data = arrays["window"][("smax_L365", seed)]
        gates = (
            np.mean(data["negative"][indices] >= 0.01) >= 0.50,
            np.median(data["deficit"][indices]) >= 1.0,
        )
        smax_passes += int(all(gates))

    avoidable_count = sum(row["avoidable"] for row in retrain_seed)
    return {
        "same_checkpoint_cross_seed_median_prevalence": float(
            np.median([row["prevalence"] for row in zero_seed])
        ),
        "same_checkpoint_cross_seed_median_maximum_deficit_mm": float(
            np.median([row["deficit"] for row in zero_seed])
        ),
        "same_checkpoint_cross_seed_median_nse_decline": float(
            np.median([row["decline"] for row in zero_seed])
        ),
        "retraining_cross_seed_median_recovery_fraction": float(
            np.nanmedian([row["recovery"] for row in retrain_seed])
        ),
        "retraining_cross_seed_median_paired_nse_gap": float(
            np.median([row["gap"] for row in retrain_seed])
        ),
        "avoidable_shortcut_seed_count": int(avoidable_count),
        "avoidable_shortcut_decision": bool(avoidable_count >= 2),
        "extended_history_seed_pass_counts": extended_counts,
        "generous_initial_storage_seed_pass_count": int(smax_passes),
        "window_initialization_decision": bool(
            all(count >= 2 for count in extended_counts.values()) and smax_passes >= 2
        ),
    }


def reproduce_core() -> dict[str, Any]:
    window = pd.read_csv(DATA / "window_panel_per_basin.csv", dtype={"basin": str})
    window["basin"] = window.basin.str.zfill(8)
    retrain = pd.read_csv(DATA / "feasible_retraining_per_basin.csv", dtype={"basin": str})
    basins = sorted(window.basin.unique().tolist())
    if len(basins) != 24 or len(window) != 4 * 3 * 24 or len(retrain) != 3 * 24:
        raise RuntimeError("Unexpected core-panel inventory")
    arrays = indexed_arrays(window, retrain, basins)
    full = np.arange(len(basins), dtype=int)
    observed = summarize_core(arrays, full)

    metric_names = (
        "same_checkpoint_cross_seed_median_prevalence",
        "same_checkpoint_cross_seed_median_maximum_deficit_mm",
        "same_checkpoint_cross_seed_median_nse_decline",
        "retraining_cross_seed_median_recovery_fraction",
        "retraining_cross_seed_median_paired_nse_gap",
    )
    boot = {name: np.empty(N_BOOTSTRAP) for name in metric_names}
    avoidable = np.empty(N_BOOTSTRAP, dtype=bool)
    window_decision = np.empty(N_BOOTSTRAP, dtype=bool)
    rng = np.random.Generator(np.random.PCG64(20260830))
    for replicate in range(N_BOOTSTRAP):
        summary = summarize_core(arrays, rng.integers(0, len(basins), size=len(basins)))
        for name in metric_names:
            boot[name][replicate] = summary[name]
        avoidable[replicate] = summary["avoidable_shortcut_decision"]
        window_decision[replicate] = summary["window_initialization_decision"]

    loo_rows = []
    for omitted in range(len(basins)):
        loo_rows.append(summarize_core(arrays, full[full != omitted]))
    loo = pd.DataFrame(loo_rows)
    bootstrap = {
        name: {"observed": observed[name], "percentile_95_interval": interval(values)}
        for name, values in boot.items()
    }
    bootstrap["avoidable_shortcut_decision_fraction"] = float(np.mean(avoidable))
    bootstrap["window_initialization_decision_fraction"] = float(np.mean(window_decision))
    return {
        "observed": observed,
        "basin_block_bootstrap": bootstrap,
        "leave_one_basin_out": {
            "deletion_count": int(len(loo)),
            "same_checkpoint_cross_seed_median_nse_decline_range": [
                float(loo.same_checkpoint_cross_seed_median_nse_decline.min()),
                float(loo.same_checkpoint_cross_seed_median_nse_decline.max()),
            ],
            "retraining_cross_seed_median_recovery_fraction_range": [
                float(loo.retraining_cross_seed_median_recovery_fraction.min()),
                float(loo.retraining_cross_seed_median_recovery_fraction.max()),
            ],
            "retraining_cross_seed_median_paired_nse_gap_range": [
                float(loo.retraining_cross_seed_median_paired_nse_gap.min()),
                float(loo.retraining_cross_seed_median_paired_nse_gap.max()),
            ],
            "avoidable_shortcut_decision_preserved_count": int(
                loo.avoidable_shortcut_decision.astype(bool).sum()
            ),
            "window_initialization_decision_preserved_count": int(
                loo.window_initialization_decision.astype(bool).sum()
            ),
        },
    }


def external_per_seed(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    result = {}
    for seed in SEEDS:
        subset = frame[frame.seed == seed]
        high = subset[subset.high_skill_original_path]
        result[str(seed)] = {
            "expected_eligible_basin_count": 506,
            "complete_basin_count": int((subset.status == "complete").sum()),
            "technical_failure_count": int((subset.status != "complete").sum()),
            "material_negative_storage_basin_count": int(subset.material_negative_storage.sum()),
            "material_negative_storage_basin_fraction_of_frozen_eligible": float(
                subset.material_negative_storage.mean()
            ),
            "high_skill_basin_count": int(len(high)),
            "high_skill_median_paired_nse_decline": float(high.nse_decline.median()),
            "high_skill_fraction_with_nse_decline": float(
                high.nse_declines_under_ordered_path.mean()
            ),
            "all_basin_original_nse_median": float(subset.original_nse.median()),
            "all_basin_ordered_nse_median": float(subset.reachable_nse.median()),
            "all_basin_paired_nse_decline_median": float(subset.nse_decline.median()),
        }
    return result


def basin_block_bootstrap(frame: pd.DataFrame) -> dict[str, Any]:
    basin_ids = np.array(sorted(frame.basin_id.unique()))
    draws = np.random.Generator(np.random.PCG64(20260829)).integers(
        0, len(basin_ids), size=(N_BOOTSTRAP, len(basin_ids))
    )
    seed_statistics = []
    observed = {}
    for seed in SEEDS:
        ordered = frame[frame.seed == seed].set_index("basin_id").loc[basin_ids]
        decline = ordered.nse_decline.to_numpy(float)[draws]
        high = ordered.high_skill_original_path.to_numpy(bool)[draws]
        seed_statistics.append(np.nanmedian(np.where(high, decline, np.nan), axis=1))
        observed[str(seed)] = float(
            frame[(frame.seed == seed) & frame.high_skill_original_path].nse_decline.median()
        )
    statistic = np.nanmedian(np.column_stack(seed_statistics), axis=1)
    lower, upper = np.quantile(statistic, [0.025, 0.975])
    return {
        "unit": "basin_id with all seed records retained",
        "replicates": N_BOOTSTRAP,
        "rng": "NumPy PCG64",
        "seed": 20260829,
        "observed_seed_medians": observed,
        "observed_median_of_seed_medians": float(np.median(list(observed.values()))),
        "percentile_95_interval": [float(lower), float(upper)],
        "lower_bound_above_zero": bool(lower > 0),
    }


def equal_region_block_bootstrap(frame: pd.DataFrame) -> dict[str, Any]:
    regions = np.array(sorted(frame.huc_02.unique()))
    region_seed_medians: dict[int, dict[str, float]] = {}
    for seed in SEEDS:
        region_seed_medians[seed] = {}
        for region in regions:
            high = frame[
                (frame.seed == seed) & (frame.huc_02 == region) & frame.high_skill_original_path
            ]
            if len(high) >= 5:
                region_seed_medians[seed][region] = float(high.nse_decline.median())
    draws = np.random.Generator(np.random.PCG64(20260829)).integers(
        0, len(regions), size=(N_BOOTSTRAP, len(regions))
    )
    statistics = np.empty(N_BOOTSTRAP)
    for index, draw in enumerate(draws):
        sampled = regions[draw]
        seed_medians = []
        for seed in SEEDS:
            values = [region_seed_medians[seed][region] for region in sampled if region in region_seed_medians[seed]]
            seed_medians.append(float(np.median(values)) if values else np.nan)
        statistics[index] = float(np.nanmedian(seed_medians))
    observed = {
        str(seed): float(np.median(list(region_seed_medians[seed].values()))) for seed in SEEDS
    }
    lower, upper = np.quantile(statistics, [0.025, 0.975])
    return {
        "unit": "HUC02 region; eligible region requires at least five high-skill basins within seed",
        "represented_region_count": int(len(regions)),
        "eligible_region_counts_by_seed": {
            str(seed): int(len(region_seed_medians[seed])) for seed in SEEDS
        },
        "replicates": N_BOOTSTRAP,
        "rng": "NumPy PCG64",
        "seed": 20260829,
        "observed_equal_region_medians_by_seed": observed,
        "observed_median_across_seeds": float(np.median(list(observed.values()))),
        "percentile_95_interval": [float(lower), float(upper)],
        "lower_bound_above_zero": bool(lower > 0),
    }


def reproduce_external() -> dict[str, Any]:
    frame = pd.read_csv(
        DATA / "external_basin_seed_outcomes.csv",
        dtype={"basin_id": str, "huc_02": str},
    )
    frame["basin_id"] = frame.basin_id.str.zfill(8)
    frame["huc_02"] = frame.huc_02.str.zfill(2)
    if len(frame) != 3 * 506 or any(frame[frame.seed == seed].basin_id.nunique() != 506 for seed in SEEDS):
        raise RuntimeError("Unexpected external-transport inventory")
    leave_out = []
    for omitted in sorted(frame.huc_02.unique()):
        retained = frame[frame.huc_02 != omitted]
        for seed in SEEDS:
            high = retained[(retained.seed == seed) & retained.high_skill_original_path]
            leave_out.append(float(high.nse_decline.median()))
    return {
        "per_seed_summary": external_per_seed(frame),
        "basin_block_bootstrap": basin_block_bootstrap(frame),
        "equal_region_block_bootstrap": equal_region_block_bootstrap(frame),
        "minimum_leave_one_huc02_out_median_nse_decline": float(min(leave_out)),
    }


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().eq("true")


def reproduce_operator_robustness() -> dict[str, Any]:
    frame = pd.read_csv(DATA / "operator_robustness_per_basin.csv", dtype={"basin": str})
    operators = (
        "source_order_runoff_first",
        "proportional_allocation",
        "euclidean_simplex_projection",
        "et_first_stress",
    )
    if len(frame) != 3 * 24 or any(len(frame[frame.seed == seed]) != 24 for seed in SEEDS):
        raise RuntimeError("Unexpected operator-robustness inventory")
    frame["high_skill_original_path"] = as_bool(frame.high_skill_original_path)
    frame["robust_lower_envelope_nse_loss_positive"] = as_bool(
        frame.robust_lower_envelope_nse_loss_positive
    )
    summaries = {}
    seed_pass = {}
    for seed in SEEDS:
        subset = frame[frame.seed == seed]
        high = subset[subset.high_skill_original_path]
        operator_summaries = {}
        for operator in operators:
            feasible = bool(
                (subset[f"{operator}_minimum_storage_mm"] >= -1e-7).all()
                and (subset[f"{operator}_maximum_release_overshoot_mm"] <= 1e-6).all()
                and (subset[f"{operator}_minimum_executed_release_mm"] >= -1e-7).all()
            )
            operator_summaries[operator] = {
                "median_nse_loss": float(subset[f"{operator}_nse_loss"].median()),
                "median_kge_loss": float(subset[f"{operator}_kge_loss"].median()),
                "median_rmse_change_mm_day": float(
                    subset[f"{operator}_rmse_change_mm_day"].median()
                ),
                "best_feasible_operator_count": int(
                    (subset.best_feasible_operator_by_nse == operator).sum()
                ),
                "feasible_all_basins": feasible,
            }
        summary = {
            "basin_count": int(len(subset)),
            "high_skill_basin_count": int(len(high)),
            "median_robust_lower_envelope_nse_loss": float(
                subset.robust_lower_envelope_nse_loss.median()
            ),
            "fraction_robust_lower_envelope_nse_loss_positive": float(
                subset.robust_lower_envelope_nse_loss_positive.mean()
            ),
            "high_skill_median_robust_lower_envelope_nse_loss": float(
                high.robust_lower_envelope_nse_loss.median()
            ),
            "high_skill_fraction_robust_lower_envelope_nse_loss_positive": float(
                high.robust_lower_envelope_nse_loss_positive.mean()
            ),
            "maximum_official_forward_parity_error_mm_day": float(
                subset.official_forward_max_abs_parity_error_mm_day.max()
            ),
            "all_expected_test_days_present": bool(
                (subset.evaluated_test_day_count == 3652).all()
            ),
            "all_frozen_operators_feasible_all_basins": bool(
                all(row["feasible_all_basins"] for row in operator_summaries.values())
            ),
            "operator_summaries": operator_summaries,
        }
        summaries[str(seed)] = summary
        seed_pass[str(seed)] = bool(
            summary["maximum_official_forward_parity_error_mm_day"] <= 1e-6
            and summary["all_expected_test_days_present"]
            and summary["all_frozen_operators_feasible_all_basins"]
            and summary["median_robust_lower_envelope_nse_loss"] >= 0.05
            and summary["high_skill_median_robust_lower_envelope_nse_loss"] >= 0.05
            and summary[
                "high_skill_fraction_robust_lower_envelope_nse_loss_positive"
            ]
            >= 0.60
        )
    passed = sum(seed_pass.values()) >= 2
    return {
        "per_seed_panel_summary": summaries,
        "operator_robustness_pass": passed,
        "decision": "operator_robust" if passed else "operator_sensitive_or_mixed",
    }


def feasible_seed_summary(frame: pd.DataFrame) -> dict[str, Any]:
    high = frame[frame.high_skill_unconstrained_original]
    eligible = frame[frame.recovery_eligible]
    return {
        "basin_count": int(len(frame)),
        "high_skill_basin_count": int(len(high)),
        "recovery_eligible_basin_count": int(len(eligible)),
        "all_basin_median_transfer_gap_nse": float(frame.transfer_gap_nse.median()),
        "high_skill_median_transfer_gap_nse": float(high.transfer_gap_nse.median()),
        "high_skill_fraction_transfer_gap_within_0_10": float(
            high.transfer_gap_within_0_10.mean()
        ),
        "recovery_eligible_median_recovery_fraction": float(
            eligible.recovery_fraction.median()
        ),
        "recovery_eligible_fraction_recovery_at_least_0_80": float(
            eligible.recovery_at_least_0_80.mean()
        ),
        "all_basin_fraction_feasible_retrained_beats_fixed_projection": float(
            frame.feasible_retrained_beats_fixed_projection.mean()
        ),
        "median_nse_unconstrained_original": float(
            frame.nse_unconstrained_original.median()
        ),
        "median_nse_feasible_retrained": float(frame.nse_feasible_retrained.median()),
        "median_kge_gap_unconstrained_minus_feasible": float(
            (frame.kge_unconstrained_original - frame.kge_feasible_retrained).median()
        ),
        "median_rmse_change_feasible_minus_unconstrained_mm_day": float(
            (
                frame.rmse_feasible_retrained_mm_day
                - frame.rmse_unconstrained_original_mm_day
            ).median()
        ),
        "median_absolute_bias_ratio_change": float(
            (
                frame.bias_ratio_feasible_retrained
                - frame.bias_ratio_unconstrained_original
            ).abs().median()
        ),
    }


def feasible_gates(summary: dict[str, Any]) -> dict[str, bool]:
    return {
        "recovery_capacity_this_seed": summary["recovery_eligible_basin_count"] >= 100,
        "recovery_materiality_this_seed": summary[
            "recovery_eligible_basin_count"
        ]
        >= 100
        and summary["recovery_eligible_median_recovery_fraction"] >= 0.80,
        "all_basin_transfer_gap_this_seed": summary[
            "all_basin_median_transfer_gap_nse"
        ]
        <= 0.05,
        "high_skill_transfer_gap_this_seed": summary[
            "high_skill_median_transfer_gap_nse"
        ]
        <= 0.05,
        "high_skill_directional_consistency_this_seed": summary[
            "high_skill_fraction_transfer_gap_within_0_10"
        ]
        >= 0.60,
    }


def reproduce_feasible_external_transfer() -> dict[str, Any]:
    frame = pd.read_csv(
        DATA / "feasible_external_transfer_per_basin.csv", dtype={"basin": str}
    )
    frame["basin"] = frame.basin.str.zfill(8)
    boolean_fields = (
        "high_skill_unconstrained_original",
        "recovery_eligible",
        "transfer_gap_within_0_10",
        "recovery_at_least_0_80",
        "feasible_retrained_beats_fixed_projection",
    )
    for field in boolean_fields:
        frame[field] = as_bool(frame[field])
    if len(frame) != 3 * 506 or any(
        frame[frame.seed == seed].basin.nunique() != 506 for seed in SEEDS
    ):
        raise RuntimeError("Unexpected feasible external-transfer inventory")

    per_seed = {
        str(seed): feasible_seed_summary(frame[frame.seed == seed]) for seed in SEEDS
    }
    gates = {str(seed): feasible_gates(per_seed[str(seed)]) for seed in SEEDS}
    seed_pass = {str(seed): all(gates[str(seed)].values()) for seed in SEEDS}

    basin_ids = np.array(sorted(frame.basin.unique()))
    draws = np.random.Generator(np.random.PCG64(20260901)).integers(
        0, len(basin_ids), size=(N_BOOTSTRAP, len(basin_ids))
    )
    fields = {
        "all_basin_median_transfer_gap_nse": [],
        "high_skill_median_transfer_gap_nse": [],
        "recovery_eligible_median_recovery_fraction": [],
        "high_skill_fraction_transfer_gap_within_0_10": [],
    }
    for seed in SEEDS:
        subset = frame[frame.seed == seed].set_index("basin").loc[basin_ids]
        gaps = subset.transfer_gap_nse.to_numpy(float)[draws]
        high = subset.high_skill_unconstrained_original.to_numpy(bool)[draws]
        eligible = subset.recovery_eligible.to_numpy(bool)[draws]
        recovery = subset.recovery_fraction.to_numpy(float)[draws]
        within = subset.transfer_gap_within_0_10.to_numpy(bool)[draws]
        fields["all_basin_median_transfer_gap_nse"].append(np.nanmedian(gaps, axis=1))
        fields["high_skill_median_transfer_gap_nse"].append(
            np.nanmedian(np.where(high, gaps, np.nan), axis=1)
        )
        fields["recovery_eligible_median_recovery_fraction"].append(
            np.nanmedian(np.where(eligible, recovery, np.nan), axis=1)
        )
        fields["high_skill_fraction_transfer_gap_within_0_10"].append(
            np.nanmean(np.where(high, within, np.nan), axis=1)
        )
    bootstrap_values = {
        field: np.nanmedian(np.column_stack(values), axis=1)
        for field, values in fields.items()
    }
    observed = {
        field: float(np.median([per_seed[str(seed)][field] for seed in SEEDS]))
        for field in fields
    }
    bootstrap_pass = (
        (bootstrap_values["all_basin_median_transfer_gap_nse"] <= 0.05)
        & (bootstrap_values["high_skill_median_transfer_gap_nse"] <= 0.05)
        & (bootstrap_values["recovery_eligible_median_recovery_fraction"] >= 0.80)
        & (bootstrap_values["high_skill_fraction_transfer_gap_within_0_10"] >= 0.60)
    )
    bootstrap = {
        "unit": "basin with all three seed records retained",
        "replicates": N_BOOTSTRAP,
        "rng": "NumPy PCG64",
        "seed": 20260901,
        "metrics": {
            field: {
                "observed": observed[field],
                "percentile_95_interval": interval(values),
            }
            for field, values in bootstrap_values.items()
        },
        "all_four_metric_gates_pass_fraction": float(np.mean(bootstrap_pass)),
    }

    attributes = pd.read_csv(
        DATA / "external_basin_seed_outcomes.csv",
        dtype={"basin_id": str, "huc_02": str},
    )
    attributes["basin_id"] = attributes.basin_id.str.zfill(8)
    attributes["huc_02"] = attributes.huc_02.str.zfill(2)
    regions = attributes[["basin_id", "huc_02"]].drop_duplicates()
    spatial = frame.merge(
        regions,
        left_on="basin",
        right_on="basin_id",
        validate="many_to_one",
    )
    leave_rows = []
    for region in sorted(spatial.huc_02.unique()):
        retained = spatial[spatial.huc_02 != region]
        summaries = {
            str(seed): feasible_seed_summary(retained[retained.seed == seed])
            for seed in SEEDS
        }
        row = {
            field: float(np.median([summaries[str(seed)][field] for seed in SEEDS]))
            for field in fields
        }
        row["pass"] = bool(
            row["all_basin_median_transfer_gap_nse"] <= 0.05
            and row["high_skill_median_transfer_gap_nse"] <= 0.05
            and row["recovery_eligible_median_recovery_fraction"] >= 0.80
            and row["high_skill_fraction_transfer_gap_within_0_10"] >= 0.60
        )
        leave_rows.append(row)
    leave_summary = {
        "region_count": int(len(leave_rows)),
        "metric_ranges": {
            field: [
                float(min(row[field] for row in leave_rows)),
                float(max(row[field] for row in leave_rows)),
            ]
            for field in fields
        },
        "all_regions_preserve_all_four_metric_gates": bool(
            all(row["pass"] for row in leave_rows)
        ),
    }
    return {
        "per_seed_summary": per_seed,
        "per_seed_scientific_gates": gates,
        "per_seed_scientific_pass": seed_pass,
        "scientific_seed_pass_count": int(sum(seed_pass.values())),
        "basin_block_bootstrap": bootstrap,
        "leave_one_huc02_out": leave_summary,
    }


def reproduce_unmodified_implementation_control() -> dict[str, Any]:
    panel = pd.read_csv(DATA / "unmodified_seed11_panel.csv", dtype={"basin": str})
    context = pd.read_csv(
        DATA / "unmodified_batch_context_control.csv", dtype={"basin": str}
    )
    if len(panel) != 24 or len(context) != 20:
        raise RuntimeError("Unexpected unmodified-implementation control inventory")
    absolute = context.maximum_pairwise_context_effect_raw_mm_day.to_numpy(float)
    relative = context.maximum_relative_context_effect_raw.to_numpy(float)
    return {
        "panel_summary": {
            "basin_fraction_negative_on_at_least_one_percent_test_days": float(
                (panel.original_test_day_any_negative_storage_fraction >= 0.01).mean()
            ),
            "median_negative_test_day_fraction": float(
                panel.original_test_day_any_negative_storage_fraction.median()
            ),
            "median_maximum_deficit_mm": float(
                panel.original_maximum_negative_storage_mm.median()
            ),
            "median_ordered_projection_nse_loss": float(
                panel.ordered_projection_nse_loss.median()
            ),
        },
        "batch_context_summary": {
            "focal_sample_count": int(len(context)),
            "maximum_absolute_context_effect_raw_mm_day": float(np.max(absolute)),
            "median_maximum_absolute_context_effect_raw_mm_day": float(
                np.median(absolute)
            ),
            "maximum_relative_context_effect_raw": float(np.max(relative)),
            "median_maximum_relative_context_effect_raw": float(np.median(relative)),
            "raw_path_technically_invariant_at_1e-6_mm_day": bool(
                np.max(absolute) <= 1e-6
            ),
            "raw_path_material_absolute_dependence": bool(np.median(absolute) >= 0.01),
            "raw_path_material_relative_dependence": bool(np.median(relative) >= 0.01),
        },
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "analysis_level": True,
        "core_panel": reproduce_core(),
        "external_transport": reproduce_external(),
        "window_initialization_decision": "NOT_A_365DAY_ZERO_INITIALIZATION_ARTIFACT",
        "operator_robustness": reproduce_operator_robustness(),
        "feasible_external_transfer": reproduce_feasible_external_transfer(),
        "unmodified_implementation_control": reproduce_unmodified_implementation_control(),
    }
    OUTPUT.write_text(json.dumps(json_safe(result), indent=2) + "\n", encoding="utf-8")
    print(OUTPUT.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
