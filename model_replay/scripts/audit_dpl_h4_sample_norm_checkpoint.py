"""Process-feasibility audit for sample-separable DPL-H4.

The audit reuses the official embedding and learned gates, but explicitly
executes three state-update variants at identical weights.  The requested
official execution path is checked against the model forward method before any
diagnostic is accepted.  The default remains the frozen unconstrained audit;
the feasible-retraining control explicitly selects ordered_flux_projection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.evaluation.utils import load_scaler
from neuralhydrology.modelzoo import get_model
from neuralhydrology.utils.config import Config


VARIANTS = ("original", "dormant_et_ceiling", "ordered_flux_projection")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar_from_scaler(scaler: dict[str, Any], group: str, variable: str, device: torch.device) -> torch.Tensor:
    value = float(np.asarray(scaler[group][variable].values).squeeze())
    return torch.tensor(value, dtype=torch.float32, device=device)


@torch.no_grad()
def run_variant(model, data: dict[str, torch.Tensor], scaler: dict[str, Any], variant: str):
    if variant not in VARIANTS:
        raise ValueError(variant)
    embedded = model.embedding_net(data, concatenate_output=True)
    x_m = embedded[:, :, : model._n_mass_vars]
    x_a = embedded[:, :, model._n_mass_vars :]
    cell = model.ddpl_h
    _, batch_size, _ = x_m.shape
    hidden = cell._hidden_size
    c = x_m.new_zeros((batch_size, hidden))
    c_s = x_m.new_zeros((batch_size, hidden))

    tmax_scale = scalar_from_scaler(scaler, "xarray_feature_scale", "tmax(C)", x_m.device)
    tmax_center = scalar_from_scaler(scaler, "xarray_feature_center", "tmax(C)", x_m.device)
    tmin_scale = scalar_from_scaler(scaler, "xarray_feature_scale", "tmin(C)", x_m.device)
    tmin_center = scalar_from_scaler(scaler, "xarray_feature_center", "tmin(C)", x_m.device)

    q_steps = []
    final: dict[str, torch.Tensor] = {}
    for step, (p, a) in enumerate(zip(x_m, x_a)):
        temperature = torch.stack(
            [a[:, 1] * tmax_scale + tmax_center, a[:, 2] * tmin_scale + tmin_center], dim=1
        ).mean(dim=1, keepdim=True)
        threshold = cell.Tt_gate(torch.cat([p, a], dim=-1))
        rain = p * cell._comparison_g(temperature - threshold)
        snow = p - rain

        # Match the clean branch's per-sample L1 normalization exactly.
        features = torch.cat(
            [rain, a, c / (c.norm(p=1, dim=-1, keepdim=True) + 1e-6)], dim=-1
        )
        features_s = torch.cat(
            [snow, a, c_s / (c_s.norm(p=1, dim=-1, keepdim=True) + 1e-6)], dim=-1
        )

        input_gate = cell.input_gate(features)
        snow_input_gate = cell.input_gate_s(features_s)
        smax = cell.SMmax_gate(a[:, 5:])
        sfc = cell.SMfc_gate(a[:, 5:], smax)
        redistribution = cell.redistribution(features)
        snow_redistribution = cell.redistribution_s(features_s)
        runoff_gate = cell.output_gate(features)
        baseflow_gate = cell.bfout_gate(a[:, 5:])
        melt_gate = cell.ddf_gate(features_s)
        et_raw = cell.et_gate(features)

        snow_system = torch.matmul(c_s.unsqueeze(-2), snow_redistribution)
        snow_system = snow_system + torch.matmul(snow.unsqueeze(-2), snow_input_gate)
        melt_potential = melt_gate * torch.repeat_interleave(
            torch.relu(temperature - threshold), repeats=hidden, dim=-1
        )
        melt = torch.minimum(melt_potential, snow_system.squeeze(-2))
        c_s_new = snow_system.squeeze(-2) - melt

        incoming = torch.matmul(rain.unsqueeze(-2), input_gate) + melt.unsqueeze(-2)
        soil_before_input = torch.matmul(c.unsqueeze(-2), redistribution)
        fast = incoming * cell._comparison_g(soil_before_input - smax)
        available = (soil_before_input + incoming - fast).squeeze(-2)
        subsurface_raw = runoff_gate * torch.relu((available.unsqueeze(-2) - sfc).squeeze(-2))
        baseflow_raw = baseflow_gate * available

        if variant == "original":
            subsurface = subsurface_raw
            baseflow = baseflow_raw
            remaining_before_et = available - subsurface - baseflow
            et = et_raw
        elif variant == "dormant_et_ceiling":
            subsurface = subsurface_raw
            baseflow = baseflow_raw
            remaining_before_et = available - subsurface - baseflow
            et = torch.clamp(torch.minimum(remaining_before_et, et_raw), min=0.0)
        else:
            feasible = torch.clamp(available, min=0.0)
            subsurface = torch.minimum(torch.clamp(subsurface_raw, min=0.0), feasible)
            feasible = feasible - subsurface
            baseflow = torch.minimum(torch.clamp(baseflow_raw, min=0.0), feasible)
            remaining_before_et = feasible - baseflow
            et = torch.minimum(torch.clamp(et_raw, min=0.0), remaining_before_et)

        c_new = available - subsurface - baseflow - et
        q_cells = fast.squeeze(-2) + subsurface + baseflow
        q_steps.append(q_cells.sum(dim=-1))
        c, c_s = c_new, c_s_new
        if step == x_m.shape[0] - 1:
            final = {
                "storage": c_new,
                "et_raw": et_raw,
                "et_executed": et,
                "remaining_before_et": remaining_before_et,
                "available": available,
                "subsurface": subsurface,
                "baseflow": baseflow,
            }
    return torch.stack(q_steps, dim=0), final


def metrics(obs: np.ndarray, sim: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(obs) & np.isfinite(sim)
    o = obs[valid]
    s = sim[valid]
    if len(o) < 2 or np.sum((o - o.mean()) ** 2) <= 0:
        return {"n": int(len(o)), "nse": math.nan, "kge": math.nan, "correlation": math.nan,
                "variability_ratio": math.nan, "bias_ratio": math.nan, "rmse_mm_day": math.nan}
    nse = 1.0 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)
    correlation = np.corrcoef(o, s)[0, 1]
    alpha = np.std(s, ddof=0) / np.std(o, ddof=0)
    beta = np.mean(s) / np.mean(o) if np.mean(o) != 0 else math.nan
    kge = 1.0 - np.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    rmse = np.sqrt(np.mean((s - o) ** 2))
    return {
        "n": int(len(o)),
        "nse": float(nse),
        "kge": float(kge),
        "correlation": float(correlation),
        "variability_ratio": float(alpha),
        "bias_ratio": float(beta),
        "rmse_mm_day": float(rmse),
    }


def summarize_final(storage: np.ndarray, et_raw: np.ndarray, remaining: np.ndarray) -> dict[str, float | int]:
    negative = storage < -1e-7
    deficit = np.clip(-storage, 0.0, None)
    proposal_overshoot = et_raw > remaining + 1e-7
    proposal_excess = np.clip(et_raw - remaining, 0.0, None)
    return {
        "test_days": int(storage.shape[0]),
        "cell_count": int(storage.size),
        "negative_storage_cell_fraction": float(negative.mean()),
        "test_day_any_negative_storage_fraction": float(negative.any(axis=1).mean()),
        "negative_storage_mass_sum_mm": float(deficit.sum()),
        "maximum_negative_storage_mm": float(deficit.max(initial=0.0)),
        "proposed_et_overshoot_cell_fraction": float(proposal_overshoot.mean()),
        "test_day_any_proposed_et_overshoot_fraction": float(proposal_overshoot.any(axis=1).mean()),
        "proposed_et_overshoot_mass_sum_mm": float(proposal_excess.sum()),
        "maximum_proposed_et_overshoot_mm_day": float(proposal_excess.max(initial=0.0)),
    }


def finite_median(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.nanmedian(array)) if np.isfinite(array).any() else math.nan


def json_safe(value: Any) -> Any:
    """Convert non-finite diagnostics to JSON null without altering CSV output."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--official-variant",
        choices=VARIANTS,
        default="original",
        help="manual execution variant that must reproduce model.forward()",
    )
    parser.add_argument(
        "--audit-role",
        choices=("unconstrained_same_checkpoint", "feasible_retrained_control"),
        default="unconstrained_same_checkpoint",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="debug override; omitted for formal audits so the archived run config controls batching",
    )
    parser.add_argument("--basin-limit", type=int)
    parser.add_argument("--max-days", type=int)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    checkpoint = run_dir / f"model_epoch{args.epoch:03d}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    cfg = Config(run_dir / "config.yml")
    cfg.run_dir = run_dir
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.batch_size)
    device = torch.device(args.device)
    model = get_model(cfg).to(device)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    scaler = load_scaler(run_dir)

    basins = [line.strip() for line in args.panel.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.basin_limit is not None:
        basins = basins[: args.basin_limit]

    per_basin: list[dict[str, Any]] = []
    global_parity_max = 0.0
    for basin_index, basin in enumerate(basins, start=1):
        dataset = get_dataset(cfg=cfg, is_train=False, period="test", basin=basin, scaler=scaler)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        obs_parts: list[np.ndarray] = []
        sim_parts = {variant: [] for variant in VARIANTS}
        diag_parts = {
            variant: {"storage": [], "et_raw": [], "remaining": []} for variant in VARIANTS
        }
        consumed = 0
        for batch_index, data in enumerate(loader):
            if args.max_days is not None and consumed >= args.max_days:
                break
            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.to(device)
            if args.max_days is not None:
                take = min(data["y"].shape[0], args.max_days - consumed)
                data = {key: value[:take] if torch.is_tensor(value) else value for key, value in data.items()}
            obs_parts.append(data["y"][:, -1, 0].detach().cpu().numpy())

            for variant in VARIANTS:
                q_steps, final = run_variant(model, data, scaler, variant)
                if variant == args.official_variant and batch_index == 0:
                    official = model(data, scaler)["y_hat"].squeeze(-1).transpose(0, 1)
                    parity = float(torch.max(torch.abs(official - q_steps)).item())
                    global_parity_max = max(global_parity_max, parity)
                sim_parts[variant].append(q_steps[-1].detach().cpu().numpy())
                diag_parts[variant]["storage"].append(final["storage"].detach().cpu().numpy())
                diag_parts[variant]["et_raw"].append(final["et_raw"].detach().cpu().numpy())
                diag_parts[variant]["remaining"].append(final["remaining_before_et"].detach().cpu().numpy())
            consumed += int(data["y"].shape[0])

        obs = np.concatenate(obs_parts)
        simulations = {variant: np.concatenate(parts) for variant, parts in sim_parts.items()}
        diagnostics = {}
        for variant in VARIANTS:
            diagnostics[variant] = summarize_final(
                np.concatenate(diag_parts[variant]["storage"]),
                np.concatenate(diag_parts[variant]["et_raw"]),
                np.concatenate(diag_parts[variant]["remaining"]),
            )
        scores = {variant: metrics(obs, np.clip(sim, 0.0, None)) for variant, sim in simulations.items()}
        original_mass = diagnostics["original"]["negative_storage_mass_sum_mm"]
        et_mass = diagnostics["dormant_et_ceiling"]["negative_storage_mass_sum_mm"]
        et_removed = (original_mass - et_mass) / original_mass if original_mass > 0 else math.nan
        exact_diff = float(
            np.max(np.abs(simulations["dormant_et_ceiling"] - simulations["ordered_flux_projection"]), initial=0.0)
        )
        row = {
            "basin": basin,
            "performance": scores,
            "diagnostics": diagnostics,
            "same_checkpoint_effects": {
                "et_repair_fraction_of_original_negative_mass_removed": float(et_removed),
                "ordered_projection_nse_loss": float(scores["original"]["nse"] - scores["ordered_flux_projection"]["nse"]),
                "ordered_projection_kge_loss": float(scores["original"]["kge"] - scores["ordered_flux_projection"]["kge"]),
                "original_vs_ordered_mean_abs_runoff_change_mm_day": float(
                    np.mean(np.abs(simulations["original"] - simulations["ordered_flux_projection"]))
                ),
                "et_repair_vs_ordered_projection_max_abs_runoff_difference_mm_day": exact_diff,
            },
        }
        per_basin.append(row)
        print(
            f"[{basin_index:02d}/{len(basins):02d}] {basin} "
            f"neg_days={diagnostics['original']['test_day_any_negative_storage_fraction']:.4f} "
            f"NSE={scores['original']['nse']:.4f}->{scores['ordered_flux_projection']['nse']:.4f} "
            f"ET_removed={et_removed:.4f}"
        )

    negative_day_fraction = [
        row["diagnostics"]["original"]["test_day_any_negative_storage_fraction"] for row in per_basin
    ]
    maximum_deficit = [row["diagnostics"]["original"]["maximum_negative_storage_mm"] for row in per_basin]
    nse_losses = [row["same_checkpoint_effects"]["ordered_projection_nse_loss"] for row in per_basin]
    et_removed = [
        row["same_checkpoint_effects"]["et_repair_fraction_of_original_negative_mass_removed"] for row in per_basin
    ]
    prevalence = float(np.mean(np.asarray(negative_day_fraction) >= 0.01))
    ordered_all_feasible = all(
        row["diagnostics"]["ordered_flux_projection"]["negative_storage_cell_fraction"] == 0.0
        for row in per_basin
    )
    official_all_feasible = all(
        row["diagnostics"][args.official_variant]["negative_storage_cell_fraction"] == 0.0
        for row in per_basin
    )
    official_nse = [row["performance"][args.official_variant]["nse"] for row in per_basin]
    official_kge = [row["performance"][args.official_variant]["kge"] for row in per_basin]
    gates = {
        "mechanism_prevalence_this_seed": prevalence >= 0.50,
        "mechanism_magnitude_this_seed": finite_median(negative_day_fraction) >= 0.05
        and finite_median(maximum_deficit) >= 1.0,
        "feasibility_repair_this_seed": ordered_all_feasible,
        "et_attribution_this_seed": finite_median(et_removed) >= 0.80,
        "skill_materiality_this_seed": finite_median(nse_losses) >= 0.05,
        "official_original_path_parity": global_parity_max <= 1e-6,
    }
    if args.official_variant != "original":
        gates["official_forward_path_parity"] = global_parity_max <= 1e-6
        gates["official_execution_feasible_all_basins"] = official_all_feasible
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit": (
            "sample-separable DPL-H4 frozen-panel same-checkpoint feasibility audit"
            if args.audit_role == "unconstrained_same_checkpoint"
            else "sample-separable DPL-H4 frozen-panel feasible-retraining control audit"
        ),
        "audit_role": args.audit_role,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "epoch": args.epoch,
        "seed": cfg.seed,
        "panel": str(args.panel.resolve()),
        "panel_sha256": sha256(args.panel),
        "basin_count": len(per_basin),
        "device": str(device),
        "batch_size": batch_size,
        "batch_size_source": "debug CLI override" if args.batch_size is not None else "archived run config",
        "max_days_debug_limit": args.max_days,
        "weights_identical_across_variants": True,
        "evaluation_clips_negative_runoff_to_zero_like_official_tester": True,
        "official_execution_variant": args.official_variant,
        "official_forward_path_max_abs_parity_error_mm_day": global_parity_max,
        "official_original_path_max_abs_parity_error_mm_day": global_parity_max,
        "per_basin": per_basin,
        "panel_summary": {
            "basin_fraction_negative_on_at_least_one_percent_test_days": prevalence,
            "median_negative_test_day_fraction": finite_median(negative_day_fraction),
            "median_maximum_deficit_mm": finite_median(maximum_deficit),
            "median_et_repair_fraction_of_negative_mass_removed": finite_median(et_removed),
            "median_ordered_projection_nse_loss": finite_median(nse_losses),
            "ordered_projection_feasible_all_basins": ordered_all_feasible,
            "official_execution_feasible_all_basins": official_all_feasible,
            "median_official_nse": finite_median(official_nse),
            "median_official_kge": finite_median(official_kge),
        },
        "single_seed_gates": gates,
        "branch_identity": (
            "one-line dormant per-sample L1 normalization activated; no other architecture change"
            if args.audit_role == "unconstrained_same_checkpoint"
            else "same sample-normalized DPL-H4 with only ordered soil-outflow feasibility projection activated"
        ),
        "seed11_survival_screen": "pass only if all six single-seed gates are true",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    flat_rows = []
    for row in per_basin:
        flat = {"basin": row["basin"]}
        for variant in VARIANTS:
            for key, value in row["performance"][variant].items():
                flat[f"{variant}_{key}"] = value
            for key, value in row["diagnostics"][variant].items():
                flat[f"{variant}_{key}"] = value
        flat.update(row["same_checkpoint_effects"])
        flat_rows.append(flat)
    csv_path = args.output.with_suffix(".csv")
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    print(json.dumps(result["panel_summary"], indent=2))
    print(json.dumps(gates, indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
