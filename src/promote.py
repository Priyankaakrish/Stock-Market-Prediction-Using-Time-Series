"""Model approval and stage promotion.

The registry records every training run as a new version. Nothing in that is a
decision. This module is the decision: which version, if any, may serve traffic.

Three checks, and the third is the one that matters here:

1. Not worse than the incumbent. A new version may be at most `tolerance`
   times the current Production RMSE. Retraining on fresher data usually helps;
   occasionally it does not, and shipping automatically on every run is how a
   silent regression reaches users.

2. Sane in absolute terms. MAPE under a ceiling, so a catastrophically broken
   run cannot be promoted merely because no incumbent exists.

3. Not worse than a random walk. Given this project's central finding, a model
   that loses to naive on its own test window has negative value: more complex
   than the baseline and less accurate. Blocked outright.

MLflow 3 deprecated stage transitions in favour of aliases, so "promote to
Production" is set_registered_model_alias(name, "Production", version). The
serving layer then resolves models:/nvda-forecaster@Production — which is what
MODEL_URI in .env.example is for, and what had no code behind it until now.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass

from .config import MLFLOW, PATHS

log = logging.getLogger(__name__)

PRODUCTION = "Production"
STAGING = "Staging"


@dataclass
class GateResult:
    version: str
    approved: bool
    reasons: list[str]
    candidate_rmse: float
    candidate_mape: float
    incumbent_rmse: float | None
    naive_rmse: float | None

    def render(self) -> str:
        head = f"version {self.version}: {'APPROVED' if self.approved else 'REJECTED'}"
        return head + "".join(f"\n  - {r}" for r in self.reasons)


def _client():
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW.tracking_uri)
    return MlflowClient()


def _metrics_for(client, version) -> dict:
    """Metrics live on the run that produced the version, not the version."""
    return client.get_run(version.run_id).data.metrics


def list_versions(name: str = MLFLOW.registered_model) -> list:
    client = _client()
    out = []
    for v in client.search_model_versions(f"name='{name}'"):
        m = _metrics_for(client, v)
        out.append({
            "version": v.version,
            "run_id": v.run_id,
            "aliases": list(v.aliases or []),
            "test_rmse": m.get("best_test_rmse", m.get("test_rmse")),
            "naive_test_rmse": m.get("naive_test_rmse"),
        })
    return sorted(out, key=lambda d: int(d["version"]))


def current_production(name: str = MLFLOW.registered_model):
    client = _client()
    try:
        return client.get_model_version_by_alias(name, PRODUCTION)
    except Exception:
        return None


def evaluate_gate(version=None, name: str = MLFLOW.registered_model,
                  tolerance: float = 1.05, max_mape: float = 5.0) -> GateResult:
    """Decide whether `version` (default: newest) may serve traffic."""
    client = _client()
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(f"No versions registered under '{name}'. Train first.")

    if version is None:
        candidate = max(versions, key=lambda v: int(v.version))
    else:
        candidate = next((v for v in versions if v.version == str(version)), None)
        if candidate is None:
            raise RuntimeError(f"Version {version} not found under '{name}'")

    m = _metrics_for(client, candidate)
    cand_rmse = m.get("best_test_rmse", m.get("test_rmse"))
    cand_mape = m.get("best_test_mape", m.get("test_mape"))
    naive_rmse = m.get("naive_test_rmse")

    meta_path = PATHS.models / "best_model_meta.json"
    if (cand_rmse is None or cand_mape is None) and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        cand_rmse = cand_rmse if cand_rmse is not None else meta["metrics"]["rmse"]
        cand_mape = cand_mape if cand_mape is not None else meta["metrics"]["mape"]
        naive_rmse = naive_rmse if naive_rmse is not None else meta.get("naive_rmse")

    if cand_rmse is None:
        raise RuntimeError(f"Version {candidate.version} has no RMSE logged")

    incumbent = current_production(name)
    inc_rmse = None
    if incumbent is not None and incumbent.version != candidate.version:
        im = _metrics_for(client, incumbent)
        inc_rmse = im.get("best_test_rmse", im.get("test_rmse"))

    reasons, approved = [], True

    if inc_rmse:
        ratio = cand_rmse / inc_rmse
        if ratio > tolerance:
            approved = False
            reasons.append(f"RMSE {cand_rmse:.4f} is {ratio:.3f}x the incumbent "
                           f"{inc_rmse:.4f} (limit {tolerance})")
        else:
            reasons.append(f"RMSE {cand_rmse:.4f} vs incumbent {inc_rmse:.4f} "
                           f"({ratio:.3f}x) - within tolerance")
    else:
        reasons.append("no incumbent in Production; absolute checks only")

    if cand_mape is not None and cand_mape > max_mape:
        approved = False
        reasons.append(f"MAPE {cand_mape:.3f}% exceeds the {max_mape}% ceiling")
    elif cand_mape is not None:
        reasons.append(f"MAPE {cand_mape:.3f}% within the {max_mape}% ceiling")

    # The check this project exists to make.
    if naive_rmse:
        if cand_rmse > naive_rmse:
            approved = False
            reasons.append(
                f"BLOCKED: RMSE {cand_rmse:.4f} is worse than the random walk "
                f"{naive_rmse:.4f}. A model that loses to naive is more complex "
                "and less accurate than doing nothing.")
        else:
            reasons.append(
                f"beats the random walk ({cand_rmse:.4f} vs {naive_rmse:.4f}), "
                "though see the walk-forward report before reading much into it")
    else:
        reasons.append("no naive baseline logged - cannot verify against a random walk")

    return GateResult(
        version=candidate.version, approved=approved, reasons=reasons,
        candidate_rmse=float(cand_rmse),
        candidate_mape=float(cand_mape) if cand_mape is not None else float("nan"),
        incumbent_rmse=float(inc_rmse) if inc_rmse else None,
        naive_rmse=float(naive_rmse) if naive_rmse else None,
    )


def promote(version=None, stage: str = PRODUCTION,
            name: str = MLFLOW.registered_model, force: bool = False) -> GateResult:
    """Run the gate and, if it passes, move the alias. This is the deploy trigger."""
    result = evaluate_gate(version, name=name)
    if not result.approved and not force:
        log.error("Promotion refused.\n%s", result.render())
        return result

    client = _client()
    client.set_registered_model_alias(name, stage, result.version)
    client.set_model_version_tag(name, result.version, "approved_by",
                                 "src.promote gate" if not force else "forced")
    log.info("Version %s now serves as '%s'. Point MODEL_URI at models:/%s@%s",
             result.version, stage, name, stage)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Model approval and stage promotion")
    ap.add_argument("action", choices=["list", "check", "promote", "stage"])
    ap.add_argument("--version", default=None, help="default: newest")
    ap.add_argument("--force", action="store_true", help="promote despite a failed gate")
    args = ap.parse_args()

    if args.action == "list":
        for v in list_versions():
            alias = f"  [{', '.join(v['aliases'])}]" if v["aliases"] else ""
            rmse = f"{v['test_rmse']:.4f}" if v["test_rmse"] else "n/a"
            print(f"v{str(v['version']):<3} rmse={rmse:<10}{alias}")
        prod = current_production()
        print(f"\nProduction: {'v' + str(prod.version) if prod else 'none assigned'}")
    elif args.action == "check":
        print(evaluate_gate(args.version).render())
    else:
        stage = PRODUCTION if args.action == "promote" else STAGING
        print(promote(args.version, stage=stage, force=args.force).render())


if __name__ == "__main__":
    main()