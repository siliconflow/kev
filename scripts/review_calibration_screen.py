import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kev.metrics import grouped_metrics, metrics, paired_bootstrap
from kev.suite import digest, read_json, write_json
from scripts.calibration_audit import describe, read_rows, tempered

LOSS_KEYS = ("label_smoothing", "brier_w", "focal_gamma")


def arm_name(config):
    if config.get("label_smoothing", 0):
        return "smoothing-005"
    if config.get("brier_w", 0):
        return "ce-brier-05"
    if config.get("focal_gamma", 0):
        return "focal-1"
    return "ce-control"


def screen_checks(candidate, controls, rule):
    checks = {}
    for name, control in controls.items():
        a, b = candidate["micro"], control["micro"]
        min_gain = rule["coverage_delta_vs_ce_control_min"] if name == "ce-control" else rule["coverage_delta_vs_recalibrated_parent_min"]
        checks[f"coverage_vs_{name}"] = a["coverage_at_5pct_error"] - b["coverage_at_5pct_error"] >= min_gain - 1e-12
        checks[f"accuracy_vs_{name}"] = a["acc"] - b["acc"] >= rule["accuracy_delta_vs_each_min"] - 1e-12
        checks[f"aurc_vs_{name}"] = a["aurc"] - b["aurc"] <= rule["aurc_delta_vs_each_max"] + 1e-12
        checks[f"sources_vs_{name}"] = all(candidate["sources"][source]["acc"] - reference["acc"] >= rule["per_source_accuracy_delta_min"] - 1e-12
                                               for source, reference in control["sources"].items())
    return checks


def load_trial(path):
    result = read_json(path / "result.json")
    if result.get("test_evaluated"):
        raise ValueError("screen cannot select using a trial marked test-evaluated")
    rows = read_rows(path / "transfer/rows.json")
    if not rows or any("logits" not in r or r.get("inference_temperature") != 1.0 for r in rows):
        raise ValueError("screen requires newly captured raw logits")
    fit = result["calibration_fit"]
    if fit["split"] != "calibration" or fit["rows_sha256"] != digest(path / "calibration/rows.json"):
        raise ValueError("temperature fit is not bound to this trial's calibration rows")
    calibrated = tempered(rows, fit["temperature"])
    description = describe(calibrated)
    description["sources"] = grouped_metrics(calibrated, "source")
    return result, rows, calibrated, description


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", default="experiments/calibration-audit-protocol.json")
    ap.add_argument("--samples", type=int, default=2000)
    args = ap.parse_args()
    protocol_path = ROOT / args.protocol
    protocol = read_json(protocol_path)
    study = ROOT / "runs" / args.study
    trials = sorted(p for p in study.iterdir() if (p / "result.json").exists())
    if len(trials) != len(protocol["screen"]["arms"]) + 1:
        raise ValueError("wait for the unchanged parent and all four matched arms; do not select from a partial study")
    results, calibrated_rows, summary = {}, {}, {}
    expected_suite = digest(ROOT / protocol["data"]["decision_suite"] / "manifest.json")
    expected_transfer = digest(ROOT / protocol["data"]["development_suite"] / "manifest.json")
    for path in trials:
        result, raw, cal, info = load_trial(path)
        provenance = result["provenance"]
        config = provenance["config"]
        name = "parent" if provenance["legacy_checkpoint"] else arm_name(config)
        if name in results:
            raise ValueError("duplicate screen arm")
        if provenance["suite_sha256"] != expected_suite or result["transfer"]["suite_sha256"] != expected_transfer:
            raise ValueError("screen data hashes differ from the registered suites")
        if not result["mechanism_checks"]["passed"]:
            raise ValueError("trial failed isolation/parity checks")
        if result["transfer"]["coverage"]["evaluated_records"] != result["transfer"]["coverage"]["requested_records"]:
            raise ValueError("incomplete development coverage")
        info.update(path=str(path.relative_to(ROOT)), result_sha256=digest(path / "result.json"),
                    rows_sha256=digest(path / "transfer/rows.json"), temperature=result["calibration_fit"]["temperature"],
                    raw=metrics(raw), fit=result["calibration_fit"], config=config, resources=result.get("training_resources"))
        results[name], calibrated_rows[name], summary[name] = result, cal, info
    parent_checkpoint = results["parent"]["provenance"]["measured_checkpoint"]
    if parent_checkpoint["requested"] != protocol["parents"]["4b"]:
        raise ValueError("unchanged parent differs from the registered revision")
    control = summary["ce-control"]
    for name, info in summary.items():
        if name == "parent":
            continue
        cfg = info["config"]
        if cfg["init_from"] != protocol["parents"]["4b"]:
            raise ValueError("trial warm-start source differs")
        if {k: v for k, v in cfg.items() if k not in LOSS_KEYS} != {k: v for k, v in control["config"].items() if k not in LOSS_KEYS}:
            raise ValueError("training conditions differ between arms")
        if any(info["resources"][k] != control["resources"][k] for k in ("optimizer_steps", "forward_tokens", "records_seen")):
            raise ValueError("training updates, tokens, or record counts differ between arms")
        expected = next(a for a in protocol["screen"]["arms"] if a["name"] == name)
        if any(cfg[k] != expected[k] for k in LOSS_KEYS):
            raise ValueError("unregistered loss settings")
    comparisons, candidates = {}, []
    for name in [a["name"] for a in protocol["screen"]["arms"] if a["name"] != "ce-control"]:
        checks = screen_checks(summary[name], {k: summary[k] for k in ("parent", "ce-control")}, protocol["screen"]["advance"])
        comparisons[name] = {"checks": checks, "passes": all(checks.values()), "paired": {}}
        for reference in ("parent", "ce-control"):
            comparisons[name]["paired"][reference] = {metric: paired_bootstrap(calibrated_rows[name], calibrated_rows[reference],
                                                                               samples=args.samples, seed=20260921,
                                                                               metric=metric, aggregation="micro")
                                                           for metric in ("acc", "coverage_at_5pct_error", "aurc", "brier")}
        if all(checks.values()):
            candidates.append(name)
    order = {a["name"]: i for i, a in enumerate(protocol["screen"]["arms"])}
    candidates.sort(key=lambda n: (-summary[n]["micro"]["coverage_at_5pct_error"], summary[n]["micro"]["aurc"], summary[n]["micro"]["nll"], order[n]))
    selected = candidates[0] if candidates else None
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    report = {"study": args.study, "protocol": args.protocol, "protocol_sha256": digest(protocol_path),
              "metric_version": 2, "models": summary, "comparisons": comparisons,
              "selected_for_replication": selected, "final_test_evaluated": False,
              "decision": "replicate the selected loss against matched CE controls" if selected else "stop: no arm passed the registered screening gate; final test remains sealed"}
    write_json(out / "report.json", report)
    for name in ("parent", *order):
        info = summary[name]
        m = info["micro"]
        print(f"{name:15} T={info['temperature']:.3f} acc={m['acc']:.4f} cov5={m['coverage_at_5pct_error']:.4f} "
              f"aurc={m['aurc']:.4f} nll={m['nll']:.4f} brier={m['brier']:.4f} ece={m['ece']:.4f}")
    for name, comparison in comparisons.items():
        print(name, "PASS" if comparison["passes"] else "FAIL", [k for k, v in comparison["checks"].items() if not v])
    print(report["decision"])


if __name__ == "__main__":
    main()
