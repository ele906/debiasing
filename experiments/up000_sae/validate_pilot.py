"""Require complete, active, pixel-matched pilot shards before the full sweep."""
import csv
import hashlib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent


def main():
    features = json.loads((RUN / "discovery/top10.json").read_text())["features"][:3]
    path = RUN / "pilot/validation.json"
    checkpoint = RUN / "training/sae/state_dict.pth"
    metrics = {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               "features": features, "shards": {}, "passed": False}
    for feature in features:
        with (RUN / "pilot" / f"feature_{feature}" / "manifest.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        keys = {(r["group"], r["seed"], int(r["alpha"])) for r in rows}
        if len(rows) != 420 or len(keys) != 420:
            raise ValueError(f"Feature {feature} does not have 420 unique conditions")
        controls = [r for r in rows if r["alpha"] == "1"]
        zero = [r for r in rows if r["zero_activation"] == "True"]
        active_edits = [r for r in rows if r["alpha"] != "1" and float(r["delta_fro_applied"]) > 0]
        no_ops = all(r["matches_clean_baseline"] == "True" and float(r["delta_fro_applied"]) == 0
                     for r in controls + zero)
        metrics["shards"][feature] = dict(conditions=len(rows), active_edits=len(active_edits),
                                         zero_activation_conditions=len(zero),
                                         no_ops_passed=no_ops and len(controls) == 60,
                                         matched_mid0_controls=sum(r["clean_matches_mid0_control"] == "True"
                                                                   for r in controls),
                                         matched_dataset_controls=sum(r["clean_matches_historical_baseline"] == "True"
                                                                      for r in controls))
    metrics["passed"] = all(s["no_ops_passed"] and s["active_edits"] and s["matched_mid0_controls"] == 60
                            for s in metrics["shards"].values())
    path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    if not metrics["passed"]:
        raise RuntimeError("Pilot failed; inspect validation.json before launching a full run")


if __name__ == "__main__":
    main()
