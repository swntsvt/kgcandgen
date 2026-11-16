"""Run every step behind the revised manuscript once, in order, and record a manifest.

Usage:
    python scripts/canonical_run.py --label v1.1-sncs-r1 [--data-root PATH] [--skip-efficiency]
    python scripts/canonical_run.py --label v1.1-sncs-r1 --rehash-only

Steps follow REPRODUCE.md. Each step's printed artefact paths feed the next step. The
manifest (results/canonical/<label>/manifest.json) records the commit, whether the working
tree was clean, the environment, every command with its duration and exit status, and the
SHA-256 of every artefact. The run refuses to start on a dirty tree unless --allow-dirty.

--rehash-only re-runs nothing: it recomputes the artefact list and hashes of an existing
complete manifest (keeping its commit and steps) and records when and at which commit the
hashes were recomputed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
CONFIG = "config/runtime.paper.yaml"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Runner:
    def __init__(self, label: str, env: dict[str, str]) -> None:
        self.label = label
        self.env = env
        self.out_dir = REPO / "results" / "canonical" / label
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.steps: list[dict[str, object]] = []

    def run(self, name: str, args: list[str]) -> dict[str, str]:
        """Run one CLI step, tee its output to a log, and return 'Label: path' pairs."""
        command = [PYTHON, "-m", "src.main", *args]
        log_path = self.out_dir / f"{len(self.steps) + 1:02d}_{name}.log"
        print(f"[{datetime.now():%H:%M:%S}] {name}: {' '.join(args)}", flush=True)
        start = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=REPO, env=self.env, stdout=subprocess.PIPE, stderr=log, text=True
            )
            log.write("\n--- stdout ---\n" + completed.stdout)
        seconds = time.perf_counter() - start
        outputs = {
            match.group(1).strip(): _absolute(match.group(2).strip())
            for match in re.finditer(r"^([A-Za-z][^:\n]*?): (\S.*)$", completed.stdout, re.M)
        }
        self.steps.append(
            {
                "name": name,
                "command": ["python", "-m", "src.main", *args],
                "exit_code": completed.returncode,
                "seconds": round(seconds, 1),
                "log": str(log_path.relative_to(REPO)),
                "printed_outputs": outputs,
            }
        )
        self.write_manifest(status="running")
        if completed.returncode != 0:
            raise SystemExit(f"Step '{name}' failed with exit code {completed.returncode}; see {log_path}")
        print(f"    done in {seconds:.0f}s", flush=True)
        return outputs

    def write_manifest(self, *, status: str, artifacts: dict[str, str] | None = None) -> None:
        manifest = {
            "label": self.label,
            "status": status,
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "working_tree_clean": _git("status", "--porcelain", "--untracked-files=no") == "",
            "config": CONFIG,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "environment": {
                key: self.env.get(key, "")
                for key in ("KGCANDGEN_DATA_ROOT", "PYTHONHASHSEED", "HF_HUB_OFFLINE")
            },
            "written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "steps": self.steps,
            "artifacts_sha256": artifacts or {},
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _absolute(value: str) -> str:
    """CLI paths are relative to the repository; make them absolute (non-paths pass through)."""
    path = Path(value)
    if path.is_absolute() or not (REPO / path).exists():
        return value
    return str((REPO / path).resolve())


def _query_csv(results_csv: str) -> str:
    sys.path.insert(0, str(REPO))
    from src.experiments.query_records import query_output_csv_path

    return str(query_output_csv_path(Path(results_csv)))


def _errors_csv(results_csv: str) -> str:
    path = Path(results_csv)
    return str(path.with_name(f"{path.stem}_errors.csv"))


def _collect_artifacts(
    *, dev: str, selected: str, kg_results: str, cls_results: str, kg_grid: str, cls_grid: str, out: Path
) -> dict[str, str]:
    """SHA-256 of every result, query, errors and report file the run produced."""
    results = [dev, kg_results, cls_results, kg_grid, cls_grid]
    candidates = {selected}
    candidates |= set(results) | {_query_csv(path) for path in results} | {_errors_csv(path) for path in results}
    candidates |= {str(path) for path in out.glob("*.csv")}
    return {
        str(Path(path).resolve().relative_to(REPO)): _sha256(Path(path))
        for path in sorted(candidates)
        if Path(path).exists()
    }


def _grid_paths(label: str) -> tuple[str, str]:
    return (
        str(REPO / "results" / f"heldout_grid_result_{label}.csv"),
        str(REPO / "results" / f"heldout_class_grid_result_{label}.csv"),
    )


def rehash(label: str) -> int:
    """Recompute the artefact hashes of an existing complete run without re-running it."""
    out = REPO / "results" / "canonical" / label
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise SystemExit(f"{manifest_path} is not a complete run; re-run it instead.")
    printed = {step["name"]: step["printed_outputs"] for step in manifest["steps"]}
    kg_grid, cls_grid = _grid_paths(label)
    artifacts = _collect_artifacts(
        dev=printed["development"]["Results CSV"],
        selected=printed["heldout_kg"]["Selected Settings JSON"],
        kg_results=printed["heldout_kg"]["Held-Out KG Results CSV"],
        cls_results=printed["heldout_bioml"]["Held-Out Class-Only Results CSV"],
        kg_grid=kg_grid,
        cls_grid=cls_grid,
        out=out,
    )
    previous = manifest.get("artifacts_sha256", {})
    changed = sorted(path for path in previous if path in artifacts and previous[path] != artifacts[path])
    missing = sorted(set(previous) - set(artifacts))
    if changed or missing:
        raise SystemExit(f"Artefacts changed since the run (changed={changed}, missing={missing}); not rewriting.")
    added = sorted(set(artifacts) - set(previous))
    manifest["artifacts_sha256"] = artifacts
    manifest["artifacts_rehashed"] = {
        "at_commit": _git("rev-parse", "HEAD"),
        "working_tree_clean": _git("status", "--porcelain", "--untracked-files=no") == "",
        "written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "added": added,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Rehashed {len(artifacts)} artefacts ({len(added)} added, existing hashes unchanged): {manifest_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True)
    parser.add_argument("--data-root", default=None, help="Sets KGCANDGEN_DATA_ROOT.")
    parser.add_argument("--skip-efficiency", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--rehash-only", action="store_true", help="Recompute hashes of an existing run.")
    args = parser.parse_args(argv)

    if args.rehash_only:
        return rehash(args.label)

    if not args.allow_dirty and _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("Working tree has uncommitted changes; commit them or pass --allow-dirty.")

    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("HF_HUB_OFFLINE", "1")
    if args.data_root:
        env["KGCANDGEN_DATA_ROOT"] = str(Path(args.data_root).resolve())
    data_root = Path(env.get("KGCANDGEN_DATA_ROOT", REPO / "data" / "oaei_track_cache"))

    runner = Runner(args.label, env)
    out = runner.out_dir
    verify = subprocess.run(
        [PYTHON, "scripts/fetch_oaei.py", "--data-root", str(data_root)],
        cwd=REPO, capture_output=True, text=True,
    )
    (out / "00_verify_data.log").write_text(verify.stdout + verify.stderr, encoding="utf-8")
    if verify.returncode != 0:
        raise SystemExit("Benchmark files failed verification; see 00_verify_data.log")
    print(verify.stdout.strip(), flush=True)

    dev = runner.run("development", ["full-run", "--config-path", CONFIG])["Results CSV"]
    dev_query = _query_csv(dev)
    runner.run("grid_diagnostics", ["grid-diagnostics", "--results-csv", dev,
                                    "--output-csv", str(out / "grid_diagnostics.csv")])

    kg = runner.run("heldout_kg", ["heldout-full-run", "--config-path", CONFIG, "--dev-results-csv", dev])
    selected = kg["Selected Settings JSON"]
    kg_results = kg["Held-Out KG Results CSV"]
    cls = runner.run("heldout_bioml", ["heldout-class-full-run", "--config-path", CONFIG,
                                       "--dev-results-csv", dev])
    cls_results = cls["Held-Out Class-Only Results CSV"]

    kg_grid, cls_grid = _grid_paths(args.label)
    runner.run("heldout_kg_full_grid", ["run-heldout-kg", "--config-path", CONFIG, "--full-grid",
                                        "--no-progress", "--output-csv-path", kg_grid])
    runner.run("heldout_bioml_full_grid", ["run-heldout-class", "--config-path", CONFIG, "--full-grid",
                                           "--no-progress", "--output-csv-path", cls_grid])
    runner.run("selection_regret", ["selection-regret", "--development-csv", dev,
                                    "--heldout-grid", f"kg={kg_grid}", f"bioml={cls_grid}",
                                    "--output-csv", str(out / "selection_regret.csv")])

    runner.run("paired_inference_heldout", ["paired-inference", "--query-csv", _query_csv(kg_results),
                                            _query_csv(cls_results),
                                            "--output-csv", str(out / "paired_inference_heldout.csv")])
    runner.run("paired_inference_development", ["paired-inference", "--query-csv", dev_query,
                                                "--selected-settings-json", selected,
                                                "--output-csv", str(out / "paired_inference_development.csv")])
    runner.run("dataset_accounting", ["dataset-accounting", "--config-path", CONFIG,
                                      "--output-csv-path", str(out / "dataset_accounting.csv")])
    if not args.skip_efficiency:
        runner.run("efficiency", ["efficiency-benchmark", "--config-path", CONFIG,
                                  "--selected-settings-json", selected,
                                  "--output-csv-path", str(out / "efficiency.csv")])

    artifacts = _collect_artifacts(
        dev=dev, selected=selected, kg_results=kg_results, cls_results=cls_results,
        kg_grid=kg_grid, cls_grid=cls_grid, out=out,
    )
    runner.write_manifest(status="complete", artifacts=artifacts)
    print(f"Canonical run complete. Manifest: {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
