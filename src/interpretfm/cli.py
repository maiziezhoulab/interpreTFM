"""The ``interpretfm`` command.

::

    interpretfm validate config.yaml [--env-file paths.env]   check a configuration (schema and referenced files)
    interpretfm run config.yaml [--stages a,b] [--force] [--dry-run] [--env-file paths.env]
    interpretfm status run_dir/               stage status from the manifests
    interpretfm report run_dir/               (re)build the report of a run whose stages are done
    interpretfm version
    interpretfm extract --config <yaml> [--run-dir <dir>] [--model <name>] [--layers a,b] [--env-file paths.env]
                                              the extract stage in this interpreter (what the runner starts in model.env)
    interpretfm stage <name> --config <resolved.yaml> --run-dir <dir>   (internal: one stage, in-process)

The 0.9 module entry points (``python -m interp_pipeline....``) are unchanged.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from interp_pipeline import __version__
from interpretfm.config import ConfigError, check_files, load_config
from interpretfm.runner import StageError, list_stages, run, run_stage_inprocess, status


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="interpretfm", description="interpreTFM pipeline (v1, in development)")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", metavar="command")
    sub.required = True

    v = sub.add_parser("validate", help="validate a configuration file")
    v.add_argument("config")
    v.add_argument("--no-files", action="store_true", help="do not check that referenced files exist")
    v.add_argument("--env-file", help="KEY=VALUE file supplying ${NAME} placeholders of the configuration")

    r = sub.add_parser("run", help="run the pipeline (cached and resumable)")
    r.add_argument("config")
    r.add_argument("--stages", help="comma-separated subset in pipeline order, e.g. ingest,extract,tis")
    r.add_argument("--force", action="store_true", help="re-run the selected stages even when cached")
    r.add_argument("--dry-run", action="store_true", help="show what would run and exit")
    r.add_argument("--env-file", help="KEY=VALUE file supplying ${NAME} placeholders of the configuration")

    s = sub.add_parser("status", help="show the status of every stage of a run")
    s.add_argument("run_dir")

    rp = sub.add_parser("report", help="(re)build the report of a run")
    rp.add_argument("run_dir")

    sub.add_parser("version", help="print the package version")

    ex = sub.add_parser("extract", help="run the extract stage in this interpreter (the runner starts it in model.env)")
    ex.add_argument("--config", required=True, help="the configuration (or a run's config.resolved.yaml)")
    ex.add_argument("--run-dir", help="run directory (default: run.out_dir of the configuration)")
    ex.add_argument("--model", help="expected model.name (checked against the configuration)")
    ex.add_argument("--layers", help="expected model.layers, comma-separated (checked against the configuration)")
    ex.add_argument("--env-file", help="KEY=VALUE file supplying ${NAME} placeholders of the configuration")

    hb = sub.add_parser("hub", help="export trained SAEs to a Hugging Face-ready folder, or check such a folder")
    hsub = hb.add_subparsers(dest="hub_command", metavar="subcommand")
    hsub.required = True
    he = hsub.add_parser("export", help="write <out_dir>/<model>/<dataset>/<layer>/ (safetensors, config, features, metrics, model card) per run")
    he.add_argument("run_dirs", nargs="+", metavar="run_dir", help="finished run directories (the last argument is out_dir)")
    he.add_argument("--layers", help="comma-separated subset of model.layers (default: all)")
    he.add_argument("--repo-id", default=None, help="repository id written into the cards (default: the release repository, YunfeiHu/interpreTFM)")
    hv = hsub.add_parser("verify", help="reload every exported layer of the runs and compare it with the run's checkpoint")
    hv.add_argument("run_dirs", nargs="+", metavar="run_dir", help="run directories (the last argument is the export folder)")

    st = sub.add_parser("stage", help=argparse.SUPPRESS)
    st.add_argument("name")
    st.add_argument("--config", required=True)
    st.add_argument("--run-dir", required=True)
    return p


def _print_plan(rows) -> None:
    w = max(len(r["stage"]) for r in rows) if rows else 8
    print(f"{'stage':<{w}}  {'env':<5}  {'action':<12}  reason")
    for r in rows:
        print(f"{r['stage']:<{w}}  {r['env']:<5}  {r.get('status') or r['action']:<12}  {r['reason']}")


def cmd_validate(args) -> int:
    try:
        cfg = load_config(args.config, env_file=args.env_file)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 2
    problems = [] if args.no_files else check_files(cfg)
    stage_names = {s.name for s in list_stages()}
    for name in cfg.stages:
        if name not in stage_names:
            problems.append(f"stages.{name}: no registered stage of that name")
    if problems:
        print("configuration is valid but refers to missing files:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        return 3
    mode = "spatial" if cfg.dataset.is_spatial else "scRNA-seq (spatial stages skipped)"
    print(f"OK: {args.config}\n  run: {cfg.run.name} -> {cfg.run.out_dir}\n  dataset: {cfg.dataset.h5ad} [{mode}]"
          f"\n  model: {cfg.model.name}, {len(cfg.model.layers)} layer(s), env: {cfg.model.env.kind}")
    return 0


def cmd_run(args) -> int:
    try:
        cfg = load_config(args.config, env_file=args.env_file)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 2
    stages = [s.strip() for s in args.stages.split(",") if s.strip()] if args.stages else None
    try:
        rows = run(cfg, stages=stages, force=args.force, dry_run=args.dry_run)
    except StageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    _print_plan(rows)
    return 0


def cmd_status(args) -> int:
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"not a run directory: {run_dir}", file=sys.stderr)
        return 2
    rows = status(run_dir)
    print(f"{'stage':<10} {'status':<8} {'duration_s':>10}  {'started':<26} error")
    for r in rows:
        dur = "" if r["duration_s"] is None else f"{r['duration_s']:.1f}"
        print(f"{r['stage']:<10} {r['status']:<8} {dur:>10}  {r['started']:<26} {r['error']}")
    return 0


def cmd_report(args) -> int:
    """(Re)build the report of a run from its resolved configuration and finished stages."""
    from interpretfm.runner import RESOLVED_CONFIG, stage_dir

    run_dir = Path(args.run_dir)
    resolved = run_dir / RESOLVED_CONFIG
    if not resolved.is_file():
        print(f"not a run directory (no {RESOLVED_CONFIG}): {run_dir}", file=sys.stderr)
        return 2
    try:
        cfg = load_config(resolved)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 2
    try:
        m = run_stage_inprocess(cfg, "report", run_dir)
    except StageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    out = stage_dir(run_dir, "report")
    print(f"report {m.status}: {out / 'report.json'}" + (f", {out / 'report.html'}" if cfg.output.html else ""))
    return 0


def cmd_stage(args) -> int:
    cfg = load_config(args.config)
    try:
        m = run_stage_inprocess(cfg, args.name, Path(args.run_dir))
    except StageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"stage {args.name}: {m.status}")
    return 0


def cmd_extract(args) -> int:
    """The extract stage in this interpreter: the entry point the runner starts inside ``model.env``."""
    try:
        cfg = load_config(args.config, env_file=args.env_file)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 2
    if args.model and args.model != cfg.model.name:
        print(f"error: --model {args.model!r} does not match model.name {cfg.model.name!r} in {args.config}", file=sys.stderr)
        return 2
    if args.layers:
        want = [s.strip() for s in args.layers.split(",") if s.strip()]
        if want != list(cfg.model.layers):
            print(f"error: --layers {want} do not match model.layers {cfg.model.layers} in {args.config}", file=sys.stderr)
            return 2
    run_dir = Path(args.run_dir) if args.run_dir else cfg.run.out_dir
    try:
        m = run_stage_inprocess(cfg, "extract", run_dir)
    except StageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"stage extract: {m.status}")
    return 0


def cmd_hub(args) -> int:
    from interpretfm import hub

    *run_dirs, out_dir = args.run_dirs
    if not run_dirs:
        print("hub: give at least one run directory and the output folder", file=sys.stderr)
        return 2
    if args.hub_command == "export":
        layers = [s.strip() for s in args.layers.split(",")] if args.layers else None
        repo_id = args.repo_id or hub.DEFAULT_REPO
        for rd in run_dirs:
            for d in hub.export_run(rd, out_dir, layers=layers, repo_id=repo_id):
                print(f"exported {d}")
        print(f"index {hub.write_index(out_dir, repo_id=repo_id)}")
        return 0
    rows = hub.verify_export(out_dir, run_dirs)
    for r in rows:
        print(f"{r['model']:12s} {r['dataset']:8s} {r['layer']:16s} {'identical' if r['identical'] else 'DIFFERENT'}  ({', '.join(r['tensors'])})")
    return 0 if rows and all(r["identical"] for r in rows) else 1


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s", stream=sys.stderr)
    if args.command == "version":
        print(__version__)
        return 0
    return {"validate": cmd_validate, "run": cmd_run, "status": cmd_status, "report": cmd_report,
            "stage": cmd_stage, "extract": cmd_extract, "hub": cmd_hub}[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
