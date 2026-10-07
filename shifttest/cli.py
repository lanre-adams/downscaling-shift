"""Command line: python -m shifttest run|serve|export."""
from __future__ import annotations

import argparse
import json
import sys

from .config import Config, preset


def _progress(msg: str, frac: float) -> None:
    print(f"[{frac * 100:5.1f}%] {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="shifttest", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the shift test and write results/ (JSON, figures, report)")
    r.add_argument("--preset", default="default", help="quick | default | future")
    r.add_argument("--config", help="JSON file with Config fields (overrides preset)")
    r.add_argument("--source", choices=["synthetic", "real"])
    r.add_argument("--coarse", help="real data: predictors NetCDF")
    r.add_argument("--fine", help="real data: target rainfall NetCDF")
    r.add_argument("--warming-rate", type=float)
    r.add_argument("--seed", type=int)
    r.add_argument("--out", default="results/custom", help="output folder (default results/custom)")

    w = sub.add_parser("sweep", help="run the three warming scenarios into results/<scenario>/")
    w.add_argument("--root", default="results")
    w.add_argument("--presets", nargs="*", help="subset of: default future stress")
    w.add_argument("--quick", action="store_true", help="use small, fast settings (CI)")

    s = sub.add_parser("serve", help="start the web dashboard")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)

    e = sub.add_parser("export", help="build a static site (e.g. for GitHub Pages) from results/")
    e.add_argument("--results", default="results")
    e.add_argument("--out", default="site")

    a = ap.parse_args(argv)
    if a.cmd == "run":
        cfg = preset(a.preset)
        if a.config:
            with open(a.config) as fh:
                cfg = Config.from_dict({**cfg.to_dict(), **json.load(fh)})
        for k, v in (("source", a.source), ("coarse_path", a.coarse), ("fine_path", a.fine),
                     ("warming_rate", a.warming_rate), ("seed", a.seed), ("out_dir", a.out)):
            if v is not None:
                setattr(cfg, k, v)
        if a.coarse or a.fine:
            cfg.source = "real"
        from .pipeline import run
        res = run(cfg, _progress)
        from pathlib import Path
        from .pipeline import update_index
        out = Path(cfg.out_dir)
        update_index(out.parent, out.name, f"Custom run: {out.name}", res)
        print("\nFindings:")
        for f in res["findings"]:
            print(" -", f)
        print(f"\nWrote {cfg.out_dir}/results.json, report.md, report.html and figures/")
        return 0
    if a.cmd == "sweep":
        from .config import PRESETS
        from .pipeline import sweep
        over = {k: v for k, v in PRESETS["quick"].items()} if a.quick else None
        idx = sweep(a.root, a.presets, over, _progress)
        for r in idx["runs"]:
            print(f"  {r['id']:8s} shift {r['shift_K']:+.2f} K  -> {a.root}/{r['path']}")
        return 0
    if a.cmd == "serve":
        import uvicorn
        uvicorn.run("shifttest.server:app", host=a.host, port=a.port)
        return 0
    if a.cmd == "export":
        from .server import export_static
        path = export_static(a.results, a.out)
        print(f"Static site written to {path}")
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
