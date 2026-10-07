"""Builds the written report from results.json.

One list of blocks is rendered twice: report.md (for GitHub) and report.html
(a fragment the dashboard embeds). Every number in the text is read from the
results, so the report cannot drift from the run that produced it.
"""
from __future__ import annotations

import html
from pathlib import Path

LABEL = {"qm": "Quantile mapping", "gbm": "ML downscaler", "gbm_phys": "ML + warming prior"}


def _f(x, fmt="{:+.1f}"):
    try:
        if x != x:  # NaN
            return "n/a"
        return fmt.format(x)
    except (TypeError, ValueError):
        return str(x)


def _span(years):
    return f"{years[0]}–{years[-1]}" if len(years) > 1 else str(years[0])


def build_blocks(r: dict) -> list[tuple]:
    m, s, cfg = r["meta"], r["scores"], r["meta"]["config"]
    synthetic = m["source"] == "synthetic"
    sp = r["splits"]
    B: list[tuple] = []

    B.append(("h1", "Does it survive a warmer climate?"))
    B.append(("p", "A distribution-shift test of machine-learning rainfall downscaling over West Africa."))
    if synthetic:
        B.append(("note", "Data status: synthetic. Every number below comes from a simulated "
                  "'West African monsoon' world built into this repository, with a warming trend of "
                  f"{cfg['warming_rate']} K per year chosen by the user. These results demonstrate that the test "
                  "harness works and show how the models behave under a known shift. They are not findings "
                  "about the real climate. Run the real-data path (ERA5 predictors, CHIRPS rainfall) to "
                  "produce evidence about West Africa."))
    else:
        B.append(("note", "Data status: real data. Coarse predictors and target rainfall were read from "
                  f"{html.escape(cfg['coarse_path'])} and {html.escape(cfg['fine_path'])}. This is perfect-"
                  "prognosis downscaling against an observational product, not emulation of a "
                  "convection-permitting model."))

    B.append(("h2", "Question"))
    B.append(("p", "Machine-learning downscalers learn the link between coarse, large-scale weather and local "
              "rainfall from past data. Climate projection asks them to apply that link in a warmer future "
              "they have never seen. This study asks: when a downscaler is trained on an earlier period, how "
              "much does its skill on extreme rainfall degrade on a later, warmer period, and does a simple "
              "physical prior help?"))

    B.append(("h2", "Design"))
    B.append(("table", ["Set", "Years", "Days", "Warming vs training (K)"], [
        ["Training", _span(sp["train"]["years"]), sp["train"]["n_days"], _f(sp["train"]["mean_dT"], "{:+.2f}")],
        ["In-distribution test (held-out early years)", ", ".join(map(str, sp["in_dist"]["years"])),
         sp["in_dist"]["n_days"], _f(sp["in_dist"]["mean_dT"], "{:+.2f}")],
        ["Shifted test (later years)", _span(sp["shifted"]["years"]), sp["shifted"]["n_days"],
         _f(sp["shifted"]["mean_dT"], "{:+.2f}")],
    ]))
    g = m["grid"]
    B.append(("p", f"Domain {cfg['lat_min']}–{cfg['lat_max']}°N, {cfg['lon_min']}–{cfg['lon_max']}°E, "
              f"June–September. Fine grid {g['fine'][0]}×{g['fine'][1]}; coarse grid {g['coarse'][0]}×{g['coarse'][1]} "
              f"(each coarse cell covers {g['factor']}×{g['factor']} fine cells). Predictors: moisture, "
              "large-scale ascent, temperature and coarse rainfall, bilinearly interpolated to the fine grid, "
              "plus elevation, position and season."))
    B.append(("list", [
        "Quantile mapping (baseline): interpolate coarse rainfall, then map its distribution onto the observed "
        "distribution pixel by pixel. Values beyond the training range are extended additively.",
        "ML downscaler: gradient-boosted trees with a Poisson loss, trained on "
        f"{cfg['train_rows']:,} sampled pixel-days. Trees cannot predict beyond the range they were trained on.",
        f"ML + warming prior: the same model trained after rescaling moisture, rainfall and temperature to the "
        f"training climate with a prior of {100 * cfg['physics_beta']:.0f}% per K (Clausius–Clapeyron), then "
        "scaled back. This is the simplest way to put physical knowledge into the model.",
        "ML 10–90% interval: the same features with quantile losses, scored by how often observations fall inside.",
    ]))

    B.append(("h2", "Results"))
    rows = []
    for k in ("qm", "gbm", "gbm_phys"):
        a, b = s["in_dist"][k], s["shifted"][k]
        rows.append([LABEL[k], _f(a["mean_bias_pct"]), _f(b["mean_bias_pct"]), _f(a["p99_bias_pct"]),
                     _f(b["p99_bias_pct"]), _f(a["rmse"], "{:.2f}"), _f(b["rmse"], "{:.2f}"),
                     _f(b["small_scale_ratio"], "{:.2f}")])
    B.append(("table", ["Model", "Mean bias % (in-dist)", "Mean bias % (shifted)", "P99 bias % (in-dist)",
                        "P99 bias % (shifted)", "RMSE (in-dist)", "RMSE (shifted)", "Fine-scale power ratio"], rows))
    B.append(("p", "Bias is (model − observed) ÷ observed. P99 is the 99th percentile of all pixel-day "
              "rainfall. The fine-scale power ratio compares spatial variability at scales finer than the "
              "coarse grid: 1.00 is right, below 1 is too smooth."))
    B.append(("h3", "What the numbers say"))
    B.append(("list", r["findings"]))

    ch = r["change"]
    B.append(("h3", "Does the model reproduce the change, not just the climate?"))
    B.append(("table", ["Model", "Observed P99 change %", "Predicted P99 change %", "Share of change captured"],
              [[LABEL[k], _f(ch[k]["obs_p99_change_pct"]), _f(ch[k]["pred_p99_change_pct"]),
                _f(ch[k]["p99_captured_pct"], "{:.0f}%")] for k in ("qm", "gbm", "gbm_phys")]))
    iv = s["in_dist"]["interval"], s["shifted"]["interval"]
    B.append(("p", f"Uncertainty interval: the ML 10–90% interval contains {100 * iv[0]['coverage_wet']:.0f}% "
              f"of wet-day observations in-distribution and {100 * iv[1]['coverage_wet']:.0f}% on the shifted "
              f"period; {100 * iv[1]['above_upper_wet']:.0f}% of shifted wet days fall above its upper bound "
              "(nominal 80% inside, 10% above)."))
    for fig in r["figures"]:
        B.append(("figure", fig["file"], fig["title"], fig["caption"]))

    if r.get("drywet"):
        dw = r["drywet"]
        B.append(("h3", "Second test: trained on dry years, tested on wet years"))
        B.append(("table", ["Model", "Mean bias %", "P99 bias %", "RMSE"],
                  [[LABEL[k], _f(v["mean_bias_pct"]), _f(v["p99_bias_pct"]), _f(v["rmse"], "{:.2f}")]
                   for k, v in dw["scores"].items()]))
        B.append(("p", f"Training years ({len(dw['dry_years'])} driest seasons) versus test years "
                  f"({len(dw['wet_years'])} wettest). This shift comes from natural variability rather than "
                  "warming, so the warming prior is not expected to help here."))

    B.append(("h2", "Interpretation"))
    if synthetic:
        B.append(("p", "In this synthetic world, storm intensity grows faster with warming than moisture "
                  f"does (by construction: {100 * cfg['cc_scaling']:.0f}% per K for moisture plus "
                  f"{100 * cfg['extreme_scaling']:.0f}% per K for storms). The warming prior assumes "
                  f"{100 * cfg['physics_beta']:.0f}% per K, so it is deliberately incomplete. Any advantage it "
                  "shows here is partly built in; the useful lesson is the test itself: in-distribution scores "
                  "can look fine while the change signal and the extremes go wrong."))
    B.append(("p", "The test generalises directly to the real problem: train an emulator on convection-"
              "permitting simulations of one period and check it, period by period, on another, scoring the "
              "change signal and extremes rather than only average skill."))

    B.append(("h2", "Limits"))
    B.append(("list", [
        "Synthetic mode is a toy world with simple, chosen physics. It is not a climate model."
        if synthetic else "Observed gridded rainfall over Africa has its own errors, especially for extremes "
        "and in data-sparse regions.",
        "Perfect-prognosis downscaling against observations is not emulation of a convection-permitting model; "
        "the Met Office CPM ensemble for Africa that the PhD would use is not public.",
        "A shift of about a degree over a few decades is a weak proxy for end-of-century warming; the test "
        "probes the method, not future climate.",
        "Gradient-boosted trees stand in for the deep models (U-Net, diffusion) used in current emulator work; "
        "they make the extrapolation failure easy to see but are not the state of the art.",
        "Pixel-wise metrics ignore whether storms are organised realistically (e.g. mesoscale convective "
        "systems); object-based evaluation is a next step.",
    ]))

    B.append(("h2", "Next steps"))
    B.append(("list", [
        "Run on real data: ERA5 predictors and CHIRPS rainfall for 1981–present (scripts included).",
        "Add a convolutional U-Net and a probabilistic generative model; compare their shift penalty with the trees.",
        "Add storm-object metrics (size, intensity, lifetime) relevant to mesoscale convective systems.",
        "With convection-permitting model output: train on present-day simulations, test on future simulations.",
    ]))

    B.append(("h2", "Reproducibility and disclosure"))
    B.append(("p", f"Generated {m['generated_at']} by downscaling-shift-test {m['version']} (Python {m['python']}) "
              f"in {m['runtime_s']} s. Seed {cfg['seed']}. The full configuration is stored in results.json; "
              "rerunning with the same configuration reproduces these numbers."))
    B.append(("p", "AI assistance: the code and the report template in this repository were written with the "
              "help of an AI assistant (Claude, Anthropic) and reviewed by the author. All numbers are "
              "computed by the code at run time."))

    B.append(("h2", "References"))
    B.append(("list", [
        "Senior, C. A. et al. (2021). Convection-permitting regional climate change simulations for "
        "understanding future climate and informing decision-making in Africa. Bulletin of the American "
        "Meteorological Society, 102(6).",
        "Kendon, E. J. et al. (2025). Potential for machine learning emulators to augment regional climate "
        "simulations in provision of local climate change information. Bulletin of the American "
        "Meteorological Society, 106(6).",
        "Addison, H., Kendon, E., Ravuri, S., Aitchison, L. & Watson, P. A. G. Machine learning emulation of "
        "precipitation from km-scale regional climate simulations using a diffusion model. arXiv:2407.14158.",
        "Hersbach, H. et al. (2020). The ERA5 global reanalysis. Quarterly Journal of the Royal "
        "Meteorological Society, 146(730).",
        "Funk, C. et al. (2015). The climate hazards infrared precipitation with stations (CHIRPS). "
        "Scientific Data, 2, 150066.",
    ]))
    return B


def to_markdown(blocks) -> str:
    out = []
    for b in blocks:
        kind = b[0]
        if kind in ("h1", "h2", "h3"):
            out.append("#" * int(kind[1]) + " " + b[1])
        elif kind == "p":
            out.append(b[1])
        elif kind == "note":
            out.append("> " + b[1])
        elif kind == "list":
            out.append("\n".join(f"- {x}" for x in b[1]))
        elif kind == "table":
            head, rows = b[1], b[2]
            lines = ["| " + " | ".join(map(str, head)) + " |", "|" + "---|" * len(head)]
            lines += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
            out.append("\n".join(lines))
        elif kind == "figure":
            out.append(f"![{b[2]}]({b[1]})\n\n*{b[3]}*")
    return "\n\n".join(out) + "\n"


def to_html(blocks) -> str:
    e = html.escape
    out = []
    for b in blocks:
        kind = b[0]
        if kind in ("h1", "h2", "h3"):
            out.append(f"<{kind}>{e(b[1])}</{kind}>")
        elif kind == "p":
            out.append(f"<p>{e(b[1])}</p>")
        elif kind == "note":
            out.append(f'<aside class="status-note">{e(b[1])}</aside>')
        elif kind == "list":
            out.append("<ul>" + "".join(f"<li>{e(str(x))}</li>" for x in b[1]) + "</ul>")
        elif kind == "table":
            head = "".join(f"<th scope=\"col\">{e(str(h))}</th>" for h in b[1])
            body = "".join("<tr>" + "".join(
                (f"<th scope=\"row\">{e(str(c))}</th>" if i == 0 else f"<td>{e(str(c))}</td>")
                for i, c in enumerate(r)) + "</tr>" for r in b[2])
            out.append(f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')
        elif kind == "figure":
            out.append(f'<figure><img src="{e(b[1])}" alt="{e(b[2])}" loading="lazy">'
                       f"<figcaption><strong>{e(b[2])}.</strong> {e(b[3])}</figcaption></figure>")
    return "\n".join(out)


def write_report(results: dict, out: Path) -> None:
    blocks = build_blocks(results)
    (Path(out) / "report.md").write_text(to_markdown(blocks), encoding="utf-8")
    (Path(out) / "report.html").write_text(to_html(blocks), encoding="utf-8")
