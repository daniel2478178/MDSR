#!/usr/bin/env python3
"""LLM-MDSR paired benchmark figures. See --help. Dependencies: numpy pandas openpyxl matplotlib."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import LOGS_ROOT

COLORS = ["#3267A8", "#E48A32", "#379582"]
LABELS = ["0%", "1%", "3%"]
LEVELS = ["Low", "Medium", "Relatively High", "High"]
NOISE = ["0", "001", "003", "005", "01"]
WASS = ["w0", "w025", "w04"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=LOGS_ROOT)
    p.add_argument("--files", type=Path, nargs=3, metavar=("TRAIN000", "TRAIN001", "TRAIN003"))
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--failure-cutoff", type=float, default=-1e50,
                   help="R2 <= cutoff is a flagged extreme result (default -1e50); not proof of numerical failure.")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--epsilon", type=float, default=1e-30,
                   help="Floor for displaying -log10(NMSE); raw CSV values are preserved.")
    args = p.parse_args()
    if args.bootstrap < 1 or args.epsilon <= 0 or args.failure_cutoff >= 0:
        p.error("bootstrap >= 1, epsilon > 0, failure-cutoff < 0 required")
    files = args.files or [args.input_dir / f"LLM_MDSR_{t}.xlsx" for t in ["000", "001", "003"]]
    out = args.output_dir or files[0].resolve().parent / "LLM_MDSR_figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "pdf.fonttype": 42})
    frames = []
    required = ["ID", "GenerationFormula", "formula", "SimilarityScore", "SimilarityLevel",
                "score", "complexity", "trained_mean_nmse"]
    required += [f"noise_{s}_{m}" for s in NOISE + WASS for m in ["R2", "NMSE"]]
    for file in files:
        if not file.exists():
            raise FileNotFoundError(f"Input missing: {file}")
        d = pd.read_excel(file, sheet_name=0)
        d.columns = d.columns.astype(str).str.strip()
        missing = sorted(set(required) - set(d.columns))
        if missing:
            raise ValueError(f"{file.name}: missing columns {missing}")
        if d.ID.isna().any():
            raise ValueError(f"{file.name}: blank ID")
        d["ID"] = d.ID.astype(str).str.strip()
        if d.ID.duplicated().any() or d.ID.eq("").any():
            raise ValueError(f"{file.name}: duplicate or empty ID")
        for col in required:
            if col not in ["ID", "GenerationFormula", "formula", "SimilarityLevel"]:
                d[col] = pd.to_numeric(d[col], errors="raise")
        d["SimilarityLevel"] = d.SimilarityLevel.astype(str).str.strip()
        unknown = set(d.SimilarityLevel) - set(LEVELS)
        if unknown:
            raise ValueError(f"Unknown similarity levels: {unknown}")
        frames.append(d.set_index("ID"))
    ids = sorted(set().union(*(set(d.index) for d in frames)))
    # Align every plot and bootstrap resample by problem ID.
    frames = [d.reindex(ids) for d in frames]
    n = len(ids)
    rng = np.random.default_rng(args.seed)
    samples = rng.integers(0, n, (args.bootstrap, n))
    summary, audit = [], []

    def save(fig, name):
        fig.savefig(out / f"{name}.png", bbox_inches="tight")
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {name}", flush=True)

    def csv(d, name):
        d.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")

    def status(d, s):
        r = d[f"noise_{s}_R2"].to_numpy(float)
        e = d[f"noise_{s}_NMSE"].to_numpy(float)
        missing = np.isnan(r) | np.isnan(e)
        extreme = ~missing & ((~np.isfinite(r)) | (~np.isfinite(e)) |
                              (r <= args.failure_cutoff) | (e < 0) | (r > 1 + 1e-8))
        return r, e, missing, extreme

    def values(d, s, metric):
        r, e, missing, extreme = status(d, s)
        v = r.copy() if metric == "r2" else -np.log10(np.maximum(e, args.epsilon))
        v[missing | extreme] = np.nan
        return v

    def interval(v):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(v)
            boot = np.nanmedian(v[samples], axis=1)
        finite = boot[np.isfinite(boot)]
        return (med, *np.quantile(finite, [.025, .975])) if len(finite) else (np.nan,)*3

    for t, d in enumerate(frames):
        for s in NOISE + WASS:
            r, e, missing, extreme = status(d, s)
            valid = ~missing & ~extreme
            row = dict(train_noise=LABELS[t], test=s, total=n,
                       valid=int(valid.sum()), missing=int(missing.sum()), extreme=int(extreme.sum()))
            row["median_r2"], row["ci_low"], row["ci_high"] = interval(values(d, s, "r2"))
            for threshold in [.99, .9]:
                # Missing and flagged outcomes are unsuccessful in the primary denominator.
                count = int((valid & (r >= threshold)).sum())
                row[f"success_{threshold}"] = count
                row[f"rate_{threshold}"] = count/n
            summary.append(row)
            for i, pid in enumerate(ids):
                audit.append(dict(ID=pid, train_noise=LABELS[t], test=s, R2=r[i], NMSE=e[i],
                                  status="missing" if missing[i] else "flagged_extreme" if extreme[i] else "valid"))
    csv(pd.DataFrame(audit), "test_results_and_status")
    csv(pd.DataFrame(summary), "test_summary")
    combined = pd.concat([d.assign(train_noise=LABELS[t]).reset_index()
                          for t, d in enumerate(frames)], ignore_index=True)
    csv(combined, "aligned_source")
    csv(combined.groupby("train_noise", sort=False)[
        ["SimilarityScore", "complexity", "score", "trained_mean_nmse"]].agg(
            ["count", "median", "mean", "min", "max"]).reset_index(), "formula_summary")

    # Figure 1: levels and paired score changes.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    bottom = np.zeros(3)
    for level, color in zip(LEVELS, ["#BDC5CE", "#F2CB82", "#7EB4B2", "#3267A8"]):
        counts = np.array([d.SimilarityLevel.eq(level).sum() for d in frames])
        heights = counts/n*100
        axes[0].bar(LABELS, heights, bottom=bottom, label=level, color=color)
        for j, count in enumerate(counts):
            if count:
                axes[0].text(j, bottom[j]+heights[j]/2, str(count), ha="center", va="center")
        bottom += heights
    axes[0].set(ylabel="Share of all problems (%)", xlabel="Training noise", ylim=(0, 100))
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, 1.18), ncol=2)
    scores = np.column_stack([d.SimilarityScore for d in frames])
    axes[1].plot(range(3), scores.T, color="#ADB5BF", alpha=.35, lw=.7)
    axes[1].plot(range(3), np.nanmedian(scores, axis=0), "o-", color=COLORS[0], lw=2, label="Median")
    axes[1].set(xticks=range(3), xticklabels=LABELS, xlabel="Training noise",
                ylabel="Similarity score", ylim=(0, 102))
    axes[1].legend()
    save(fig, "fig01_structure_recovery")

    def curves(name, suffixes, x, xlabel, delta=False):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
        for ax, metric in zip(axes, ["r2", "quality"]):
            for t, d in enumerate(frames):
                triples = []
                for s in suffixes:
                    v = values(d, s, metric)
                    if delta:
                        v = v - values(d, suffixes[0], metric)
                    triples.append(interval(v))
                a = np.array(triples)
                ax.plot(x, a[:, 0], "o-", color=COLORS[t], label=f"Train {LABELS[t]}")
                ax.fill_between(x, a[:, 1], a[:, 2], color=COLORS[t], alpha=.13)
            ax.set(xlabel=xlabel, xticks=x)
            ax.grid(alpha=.18)
            ax.legend(fontsize=8)
        axes[0].set_ylabel(("Change in " if delta else "Median ") + "$R^2$")
        axes[1].set_ylabel(("Change in " if delta else "Median ") + r"$-\log_{10}(\mathrm{NMSE})$")
        fig.suptitle("Median with pointwise 95% bootstrap CI across problem IDs", fontsize=11)
        save(fig, name)

    curves("fig02_noise_robustness", NOISE, [0, 1, 3, 5, 10], "Test noise (%)")
    curves("fig04a_wasserstein_performance", WASS, [0, .25, .4], "Wasserstein distance")
    curves("fig04b_wasserstein_change", WASS, [0, .25, .4], "Wasserstein distance", delta=True)

    def heatmap(name, suffixes, xticks, xlabel):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
        for ax, threshold in zip(axes, [.99, .9]):
            counts = []
            for d in frames:
                counts.append([np.sum(np.isfinite(values(d, s, "r2")) &
                                      (values(d, s, "r2") >= threshold)) for s in suffixes])
            counts = np.array(counts)
            im = ax.imshow(counts/n*100, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
            for i in range(3):
                for j in range(len(suffixes)):
                    ax.text(j, i, f"{counts[i,j]}/{n}\n{counts[i,j]/n:.1%}",
                            ha="center", va="center", fontsize=9,
                            color="white" if counts[i,j]/n > .6 else "black")
            ax.set(xticks=range(len(suffixes)), xticklabels=xticks,
                   yticks=range(3), yticklabels=LABELS, xlabel=xlabel,
                   ylabel="Training noise", title=f"Success: R² ≥ {threshold}")
        fig.colorbar(im, ax=axes, label="Success (%)", shrink=.85)
        save(fig, name)
    heatmap("fig03_noise_success", NOISE, ["0", "1", "3", "5", "10"], "Test noise (%)")
    heatmap("fig04c_wasserstein_success", WASS, ["0", "0.25", "0.40"], "Wasserstein distance")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    cs = np.column_stack([d.complexity for d in frames])
    axes[0].plot(range(3), cs.T, color="#ADB5BF", alpha=.3, lw=.7)
    axes[0].plot(range(3), np.nanmedian(cs, axis=0), "o-", color=COLORS[0], lw=2)
    axes[0].set(xticks=range(3), xticklabels=LABELS, xlabel="Training noise", ylabel="Complexity")
    for t, d in enumerate(frames):
        axes[1].scatter(d.complexity, d.SimilarityScore, color=COLORS[t], alpha=.65, s=22, label=LABELS[t])
        axes[2].scatter(d.complexity, values(d, "0", "quality"), color=COLORS[t], alpha=.65, s=22, label=LABELS[t])
    axes[1].set(xlabel="Complexity", ylabel="Similarity score")
    axes[2].set(xlabel="Complexity", ylabel=r"Clean test $-\log_{10}(\mathrm{NMSE})$")
    axes[1].legend(title="Training noise", fontsize=8)
    save(fig, "fig05_complexity_tradeoff")

    # Within-training-condition rank correlations; no pooling of score scales.
    corr_rows = []
    for t, d in enumerate(frames):
        ys = {"SimilarityScore": d.SimilarityScore.to_numpy(),
              "complexity": d.complexity.to_numpy(),
              "trained_mean_nmse": d.trained_mean_nmse.to_numpy()}
        ys.update({f"{s}_quality": values(d, s, "quality") for s in NOISE+WASS})
        for target, y in ys.items():
            pair = pd.DataFrame({"x":d.score.to_numpy(), "y":y}).replace([np.inf,-np.inf],np.nan).dropna()
            rho = pair.rank().corr().iloc[0,1] if len(pair)>2 else np.nan
            corr_rows.append(dict(train_noise=LABELS[t], target=target, n=len(pair), spearman_rho=rho))
    corr = pd.DataFrame(corr_rows)
    csv(corr, "score_correlations")
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    for t, label in enumerate(LABELS):
        rows = corr[corr.train_noise.eq(label)]
        ax.plot(range(len(rows)), rows.spearman_rho, "o-", color=COLORS[t], label=label)
    ax.set(xticks=range(len(ys)), xticklabels=list(ys), ylabel="Spearman correlation with score", ylim=(-1.05,1.05))
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    ax.axhline(0, color="gray", lw=.7)
    ax.legend(title="Training noise")
    save(fig, "fig06_score_correlations")
    metadata = dict(inputs=[str(f.resolve()) for f in files], total_ids=n,
                    arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    notes=[
                        "R2<=cutoff flags extreme outcomes; it does not establish a numerical failure cause.",
                        "Continuous summaries exclude missing/flagged results; see per-cell valid counts.",
                        "Success fractions use union of IDs; missing/flagged results count as unsuccessful.",
                        "Wasserstein changes subtract w0 within the same ID and training condition.",
                        "Bootstrap resamples problem IDs, not seeds or individual observations.",
                        "Noise/Wasserstein suffixes are explicitly mapped, not inferred numerically.",
                        "Similarity labels are supplied LLM assessments, not proof of symbolic equivalence.",
                        "No theoretical complexity ratio: ground-truth and discovered complexity need a shared definition.",
                        "Score correlations across different problems are exploratory, not within-problem candidate selection validation.",
                        "Test coefficients may have been refitted; these tables alone do not identify the evaluation protocol."
                    ])
    metadata["arguments"]["files"] = [str(f) for f in args.files] if args.files else None
    (out/"analysis_settings.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Completed. Output: {out.resolve()}")


if __name__ == "__main__":
    main()
