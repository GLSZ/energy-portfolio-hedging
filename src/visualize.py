# src/visualize.py

"""
visualize.py — Dashboard complet de visualisation du projet
Portfolio Management & Hedging sur Commodités Énergie.

Figures générées (8 figures) :
  1. Prix & rendements de l'univers d'investissement
  2. Statistiques descriptives & matrice de corrélation
  3. Frontière efficiente & comparaison des portefeuilles
  4. Analyse de risque : VaR (3 méthodes) & distribution des rendements
  5. Stress tests
  6. Hedging : rolling beta, hedge ratio & performance hedged vs unhedged
  7. Backtest walk-forward : performance cumulée & métriques par fenêtre
  8. Comparaison finale des trois stratégies
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm
from scipy import stats as scipy_stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TICKERS, START_DATE, END_DATE,
    RISK, PATHS, HEDGING
)

# ─────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────

COLORS = {
    "primary"    : "#1A3A5C",
    "secondary"  : "#2563EB",
    "accent"     : "#0891B2",
    "green"      : "#059669",
    "red"        : "#DC2626",
    "orange"     : "#D97706",
    "purple"     : "#7C3AED",
    "gray"       : "#6B7280",
    "light"      : "#F3F4F6",
    "grid"       : "#E5E7EB",
    "markowitz"  : "#2563EB",
    "cvar"       : "#059669",
    "equal"      : "#D97706",
    "benchmark"  : "#6B7280",
    "hedged"     : "#059669",
    "unhedged"   : "#DC2626",
}

# Palette par ticker (cohérente sur toutes les figures)
TICKER_COLORS = [
    "#1A3A5C", "#2563EB", "#0891B2", "#059669",
    "#D97706", "#DC2626", "#7C3AED", "#DB2777", "#0D9488",
]

plt.rcParams.update({
    "figure.facecolor"   : "white",
    "axes.facecolor"     : "white",
    "axes.grid"          : True,
    "grid.color"         : COLORS["grid"],
    "grid.linewidth"     : 0.7,
    "axes.spines.top"    : False,
    "axes.spines.right"  : False,
    "font.family"        : "DejaVu Sans",
    "axes.titlesize"     : 12,
    "axes.titleweight"   : "bold",
    "axes.labelsize"     : 10,
    "xtick.labelsize"    : 9,
    "ytick.labelsize"    : 9,
    "legend.fontsize"    : 9,
    "legend.framealpha"  : 0.9,
})

TRADING_DAYS = RISK["trading_days"]


# ─────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────

def _pct_formatter(x, pos):
    return f"{x*100:.1f}%"

def _pct2_formatter(x, pos):
    return f"{x:.1f}%"

def _eur_formatter(x, pos):
    return f"{x:,.0f} €"

def _save(fig: plt.Figure, output_dir: str, name: str):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{name}_{START_DATE}_{END_DATE}.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"[SAVE] {filepath}")
    plt.show()
    plt.close(fig)

def _section_title(text: str) -> str:
    """Formate un titre de section pour les prints."""
    return f"\n{'─'*60}\n  {text}\n{'─'*60}"


# ─────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────

def load_all_data(data_dir: str = None) -> dict:
    """
    Charge tous les fichiers produits par les modules précédents.
    Retourne un dictionnaire centralisé de DataFrames.
    """
    proc = data_dir or PATHS["processed"]
    raw  = PATHS["raw"]

    def _read(path, **kwargs):
        if not os.path.exists(path):
            print(f"[WARN] Fichier absent : {path}")
            return None
        df = pd.read_csv(path, **kwargs)
        if isinstance(df.index, pd.DatetimeIndex):
            pass
        return df

    def _read_ts(path):
        """Charge un CSV avec index datetime."""
        if not os.path.exists(path):
            print(f"[WARN] Fichier absent : {path}")
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
        return df

    suf = f"{START_DATE}_{END_DATE}"

    data = {
        # Preprocess
        "prices"       : _read_ts(f"{proc}/prices_clean_{suf}.csv"),
        "returns"      : _read_ts(f"{proc}/returns_{suf}.csv"),
        "stats"        : _read(f"{proc}/stats_{suf}.csv", index_col=0),
        "correlation"  : _read_ts(f"{proc}/correlation_{suf}.csv"),

        # Portfolio
        "weights"      : _read(f"{proc}/portfolio_weights_{suf}.csv", index_col=0),
        "port_metrics" : _read(f"{proc}/portfolio_metrics_{suf}.csv", index_col=0),
        "frontier"     : _read(f"{proc}/efficient_frontier_{suf}.csv"),

        # Risk
        "var_comp"     : _read(f"{proc}/var_comparison_{suf}.csv", index_col=0),
        "stress"       : _read(f"{proc}/stress_tests_{suf}.csv", index_col=0),

        # Hedging
        "hedging_ts"   : _read_ts(f"{proc}/hedging_timeseries_{suf}.csv"),
        "hedging_perf" : _read(f"{proc}/hedging_performance_{suf}.csv", index_col=0),
        "effectiveness": _read_ts(f"{proc}/hedging_effectiveness_{suf}.csv"),

        # Backtest
        "bt_markowitz" : _read_ts(f"{proc}/backtest_returns_markowitz_{suf}.csv"),
        "bt_cvar"      : _read_ts(f"{proc}/backtest_returns_cvar_{suf}.csv"),
        "bt_equal"     : _read_ts(f"{proc}/backtest_returns_equal_weight_{suf}.csv"),
        "bt_windows"   : _read(f"{proc}/backtest_windows_markowitz_{suf}.csv", index_col=0),
        "bt_metrics"   : _read(f"{proc}/backtest_metrics_markowitz_{suf}.csv", index_col=0),
        "bt_drawdowns" : _read(f"{proc}/backtest_drawdowns_markowitz_{suf}.csv"),
    }

    loaded = sum(1 for v in data.values() if v is not None)
    print(f"[LOAD] {loaded}/{len(data)} fichiers chargés")

    return data


# ─────────────────────────────────────────────
# FIGURE 1 — UNIVERS D'INVESTISSEMENT
# ─────────────────────────────────────────────

def plot_universe(data: dict, output_dir: str):
    """
    Prix normalisés et rendements cumulés de tous les actifs.
    Permet de visualiser les performances relatives sur la période.
    """
    prices  = data.get("prices")
    returns = data.get("returns")

    if prices is None or returns is None:
        print("[SKIP] Figure 1 — données manquantes")
        return

    tickers = prices.columns.tolist()
    colors  = TICKER_COLORS[:len(tickers)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        f"Univers d'Investissement — Prix Normalisés & Rendements Cumulés\n"
        f"{START_DATE} → {END_DATE}",
        fontsize=14, fontweight="bold"
    )

    # ── Panneau 1 : prix normalisés (base 100) ───────────────────────────
    ax1 = axes[0]
    for i, ticker in enumerate(tickers):
        series = prices[ticker].dropna()
        if series.empty:
            continue
        normalized = series / series.iloc[0] * 100
        ax1.plot(normalized.index, normalized, color=colors[i],
                 linewidth=1.8, label=TICKERS.get(ticker, ticker)[:20])

    ax1.axhline(100, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.set_ylabel("Valeur base 100")
    ax1.set_title("Prix normalisés (base 100 au départ)", loc="left")
    ax1.legend(loc="upper left", ncol=2, fontsize=8)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.0f}"))

    # ── Panneau 2 : rendement cumulé (log) ───────────────────────────────
    ax2 = axes[1]
    for i, ticker in enumerate(tickers):
        r = returns[ticker].dropna()
        if r.empty:
            continue
        cumul = (1 + r).cumprod() - 1
        ax2.plot(cumul.index, cumul * 100, color=colors[i],
                 linewidth=1.8, label=ticker)

    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_ylabel("Rendement cumulé (%)")
    ax2.set_title("Rendement cumulé total (%)", loc="left")
    ax2.legend(loc="upper left", ncol=3, fontsize=8)
    ax2.yaxis.set_major_formatter(FuncFormatter(_pct2_formatter))

    plt.tight_layout()
    _save(fig, output_dir, "01_universe")


# ─────────────────────────────────────────────
# FIGURE 2 — STATISTIQUES & CORRÉLATIONS
# ─────────────────────────────────────────────

def plot_stats_and_correlation(data: dict, output_dir: str):
    """
    Scatter risk/return + heatmap de corrélation.
    Vue d'ensemble des propriétés statistiques de l'univers.
    """
    stats = data.get("stats")
    corr  = data.get("correlation")

    if stats is None:
        print("[SKIP] Figure 2 — données manquantes")
        return

    fig = plt.figure(figsize=(16, 7))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    fig.suptitle("Statistiques Descriptives & Corrélations",
                 fontsize=14, fontweight="bold")

    # ── Scatter risk/return ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    tickers = stats.index.tolist()
    colors  = TICKER_COLORS[:len(tickers)]

    for i, ticker in enumerate(tickers):
        vol = stats.loc[ticker, "volatility_ann"] * 100
        ret = stats.loc[ticker, "mean_return_ann"] * 100
        ax1.scatter(vol, ret, color=colors[i], s=120, zorder=3)
        ax1.annotate(
            ticker,
            xy=(vol, ret),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=8, color=colors[i], fontweight="bold"
        )

    # Ligne iso-Sharpe (Sharpe = 0.5 comme référence)
    vols = np.linspace(
        stats["volatility_ann"].min() * 90,
        stats["volatility_ann"].max() * 110, 100
    )
    for sharpe_ref, ls in [(0.3, ":"), (0.5, "--"), (1.0, "-.")]:
        ax1.plot(
            vols * 100,
            (sharpe_ref * vols + 0.03) * 100,
            color="gray", linewidth=1, linestyle=ls, alpha=0.5,
            label=f"Sharpe = {sharpe_ref}"
        )

    ax1.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax1.set_xlabel("Volatilité annualisée (%)")
    ax1.set_ylabel("Rendement annualisé (%)")
    ax1.set_title("Espace Risque / Rendement", loc="left")
    ax1.legend(fontsize=8)

    # ── Heatmap de corrélation ────────────────────────────────────────────
    if corr is not None:
        ax2  = fig.add_subplot(gs[1])
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        corr_lower = corr.copy()
        corr_lower[mask] = np.nan

        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        im   = ax2.imshow(
            corr_lower.values,
            cmap="RdYlGn", norm=norm,
            aspect="auto", interpolation="nearest"
        )

        n = len(corr.columns)
        ax2.set_xticks(range(n))
        ax2.set_yticks(range(n))
        ax2.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax2.set_yticklabels(corr.columns, fontsize=8)

        # Valeurs dans chaque cellule
        for i in range(n):
            for j in range(n):
                if not np.isnan(corr_lower.values[i, j]):
                    val  = corr_lower.values[i, j]
                    col  = "white" if abs(val) > 0.6 else "black"
                    ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                             fontsize=7, color=col)

        plt.colorbar(im, ax=ax2, shrink=0.8, label="Corrélation de Pearson")
        ax2.set_title("Matrice de Corrélation", loc="left")

    plt.tight_layout()
    _save(fig, output_dir, "02_stats_correlation")


# ─────────────────────────────────────────────
# FIGURE 3 — FRONTIÈRE EFFICIENTE
# ─────────────────────────────────────────────

def plot_efficient_frontier(data: dict, output_dir: str):
    """
    Frontière efficiente + positions des trois portefeuilles optimaux.
    Figure centrale du module portfolio.
    """
    frontier = data.get("frontier")
    weights  = data.get("weights")
    metrics  = data.get("port_metrics")
    stats    = data.get("stats")

    if frontier is None:
        print("[SKIP] Figure 3 — frontière manquante")
        return

    fig, (ax_front, ax_weights) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Frontière Efficiente & Allocation des Portefeuilles",
                 fontsize=14, fontweight="bold")

    # ── Frontière efficiente ──────────────────────────────────────────────
    vols = frontier["volatility_ann"] * 100
    rets = frontier["return_ann"] * 100

    # Colore selon le Sharpe ratio
    sharpe_vals = frontier["sharpe_ratio"]
    sc = ax_front.scatter(
        vols, rets,
        c=sharpe_vals, cmap="RdYlGn",
        s=25, zorder=2, alpha=0.8
    )
    plt.colorbar(sc, ax=ax_front, label="Sharpe ratio", shrink=0.8)

    # Portefeuille Sharpe max sur la frontière
    idx_max = frontier["sharpe_ratio"].idxmax()
    ax_front.scatter(
        vols.iloc[idx_max], rets.iloc[idx_max],
        color=COLORS["markowitz"], s=200, zorder=5,
        marker="*", label="Sharpe Maximum"
    )

    # Actifs individuels
    if stats is not None:
        tickers = stats.index.tolist()
        colors  = TICKER_COLORS[:len(tickers)]
        for i, t in enumerate(tickers):
            vol = stats.loc[t, "volatility_ann"] * 100
            ret = stats.loc[t, "mean_return_ann"] * 100
            ax_front.scatter(vol, ret, color=colors[i], s=80,
                             marker="D", zorder=4, alpha=0.8)
            ax_front.annotate(t, xy=(vol, ret), xytext=(4, 2),
                              textcoords="offset points", fontsize=7)

    # Portefeuilles optimaux
    if metrics is not None:
        port_info = [
            ("Markowitz",  COLORS["markowitz"], "o"),
            ("CVaR 95%",   COLORS["cvar"],      "s"),
            ("1/N",        COLORS["equal"],     "^"),
        ]
        for label, color, marker in port_info:
            if label in metrics.index:
                v = float(metrics.loc[label, "volatility_ann"]) * 100
                r = float(metrics.loc[label, "return_ann"]) * 100
                ax_front.scatter(v, r, color=color, s=150,
                                 marker=marker, zorder=6, label=label,
                                 edgecolors="black", linewidths=0.8)

    ax_front.set_xlabel("Volatilité annualisée (%)")
    ax_front.set_ylabel("Rendement annualisé (%)")
    ax_front.set_title("Frontière Efficiente (Markowitz)", loc="left")
    ax_front.legend(loc="lower right")

    # ── Allocation des portefeuilles ──────────────────────────────────────
    if weights is not None:
        port_labels = weights.columns.tolist()
        tickers_w   = weights.index.tolist()
        n_tickers   = len(tickers_w)
        colors_w    = TICKER_COLORS[:n_tickers]

        x     = np.arange(len(port_labels))
        width = 0.6
        bottoms = np.zeros(len(port_labels))

        for i, ticker in enumerate(tickers_w):
            vals = weights.loc[ticker].values * 100
            ax_weights.bar(
                x, vals, width,
                bottom=bottoms,
                color=colors_w[i],
                alpha=0.85,
                label=ticker
            )
            # Annotation si poids > 5%
            for j, (v, b) in enumerate(zip(vals, bottoms)):
                if v > 5:
                    ax_weights.text(
                        x[j], b + v/2,
                        f"{v:.0f}%",
                        ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold"
                    )
            bottoms += vals

        ax_weights.set_xticks(x)
        ax_weights.set_xticklabels(port_labels, rotation=15)
        ax_weights.set_ylabel("Poids (%)")
        ax_weights.set_ylim(0, 105)
        ax_weights.set_title("Allocation par Portefeuille", loc="left")
        ax_weights.legend(loc="upper right", ncol=2, fontsize=8)

    plt.tight_layout()
    _save(fig, output_dir, "03_efficient_frontier")


# ─────────────────────────────────────────────
# FIGURE 4 — ANALYSE DE RISQUE (VaR)
# ─────────────────────────────────────────────

def plot_risk_analysis(data: dict, output_dir: str):
    """
    VaR par méthode + distribution des rendements du portefeuille.
    Montre les queues épaisses et compare les estimateurs de risque.
    """
    returns  = data.get("returns")
    var_comp = data.get("var_comp")

    if returns is None:
        print("[SKIP] Figure 4 — données manquantes")
        return

    # Construit le portefeuille équipondéré pour l'illustration
    n        = len(returns.columns)
    w        = np.ones(n) / n
    port_ret = returns @ w
    port_ret = port_ret.dropna()

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)
    fig.suptitle("Analyse de Risque — VaR & Distribution des Rendements",
                 fontsize=14, fontweight="bold")

    # ── Distribution des rendements + VaR ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])   # pleine largeur

    ax1.hist(
        port_ret * 100, bins=80,
        color=COLORS["secondary"], alpha=0.6,
        density=True, label="Rendements journaliers"
    )

    # Superpose une loi normale ajustée
    mu, sigma = port_ret.mean() * 100, port_ret.std() * 100
    x_range   = np.linspace(port_ret.min()*100 - 1, port_ret.max()*100 + 1, 300)
    normal_pdf = scipy_stats.norm.pdf(x_range, mu, sigma)
    ax1.plot(x_range, normal_pdf, color=COLORS["orange"], linewidth=2,
             label=f"Loi normale N({mu:.2f}%, {sigma:.2f}%)")

    # Lignes VaR
    var_95_hist = -np.percentile(port_ret, 5) * 100
    var_99_hist = -np.percentile(port_ret, 1) * 100

    ax1.axvline(-var_95_hist, color=COLORS["orange"], linewidth=2,
                linestyle="--", label=f"VaR Hist. 95% = {var_95_hist:.2f}%")
    ax1.axvline(-var_99_hist, color=COLORS["red"], linewidth=2,
                linestyle="--", label=f"VaR Hist. 99% = {var_99_hist:.2f}%")

    # Zone de queue
    ax1.fill_betweenx(
        [0, normal_pdf.max() * 0.5],
        x_range.min(), -var_95_hist,
        alpha=0.15, color=COLORS["red"], label="Queue 5%"
    )

    ax1.set_xlabel("Rendement journalier (%)")
    ax1.set_ylabel("Densité")
    ax1.set_title("Distribution des Rendements — Portefeuille Équipondéré", loc="left")
    ax1.legend(loc="upper left", ncol=2, fontsize=8)

    # ── Comparaison VaR par méthode ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])

    if var_comp is not None:
        var_cols = [c for c in var_comp.columns if c.startswith("var_")]
        methods  = var_comp.index.tolist()
        x        = np.arange(len(methods))
        width    = 0.35
        cols_plot = ["var_historical", "var_parametric",
                     "var_mc_gaussian", "var_mc_bootstrap"]
        bar_colors = [COLORS["secondary"], COLORS["orange"],
                      COLORS["green"], COLORS["purple"]]
        bar_labels = ["Historique", "Paramétrique", "MC Gaussien", "MC Bootstrap"]

        confidence_idx = 0   # 95%
        row = var_comp.index[confidence_idx]

        for i, (col, color, label) in enumerate(
            zip(cols_plot, bar_colors, bar_labels)
        ):
            if col in var_comp.columns:
                val = float(var_comp.loc[row, col]) * 100
                ax2.bar(i, val, color=color, alpha=0.8, label=label)
                ax2.text(i, val + 0.01, f"{val:.2f}%",
                         ha="center", va="bottom", fontsize=8)

        ax2.set_xticks(range(len(cols_plot)))
        ax2.set_xticklabels(bar_labels, rotation=15, fontsize=8)
        ax2.set_ylabel("VaR journalière (%)")
        ax2.set_title(f"VaR 95% — Comparaison des méthodes", loc="left")

    # ── VaR en rolling (évolution dans le temps) ─────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])

    window = RISK["var_window_days"]
    roll_var = port_ret.rolling(window).apply(
        lambda x: -np.percentile(x, 5)
    )
    roll_vol = port_ret.rolling(window).std() * np.sqrt(TRADING_DAYS)

    ax3.fill_between(roll_var.index, 0, roll_var * 100,
                     color=COLORS["red"], alpha=0.3, label="VaR 95% rolling")
    ax3.plot(roll_var.index, roll_var * 100,
             color=COLORS["red"], linewidth=1.5)

    ax3_vol = ax3.twinx()
    ax3_vol.plot(roll_vol.index, roll_vol * 100,
                 color=COLORS["secondary"], linewidth=1.5,
                 linestyle="--", alpha=0.7, label="Vol. ann. rolling")
    ax3_vol.set_ylabel("Volatilité annualisée (%)", color=COLORS["secondary"])
    ax3_vol.tick_params(axis="y", labelcolor=COLORS["secondary"])

    ax3.set_ylabel("VaR 95% journalière (%)", color=COLORS["red"])
    ax3.set_title(f"VaR & Volatilité rolling ({window}j)", loc="left")

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_vol.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    plt.tight_layout()
    _save(fig, output_dir, "04_risk_var")


# ─────────────────────────────────────────────
# FIGURE 5 — STRESS TESTS
# ─────────────────────────────────────────────

def plot_stress_tests(data: dict, output_dir: str):
    """
    Impacts des scénarios de stress sur le portefeuille.
    """
    stress = data.get("stress")

    if stress is None:
        print("[SKIP] Figure 5 — données manquantes")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Stress Tests — Impact des Scénarios Extrêmes",
                 fontsize=14, fontweight="bold")

    scenarios = stress.index.tolist()
    impacts   = stress["impact_portfolio"].values * 100
    impacts_e = stress["impact_eur"].values

    # ── Barres horizontales — impact en % ─────────────────────────────────
    colors_bar = [COLORS["red"] if v < 0 else COLORS["green"] for v in impacts]
    y_pos      = range(len(scenarios))

    ax1.barh(y_pos, impacts, color=colors_bar, alpha=0.8, height=0.6)
    ax1.axvline(0, color="black", linewidth=0.8)

    for i, (val, name) in enumerate(zip(impacts, scenarios)):
        offset = -0.3 if val < 0 else 0.3
        ax1.text(val + offset, i, f"{val:+.1f}%",
                 va="center", fontsize=8, fontweight="bold",
                 color=COLORS["red"] if val < 0 else COLORS["green"])

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([s.replace("_", " ") for s in scenarios], fontsize=9)
    ax1.set_xlabel("Impact sur le portefeuille (%)")
    ax1.set_title("Impact en % (sur capital 1M€)", loc="left")
    ax1.invert_yaxis()

    # ── Impact en euros ───────────────────────────────────────────────────
    colors_eur = [COLORS["red"] if v < 0 else COLORS["green"] for v in impacts_e]
    ax2.barh(y_pos, impacts_e / 1000, color=colors_eur, alpha=0.8, height=0.6)
    ax2.axvline(0, color="black", linewidth=0.8)

    for i, val in enumerate(impacts_e):
        offset = -5 if val < 0 else 5
        ax2.text(val/1000 + offset, i, f"{val:+,.0f} €",
                 va="center", fontsize=8)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([s.replace("_", " ") for s in scenarios], fontsize=9)
    ax2.set_xlabel("Impact (k€)")
    ax2.set_title("Impact en Euros (base 1 000 000 €)", loc="left")
    ax2.invert_yaxis()

    plt.tight_layout()
    _save(fig, output_dir, "05_stress_tests")


# ─────────────────────────────────────────────
# FIGURE 6 — HEDGING
# ─────────────────────────────────────────────

def plot_hedging(data: dict, output_dir: str):
    """
    Rolling beta, hedge ratio, performance hedged vs unhedged
    et efficacité du hedge dans le temps.
    """
    hedging_ts   = data.get("hedging_ts")
    hedging_perf = data.get("hedging_perf")
    effectiveness = data.get("effectiveness")
    returns       = data.get("returns")

    if hedging_ts is None or returns is None:
        print("[SKIP] Figure 6 — données manquantes")
        return

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Stratégie de Couverture Dynamique (Rolling Beta Hedging)",
                 fontsize=14, fontweight="bold")

    # Reconstruction du portefeuille unhedged
    n       = len(returns.columns)
    w       = np.ones(n) / n
    port_uh = returns @ w
    port_uh.name = "portfolio_unhedged"

    # ── Rolling beta ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])

    if "rolling_beta" in hedging_ts.columns:
        beta = hedging_ts["rolling_beta"].dropna()
        ax1.plot(beta.index, beta, color=COLORS["primary"],
                 linewidth=1.8, label="Rolling beta (60j)")
        ax1.fill_between(beta.index, 0, beta,
                         alpha=0.15, color=COLORS["primary"])
        ax1.axhline(1.0, color=COLORS["orange"], linewidth=1,
                    linestyle="--", label="β = 1 (même risque)")
        ax1.axhline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
        ax1.axhline(beta.mean(), color=COLORS["green"], linewidth=1,
                    linestyle=":", label=f"β moyen = {beta.mean():.2f}")

    if "hedge_ratio" in hedging_ts.columns:
        hr  = hedging_ts["hedge_ratio"].dropna()
        ax1b = ax1.twinx()
        ax1b.plot(hr.index, hr, color=COLORS["red"],
                  linewidth=1.5, linestyle="--", alpha=0.7,
                  label="Hedge ratio")
        ax1b.set_ylabel("Hedge ratio", color=COLORS["red"])
        ax1b.tick_params(axis="y", labelcolor=COLORS["red"])

    ax1.set_ylabel("Beta")
    ax1.set_title("Rolling Beta & Hedge Ratio", loc="left")

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc="upper left", fontsize=8)

    # ── Performance hedged vs unhedged ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])

    cumul_uh = (1 + port_uh).cumprod() - 1

    ax2.fill_between(cumul_uh.index, 0, cumul_uh * 100,
                     alpha=0.2, color=COLORS["unhedged"])
    ax2.plot(cumul_uh.index, cumul_uh * 100,
             color=COLORS["unhedged"], linewidth=2,
             label="Unhedged (1/N)")

    if "portfolio_hedged" in hedging_ts.columns:
        r_h    = hedging_ts["portfolio_hedged"].dropna()
        cumul_h = (1 + r_h).cumprod() - 1
        ax2.plot(cumul_h.index, cumul_h * 100,
                 color=COLORS["hedged"], linewidth=2,
                 label="Hedged (Brent short)")

    ax2.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax2.set_ylabel("Rendement cumulé (%)")
    ax2.set_title("Performance Cumulée : Hedged vs Unhedged", loc="left")
    ax2.legend(loc="upper left", fontsize=9)

    # ── Efficacité rolling du hedge ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])

    if effectiveness is not None and "effectiveness" in effectiveness.columns:
        eff = effectiveness["effectiveness"].dropna()
        colors_eff = [COLORS["green"] if v > 0 else COLORS["red"]
                      for v in eff.values]
        ax3.fill_between(eff.index, 0, eff,
                         where=eff >= 0,
                         alpha=0.4, color=COLORS["green"],
                         label="Hedge efficace")
        ax3.fill_between(eff.index, 0, eff,
                         where=eff < 0,
                         alpha=0.4, color=COLORS["red"],
                         label="Hedge nuit")
        ax3.plot(eff.index, eff,
                 color=COLORS["primary"], linewidth=1.5)
        ax3.axhline(0, color="black", linewidth=0.8)
        ax3.axhline(eff.mean(), color=COLORS["orange"],
                    linewidth=1, linestyle="--",
                    label=f"Moy = {eff.mean():.2f}")
        ax3.set_ylabel("Hedge Effectiveness")
        ax3.set_title("Efficacité du Hedge (rolling trimestriel)", loc="left")
        ax3.legend(fontsize=8)
        ax3.set_ylim(-1, 1.1)

    # ── Tableau comparatif hedged vs unhedged ─────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis("off")

    if hedging_perf is not None:
        rows_display = [
            ("Rendement ann.",  "return_ann",     True),
            ("Volatilité ann.", "volatility_ann", True),
            ("Sharpe ratio",    "sharpe_ratio",   False),
            ("Sortino ratio",   "sortino_ratio",  False),
            ("Max Drawdown",    "max_drawdown",   True),
            ("CVaR 95%",        "cvar_95",        True),
        ]

        y_start = 0.92
        ax4.text(0.05, y_start + 0.05, "Comparaison Hedged / Unhedged",
                 fontsize=10, fontweight="bold", transform=ax4.transAxes)

        col_positions = [0.05, 0.45, 0.72]
        headers = ["Métrique", "Unhedged", "Hedged"]
        for col_x, header in zip(col_positions, headers):
            ax4.text(col_x, y_start, header,
                     fontsize=9, fontweight="bold",
                     transform=ax4.transAxes,
                     color=COLORS["primary"])

        y = y_start - 0.08
        for label, key, is_pct in rows_display:
            if key not in hedging_perf.index:
                continue

            uh_val = hedging_perf.loc[key, "Unhedged"] if "Unhedged" in hedging_perf.columns else np.nan
            h_val  = hedging_perf.loc[key, "Hedged"]   if "Hedged"   in hedging_perf.columns else np.nan

            fmt = lambda v: f"{v*100:+.2f}%" if is_pct else f"{v:+.3f}"
            uh_str = fmt(uh_val) if not np.isnan(uh_val) else "N/A"
            h_str  = fmt(h_val)  if not np.isnan(h_val)  else "N/A"

            ax4.text(col_positions[0], y, label, fontsize=8.5,
                     transform=ax4.transAxes)
            ax4.text(col_positions[1], y, uh_str, fontsize=8.5,
                     transform=ax4.transAxes, color=COLORS["unhedged"])
            ax4.text(col_positions[2], y, h_str,  fontsize=8.5,
                     transform=ax4.transAxes, color=COLORS["hedged"])
            y -= 0.12

    plt.tight_layout()
    _save(fig, output_dir, "06_hedging")


# ─────────────────────────────────────────────
# FIGURE 7 — BACKTEST WALK-FORWARD
# ─────────────────────────────────────────────

def plot_backtest(data: dict, output_dir: str):
    """
    Performance cumulée hors-sample, drawdown, métriques par fenêtre
    et analyse des drawdowns historiques.
    """
    bt_ret  = data.get("bt_markowitz")
    bt_win  = data.get("bt_windows")
    bt_dd   = data.get("bt_drawdowns")
    returns = data.get("returns")

    if bt_ret is None:
        print("[SKIP] Figure 7 — données de backtest manquantes")
        return

    # Série de rendements propre
    if isinstance(bt_ret, pd.DataFrame):
        r_strat = bt_ret.iloc[:, 0].dropna()
    else:
        r_strat = bt_ret.dropna()

    # Benchmark équipondéré
    if returns is not None:
        n     = len(returns.columns)
        w     = np.ones(n) / n
        bench = (returns @ w).reindex(r_strat.index).dropna()
    else:
        bench = None

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Backtest Walk-Forward — Stratégie Markowitz",
                 fontsize=14, fontweight="bold")

    # ── Performance cumulée ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])

    cumul_strat = (1 + r_strat).cumprod() - 1
    ax1.plot(cumul_strat.index, cumul_strat * 100,
             color=COLORS["markowitz"], linewidth=2.2,
             label="Markowitz (hors-sample)")

    if bench is not None:
        cumul_bench = (1 + bench).cumprod() - 1
        ax1.plot(cumul_bench.index, cumul_bench * 100,
                 color=COLORS["benchmark"], linewidth=1.8,
                 linestyle="--", label="Benchmark 1/N", alpha=0.8)

    ax1.fill_between(cumul_strat.index, 0, cumul_strat * 100,
                     alpha=0.1, color=COLORS["markowitz"])
    ax1.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)

    # Annotations des fenêtres de test
    if bt_win is not None and "test_start" in bt_win.columns:
        for _, row in bt_win.iterrows():
            try:
                ts = pd.to_datetime(row["test_start"])
                ax1.axvline(ts, color="gray", linewidth=0.8,
                            linestyle=":", alpha=0.6)
            except Exception:
                pass

    ax1.set_ylabel("Rendement cumulé (%)")
    ax1.set_title("Performance Cumulée Hors-Sample", loc="left")
    ax1.legend(loc="upper left", fontsize=9)

    # ── Drawdown ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])

    peak    = (1 + r_strat).cumprod().cummax()
    cumul_c = (1 + r_strat).cumprod()
    dd_ser  = ((cumul_c / peak) - 1) * 100

    ax2.fill_between(dd_ser.index, 0, dd_ser,
                     color=COLORS["red"], alpha=0.5)
    ax2.plot(dd_ser.index, dd_ser,
             color=COLORS["red"], linewidth=1.5)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title("Drawdown Hors-Sample", loc="left")

    # ── Métriques par fenêtre ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])

    if bt_win is not None and "sharpe_ratio" in bt_win.columns:
        sharpes = bt_win["sharpe_ratio"].values
        labels  = [f"F{i}" for i in bt_win.index]
        colors_s = [COLORS["green"] if s > 0 else COLORS["red"]
                    for s in sharpes]

        bars = ax3.bar(labels, sharpes, color=colors_s, alpha=0.8)
        ax3.axhline(0, color="black", linewidth=0.8)
        ax3.axhline(bt_win["sharpe_ratio"].mean(),
                    color=COLORS["orange"], linewidth=1.5,
                    linestyle="--",
                    label=f"Sharpe moy = {bt_win['sharpe_ratio'].mean():.2f}")

        for bar, val in zip(bars, sharpes):
            ax3.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        ax3.set_ylabel("Sharpe ratio")
        ax3.set_title("Sharpe ratio par Fenêtre de Test", loc="left")
        ax3.legend(fontsize=8)

    # ── Top drawdowns ─────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis("off")

    if bt_dd is not None and not bt_dd.empty:
        ax4.text(0.05, 0.95, "Top Drawdowns Identifiés",
                 fontsize=10, fontweight="bold",
                 transform=ax4.transAxes)

        headers = ["Début", "Creux", "Profondeur", "Durée"]
        x_cols  = [0.02, 0.27, 0.52, 0.75]

        y = 0.82
        for h, x in zip(headers, x_cols):
            ax4.text(x, y, h, fontsize=8.5, fontweight="bold",
                     transform=ax4.transAxes, color=COLORS["primary"])

        y -= 0.10
        for _, row in bt_dd.head(5).iterrows():
            vals = [
                str(row.get("start", ""))[:10],
                str(row.get("trough", ""))[:10],
                f"{row.get('depth', 0)*100:.1f}%",
                f"{int(row.get('days_to_trough', 0))}j",
            ]
            for val, x in zip(vals, x_cols):
                ax4.text(x, y, val, fontsize=8,
                         transform=ax4.transAxes)
            y -= 0.10

    # ── Rendements mensuels ───────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])

    monthly = r_strat.resample("ME").apply(
        lambda x: (1 + x).prod() - 1
    ) * 100

    colors_m = [COLORS["green"] if v >= 0 else COLORS["red"]
                for v in monthly.values]
    ax5.bar(range(len(monthly)), monthly.values,
            color=colors_m, alpha=0.8, width=0.8)
    ax5.axhline(0, color="black", linewidth=0.8)
    ax5.set_xlabel("Mois (chronologique)")
    ax5.set_ylabel("Rendement mensuel (%)")
    ax5.set_title("Rendements Mensuels Hors-Sample", loc="left")

    hit_ratio = (monthly > 0).mean() * 100
    ax5.text(0.02, 0.95, f"Hit ratio : {hit_ratio:.1f}%",
             transform=ax5.transAxes, fontsize=9,
             color=COLORS["green"] if hit_ratio > 50 else COLORS["red"])

    plt.tight_layout()
    _save(fig, output_dir, "07_backtest")


# ─────────────────────────────────────────────
# FIGURE 8 — COMPARAISON DES STRATÉGIES
# ─────────────────────────────────────────────

def plot_strategy_comparison(data: dict, output_dir: str):
    """
    Comparaison finale des trois stratégies sur tous les axes.
    Figure de synthèse du projet.
    """
    bt_mkt   = data.get("bt_markowitz")
    bt_cvar  = data.get("bt_cvar")
    bt_equal = data.get("bt_equal")
    returns  = data.get("returns")

    # Vérifie qu'on a au moins une stratégie
    available = {k: v for k, v in {
        "Markowitz" : bt_mkt,
        "CVaR"      : bt_cvar,
        "1/N"       : bt_equal,
    }.items() if v is not None}

    if not available:
        print("[SKIP] Figure 8 — données de backtest manquantes")
        return

    colors_strat = {
        "Markowitz" : COLORS["markowitz"],
        "CVaR"      : COLORS["cvar"],
        "1/N"       : COLORS["equal"],
    }

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)
    fig.suptitle("Comparaison Finale des Stratégies — Backtest Hors-Sample",
                 fontsize=14, fontweight="bold")

    # ── Performance cumulée ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])

    all_series = {}
    for label, bt in available.items():
        if isinstance(bt, pd.DataFrame):
            r = bt.iloc[:, 0].dropna()
        else:
            r = bt.dropna()
        all_series[label] = r
        cumul = (1 + r).cumprod() - 1
        ax1.plot(cumul.index, cumul * 100,
                 color=colors_strat[label], linewidth=2,
                 label=label)

    ax1.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax1.set_ylabel("Rendement cumulé (%)")
    ax1.set_title("Performance Cumulée Comparative", loc="left")
    ax1.legend(loc="upper left")

    # ── Radar chart des métriques ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0], projection="polar")

    metrics_radar = {
        "Sharpe"    : [],
        "Sortino"   : [],
        "Hit ratio" : [],
        "1-MaxDD"   : [],
        "1-CVaR95"  : [],
    }

    strat_values = {}
    for label, r in all_series.items():
        r = r.dropna()
        ret_ann  = r.mean() * TRADING_DAYS
        vol_ann  = r.std()  * np.sqrt(TRADING_DAYS)
        sharpe   = (ret_ann - 0.03) / vol_ann if vol_ann > 0 else 0

        downside = r[r < 0]
        dd_ann   = downside.std() * np.sqrt(TRADING_DAYS) if len(downside) > 1 else vol_ann
        sortino  = (ret_ann - 0.03) / dd_ann if dd_ann > 0 else 0

        cumul    = (1 + r).cumprod()
        peak     = cumul.cummax()
        max_dd   = ((cumul / peak) - 1).min()
        var_95   = -np.percentile(r, 5)
        monthly  = r.resample("ME").apply(lambda x: (1+x).prod()-1)
        hit      = (monthly > 0).mean()

        strat_values[label] = [
            np.clip(sharpe, -2, 3),
            np.clip(sortino, -2, 3),
            hit,
            1 + max_dd,
            1 - var_95 * 10,
        ]

    categories = ["Sharpe", "Sortino", "Hit ratio", "1+MaxDD", "1-10×VaR"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    for label, vals in strat_values.items():
        vals_plot = vals + vals[:1]
        ax2.plot(angles, vals_plot, color=colors_strat[label],
                 linewidth=2, label=label)
        ax2.fill(angles, vals_plot, color=colors_strat[label], alpha=0.1)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=9)
    ax2.set_title("Profil de Performance (Radar)", loc="center",
                  pad=20, fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    # ── Tableau de métriques comparatif ───────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")

    col_labels  = ["Métrique"] + list(available.keys())
    col_x       = [0.02] + [0.30 + i * 0.22 for i in range(len(available))]
    row_metrics = [
        ("Rend. ann.",   lambda r: f"{r.mean()*TRADING_DAYS*100:+.1f}%"),
        ("Vol. ann.",    lambda r: f"{r.std()*np.sqrt(TRADING_DAYS)*100:.1f}%"),
        ("Sharpe",       lambda r: f"{(r.mean()*TRADING_DAYS-0.03)/(r.std()*np.sqrt(TRADING_DAYS)):.2f}"),
        ("Max DD",       lambda r: f"{((1+r).cumprod()/((1+r).cumprod().cummax())-1).min()*100:.1f}%"),
        ("Hit ratio",    lambda r: f"{(r.resample('ME').apply(lambda x:(1+x).prod()-1)>0).mean()*100:.0f}%"),
        ("VaR 95%",      lambda r: f"{-np.percentile(r,5)*100:.2f}%"),
    ]

    y = 0.92
    # En-têtes
    for label, x in zip(col_labels, col_x):
        ax3.text(x, y, label, fontsize=9, fontweight="bold",
                 transform=ax3.transAxes, color=COLORS["primary"])

    y -= 0.10
    for row_label, func in row_metrics:
        ax3.text(col_x[0], y, row_label, fontsize=8.5,
                 transform=ax3.transAxes)
        for i, (strat_label, r) in enumerate(all_series.items()):
            try:
                val_str = func(r.dropna())
            except Exception:
                val_str = "N/A"
            ax3.text(col_x[i+1], y, val_str, fontsize=8.5,
                     transform=ax3.transAxes,
                     color=colors_strat[strat_label])
        y -= 0.12

    ax3.set_title("Tableau Comparatif", loc="left",
                  fontsize=11, fontweight="bold", pad=10)

    plt.tight_layout()
    _save(fig, output_dir, "08_strategy_comparison")


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run_visualization(
    data_dir  : str = None,
    output_dir: str = None,
) -> None:
    """
    Génère les 8 figures du dashboard complet.
    """
    output_dir = output_dir or PATHS["outputs"]

    print("=" * 60)
    print(f"VISUALISATION — {START_DATE} → {END_DATE}")
    print("=" * 60)

    data = load_all_data(data_dir)

    plot_universe(data, output_dir)
    plot_stats_and_correlation(data, output_dir)
    plot_efficient_frontier(data, output_dir)
    plot_risk_analysis(data, output_dir)
    plot_stress_tests(data, output_dir)
    plot_hedging(data, output_dir)
    plot_backtest(data, output_dir)
    plot_strategy_comparison(data, output_dir)

    print("\n" + "=" * 60)
    print(f"VISUALISATION TERMINÉE — 8 figures dans {output_dir}/")
    print("=" * 60)


# ─────────────────────────────────────────────
# TEST STANDALONE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_visualization()