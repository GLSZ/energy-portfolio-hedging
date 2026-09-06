# src/hedging.py

"""
hedging.py — Stratégie de couverture dynamique par rolling beta.

Principe :
  Un portefeuille long sur des actions/ETFs énergie est exposé aux
  mouvements du marché (risque systématique). Le hedging consiste à
  prendre une position courte sur un instrument de couverture
  (ici : Brent Futures BZ=F) proportionnelle au beta du portefeuille.

  Beta mesure la sensibilité du portefeuille aux mouvements de
  l'instrument de couverture :
    β = Cov(r_portfolio, r_hedge) / Var(r_hedge)

  Hedge ratio = -β × (valeur portefeuille / valeur instrument de couverture)
  → Position courte sur le Brent qui neutralise l'exposition pétrolière

Stratégie implémentée :
  1. Rolling beta (fenêtre glissante de 60 jours de trading)
  2. Rebalancing mensuel (recalcul du hedge ratio)
  3. Comparaison hedged vs unhedged sur toute la période

Structure de sortie :
  - Série temporelle du beta glissant
  - Série temporelle du hedge ratio
  - Rendements hedged vs unhedged
  - Métriques de performance comparées
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HEDGING, RISK, PATHS, START_DATE, END_DATE

TRADING_DAYS = RISK["trading_days"]


# ─────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────

def load_data(data_dir: str = None) -> tuple:
    """
    Charge les rendements et les prix de l'instrument de couverture.

    Retourne : (returns, hedge_returns)
      returns       : pd.DataFrame — rendements de tous les actifs
      hedge_returns : pd.Series    — rendements du Brent (BZ=F)

    TODO :
      - Charge returns depuis data/processed/
      - Extrait la colonne HEDGING["hedge_instrument"] comme hedge_returns
      - Vérifie que l'instrument de couverture est bien présent
      - Affiche un résumé des données chargées
    """
    data_dir = data_dir or PATHS["processed"]
    filepath = os.path.join(data_dir, f"returns_{START_DATE}_{END_DATE}.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Fichier introuvable : {filepath}\n"
            "Lance d'abord : python src/preprocess.py"
        )

    returns = pd.read_csv(filepath, index_col=0, parse_dates=True)

    if isinstance(returns.columns, pd.MultiIndex):
        returns.columns = returns.columns.get_level_values(-1)

    hedge_instrument = HEDGING["hedge_instrument"]   # "BZ=F"

    if hedge_instrument not in returns.columns:
        raise ValueError(
            f"Instrument de couverture '{hedge_instrument}' absent des rendements.\n"
            f"Tickers disponibles : {returns.columns.tolist()}"
        )

    hedge_returns = returns[hedge_instrument].copy()
    hedge_returns.name = hedge_instrument

    print(f"[LOAD] Rendements — {len(returns)} jours × {len(returns.columns)} actifs")
    print(f"[LOAD] Instrument de couverture : {hedge_instrument}")
    print(f"       Rendement moy  : {hedge_returns.mean()*100:.4f}%/jour")
    print(f"       Volatilité     : {hedge_returns.std()*100:.4f}%/jour")

    return returns, hedge_returns




# ─────────────────────────────────────────────
# 2. RENDEMENTS DU PORTEFEUILLE
# ─────────────────────────────────────────────

def compute_portfolio_returns(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """
    Calcule les rendements du portefeuille pondéré.
    Même logique que dans risk.py — factorisée ici pour autonomie du module.

    TODO :
      - Aligne weights sur returns.columns
      - Renormalise à 1.0
      - Retourne returns @ weights comme pd.Series
    """
    common    = [t for t in weights.index if t in returns.columns]
    n_dropped = len(weights) - len(common)

    if n_dropped > 0:
        print(f"[WARN] {n_dropped} ticker(s) absent(s) → exclus")

    w_aligned = weights[common]
    w_aligned = w_aligned / w_aligned.sum()

    port_returns       = returns[common] @ w_aligned
    port_returns.name  = "portfolio"

    print(f"\n[PORTFOLIO] Rendements calculés — {len(port_returns)} jours")
    print(f"            Rendement moy : {port_returns.mean()*100:.4f}%/jour")
    print(f"            Volatilité    : {port_returns.std()*100:.4f}%/jour")

    return port_returns



# ─────────────────────────────────────────────
# 3. ROLLING BETA
# ─────────────────────────────────────────────

def compute_rolling_beta(
    portfolio_returns: pd.Series,
    hedge_returns: pd.Series,
    window: int = None,
) -> pd.Series:
    """
    Calcule le beta glissant entre le portefeuille et l'instrument de hedge.

    Formule :
      β(t) = Cov(r_p[t-w:t], r_h[t-w:t]) / Var(r_h[t-w:t])

    Interprétation :
      β > 1  → portefeuille amplifie les mouvements du Brent
      β = 1  → mouvements identiques
      β < 1  → portefeuille moins sensible que le Brent
      β < 0  → corrélation inverse (rare sur actions énergie)

    Note : les w premiers jours ont β = NaN (fenêtre incomplète)
    → on démarre le hedging uniquement quand β est disponible

    TODO :
      - window = window or HEDGING["rolling_window_days"]
      - Aligne les deux séries sur le même index
      - Pour chaque date t ≥ window :
          cov = Cov(r_p[t-w:t], r_h[t-w:t])
          var = Var(r_h[t-w:t])
          beta(t) = cov / var
      - Indice : utilise pd.Series.rolling(window).cov() et .var()
        → plus efficace que la boucle explicite
      - Affiche : beta moyen, min, max sur la période
      - Retourne une pd.Series indexée par date
    """
    window = window or HEDGING["rolling_window_days"]   # 60

    # Aligne les deux séries sur l'intersection des dates
    aligned   = pd.concat([portfolio_returns, hedge_returns], axis=1).dropna()
    r_p       = aligned.iloc[:, 0]
    r_h       = aligned.iloc[:, 1]

    # Rolling covariance et variance avec pandas
    # rolling(window).cov() → covariance glissante sur `window` jours
    # rolling(window).var() → variance glissante sur `window` jours
    roll_cov  = r_p.rolling(window).cov(r_h)
    roll_var  = r_h.rolling(window).var()

    # β = Cov / Var — division terme à terme
    # Protection contre division par zéro (si Var = 0, β = NaN)
    rolling_beta       = roll_cov / roll_var.replace(0, np.nan)
    rolling_beta.name  = "rolling_beta"

    # Statistiques sur les valeurs non-NaN (fenêtre complète disponible)
    beta_valid = rolling_beta.dropna()
    print(f"\n[BETA] Rolling beta calculé (fenêtre = {window} jours)")
    print(f"       Observations valides : {len(beta_valid)} / {len(rolling_beta)}")
    print(f"       Beta moyen : {beta_valid.mean():.3f}")
    print(f"       Beta min   : {beta_valid.min():.3f}")
    print(f"       Beta max   : {beta_valid.max():.3f}")
    print(f"       Beta std   : {beta_valid.std():.3f}  "
          f"({'instable' if beta_valid.std() > 0.5 else 'stable'})")

    return rolling_beta


def compute_rolling_correlation(
    portfolio_returns: pd.Series,
    hedge_returns: pd.Series,
    window: int = None,
) -> pd.Series:
    """
    Calcule la corrélation glissante entre portefeuille et hedge.

    Corrélation ∈ [-1, 1] — mesure la direction de la relation
    indépendamment de l'amplitude (contrairement au beta).

    Un bon instrument de hedge doit avoir :
      - Corrélation élevée et stable → hedge ratio prévisible
      - Beta modéré → pas de sur-hedge
    """
    window = window or HEDGING["rolling_window_days"]

    aligned  = pd.concat([portfolio_returns, hedge_returns], axis=1).dropna()
    r_p      = aligned.iloc[:, 0]
    r_h      = aligned.iloc[:, 1]

    rolling_corr      = r_p.rolling(window).corr(r_h)
    rolling_corr.name = "rolling_correlation"

    corr_valid = rolling_corr.dropna()
    print(f"\n[CORR] Corrélation glissante portefeuille / {hedge_returns.name}")
    print(f"       Corrélation moyenne : {corr_valid.mean():.3f}")
    print(f"       Corrélation min     : {corr_valid.min():.3f}")
    print(f"       Corrélation max     : {corr_valid.max():.3f}")

    return rolling_corr


def compute_rolling_correlation(
    portfolio_returns: pd.Series,
    hedge_returns: pd.Series,
    window: int = None,
) -> pd.Series:
    """
    Calcule la corrélation glissante entre portefeuille et hedge.

    Complémentaire au beta : la corrélation mesure la DIRECTION
    de la relation (entre -1 et 1) indépendamment de l'amplitude.

    Un bon instrument de hedge doit avoir :
      - Corrélation élevée avec le portefeuille (β pertinent)
      - β stable dans le temps (hedge ratio prévisible)

    TODO :
      - Même logique que compute_rolling_beta
      - Utilise pd.Series.rolling(window).corr()
      - Retourne une pd.Series
    """
    rebalancing_freq = rebalancing_freq or HEDGING["rebalancing_frequency"]
    max_ratio        = HEDGING["max_hedge_ratio"]   # 2.0

    # Position courte → hedge ratio négatif
    hedge_ratio_raw = -rolling_beta

    # Clippe entre [-max_ratio, 0]
    # On est long portefeuille → hedge toujours short (ratio ≤ 0)
    hedge_ratio_clipped = hedge_ratio_raw.clip(lower=-max_ratio, upper=0)

    # ── Rebalancing : extraction des valeurs aux dates de fin de mois ─────
    # resample("ME").last() → prend la dernière valeur de chaque mois
    # Ces valeurs sont les hedge ratios applicables le mois suivant
    rebalancing_values = hedge_ratio_clipped.resample(rebalancing_freq).last()

    # Réindexe sur l'index original et forward fill
    # → entre deux rebalancings, le ratio est maintenu constant
    hedge_ratio = (
        rebalancing_values
        .reindex(rolling_beta.index)
        .ffill()
    )
    hedge_ratio.name = "hedge_ratio"

    # Statistiques
    hr_valid      = hedge_ratio.dropna()
    n_rebalancing = rebalancing_values.dropna().shape[0]

    print(f"\n[HEDGE RATIO] Fréquence : {rebalancing_freq}")
    print(f"              Nombre de rebalancings : {n_rebalancing}")
    print(f"              Ratio moyen  : {hr_valid.mean():.3f}")
    print(f"              Ratio min    : {hr_valid.min():.3f}")
    print(f"              Ratio max    : {hr_valid.max():.3f}")

    # Détecte les périodes où la contrainte max a été atteinte
    n_clipped = (hedge_ratio_raw < -max_ratio).sum()
    if n_clipped > 0:
        print(f"[WARN] {n_clipped} jours où beta > {max_ratio} "
              f"→ hedge ratio clippé à -{max_ratio}")

    return hedge_ratio



# ─────────────────────────────────────────────
# 4. HEDGE RATIO & REBALANCING
# ─────────────────────────────────────────────

def compute_hedge_ratio(
    rolling_beta: pd.Series,
    rebalancing_freq: str = None,
) -> pd.Series:
    """
    Calcule le hedge ratio à appliquer, réévalué à chaque rebalancing.

    Hedge ratio = -β
    → Si β = 0.8, on vend à découvert 0.8 € de Brent par € de portefeuille
    → Le signe négatif : position courte sur le hedge pour neutraliser
      l'exposition longue du portefeuille

    Rebalancing :
      On ne recalcule pas le hedge ratio chaque jour (trop coûteux en
      coûts de transaction). On le réévalue à fréquence fixe (mensuelle).
      Entre deux dates de rebalancing, le ratio est maintenu constant
      (forward fill).

    Contrainte : max_hedge_ratio depuis config
      Évite le sur-hedge si β explose temporairement (ex : crise 2022)

    Paramètres
    ----------
    rebalancing_freq : str — fréquence pandas (ex: "ME" = Month End)

    TODO :
      - rebalancing_freq = rebalancing_freq or HEDGING["rebalancing_frequency"]
      - hedge_ratio_raw = -rolling_beta (position courte)
      - Clippe entre [-max_ratio, 0] (long-only portefeuille → hedge toujours short)
      - Rééchantillonne à la fréquence de rebalancing :
          rebalancing_dates = rolling_beta.resample(freq).last().index
          Pour chaque date de rebalancing → prend la valeur du beta ce jour
          Entre deux dates → maintient la valeur (ffill)
      - Affiche : nb de rebalancings, hedge ratio moyen
      - Retourne la série du hedge ratio (même index que rolling_beta)
    """
    rebalancing_freq = rebalancing_freq or HEDGING["rebalancing_frequency"]
    max_ratio        = HEDGING["max_hedge_ratio"]   # 2.0

    # Position courte → hedge ratio négatif
    hedge_ratio_raw = -rolling_beta

    # Clippe entre [-max_ratio, 0]
    # On est long portefeuille → hedge toujours short (ratio ≤ 0)
    hedge_ratio_clipped = hedge_ratio_raw.clip(lower=-max_ratio, upper=0)

    # ── Rebalancing : extraction des valeurs aux dates de fin de mois ─────
    # resample("ME").last() → prend la dernière valeur de chaque mois
    # Ces valeurs sont les hedge ratios applicables le mois suivant
    rebalancing_values = hedge_ratio_clipped.resample(rebalancing_freq).last()

    # Réindexe sur l'index original et forward fill
    # → entre deux rebalancings, le ratio est maintenu constant
    hedge_ratio = (
        rebalancing_values
        .reindex(rolling_beta.index)
        .ffill()
    )
    hedge_ratio.name = "hedge_ratio"

    # Statistiques
    hr_valid      = hedge_ratio.dropna()
    n_rebalancing = rebalancing_values.dropna().shape[0]

    print(f"\n[HEDGE RATIO] Fréquence : {rebalancing_freq}")
    print(f"              Nombre de rebalancings : {n_rebalancing}")
    print(f"              Ratio moyen  : {hr_valid.mean():.3f}")
    print(f"              Ratio min    : {hr_valid.min():.3f}")
    print(f"              Ratio max    : {hr_valid.max():.3f}")

    # Détecte les périodes où la contrainte max a été atteinte
    n_clipped = (hedge_ratio_raw < -max_ratio).sum()
    if n_clipped > 0:
        print(f"[WARN] {n_clipped} jours où beta > {max_ratio} "
              f"→ hedge ratio clippé à -{max_ratio}")

    return hedge_ratio


# ─────────────────────────────────────────────
# 5. RENDEMENTS HEDGÉS
# ─────────────────────────────────────────────

def compute_hedged_returns(
    portfolio_returns: pd.Series,
    hedge_returns: pd.Series,
    hedge_ratio: pd.Series,
) -> pd.Series:
    """
    Calcule les rendements du portefeuille après couverture.

    Formule :
      r_hedged(t) = r_portfolio(t) + hedge_ratio(t) × r_hedge(t)

    Décomposition :
      r_portfolio(t)              → exposition longue au portefeuille
      hedge_ratio(t) × r_hedge(t) → exposition courte au Brent
                                    (hedge_ratio < 0 → position courte)

    Exemple numérique :
      r_portfolio = -2.0%  (le portefeuille baisse)
      r_hedge     = -3.0%  (le Brent baisse aussi)
      hedge_ratio = -0.8   (on est court 0.8× le Brent)
      r_hedged    = -2.0% + (-0.8) × (-3.0%) = -2.0% + 2.4% = +0.4%
      → La position courte sur le Brent compense la baisse du portefeuille

    TODO :
      - Aligne les trois séries sur le même index (inner join)
      - r_hedged = portfolio_returns + hedge_ratio × hedge_returns
      - Affiche la réduction de volatilité : vol_unhedged vs vol_hedged
      - Retourne la série hedgée avec name="portfolio_hedged"
    """
    aligned = pd.concat(
        [portfolio_returns, hedge_returns, hedge_ratio],
        axis=1,
        join="inner"
    ).dropna()

    r_p  = aligned.iloc[:, 0]   # rendements portefeuille
    r_h  = aligned.iloc[:, 1]   # rendements hedge instrument
    hr   = aligned.iloc[:, 2]   # hedge ratio

    hedged_returns      = r_p + hr * r_h
    hedged_returns.name = "portfolio_hedged"

    # Réduction de risque
    vol_uh = r_p.std() * np.sqrt(TRADING_DAYS)
    vol_h  = hedged_returns.std() * np.sqrt(TRADING_DAYS)
    vol_reduction = (vol_uh - vol_h) / vol_uh * 100

    print(f"\n[HEDGED] Rendements hedgés calculés — {len(hedged_returns)} jours")
    print(f"         Vol. unhedged : {vol_uh*100:.2f}%")
    print(f"         Vol. hedged   : {vol_h*100:.2f}%")
    print(f"         Réduction vol : {vol_reduction:.1f}%")

    return hedged_returns


# ─────────────────────────────────────────────
# 6. MÉTRIQUES DE PERFORMANCE COMPARÉES
# ─────────────────────────────────────────────

def compare_performance(
    unhedged_returns: pd.Series,
    hedged_returns: pd.Series,
    risk_free_rate: float = 0.03,
) -> pd.DataFrame:
    """
    Compare les performances hedged vs unhedged sur les métriques clés.

    Métriques calculées pour les deux séries :
    ┌─────────────────────────────────────────────────────────┐
    │ return_ann      : rendement annualisé                   │
    │ volatility_ann  : volatilité annualisée                 │
    │ sharpe_ratio    : (return_ann - r_f) / vol_ann          │
    │ sortino_ratio   : return_ann / downside_deviation       │
    │                   (ne pénalise que la volatilité basse) │
    │ max_drawdown    : perte max depuis un pic               │
    │ var_95          : VaR historique 95%                    │
    │ cvar_95         : CVaR historique 95%                   │
    │ hedge_effectiveness : 1 - var(hedged) / var(unhedged)   │
    │                      0 = hedge inutile, 1 = hedge parfait│
    └─────────────────────────────────────────────────────────┘

    Sortino ratio :
      Sharpe utilise la volatilité totale (hausse + baisse).
      Sortino utilise uniquement la downside deviation (baisses).
      Plus adapté car un gérant ne veut pas être pénalisé pour
      les fortes hausses — seulement pour les fortes baisses.
      downside_dev = std(min(r, 0)) × √252

    Hedge effectiveness :
      Métrique standard pour évaluer la qualité d'une couverture.
      = 1 - Var(r_hedged) / Var(r_unhedged)
      = fraction du risque éliminée par le hedge
      Ex: 0.35 → le hedge réduit la variance de 35%

    TODO :
      - Calcule chaque métrique pour unhedged et hedged
      - Assemble en DataFrame avec colonnes ["Unhedged", "Hedged", "Amélioration"]
      - Amélioration = (hedged - unhedged) / |unhedged| × 100 (en %)
      - Affiche un tableau comparatif formaté
      - Retourne le DataFrame
    """
    def _metrics(r: pd.Series, label: str) -> dict:
        """Calcule toutes les métriques pour une série de rendements."""
        r = r.dropna()

        # ── Annualisation ─────────────────────────────────────────────────
        ret_ann = r.mean() * TRADING_DAYS
        vol_ann = r.std()  * np.sqrt(TRADING_DAYS)

        # ── Sharpe ratio ──────────────────────────────────────────────────
        sharpe = (ret_ann - risk_free_rate) / vol_ann if vol_ann > 0 else np.nan

        # ── Sortino ratio ─────────────────────────────────────────────────
        # Downside deviation = écart-type des rendements négatifs uniquement
        # Annualisé avec √252 comme la volatilité classique
        downside  = r[r < 0]
        dd_ann    = downside.std() * np.sqrt(TRADING_DAYS) if len(downside) > 0 else np.nan
        sortino   = (ret_ann - risk_free_rate) / dd_ann if dd_ann and dd_ann > 0 else np.nan

        # ── Max Drawdown ──────────────────────────────────────────────────
        cumulative = (1 + r).cumprod()
        peak       = cumulative.cummax()
        drawdown   = (cumulative / peak) - 1
        max_dd     = drawdown.min()

        # ── VaR & CVaR historiques ────────────────────────────────────────
        var_95  = -np.percentile(r, 5)
        var_99  = -np.percentile(r, 1)
        tail_95 = r[r <= -var_95]
        cvar_95 = -tail_95.mean() if len(tail_95) > 0 else var_95

        # ── Calmar ratio ──────────────────────────────────────────────────
        calmar = ret_ann / abs(max_dd) if max_dd != 0 else np.nan

        return {
            "label"          : label,
            "return_ann"     : ret_ann,
            "volatility_ann" : vol_ann,
            "sharpe_ratio"   : sharpe,
            "sortino_ratio"  : sortino,
            "max_drawdown"   : max_dd,
            "var_95"         : var_95,
            "var_99"         : var_99,
            "cvar_95"        : cvar_95,
            "calmar_ratio"   : calmar,
            "n_obs"          : len(r),
        }

    uh_metrics = _metrics(unhedged_returns, "Unhedged")
    h_metrics  = _metrics(hedged_returns,   "Hedged")

    # ── Hedge effectiveness ───────────────────────────────────────────────
    # = fraction de la variance éliminée par le hedge
    # 0 = hedge inutile, 1 = hedge parfait, < 0 = hedge aggrave le risque
    var_uh      = unhedged_returns.dropna().var()
    var_h       = hedged_returns.dropna().var()
    effectiveness = 1 - var_h / var_uh if var_uh > 0 else np.nan

    # ── Assemblage en DataFrame ───────────────────────────────────────────
    metrics_order = [
        "return_ann", "volatility_ann", "sharpe_ratio",
        "sortino_ratio", "max_drawdown",
        "var_95", "var_99", "cvar_95", "calmar_ratio",
    ]

    df = pd.DataFrame({
        "Unhedged" : {k: uh_metrics[k] for k in metrics_order},
        "Hedged"   : {k: h_metrics[k]  for k in metrics_order},
    })

    # Amélioration relative : (hedged - unhedged) / |unhedged|
    # Note : pour les métriques négatives (drawdown, VaR), une amélioration
    # hedged > unhedged (moins négatif) = bonne chose → signe du calcul adapté
    df["Delta"] = df["Hedged"] - df["Unhedged"]

    # ── Affichage formaté ─────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  COMPARAISON HEDGED vs UNHEDGED")
    print(f"{'─'*65}")
    print(f"  {'Métrique':<22} {'Unhedged':>12} {'Hedged':>12} {'Delta':>10}")
    print(f"  {'─'*60}")

    display = [
        ("Rendement ann.",   "return_ann",     True),
        ("Volatilité ann.",  "volatility_ann", True),
        ("Sharpe ratio",     "sharpe_ratio",   False),
        ("Sortino ratio",    "sortino_ratio",  False),
        ("Max Drawdown",     "max_drawdown",   True),
        ("VaR 95% /jour",   "var_95",         True),
        ("CVaR 95% /jour",  "cvar_95",        True),
        ("Calmar ratio",     "calmar_ratio",   False),
    ]

    for label, key, is_pct in display:
        uh_val = df.loc[key, "Unhedged"]
        h_val  = df.loc[key, "Hedged"]
        delta  = df.loc[key, "Delta"]

        if is_pct:
            print(f"  {label:<22} {uh_val*100:>11.2f}%"
                  f" {h_val*100:>11.2f}%"
                  f" {delta*100:>+9.2f}%")
        else:
            print(f"  {label:<22} {uh_val:>12.3f}"
                  f" {h_val:>12.3f}"
                  f" {delta:>+10.3f}")

    print(f"{'─'*65}")
    print(f"  Hedge effectiveness : {effectiveness*100:.1f}% de la variance éliminée")

    flag = ("✓ efficace"   if effectiveness > 0.20 else
            "⚠ partiel"    if effectiveness > 0.05 else
            "✗ inefficace")
    print(f"  Évaluation         : {flag}")
    print(f"{'─'*65}")

    # Ajoute hedge effectiveness comme attribut du DataFrame
    df.attrs["hedge_effectiveness"] = effectiveness

    return df



# ─────────────────────────────────────────────
# 7. ANALYSE DE L'EFFICACITÉ DU HEDGE
# ─────────────────────────────────────────────

def analyze_hedge_effectiveness(
    unhedged_returns: pd.Series,
    hedged_returns: pd.Series,
    rolling_beta: pd.Series,
    window: int = 63,   # trimestre de trading
) -> pd.DataFrame:
    """
    Analyse l'efficacité du hedge dans le temps (rolling).

    Pour chaque fenêtre de `window` jours, calcule :
      - Variance hedged vs unhedged
      - Hedge effectiveness locale = 1 - var(hedged) / var(unhedged)
      - Corrélation résiduelle (risque non couvert)

    Permet d'identifier les périodes où le hedge a bien fonctionné
    (ex: crise 2022 sur le gaz) et celles où il a été inefficace
    (ex: décorrélation soudaine Brent/actions énergie EU).

    TODO :
      - Aligne les séries sur le même index
      - Calcule en rolling(window) :
          var_uh  = unhedged.rolling(window).var()
          var_h   = hedged.rolling(window).var()
          eff     = 1 - var_h / var_uh
      - Assemble en DataFrame avec : var_unhedged, var_hedged, effectiveness
      - Affiche : effectiveness moyenne, min, max
      - Retourne le DataFrame
    """
    # Aligne toutes les séries
    aligned = pd.concat(
        [unhedged_returns, hedged_returns, rolling_beta],
        axis=1, join="inner"
    ).dropna()

    r_uh = aligned.iloc[:, 0]
    r_h  = aligned.iloc[:, 1]

    # Variance glissante sur fenêtre trimestrielle
    var_uh  = r_uh.rolling(window).var()
    var_h   = r_h.rolling(window).var()

    # Hedge effectiveness locale = 1 - var_hedged / var_unhedged
    # Valeur > 0 → le hedge réduit la variance
    # Valeur < 0 → le hedge augmente la variance (nuisible)
    effectiveness = 1 - var_h / var_uh.replace(0, np.nan)

    # Volatilité glissante annualisée
    vol_uh_roll = r_uh.rolling(window).std() * np.sqrt(TRADING_DAYS)
    vol_h_roll  = r_h.rolling(window).std()  * np.sqrt(TRADING_DAYS)

    df = pd.DataFrame({
        "var_unhedged"    : var_uh,
        "var_hedged"      : var_h,
        "effectiveness"   : effectiveness,
        "vol_unhedged"    : vol_uh_roll,
        "vol_hedged"      : vol_h_roll,
        "rolling_beta"    : aligned.iloc[:, 2],
    })

    # Statistiques globales
    eff_valid = effectiveness.dropna()
    print(f"\n[EFFECTIVENESS] Analyse rolling (fenêtre = {window}j = 1 trimestre)")
    print(f"  Effectiveness moyenne : {eff_valid.mean():.3f}")
    print(f"  Effectiveness min     : {eff_valid.min():.3f} "
          f"({eff_valid.idxmin().date() if not eff_valid.empty else 'N/A'})")
    print(f"  Effectiveness max     : {eff_valid.max():.3f} "
          f"({eff_valid.idxmax().date() if not eff_valid.empty else 'N/A'})")

    # Périodes où le hedge est nuisible (effectiveness < 0)
    n_negative = (eff_valid < 0).sum()
    pct_neg    = n_negative / len(eff_valid) * 100
    print(f"  Périodes hedge nuit   : {n_negative} jours ({pct_neg:.1f}%)")

    return df


# ─────────────────────────────────────────────
# 8. EXPORT
# ─────────────────────────────────────────────

def save_hedging_results(
    rolling_beta: pd.Series,
    hedge_ratio: pd.Series,
    hedged_returns: pd.Series,
    performance: pd.DataFrame,
    effectiveness: pd.DataFrame,
) -> None:
    """
    Sauvegarde tous les outputs dans data/processed/.

    TODO :
      - Sauvegarde rolling_beta, hedge_ratio, hedged_returns dans un CSV
      - Sauvegarde performance dans un CSV séparé
      - Sauvegarde effectiveness dans un CSV séparé
      - Affiche les filepaths
    """
    output_dir = PATHS["processed"]
    os.makedirs(output_dir, exist_ok=True)

    # Séries temporelles groupées dans un seul CSV
    ts = pd.concat(
        [rolling_beta, hedge_ratio, hedged_returns],
        axis=1
    )
    fp_ts   = os.path.join(output_dir, f"hedging_timeseries_{START_DATE}_{END_DATE}.csv")
    fp_perf = os.path.join(output_dir, f"hedging_performance_{START_DATE}_{END_DATE}.csv")
    fp_eff  = os.path.join(output_dir, f"hedging_effectiveness_{START_DATE}_{END_DATE}.csv")

    ts.to_csv(fp_ts)
    performance.to_csv(fp_perf)
    effectiveness.to_csv(fp_eff)

    print(f"\n[SAVE] Séries temporelles → {fp_ts}")
    print(f"[SAVE] Performance        → {fp_perf}")
    print(f"[SAVE] Effectiveness      → {fp_eff}")



# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run_hedging(weights: pd.Series) -> dict:
    """
    Pipeline complet :
      load → portfolio_returns → rolling_beta → hedge_ratio
      → hedged_returns → compare → effectiveness → save
    """
    print("=" * 60)
    print("HEDGING ANALYSIS")
    print("=" * 60)

    # Chargement
    returns, hedge_returns = load_data()

    # Rendements du portefeuille non hedgé
    portfolio_ret = compute_portfolio_returns(returns, weights)

    # Rolling beta & corrélation
    rolling_beta = compute_rolling_beta(portfolio_ret, hedge_returns)
    rolling_corr = compute_rolling_correlation(portfolio_ret, hedge_returns)

    # Hedge ratio avec rebalancing mensuel
    hedge_ratio = compute_hedge_ratio(rolling_beta)

    # Rendements hedgés
    hedged_ret = compute_hedged_returns(portfolio_ret, hedge_returns, hedge_ratio)

    # Comparaison performance
    performance = compare_performance(portfolio_ret, hedged_ret)

    # Efficacité rolling
    effectiveness = analyze_hedge_effectiveness(
        portfolio_ret, hedged_ret, rolling_beta
    )

    # Sauvegarde
    save_hedging_results(
        rolling_beta, hedge_ratio, hedged_ret,
        performance, effectiveness
    )

    print("\n" + "=" * 60)
    print("HEDGING TERMINÉ")
    print("=" * 60)

    return {
        "portfolio_returns" : portfolio_ret,
        "hedge_returns"     : hedge_returns,
        "rolling_beta"      : rolling_beta,
        "rolling_corr"      : rolling_corr,
        "hedge_ratio"       : hedge_ratio,
        "hedged_returns"    : hedged_ret,
        "performance"       : performance,
        "effectiveness"     : effectiveness,
    }


# ─────────────────────────────────────────────
# TEST STANDALONE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    returns_test = pd.read_csv(
        f"data/processed/returns_{START_DATE}_{END_DATE}.csv",
        index_col=0, parse_dates=True
    )

    if isinstance(returns_test.columns, pd.MultiIndex):
        returns_test.columns = returns_test.columns.get_level_values(-1)

    n = len(returns_test.columns)
    weights_test = pd.Series(
        np.ones(n) / n,
        index=returns_test.columns,
        name="weight"
    )

    results = run_hedging(weights_test)

    print("\nRolling Beta — 5 premières valeurs non-NaN :")
    print(results["rolling_beta"].dropna().head())

    print("\nComparaison Hedged vs Unhedged :")
    print(results["performance"].to_string())

    eff = results["effectiveness"]["effectiveness"].dropna()
    print(f"\nEfficacité moyenne : {eff.mean():.3f}")
    print(f"Hedge effectiveness globale : "
          f"{results['performance'].attrs.get('hedge_effectiveness', 'N/A'):.3f}")