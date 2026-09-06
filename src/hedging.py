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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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

    # Poids équipondérés pour le test standalone
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

    print("\nEfficacité du hedge — statistiques :")
    eff = results["effectiveness"]["effectiveness"]
    print(f"  Moyenne : {eff.mean():.3f}")
    print(f"  Min     : {eff.min():.3f}")
    print(f"  Max     : {eff.max():.3f}")