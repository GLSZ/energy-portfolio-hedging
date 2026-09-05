# src/risk.py

"""
risk.py — Mesures de risque du portefeuille.

Trois méthodes de calcul de la VaR implémentées :
  1. VaR Historique   — percentile empirique, aucune hypothèse distributionnelle
  2. VaR Paramétrique — hypothèse gaussienne (µ − z_α × σ)
  3. VaR Monte Carlo  — simulation de scénarios de rendements

Plus :
  - Expected Shortfall (CVaR) = moyenne des pertes au-delà de la VaR
  - Stress tests sur scénarios de chocs de prix
  - Backtesting de la VaR (taux de violations)
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RISK, PATHS, START_DATE, END_DATE

TRADING_DAYS = RISK["trading_days"]


# ─────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────

def load_returns(data_dir: str = None) -> pd.DataFrame:
    """
    Charge les rendements depuis data/processed/.

    TODO : même logique que dans portfolio.py
    """
    pass


def compute_portfolio_returns(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """
    Calcule la série de rendements du portefeuille pondéré.

    r_p(t) = Σ_i w_i × r_i(t) = returns @ weights

    Aligne les poids sur les colonnes de returns
    (au cas où preprocess a exclu un ticker).

    TODO :
      - Aligne weights sur returns.columns
      - Renormalise les poids à 1.0 après alignement
      - Calcule returns @ weights (produit matriciel)
      - Retourne une pd.Series avec name="portfolio"
    """
    pass


# ─────────────────────────────────────────────
# 2. VAR HISTORIQUE
# ─────────────────────────────────────────────

def var_historical(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
    window: int = None,
) -> float:
    """
    VaR historique au niveau de confiance donné.

    Principe :
      On trie les rendements passés et on prend le percentile (1-α).
      Aucune hypothèse sur la distribution — on utilise les données brutes.

    Exemple avec α = 95% :
      VaR 95% = -percentile(rendements, 5%)
      Si VaR = 0.023 → perte de 2.3% à ne pas dépasser 95% du temps

    Avantage : capture les queues épaisses et la non-normalité
    Inconvénient : dépend fortement de la période historique choisie

    Paramètres
    ----------
    window : int ou None
      Si None → utilise tout l'historique
      Si int  → fenêtre glissante (ex: 252 = 1 an de trading)

    TODO :
      - Si window fourni : utilise les `window` derniers jours uniquement
      - VaR = -np.percentile(returns, (1 - confidence_level) * 100)
      - Retourne la VaR en positif (convention : perte exprimée positivement)
    """
    pass


def cvar_historical(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
    window: int = None,
) -> float:
    """
    CVaR (Expected Shortfall) historique.

    CVaR = moyenne des pertes au-delà de la VaR
         = E[perte | perte > VaR]
         = -mean(returns[returns < -VaR])

    La CVaR est toujours ≥ VaR (elle est plus conservative).
    C'est la mesure de risque cohérente (au sens d'Artzner) préférée
    par les régulateurs depuis Bâle IV et FRTB.

    TODO :
      - Calcule d'abord la VaR avec var_historical()
      - CVaR = -mean(returns[returns <= -var])
      - Retourne la CVaR en positif
    """
    pass


# ─────────────────────────────────────────────
# 3. VAR PARAMÉTRIQUE (GAUSSIENNE)
# ─────────────────────────────────────────────

def var_parametric(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
) -> tuple:
    """
    VaR paramétrique sous hypothèse de normalité.

    Formule :
      VaR_α = -(µ + z_α × σ)

    où z_α = quantile de la loi normale standard au niveau (1-α)
      z_95% = -1.645  (scipy_stats.norm.ppf(0.05))
      z_99% = -2.326  (scipy_stats.norm.ppf(0.01))

    Avantage : simple, analytique, pas besoin d'historique long
    Inconvénient : sous-estime le risque si la distribution a des
      queues épaisses (kurtosis > 3) — cas fréquent sur les commodités

    Pour quantifier ce biais, on compare VaR paramétrique vs historique.
    Un ratio > 1 (historique > paramétrique) indique des queues épaisses.

    Retourne : (var, cvar, mu, sigma)
      var   : VaR gaussienne
      cvar  : CVaR gaussienne = µ + σ × φ(z_α) / (1-α)
              où φ = densité de la loi normale standard
      mu    : rendement moyen journalier
      sigma : volatilité journalière

    TODO :
      - Estime µ et σ sur la série de rendements
      - Calcule z_alpha = scipy_stats.norm.ppf(1 - confidence_level)
      - VaR  = -(µ + z_alpha × σ)
      - CVaR = -(µ - σ × scipy_stats.norm.pdf(z_alpha) / (1 - confidence_level))
      - Retourne (var, cvar, mu, sigma)
    """
    pass


# ─────────────────────────────────────────────
# 4. VAR MONTE CARLO
# ─────────────────────────────────────────────

def var_monte_carlo(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
    n_simulations: int = None,
    method: str = "gaussian",
) -> tuple:
    """
    VaR Monte Carlo par simulation de rendements.

    Deux méthodes de simulation :
      "gaussian"  : tire des rendements N(µ, σ²) indépendants
      "bootstrap" : rééchantillonne les rendements historiques
                    avec remise (préserve les queues et la corrélation
                    temporelle — plus réaliste pour les commodités)

    Avantage vs paramétrique : peut capturer les queues épaisses
    Avantage vs historique   : plus de scénarios que l'historique réel
    Inconvénient : dépend des hypothèses de simulation

    Retourne : (var, cvar, simulated_returns)
      simulated_returns : np.ndarray des rendements simulés (pour les graphiques)

    TODO :
      - n_simulations = n_simulations or RISK["n_simulations"]  (10_000)
      - Si method="gaussian" :
          tire N(µ, σ) avec np.random.normal(mu, sigma, n_simulations)
      - Si method="bootstrap" :
          rééchantillonne avec np.random.choice(returns, n_simulations, replace=True)
      - VaR  = -np.percentile(simulated_returns, (1-confidence_level)*100)
      - CVaR = -mean(simulated[simulated <= -var])
      - Retourne (var, cvar, simulated_returns)
    """
    pass


# ─────────────────────────────────────────────
# 5. COMPARAISON DES TROIS MÉTHODES
# ─────────────────────────────────────────────

def compare_var_methods(
    portfolio_returns: pd.Series,
    confidence_levels: list = None,
) -> pd.DataFrame:
    """
    Calcule et compare la VaR et CVaR selon les trois méthodes
    pour chaque niveau de confiance.

    Retourne un DataFrame :
      index   : méthodes (Historique, Paramétrique, Monte Carlo Gauss,
                          Monte Carlo Bootstrap)
      colonnes: VaR_95, CVaR_95, VaR_99, CVaR_99

    Interprétation des écarts entre méthodes :
      Si historique >> paramétrique → queues épaisses (kurtosis élevé)
        → la VaR gaussienne sous-estime le risque réel
        → utiliser CVaR ou méthode historique en pratique
      Si Monte Carlo Gauss ≈ paramétrique → cohérent (les deux supposent la normalité)
      Si Monte Carlo Bootstrap ≈ historique → cohérent (même données source)

    TODO :
      - Appelle les 4 variantes pour chaque confidence_level
      - Assemble en DataFrame propre
      - Affiche le tableau formaté avec les écarts entre méthodes
      - Calcule le ratio historique/paramétrique (indicateur de kurtosis)
    """
    confidence_levels = confidence_levels or RISK["confidence_levels"]
    pass


# ─────────────────────────────────────────────
# 6. STRESS TESTS
# ─────────────────────────────────────────────

def run_stress_tests(
    weights: pd.Series,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Applique les scénarios de stress définis dans config.py.

    Principe :
      Pour chaque scénario, on choque les rendements de certains actifs
      et on calcule l'impact sur la valeur du portefeuille.

    Exemple : scénario "crise_2022_energy"
      NG=F   : +200% (choc TTF historique d'août 2022)
      BZ=F   : +60%
      CARB.L : +30%
      ENGI.PA: -35%
      → Impact portefeuille = Σ_i w_i × choc_i

    Retourne un DataFrame :
      index   : noms des scénarios
      colonnes: impact_portfolio (%), impact_eur (€ sur 1M€), 
                actifs_les_plus_impactés

    TODO :
      - Importe RISK["stress_scenarios"] depuis config
      - Pour chaque scénario :
          impact = Σ_i (w_i × choc_i) pour les actifs choqués
          les actifs non mentionnés ont un choc = 0
      - Trie par impact décroissant (pire scénario en premier)
      - Affiche un tableau lisible avec les impacts en % et en €
    """
    from config import RISK
    scenarios = RISK["stress_scenarios"]
    capital   = 1_000_000   # € — capital de référence pour l'impact en €
    pass


# ─────────────────────────────────────────────
# 7. BACKTESTING DE LA VAR
# ─────────────────────────────────────────────

def backtest_var(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
    window: int = 252,
) -> pd.DataFrame:
    """
    Backteste la VaR historique en rolling window.

    Principe :
      À chaque date t, on calcule la VaR sur la fenêtre [t-window, t-1]
      puis on vérifie si la perte réelle au jour t dépasse cette VaR.
      Si oui → "violation" (exception).

    Taux de violation attendu = (1 - confidence_level)
      Pour VaR 95% → 5% de violations attendues
      Pour VaR 99% → 1% de violations attendues

    Test de Kupiec (POF test) :
      Teste si le taux de violations observé est statistiquement
      compatible avec le taux théorique.
      H0 : taux_observé = taux_théorique
      Statistique LR = -2 × log(L0/L1) → loi χ²(1)
      Si p-value < 0.05 → le modèle de VaR est rejeté

    Retourne un DataFrame avec par date :
      - var_predicted : VaR prédite ce jour
      - return_actual : rendement réel
      - violation     : bool (perte > VaR)

    TODO :
      - Boucle sur les dates à partir du jour `window`
      - Pour chaque date t :
          hist = returns.iloc[t-window:t]
          var_t = var_historical(hist, confidence_level)
          violation = returns.iloc[t] < -var_t
      - Calcule le taux de violations
      - Implémente le test de Kupiec
      - Affiche : nb violations, taux observé vs attendu, p-value Kupiec
    """
    pass


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run_risk_analysis(weights: pd.Series) -> dict:
    """
    Pipeline complet de l'analyse de risque.

    Paramètre
    ---------
    weights : pd.Series — poids du portefeuille à analyser
              (typiquement le portefeuille Markowitz ou CVaR optimal)
    """
    print("=" * 60)
    print("RISK ANALYSIS")
    print("=" * 60)

    returns          = load_returns()
    portfolio_ret    = compute_portfolio_returns(returns, weights)

    var_comparison   = compare_var_methods(portfolio_ret)
    stress_results   = run_stress_tests(weights, returns)
    backtest_results = backtest_var(portfolio_ret)

    # Sauvegarde
    output_dir = PATHS["processed"]
    os.makedirs(output_dir, exist_ok=True)
    var_comparison.to_csv(
        os.path.join(output_dir, f"var_comparison_{START_DATE}_{END_DATE}.csv")
    )
    stress_results.to_csv(
        os.path.join(output_dir, f"stress_tests_{START_DATE}_{END_DATE}.csv")
    )

    print("\n" + "=" * 60)
    print("RISK ANALYSIS TERMINÉE")
    print("=" * 60)

    return {
        "portfolio_returns" : portfolio_ret,
        "var_comparison"    : var_comparison,
        "stress_tests"      : stress_results,
        "var_backtest"      : backtest_results,
    }


# ─────────────────────────────────────────────
# TEST STANDALONE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Poids de test : équipondéré pour le standalone
    returns_test = pd.read_csv(
        f"data/processed/returns_{START_DATE}_{END_DATE}.csv",
        index_col=0, parse_dates=True
    )
    n = len(returns_test.columns)
    weights_test = pd.Series(
        np.ones(n) / n,
        index=returns_test.columns,
        name="weight"
    )

    results = run_risk_analysis(weights_test)

    print("\nComparaison VaR :")
    print(results["var_comparison"].to_string())

    print("\nStress tests :")
    print(results["stress_tests"].to_string())

    print("\nBacktest VaR — 5 premières violations :")
    violations = results["var_backtest"]
    print(violations[violations["violation"]].head())