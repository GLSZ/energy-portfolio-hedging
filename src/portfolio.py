"""
portfolio.py — Construction et optimisation du portefeuille.

Deux approches d'optimisation implémentées :
  1. Markowitz (Mean-Variance) — minimise la variance pour un rendement cible
  2. Mean-CVaR — minimise la CVaR (plus robuste aux queues de distribution)

Librairie : cvxpy — solver convexe open-source
  → Formule le problème d'optimisation de façon déclarative
  → Résout avec des solvers internes (CLARABEL, SCS, ECOS)
  → Beaucoup plus élégant que scipy.optimize pour ce type de problème

Structure de sortie :
  - Poids optimaux par actif (vecteur w ∈ R^n, Σw = 1)
  - Métriques du portefeuille optimal (rendement, vol, Sharpe, CVaR)
  - Frontière efficiente complète (ensemble de portefeuilles optimaux)
"""

import os
import sys
import pandas as pd
import numpy as np
import cvxpy as cp
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TICKERS, PORTFOLIO, RISK, PATHS,
    START_DATE, END_DATE
)

TRADING_DAYS = RISK["trading_days"] 
N_FRONTIER   = 50   # nombre de points sur la frontière efficiente

# ─────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────

def load_returns(data_dir: str = None) -> pd.DataFrame:
    """
    Charge les log-rendements produits par preprocess.py.

    TODO :
      - Pattern : f"returns_{START_DATE}_{END_DATE}.csv"
      - Parse l'index en datetime
      - Vérifie que les colonnes correspondent aux tickers attendus
      - Affiche le nombre d'actifs et de jours chargés
    """
    pass

def load_excess_returns(data_dir: str = None) -> pd.DataFrame:
    """
    Charge les rendements en excès du taux sans risque.

    TODO :
      - Pattern : f"excess_returns_{START_DATE}_{END_DATE}.csv"
      - Même logique que load_returns
    """
    pass


# ─────────────────────────────────────────────
# 2. PARAMÈTRES DU PROBLÈME
# ─────────────────────────────────────────────

def compute_inputs(returns: pd.DataFrame) -> tuple:
    """
    Calcule les paramètres d'entrée du problème d'optimisation.

    En finance de portefeuille, deux objets suffisent à décrire
    l'univers d'investissement :
      - mu    : vecteur des rendements espérés (n actifs)
      - Sigma : matrice de covariance (n × n)

    Estimation classique (historique) :
      mu    = moyenne des rendements journaliers × 252 (annualisé)
      Sigma = matrice de covariance journalière × 252 (annualisée)

    Retourne : (mu, Sigma, tickers)
      mu     : np.ndarray shape (n,)
      Sigma  : np.ndarray shape (n, n)
      tickers: list[str]

    TODO :
      - Calcule mu = returns.mean() * TRADING_DAYS
      - Calcule Sigma = returns.cov() * TRADING_DAYS
      - Vérifie que Sigma est définie positive (np.linalg.eigvalsh)
        → si eigenvalues négatives : ajoute une régularisation de Ledoit-Wolf
      - Affiche mu et les volatilités diagonales de Sigma
      - Retourne (mu.values, Sigma.values, returns.columns.tolist())

    Note sur la régularisation Ledoit-Wolf :
      La matrice de covariance empirique est souvent mal conditionnée
      (surtout si n_actifs proche de n_observations).
      Ledoit-Wolf réduit le bruit en "shrinkant" vers une cible structurée.
      → from sklearn.covariance import LedoitWolf
    """
    pass


# ─────────────────────────────────────────────
# 3. OPTIMISATION MARKOWITZ (MEAN-VARIANCE)
# ─────────────────────────────────────────────

def optimize_markowitz(
    mu: np.ndarray,
    Sigma: np.ndarray,
    tickers: list,
    target_return: float = None,
    risk_free_rate: float = 0.03,
) -> dict:
    """
    Optimisation Mean-Variance de Markowitz.

    Deux modes selon target_return :
      - target_return = None  → maximise le Sharpe ratio
      - target_return = x     → minimise la variance pour un rendement ≥ x

    Formulation du problème (maximisation Sharpe) :
    ┌────────────────────────────────────────────────────────────────┐
    │  MAX  (w^T μ - r_f) / √(w^T Σ w)                              │
    │                                                                │
    │  sous contraintes :                                            │
    │    Σ w_i = 1          (budget)                                 │
    │    w_i ≥ w_min        (poids minimum)                          │
    │    w_i ≤ w_max        (poids maximum)                          │
    │    w_i ≥ 0            (long only si PORTFOLIO["long_only"])    │
    └────────────────────────────────────────────────────────────────┘

    Astuce cvxpy : maximiser le Sharpe n'est pas convexe directement.
    On utilise la reformulation de Sharpe (Cornuejols & Tütüncü) :
      Pose y = w / (w^T μ - r_f), κ = 1 / (w^T μ - r_f)
      → minimise y^T Σ y  sous  μ^T y - r_f × κ = 1, Σy_i = κ
      → les poids finaux sont w = y / κ

    Pour la minimisation de variance à rendement cible :
      MIN  w^T Σ w
      s.t. w^T μ ≥ target_return
           Σ w_i = 1
           w_min ≤ w_i ≤ w_max

    Retourne un dict :
    {
      "weights"        : pd.Series  — poids optimaux par ticker
      "return_ann"     : float      — rendement annualisé attendu
      "volatility_ann" : float      — volatilité annualisée
      "sharpe_ratio"   : float      — ratio de Sharpe
      "status"         : str        — statut du solver ("optimal", ...)
    }

    TODO :
      - Implémente les deux modes (Sharpe max et variance min)
      - Utilise les contraintes de PORTFOLIO (weight_min, weight_max, long_only)
      - Calcule les métriques du portefeuille optimal
      - Affiche les poids optimaux de façon lisible
    """
    n = len(tickers)
    w_min = PORTFOLIO["weight_min"]   # 0.02
    w_max = PORTFOLIO["weight_max"]   # 0.35

    pass


# ─────────────────────────────────────────────
# 4. OPTIMISATION MEAN-CVaR
# ─────────────────────────────────────────────

def optimize_cvar(
    returns: pd.DataFrame,
    tickers: list,
    confidence_level: float = 0.95,
    target_return: float = None,
) -> dict:
    """
    Optimisation Mean-CVaR (Rockafellar & Uryasev, 2000).

    Pourquoi CVaR plutôt que variance ?
      → La variance pénalise symétriquement hausses et baisses
      → La CVaR capture uniquement les pertes extrêmes (queue gauche)
      → Plus pertinent pour les commodités énergie à distribution asymétrique
      → CVaR est convexe → problème LP résolu efficacement

    Formulation LP (Rockafellar & Uryasev) :
    ┌────────────────────────────────────────────────────────────────┐
    │  MIN  VaR_α + (1/(T(1-α))) × Σ_t max(-r_p(t) - VaR_α, 0)    │
    │                                                                │
    │  où r_p(t) = w^T r(t)  rendement du portefeuille au temps t   │
    │       α    = niveau de confiance (0.95 ou 0.99)               │
    │       T    = nombre d'observations                             │
    │                                                                │
    │  Variables : w (poids), VaR_α (scalaire), z_t (pertes excès)  │
    │                                                                │
    │  Reformulation cvxpy :                                         │
    │    z_t ≥ 0                                                     │
    │    z_t ≥ -w^T r(t) - VaR_α   pour tout t                      │
    │    CVaR = VaR_α + (1/T(1-α)) × Σ z_t                          │
    └────────────────────────────────────────────────────────────────┘

    Retourne le même format de dict que optimize_markowitz
    avec en plus "cvar" : float — CVaR optimale

    TODO :
      - Implémente la formulation LP de Rockafellar & Uryasev avec cvxpy
      - Variables : w (n,), alpha_var (scalaire = VaR), z (T,) = pertes excès
      - Contraintes : budget, long_only, weight_min/max, z ≥ 0,
                      z ≥ -returns @ w - alpha_var
      - Objectif : MIN alpha_var + (1 / (T × (1-confidence_level))) × sum(z)
      - Si target_return fourni : ajoute la contrainte mu^T w ≥ target_return
      - Calcule les métriques finales (return, vol, Sharpe, CVaR réalisée)
    """
    n = len(tickers)
    T = len(returns)
    R = returns.values   # matrice (T × n) des rendements historiques

    w_min = PORTFOLIO["weight_min"]
    w_max = PORTFOLIO["weight_max"]

    pass


# ─────────────────────────────────────────────
# 5. FRONTIÈRE EFFICIENTE
# ─────────────────────────────────────────────

def compute_efficient_frontier(
    mu: np.ndarray,
    Sigma: np.ndarray,
    returns: pd.DataFrame,
    tickers: list,
    n_points: int = N_FRONTIER,
) -> pd.DataFrame:
    """
    Génère la frontière efficiente complète.

    Principe :
      On résout le problème d'optimisation pour N rendements cibles
      couvrant la plage [min(mu), max(mu)].
      Chaque résolution donne un portefeuille optimal.
      L'ensemble de ces portefeuilles forme la frontière efficiente.

    Pour chaque point de la frontière, on calcule :
      - Les poids optimaux (Markowitz)
      - Le rendement attendu
      - La volatilité
      - Le Sharpe ratio
      - La CVaR 95%

    Le portefeuille de Sharpe maximum = le point le plus "au nord-ouest"
    de la frontière dans l'espace (volatilité, rendement).

    Retourne un DataFrame de N_FRONTIER lignes :
      - target_return  : rendement cible
      - volatility     : volatilité du portefeuille optimal
      - sharpe         : Sharpe ratio
      - cvar_95        : CVaR à 95%
      - w_{ticker}     : poids de chaque actif

    TODO :
      - Génère la grille de rendements cibles avec np.linspace
        entre min(mu) * 1.01 et max(mu) * 0.99
        (légèrement à l'intérieur pour éviter les problèmes aux bornes)
      - Pour chaque target, appelle optimize_markowitz(target_return=target)
      - Si le solver échoue (status != "optimal"), skip ce point
      - Assemble les résultats en DataFrame
      - Identifie le portefeuille de Sharpe max sur la frontière
      - Affiche un résumé : nb de points résolus, Sharpe max
    """
    pass


# ─────────────────────────────────────────────
# 6. PORTEFEUILLE ÉQUIPONDÉRÉ (BENCHMARK NAÏF)
# ─────────────────────────────────────────────

def equal_weight_portfolio(
    returns: pd.DataFrame,
    tickers: list,
    risk_free_rate: float = 0.03,
) -> dict:
    """
    Calcule les métriques d'un portefeuille équipondéré (1/N).

    Le portefeuille 1/N est le benchmark naturel de tout modèle
    d'optimisation : si ton modèle ne bat pas 1/N, il ne sert à rien.
    DeMiguel et al. (2009) ont montré que 1/N bat souvent Markowitz
    out-of-sample → ça justifie la comparaison systématique.

    TODO :
      - Poids uniformes : w_i = 1/N pour tout i
      - Calcule rendement, vol, Sharpe, CVaR du portefeuille 1/N
      - Retourne le même format de dict que optimize_markowitz
    """
    n = len(tickers)
    w = np.ones(n) / n   # 1/N
    pass


# ─────────────────────────────────────────────
# 7. EXPORT
# ─────────────────────────────────────────────

def save_portfolio_results(
    markowitz: dict,
    cvar: dict,
    equal_weight: dict,
    frontier: pd.DataFrame,
) -> None:
    """
    Sauvegarde les résultats dans data/processed/.

    TODO :
      - Sauvegarde les poids Markowitz, CVaR et 1/N dans un CSV comparatif
      - Sauvegarde la frontière efficiente dans un CSV séparé
      - Affiche un tableau comparatif des trois portefeuilles
    """
    pass


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run_portfolio_optimization() -> dict:
    """
    Pipeline complet :
      load → compute_inputs → optimize → frontier → compare → save
    """
    print("=" * 60)
    print("PORTFOLIO OPTIMIZATION")
    print("=" * 60)

    # Chargement
    returns        = load_returns()
    excess_returns = load_excess_returns()

    # Paramètres
    mu, Sigma, tickers = compute_inputs(returns)

    # Optimisation
    markowitz    = optimize_markowitz(mu, Sigma, tickers)
    cvar_port    = optimize_cvar(returns, tickers)
    equal_weight = equal_weight_portfolio(returns, tickers)

    # Frontière efficiente
    frontier = compute_efficient_frontier(mu, Sigma, returns, tickers)

    # Sauvegarde
    save_portfolio_results(markowitz, cvar_port, equal_weight, frontier)

    print("\n" + "=" * 60)
    print("OPTIMISATION TERMINÉE")
    print("=" * 60)

    return {
        "markowitz"    : markowitz,
        "cvar"         : cvar_port,
        "equal_weight" : equal_weight,
        "frontier"     : frontier,
        "mu"           : mu,
        "Sigma"        : Sigma,
        "tickers"      : tickers,
        "returns"      : returns,
    }


# ─────────────────────────────────────────────
# TEST STANDALONE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    results = run_portfolio_optimization()

    print("\nMarkowitz optimal :")
    print(results["markowitz"]["weights"].to_string())

    print("\nCVaR optimal :")
    print(results["cvar"]["weights"].to_string())

    print("\nÉquipondéré (1/N) :")
    print(results["equal_weight"]["weights"].to_string())

    print("\nFrontière efficiente — 5 premiers points :")
    print(results["frontier"].head().to_string())