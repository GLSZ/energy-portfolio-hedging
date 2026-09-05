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
from sklearn.covariance import LedoitWolf

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

    data_dir = data_dir or PATHS["processed"]
    filepath = os.path.join(data_dir, f"returns_{START_DATE}_{END_DATE}.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Fichier introuvable : {filepath}\n"
            "Lance d'abord : python src/preprocess.py"
        )

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    print(f"[LOAD] Rendements — {len(df)} jours × {len(df.columns)} actifs")
    print(f"       Période : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"       Actifs  : {df.columns.tolist()}")

    return df

def load_excess_returns(data_dir: str = None) -> pd.DataFrame:
    """
    Charge les rendements en excès du taux sans risque.

    TODO :
      - Pattern : f"excess_returns_{START_DATE}_{END_DATE}.csv"
      - Même logique que load_returns
    """
    data_dir = data_dir or PATHS["processed"]
    filepath = os.path.join(data_dir, f"excess_returns_{START_DATE}_{END_DATE}.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Fichier introuvable : {filepath}\n"
            "Lance d'abord : python src/preprocess.py"
        )

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    print(f"[LOAD] Excès de rendements — {len(df)} jours × {len(df.columns)} actifs")

    return df


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
    tickers = returns.columns.tolist()
    n       = len(tickers)

    # ── Rendements espérés annualisés ─────────────────────────────────────
    mu = returns.mean().values * TRADING_DAYS   # shape (n,)

    # ── Matrice de covariance annualisée ──────────────────────────────────
    # Estimation Ledoit-Wolf sur les rendements journaliers
    lw = LedoitWolf().fit(returns.values)
    Sigma = lw.covariance_ * TRADING_DAYS       # annualisation × 252

    # ── Vérification définie positive ────────────────────────────────────
    # Eigenvalues > 0 → Sigma est définie positive → problème QP bien posé
    eigenvalues = np.linalg.eigvalsh(Sigma)
    min_eig     = eigenvalues.min()

    if min_eig < 0:
        # Régularisation additionnelle si Ledoit-Wolf insuffisant
        # (rare mais peut arriver sur des séries très courtes)
        eps   = abs(min_eig) + 1e-8
        Sigma = Sigma + eps * np.eye(n)
        print(f"[WARN] Sigma non définie positive (min eigenvalue={min_eig:.2e})"
              f" → régularisation +{eps:.2e}I appliquée")
    else:
        print(f"[INPUTS] Sigma définie positive ✓ (min eigenvalue={min_eig:.4f})")

    # ── Affichage des paramètres ──────────────────────────────────────────
    print(f"\n[INPUTS] Paramètres du problème d'optimisation :")
    print(f"  {'Actif':<12} {'μ ann.':>9} {'σ ann.':>9}")
    print(f"  {'-'*32}")
    for i, t in enumerate(tickers):
        vol_i = np.sqrt(Sigma[i, i])
        print(f"  {t:<12} {mu[i]*100:>8.2f}%  {vol_i*100:>8.2f}%")

    print(f"\n[INPUTS] Shrinkage Ledoit-Wolf : {lw.shrinkage_:.4f}")

    return mu, Sigma, tickers



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
    n     = len(tickers)
    w_min = PORTFOLIO["weight_min"]   # 0.02
    w_max = PORTFOLIO["weight_max"]   # 0.35

    if target_return is None:
        # ── Mode 1 : maximisation Sharpe (reformulation Cornuejols) ──────
        mu_excess = mu - risk_free_rate   # vecteur des excès de rendement

        # Variables de la reformulation
        y   = cp.Variable(n, name="y")       # y = w / (w^T μ - r_f)
        kap = cp.Variable(name="kappa")      # κ = 1 / (w^T μ - r_f)

        # Objectif : MIN y^T Σ y (= variance du portefeuille normalisé)
        objective = cp.Minimize(cp.quad_form(y, Sigma))

        constraints = [
            mu_excess @ y == 1,            # normalisation Sharpe
            cp.sum(y) == kap,              # budget normalisé
            y >= w_min * kap,              # poids minimum
            y <= w_max * kap,              # poids maximum
            kap >= 0,                      # κ positif (rendement > r_f)
        ]

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False)   # fallback

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(
                f"Markowitz Sharpe — solver status : {prob.status}\n"
                "Vérifie que μ > r_f pour au moins un actif."
            )

        # Récupération des poids finaux
        kap_val = float(kap.value)
        w_opt   = y.value / kap_val          # w = y / κ
        label   = "Sharpe Maximum"

    else:
        # ── Mode 2 : minimisation variance à rendement cible ─────────────
        w = cp.Variable(n, name="w")

        objective = cp.Minimize(cp.quad_form(w, Sigma))

        constraints = [
            mu @ w >= target_return,       # contrainte de rendement
            cp.sum(w) == 1,                # budget
            w >= w_min,                    # poids minimum
            w <= w_max,                    # poids maximum
        ]

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return {"status": prob.status, "weights": None}

        w_opt = w.value
        label = f"Variance Min (target={target_return*100:.1f}%)"

    # ── Métriques du portefeuille optimal ────────────────────────────────
    w_opt = np.clip(w_opt, 0, 1)           # clip numérique (évite -1e-10)
    w_opt = w_opt / w_opt.sum()            # renormalise à 1.0 exactement

    ret_ann = float(mu @ w_opt)
    vol_ann = float(np.sqrt(w_opt @ Sigma @ w_opt))
    sharpe  = (ret_ann - risk_free_rate) / vol_ann if vol_ann > 0 else np.nan

    weights_series = pd.Series(w_opt, index=tickers, name="weight")

    # ── Affichage ─────────────────────────────────────────────────────────
    print(f"\n[MARKOWITZ] {label}")
    print(f"  Rendement ann. : {ret_ann*100:.2f}%")
    print(f"  Volatilité ann.: {vol_ann*100:.2f}%")
    print(f"  Sharpe ratio   : {sharpe:.3f}")
    print(f"  Statut solver  : {prob.status}")
    print(f"\n  Poids optimaux :")
    for t, w in weights_series.items():
        bar = "█" * int(w * 40)
        print(f"    {t:<12} : {w*100:>5.1f}%  {bar}")

    return {
        "weights"        : weights_series,
        "return_ann"     : ret_ann,
        "volatility_ann" : vol_ann,
        "sharpe_ratio"   : sharpe,
        "status"         : prob.status,
        "label"          : label,
    }


# ─────────────────────────────────────────────
# 4. OPTIMISATION MEAN-CVaR
# ─────────────────────────────────────────────

def optimize_cvar(
    returns: pd.DataFrame,
    tickers: list,
    confidence_level: float = 0.95,
    target_return: float = None,
    risk_free_rate: float = 0.03,
) -> dict:
    """
    Optimisation Mean-CVaR (Rockafellar & Uryasev, 2000).

    Formulation LP (linéaire en w, alpha, z) :
    ┌────────────────────────────────────────────────────────────────┐
    │  Variables :                                                   │
    │    w       : poids (n,)                                        │
    │    alpha   : VaR scalaire (Value-at-Risk au niveau α)          │
    │    z       : pertes excédentaires (T,) — z_t ≥ 0              │
    │                                                                │
    │  Objectif :                                                    │
    │    MIN  alpha + (1 / (T × (1-α))) × Σ_t z_t                   │
    │                                                                │
    │  Contraintes :                                                 │
    │    z_t ≥ -r_p(t) - alpha    ∀t    (pertes > VaR)             │
    │    z_t ≥ 0                  ∀t    (pas de pertes négatives)   │
    │    r_p(t) = R[t,:] @ w            (rendement portefeuille)    │
    │    Σ w_i = 1                      (budget)                    │
    │    w_min ≤ w_i ≤ w_max            (bornes)                    │
    │    μ^T w ≥ target_return          (si spécifié)               │
    └────────────────────────────────────────────────────────────────┘

    Interprétation :
      alpha = VaR au niveau α (perte dépassée dans (1-α)% des cas)
      CVaR  = alpha + moyenne des pertes au-delà de la VaR
            = perte moyenne dans le pire (1-α)% des scénarios
    

    TODO :
      - Implémente la formulation LP de Rockafellar & Uryasev avec cvxpy
      - Variables : w (n,), alpha_var (scalaire = VaR), z (T,) = pertes excès
      - Contraintes : budget, long_only, weight_min/max, z ≥ 0,
                      z ≥ -returns @ w - alpha_var
      - Objectif : MIN alpha_var + (1 / (T × (1-confidence_level))) × sum(z)
      - Si target_return fourni : ajoute la contrainte mu^T w ≥ target_return
      - Calcule les métriques finales (return, vol, Sharpe, CVaR réalisée)
    """
    n   = len(tickers)
    T   = len(returns)
    R   = returns.values             # matrice (T × n) des rendements
    mu  = returns.mean().values * TRADING_DAYS

    w_min = PORTFOLIO["weight_min"]
    w_max = PORTFOLIO["weight_max"]
    alpha = 1 - confidence_level     # 0.05 pour CVaR 95%

    # ── Variables cvxpy ───────────────────────────────────────────────────
    w       = cp.Variable(n, name="w")
    var_val = cp.Variable(name="VaR")    # VaR scalaire au niveau α
    z       = cp.Variable(T, name="z")  # pertes excédentaires par scénario

    # ── Rendements du portefeuille sur l'historique ───────────────────────
    # r_p = R @ w  → vecteur (T,) des rendements du portefeuille
    r_portfolio = R @ w

    # ── Objectif : minimiser la CVaR ─────────────────────────────────────
    # CVaR = VaR + (1 / (T × α)) × Σ z_t
    cvar_expr = var_val + (1.0 / (T * alpha)) * cp.sum(z)
    objective = cp.Minimize(cvar_expr)

    # ── Contraintes ───────────────────────────────────────────────────────
    constraints = [
        z >= 0,                            # pertes excédentaires positives
        z >= -r_portfolio - var_val,       # z_t ≥ max(-r_p(t) - VaR, 0)
        cp.sum(w) == 1,                    # budget
        w >= w_min,                        # poids minimum
        w <= w_max,                        # poids maximum
    ]

    # Contrainte de rendement minimum (optionnelle)
    if target_return is not None:
        constraints.append(mu @ w >= target_return)

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        return {"status": prob.status, "weights": None}

    # ── Extraction des résultats ──────────────────────────────────────────
    w_opt   = np.clip(w.value, 0, 1)
    w_opt   = w_opt / w_opt.sum()

    # Métriques annualisées
    ret_ann  = float(mu @ w_opt)
    vol_ann  = float(np.sqrt(w_opt @ (returns.cov().values * TRADING_DAYS) @ w_opt))
    sharpe   = (ret_ann - risk_free_rate) / vol_ann if vol_ann > 0 else np.nan

    # CVaR réalisée sur l'historique (pour vérification)
    r_port_hist = R @ w_opt               # rendements historiques du portefeuille
    var_hist    = np.percentile(r_port_hist, (1 - confidence_level) * 100)
    cvar_hist   = -r_port_hist[r_port_hist <= var_hist].mean()   # perte positive

    weights_series = pd.Series(w_opt, index=tickers, name="weight")

    # ── Affichage ─────────────────────────────────────────────────────────
    print(f"\n[CVaR] Optimisation CVaR {confidence_level*100:.0f}%")
    print(f"  Rendement ann. : {ret_ann*100:.2f}%")
    print(f"  Volatilité ann.: {vol_ann*100:.2f}%")
    print(f"  Sharpe ratio   : {sharpe:.3f}")
    print(f"  VaR  {confidence_level*100:.0f}% (hist.): {-var_hist*100:.2f}% / jour")
    print(f"  CVaR {confidence_level*100:.0f}% (hist.): {cvar_hist*100:.2f}% / jour")
    print(f"  Statut solver  : {prob.status}")
    print(f"\n  Poids optimaux :")
    for t, w_i in weights_series.items():
        bar = "█" * int(w_i * 40)
        print(f"    {t:<12} : {w_i*100:>5.1f}%  {bar}")

    return {
        "weights"        : weights_series,
        "return_ann"     : ret_ann,
        "volatility_ann" : vol_ann,
        "sharpe_ratio"   : sharpe,
        "var_95"         : -var_hist,
        "cvar_95"        : cvar_hist,
        "status"         : prob.status,
        "label"          : f"CVaR Min {confidence_level*100:.0f}%",
    }


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
    r_min = mu.min() * 1.05   # légèrement au-dessus du minimum
    r_max = mu.max() * 0.95   # légèrement en dessous du maximum
    targets = np.linspace(r_min, r_max, n_points)

    print(f"\n[FRONTIER] Calcul de {n_points} portefeuilles optimaux...")
    print(f"           Rendement cible : [{r_min*100:.1f}%, {r_max*100:.1f}%]")

    records  = []
    n_solved = 0

    for i, target in enumerate(targets):
        result = optimize_markowitz(
            mu, Sigma, tickers,
            target_return=target,
        )

        # Si le solver échoue pour ce point → on le saute
        if result.get("weights") is None:
            continue

        row = {
            "target_return" : target,
            "return_ann"    : result["return_ann"],
            "volatility_ann": result["volatility_ann"],
            "sharpe_ratio"  : result["sharpe_ratio"],
        }

        # Poids de chaque actif sur ce point de frontière
        for t, w_i in result["weights"].items():
            row[f"w_{t}"] = w_i

        records.append(row)
        n_solved += 1

        # Affichage progression
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_points} points résolus...")

    frontier = pd.DataFrame(records)

    # ── Identification du portefeuille de Sharpe maximum ─────────────────
    if not frontier.empty:
        idx_sharpe = frontier["sharpe_ratio"].idxmax()
        best       = frontier.loc[idx_sharpe]
        print(f"\n[FRONTIER] {n_solved}/{n_points} points résolus avec succès")
        print(f"[FRONTIER] Portefeuille Sharpe Maximum :")
        print(f"  Rendement  : {best['return_ann']*100:.2f}%")
        print(f"  Volatilité : {best['volatility_ann']*100:.2f}%")
        print(f"  Sharpe     : {best['sharpe_ratio']:.3f}")

    return frontier

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
    n   = len(tickers)
    w   = np.ones(n) / n   # poids uniformes 1/N
    mu  = returns.mean().values * TRADING_DAYS
    Cov = returns.cov().values * TRADING_DAYS

    ret_ann = float(mu @ w)
    vol_ann = float(np.sqrt(w @ Cov @ w))
    sharpe  = (ret_ann - risk_free_rate) / vol_ann if vol_ann > 0 else np.nan

    # VaR et CVaR historiques
    r_port  = returns.values @ w
    var_95  = -np.percentile(r_port, 5)
    cvar_95 = -r_port[r_port <= -var_95].mean()

    weights_series = pd.Series(w, index=tickers, name="weight")

    print(f"\n[1/N] Portefeuille équipondéré (benchmark)")
    print(f"  Rendement ann. : {ret_ann*100:.2f}%")
    print(f"  Volatilité ann.: {vol_ann*100:.2f}%")
    print(f"  Sharpe ratio   : {sharpe:.3f}")
    print(f"  CVaR 95%       : {cvar_95*100:.2f}% / jour")

    return {
        "weights"        : weights_series,
        "return_ann"     : ret_ann,
        "volatility_ann" : vol_ann,
        "sharpe_ratio"   : sharpe,
        "var_95"         : var_95,
        "cvar_95"        : cvar_95,
        "status"         : "analytical",
        "label"          : "Équipondéré 1/N",
    }



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
    output_dir = PATHS["processed"]
    os.makedirs(output_dir, exist_ok=True)

    # ── Tableau comparatif des trois portefeuilles ────────────────────────
    comparison = pd.DataFrame({
        "Markowitz"  : markowitz["weights"],
        "CVaR 95%"   : cvar["weights"],
        "1/N"        : equal_weight["weights"],
    }).round(4)

    # Ajout des métriques en bas du tableau
    metrics_rows = pd.DataFrame({
        "Markowitz" : {
            "return_ann"     : markowitz["return_ann"],
            "volatility_ann" : markowitz["volatility_ann"],
            "sharpe_ratio"   : markowitz["sharpe_ratio"],
            "cvar_95"        : markowitz.get("cvar_95", np.nan),
        },
        "CVaR 95%" : {
            "return_ann"     : cvar["return_ann"],
            "volatility_ann" : cvar["volatility_ann"],
            "sharpe_ratio"   : cvar["sharpe_ratio"],
            "cvar_95"        : cvar.get("cvar_95", np.nan),
        },
        "1/N" : {
            "return_ann"     : equal_weight["return_ann"],
            "volatility_ann" : equal_weight["volatility_ann"],
            "sharpe_ratio"   : equal_weight["sharpe_ratio"],
            "cvar_95"        : equal_weight.get("cvar_95", np.nan),
        },
    }).T

    # Sauvegarde
    fp_weights = os.path.join(output_dir, f"portfolio_weights_{START_DATE}_{END_DATE}.csv")
    fp_metrics = os.path.join(output_dir, f"portfolio_metrics_{START_DATE}_{END_DATE}.csv")
    fp_frontier= os.path.join(output_dir, f"efficient_frontier_{START_DATE}_{END_DATE}.csv")

    comparison.to_csv(fp_weights)
    metrics_rows.to_csv(fp_metrics)
    frontier.to_csv(fp_frontier, index=False)

    print(f"\n[SAVE] Poids      → {fp_weights}")
    print(f"[SAVE] Métriques  → {fp_metrics}")
    print(f"[SAVE] Frontière  → {fp_frontier}")

    # ── Tableau comparatif affiché ────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  COMPARAISON DES PORTEFEUILLES")
    print(f"{'─'*60}")
    print(f"  {'Métrique':<20} {'Markowitz':>12} {'CVaR 95%':>12} {'1/N':>10}")
    print(f"  {'─'*56}")

    rows = [
        ("Rendement ann.",   "return_ann",     True,  "%"),
        ("Volatilité ann.",  "volatility_ann", True,  "%"),
        ("Sharpe ratio",     "sharpe_ratio",   False, ""),
        ("CVaR 95% /jour",  "cvar_95",        True,  "%"),
    ]

    for label, key, is_pct, unit in rows:
        m_val = markowitz.get(key, np.nan)
        c_val = cvar.get(key, np.nan)
        e_val = equal_weight.get(key, np.nan)

        if is_pct:
            print(f"  {label:<20} {m_val*100:>11.2f}%"
                  f" {c_val*100:>11.2f}%"
                  f" {e_val*100:>9.2f}%")
        else:
            print(f"  {label:<20} {m_val:>12.3f}"
                  f" {c_val:>12.3f}"
                  f" {e_val:>10.3f}")

    print(f"{'─'*60}")

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

    print("\nMarkowitz — poids optimaux :")
    print(results["markowitz"]["weights"].to_string())

    print("\nCVaR — poids optimaux :")
    print(results["cvar"]["weights"].to_string())

    print("\nÉquipondéré (1/N) :")
    print(results["equal_weight"]["weights"].to_string())

    print(f"\nFrontière — {len(results['frontier'])} points calculés")
    print(results["frontier"][["return_ann", "volatility_ann", "sharpe_ratio"]].head())