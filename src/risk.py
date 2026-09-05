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
    return df


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
    # Garde uniquement les actifs présents dans les deux
    common = [t for t in weights.index if t in returns.columns]
    n_dropped = len(weights) - len(common)

    if n_dropped > 0:
        print(f"[WARN] {n_dropped} ticker(s) absents des rendements → exclus du portefeuille")

    w_aligned = weights[common]
    w_aligned = w_aligned / w_aligned.sum()   # renormalise à 1.0

    port_returns = returns[common] @ w_aligned
    port_returns.name = "portfolio"

    print(f"[PORTFOLIO] Rendements calculés — {len(port_returns)} jours")
    print(f"            Rendement moy journalier : {port_returns.mean()*100:.4f}%")
    print(f"            Volatilité journalière   : {port_returns.std()*100:.4f}%")

    return port_returns


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
    r = portfolio_returns.dropna()

    if window is not None:
        r = r.iloc[-window:]    # fenêtre glissante : n derniers jours

    var = -np.percentile(r, (1 - confidence_level) * 100)
    return float(var)


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
    r   = portfolio_returns.dropna()

    if window is not None:
        r = r.iloc[-window:]

    var  = var_historical(pd.Series(r), confidence_level)
    tail = r[r <= -var]   # rendements dans la queue gauche

    if len(tail) == 0:
        # Cas dégénéré (très peu de données) → retourne la VaR
        return var

    cvar = -tail.mean()
    return float(cvar)


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
    r = portfolio_returns.dropna()

    mu    = r.mean()
    sigma = r.std(ddof=1)

    # Quantile normal au niveau (1-confidence_level)
    # Ex : confidence=0.95 → ppf(0.05) = -1.6449
    z_alpha = scipy_stats.norm.ppf(1 - confidence_level)

    # VaR gaussienne
    var = -(mu + z_alpha * sigma)

    # CVaR gaussienne (formule analytique)
    # phi = densité normale standard évaluée en z_alpha
    phi  = scipy_stats.norm.pdf(z_alpha)
    cvar = -(mu - sigma * phi / (1 - confidence_level))

    return float(var), float(cvar), float(mu), float(sigma)


# ─────────────────────────────────────────────
# 4. VAR MONTE CARLO
# ─────────────────────────────────────────────

def var_monte_carlo(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
    n_simulations: int = None,
    method: str = "gaussian",
    seed: int = 42,
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
    n_simulations = n_simulations or RISK["n_simulations"]   # 10_000
    r = portfolio_returns.dropna()
    rng = np.random.default_rng(seed)   # générateur moderne (remplace np.random.seed)

    if method == "gaussian":
        mu, sigma = r.mean(), r.std(ddof=1)
        simulated = rng.normal(mu, sigma, n_simulations)

    elif method == "bootstrap":
        # Rééchantillonnage avec remise des rendements historiques
        simulated = rng.choice(r.values, size=n_simulations, replace=True)

    else:
        raise ValueError(f"Méthode Monte Carlo inconnue : '{method}'. "
                         "Utilise 'gaussian' ou 'bootstrap'.")

    var  = -np.percentile(simulated, (1 - confidence_level) * 100)
    tail = simulated[simulated <= -var]
    cvar = -tail.mean() if len(tail) > 0 else var

    return float(var), float(cvar), simulated


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
    records = []

    print(f"\n[VAR] Comparaison des méthodes — "
          f"niveaux : {[f'{c*100:.0f}%' for c in confidence_levels]}")

    for cl in confidence_levels:
        cl_label = f"{cl*100:.0f}%"

        # 1. Historique
        var_h  = var_historical(portfolio_returns, cl)
        cvar_h = cvar_historical(portfolio_returns, cl)

        # 2. Paramétrique
        var_p, cvar_p, mu_p, sigma_p = var_parametric(portfolio_returns, cl)

        # 3. Monte Carlo gaussien
        var_mc_g, cvar_mc_g, _ = var_monte_carlo(
            portfolio_returns, cl, method="gaussian"
        )

        # 4. Monte Carlo bootstrap
        var_mc_b, cvar_mc_b, _ = var_monte_carlo(
            portfolio_returns, cl, method="bootstrap"
        )

        # Ratio historique / paramétrique (indicateur queues épaisses)
        ratio = var_h / var_p if var_p > 0 else np.nan

        records.append({
            "confidence"       : cl_label,
            "var_historical"   : var_h,
            "var_parametric"   : var_p,
            "var_mc_gaussian"  : var_mc_g,
            "var_mc_bootstrap" : var_mc_b,
            "cvar_historical"  : cvar_h,
            "cvar_parametric"  : cvar_p,
            "cvar_mc_gaussian" : cvar_mc_g,
            "cvar_mc_bootstrap": var_mc_b,
            "ratio_hist_param" : ratio,
        })

    df = pd.DataFrame(records).set_index("confidence")

    # ── Affichage formaté ─────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  {'':20} {'VaR 95%':>10} {'VaR 99%':>10} "
          f"{'CVaR 95%':>10} {'CVaR 99%':>10}")
    print(f"  {'─'*68}")

    methods = [
        ("Historique",        "var_historical",   "cvar_historical"),
        ("Paramétrique",      "var_parametric",   "cvar_parametric"),
        ("MC Gaussien",       "var_mc_gaussian",  "cvar_mc_gaussian"),
        ("MC Bootstrap",      "var_mc_bootstrap", "cvar_mc_bootstrap"),
    ]

    for label, var_col, cvar_col in methods:
        vals_var  = [df.loc[f"{c*100:.0f}%", var_col]  * 100
                     for c in confidence_levels]
        vals_cvar = [df.loc[f"{c*100:.0f}%", cvar_col] * 100
                     for c in confidence_levels]
        # Affiche jusqu'à 2 niveaux de confiance
        v95  = vals_var[0]  if len(vals_var)  > 0 else np.nan
        v99  = vals_var[1]  if len(vals_var)  > 1 else np.nan
        cv95 = vals_cvar[0] if len(vals_cvar) > 0 else np.nan
        cv99 = vals_cvar[1] if len(vals_cvar) > 1 else np.nan
        print(f"  {label:<20} {v95:>9.3f}%  {v99:>9.3f}%  "
              f"{cv95:>9.3f}%  {cv99:>9.3f}%")

    print(f"{'─'*72}")

    # Ratio queues épaisses
    for cl in confidence_levels:
        cl_label = f"{cl*100:.0f}%"
        ratio = df.loc[cl_label, "ratio_hist_param"]
        flag  = " ⚠ queues épaisses" if ratio > 1.1 else " ✓ proche normale"
        print(f"  Ratio hist/param {cl_label} : {ratio:.3f}{flag}")

    return df


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
    capital   = 1_000_000   # €

    # Aligne les poids sur les actifs disponibles
    common   = [t for t in weights.index if t in returns.columns]
    w_aligned = weights[common] / weights[common].sum()

    records = []
    print(f"\n[STRESS] Application de {len(scenarios)} scénarios de stress")
    print(f"{'─'*65}")

    for scenario_name, shocks in scenarios.items():

        # Impact par actif : choc × poids (0 si actif non choqué)
        impact_total = 0.0
        details      = []

        for ticker in common:
            w_i    = float(w_aligned.get(ticker, 0.0))
            choc_i = shocks.get(ticker, 0.0)   # 0.0 si non mentionné
            contrib = w_i * choc_i
            impact_total += contrib

            if choc_i != 0.0:
                details.append(f"{ticker}({choc_i*100:+.0f}%→{contrib*100:+.2f}%)")

        impact_eur = impact_total * capital

        records.append({
            "scenario"         : scenario_name,
            "impact_portfolio" : impact_total,
            "impact_eur"       : impact_eur,
            "details"          : "  |  ".join(details),
        })

        # Affichage coloré (pire impact en premier)
        flag = "🔴" if impact_total < -0.10 else "🟡" if impact_total < 0 else "🟢"
        print(f"  {flag} {scenario_name:<30} "
              f"{impact_total*100:>+7.2f}%  "
              f"({impact_eur:>+10,.0f} €)")
        if details:
            print(f"     ↳ {' | '.join(details)}")

    print(f"{'─'*65}")

    df = pd.DataFrame(records).set_index("scenario")
    df = df.sort_values("impact_portfolio")   # pire scénario en premier

    return df


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
    r          = portfolio_returns.dropna()
    alpha      = 1 - confidence_level   # taux de violation attendu (ex: 0.05)
    n_total    = len(r) - window

    if n_total <= 0:
        raise ValueError(
            f"Historique insuffisant : {len(r)} jours < fenêtre {window} jours"
        )

    print(f"\n[BACKTEST] VaR {confidence_level*100:.0f}% — "
          f"fenêtre {window}j — {n_total} jours testés")

    records = []
    for t in range(window, len(r)):
        # Fenêtre d'estimation : [t-window, t-1]
        hist  = r.iloc[t - window : t]
        var_t = var_historical(hist, confidence_level)

        # Rendement réel au jour t
        r_actual   = float(r.iloc[t])
        violation  = r_actual < -var_t

        records.append({
            "date"          : r.index[t],
            "var_predicted" : var_t,
            "return_actual" : r_actual,
            "violation"     : violation,
        })

    df = pd.DataFrame(records).set_index("date")

    # ── Statistiques de violations ────────────────────────────────────────
    n_violations = df["violation"].sum()
    rate_observed = n_violations / n_total
    rate_expected = alpha

    print(f"  Violations observées : {n_violations} / {n_total}")
    print(f"  Taux observé  : {rate_observed*100:.2f}%")
    print(f"  Taux attendu  : {rate_expected*100:.2f}%")

    # ── Test de Kupiec (POF) ──────────────────────────────────────────────
    # Statistique de rapport de vraisemblance
    T0 = n_violations          # nombre de violations
    T1 = n_total - n_violations # nombre de non-violations
    p  = rate_observed          # taux observé
    p0 = rate_expected          # taux théorique

    # Protection contre log(0) si aucune violation ou 100% de violations
    eps = 1e-10
    p   = np.clip(p, eps, 1 - eps)

    lr_stat = -2 * (
        T0 * np.log(p0 / p) + T1 * np.log((1 - p0) / (1 - p))
    )

    # Distribution χ²(1) sous H0
    p_value = 1 - scipy_stats.chi2.cdf(lr_stat, df=1)

    print(f"\n  Test de Kupiec (POF) :")
    print(f"    LR stat  : {lr_stat:.4f}")
    print(f"    p-value  : {p_value:.4f}")

    if p_value < 0.05:
        print(f"    Résultat : ❌ Modèle VaR rejeté (p < 0.05)")
        if rate_observed > rate_expected:
            print(f"    Cause    : trop de violations → VaR sous-estimée")
        else:
            print(f"    Cause    : trop peu de violations → VaR sur-estimée (trop conservatrice)")
    else:
        print(f"    Résultat : ✓ Modèle VaR non rejeté (p ≥ 0.05)")

    # Dates des violations (utile pour visualize.py)
    violation_dates = df[df["violation"]].index.tolist()
    print(f"\n  Premières violations :")
    for d in violation_dates[:5]:
        r_val  = df.loc[d, "return_actual"] * 100
        v_val  = df.loc[d, "var_predicted"] * 100
        print(f"    {d.date()} : rendement {r_val:+.2f}%  VaR {v_val:.2f}%")

    # Stocke les stats dans les attributs du DataFrame pour visualize.py
    df.attrs["n_violations"]   = int(n_violations)
    df.attrs["rate_observed"]  = float(rate_observed)
    df.attrs["rate_expected"]  = float(rate_expected)
    df.attrs["kupiec_lr"]      = float(lr_stat)
    df.attrs["kupiec_pvalue"]  = float(p_value)
    df.attrs["confidence"]     = confidence_level

    return df


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

    returns       = load_returns()
    portfolio_ret = compute_portfolio_returns(returns, weights)

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
    backtest_results.to_csv(
        os.path.join(output_dir, f"var_backtest_{START_DATE}_{END_DATE}.csv")
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
    import numpy as np

    # Charge les rendements
    returns_test = pd.read_csv(
        f"data/processed/returns_{START_DATE}_{END_DATE}.csv",
        index_col=0, parse_dates=True
    )

    if isinstance(returns_test.columns, pd.MultiIndex):
        returns_test.columns = returns_test.columns.get_level_values(-1)

    # Poids équipondérés pour le test standalone
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
    print(results["stress_tests"][["impact_portfolio", "impact_eur"]].to_string())

    print("\nBacktest VaR — premières violations :")
    bt = results["var_backtest"]
    print(bt[bt["violation"]].head())

    print(f"\nKupiec p-value : {bt.attrs['kupiec_pvalue']:.4f}")