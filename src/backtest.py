# src/backtest.py

"""
backtest.py — Backtesting walk-forward de la stratégie de portefeuille.

Principe du walk-forward :
  On divise l'historique en fenêtres glissantes :
    - Fenêtre d'entraînement (train) : on calibre le modèle (Markowitz/CVaR)
    - Fenêtre de test (test)         : on évalue les poids obtenus hors-sample

  À chaque pas, on avance d'une période et on répète.
  → Simule ce qu'un gérant aurait réellement obtenu en production.
  → Évite le biais de look-ahead (utiliser des données futures).

  Exemple avec train=3 ans, test=1 an, rebalancing mensuel :
    Période 1 : train [2019-2021] → test [2022]
    Période 2 : train [2020-2022] → test [2023]
    Période 3 : train [2021-2023] → test [2024]

Métriques de performance calculées :
  - Sharpe ratio, Sortino ratio, Calmar ratio
  - Maximum drawdown et durée du drawdown
  - Hit ratio (% de mois positifs)
  - Turnover (coût de rotation du portefeuille)
  - Alpha et Beta vs benchmark
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BACKTEST, RISK, PORTFOLIO, PATHS,
    START_DATE, END_DATE
)

TRADING_DAYS = RISK["trading_days"]


# ─────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────

def load_data(data_dir: str = None) -> tuple:
    """
    Charge les rendements et les données nécessaires au backtest.

    Retourne : (returns, benchmark_returns)
      returns           : pd.DataFrame — tous les actifs
      benchmark_returns : pd.Series    — benchmark (ENGI.PA par défaut)

    TODO :
      - Charge returns depuis data/processed/
      - Extrait la colonne config.BENCHMARK comme benchmark
      - Vérifie que le benchmark est présent
      - Retourne (returns, benchmark_returns)
    """
    from config import BENCHMARK

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

    # Extraction du benchmark
    if BENCHMARK not in returns.columns:
        print(f"[WARN] Benchmark '{BENCHMARK}' absent → utilisation du premier ticker")
        benchmark_returns = returns.iloc[:, 0].copy()
        benchmark_returns.name = returns.columns[0]
    else:
        benchmark_returns = returns[BENCHMARK].copy()
        benchmark_returns.name = BENCHMARK

    print(f"[LOAD] Rendements — {len(returns)} jours × {len(returns.columns)} actifs")
    print(f"[LOAD] Benchmark  — {benchmark_returns.name}")

    return returns, benchmark_returns


# ─────────────────────────────────────────────
# 2. WALK-FORWARD ENGINE
# ─────────────────────────────────────────────

def generate_walk_forward_windows(
    returns: pd.DataFrame,
    train_years: int = None,
    test_years: int = None,
) -> list:
    """
    Génère les fenêtres train/test pour le walk-forward.

    Stratégie "expanding window" vs "rolling window" :
      - Rolling  : fenêtre train de taille fixe → se déplace
      - Expanding: fenêtre train grandit avec le temps
      Ici on implémente le rolling (plus courant en pratique).

    Chaque fenêtre est un dict avec les métadonnées et les données.
    """
    train_years = train_years or BACKTEST["train_years"]   # 3
    test_years  = test_years  or BACKTEST["test_years"]    # 1

    start = returns.index[0]
    end   = returns.index[-1]

    # Vérification : assez d'historique pour au moins une fenêtre
    min_required = relativedelta(years=train_years + test_years)
    if end - start < pd.Timedelta(days=(train_years + test_years) * 252):
        raise ValueError(
            f"Historique insuffisant : {(end-start).days} jours disponibles, "
            f"{(train_years + test_years) * 252} requis "
            f"({train_years}+{test_years} ans)."
        )

    windows    = []
    window_id  = 0
    train_start = start

    while True:
        # Dates de la fenêtre courante
        train_end  = train_start + relativedelta(years=train_years)
        test_start = train_end
        test_end   = test_start + relativedelta(years=test_years)

        # Arrêt si la fenêtre de test dépasse l'historique disponible
        if test_end > end:
            break

        # Extraction des données correspondantes
        train_mask = (returns.index >= train_start) & (returns.index < train_end)
        test_mask  = (returns.index >= test_start)  & (returns.index < test_end)

        train_df = returns[train_mask].copy()
        test_df  = returns[test_mask].copy()

        # Vérifie qu'on a assez de données dans chaque split
        if len(train_df) < 60 or len(test_df) < 20:
            break

        windows.append({
            "window_id"     : window_id,
            "train_start"   : train_start,
            "train_end"     : train_end,
            "test_start"    : test_start,
            "test_end"      : test_end,
            "train_returns" : train_df,
            "test_returns"  : test_df,
        })

        # Avance d'une période de test (rolling)
        train_start = train_start + relativedelta(years=test_years)
        window_id  += 1

    print(f"\n[WALK-FORWARD] {len(windows)} fenêtres générées")
    print(f"  Train : {train_years} ans  |  Test : {test_years} an(s)")
    print(f"  {'Fenêtre':<8} {'Train':<25} {'Test':<25} {'N_train':>8} {'N_test':>8}")
    print(f"  {'─'*72}")
    for w in windows:
        print(f"  {w['window_id']:<8} "
              f"{str(w['train_start'].date())+'→'+str(w['train_end'].date()):<25} "
              f"{str(w['test_start'].date())+'→'+str(w['test_end'].date()):<25} "
              f"{len(w['train_returns']):>8} "
              f"{len(w['test_returns']):>8}")

    return windows


# ─────────────────────────────────────────────
# 3. OPTIMISATION SUR FENÊTRE D'ENTRAÎNEMENT
# ─────────────────────────────────────────────

def optimize_on_window(
    train_returns: pd.DataFrame,
    method: str = "markowitz",
) -> pd.Series:
    """
    Optimise le portefeuille sur la fenêtre d'entraînement.

    Méthodes disponibles :
      "markowitz"    → maximise le Sharpe (portfolio.optimize_markowitz)
      "cvar"         → minimise la CVaR (portfolio.optimize_cvar)
      "equal_weight" → 1/N (benchmark naïf)

    Retourne les poids optimaux comme pd.Series.

    TODO :
      - Importe les fonctions depuis portfolio.py
      - Selon method, appelle la bonne fonction d'optimisation
      - Si le solver échoue → fallback sur 1/N avec warning
      - Retourne les poids
    """
    from portfolio import (
        compute_inputs,
        optimize_markowitz,
        optimize_cvar,
        equal_weight_portfolio,
    )

    tickers = train_returns.columns.tolist()
    n       = len(tickers)

    try:
        if method == "equal_weight":
            result = equal_weight_portfolio(train_returns, tickers)
            return result["weights"]

        # Calcule mu et Sigma sur la fenêtre d'entraînement
        mu, Sigma, _ = compute_inputs(train_returns)

        if method == "markowitz":
            result = optimize_markowitz(mu, Sigma, tickers)
        elif method == "cvar":
            result = optimize_cvar(train_returns, tickers)
        else:
            raise ValueError(f"Méthode inconnue : '{method}'")

        if result.get("weights") is None:
            raise RuntimeError(f"Solver status : {result.get('status')}")

        return result["weights"]

    except Exception as e:
        # Fallback 1/N si l'optimisation échoue
        print(f"[WARN] Optimisation '{method}' échouée ({e}) → fallback 1/N")
        w = pd.Series(np.ones(n) / n, index=tickers, name="weight")
        return w


# ─────────────────────────────────────────────
# 4. ÉVALUATION SUR FENÊTRE DE TEST
# ─────────────────────────────────────────────

def evaluate_on_window(
    test_returns: pd.DataFrame,
    weights: pd.Series,
    prev_weights: pd.Series = None,
    transaction_costs_bps: int = None,
) -> dict:
    """
    Évalue les poids obtenus sur la fenêtre de test (hors-sample).

    Coûts de transaction :
      À chaque rebalancing, on paye des frais sur le turnover.
      Turnover = Σ |w_new_i - w_old_i|  (variation totale des poids)
      Coût = turnover × transaction_costs_bps / 10_000

      Ex : turnover=0.40, costs=5bps → 0.40 × 0.0005 = 0.02% de frais

    Retourne un dict avec :
      - returns_series : pd.Series des rendements journaliers du test
      - return_ann     : rendement annualisé
      - volatility_ann : volatilité annualisée
      - sharpe_ratio   : Sharpe ratio
      - max_drawdown   : drawdown maximum
      - turnover       : rotation du portefeuille
      - transaction_cost: coût en % appliqué au début de la période

    TODO :
      - Aligne weights sur test_returns.columns
      - Calcule le turnover si prev_weights fourni
      - Calcule le coût de transaction et soustrait du premier rendement
      - Calcule toutes les métriques de performance
    """
    tc_bps = transaction_costs_bps or BACKTEST["transaction_costs_bps"]   # 5
    sl_bps = BACKTEST["slippage_bps"]                                       # 2
    total_costs_bps = tc_bps + sl_bps   # 7 bps au total

    # Aligne les poids sur les colonnes disponibles dans le test
    common    = [t for t in weights.index if t in test_returns.columns]
    w_aligned = weights[common]
    w_aligned = w_aligned / w_aligned.sum()

    # Calcul du turnover si on a les poids précédents
    turnover = 0.0
    if prev_weights is not None:
        prev_common  = [t for t in prev_weights.index if t in common]
        prev_aligned = prev_weights.reindex(common).fillna(0.0)
        turnover     = float((w_aligned - prev_aligned).abs().sum())

    # Coût de transaction en fraction (pas en bps)
    # On multiplie par le turnover : seule la fraction changée est coûteuse
    transaction_cost = turnover * total_costs_bps / 10_000

    # Rendements du portefeuille sur la fenêtre de test
    port_returns = test_returns[common] @ w_aligned
    port_returns.name = "portfolio"

    # Soustraction du coût de transaction sur le premier jour
    if len(port_returns) > 0:
        port_returns.iloc[0] -= transaction_cost

    # ── Métriques de performance ──────────────────────────────────────────
    r = port_returns.dropna()

    ret_ann  = r.mean() * TRADING_DAYS
    vol_ann  = r.std()  * np.sqrt(TRADING_DAYS)
    sharpe   = (ret_ann - 0.03) / vol_ann if vol_ann > 0 else np.nan

    # Sortino
    downside = r[r < 0]
    dd_ann   = downside.std() * np.sqrt(TRADING_DAYS) if len(downside) > 1 else np.nan
    sortino  = (ret_ann - 0.03) / dd_ann if dd_ann and dd_ann > 0 else np.nan

    # Max drawdown
    cumul    = (1 + r).cumprod()
    peak     = cumul.cummax()
    drawdown = (cumul / peak) - 1
    max_dd   = float(drawdown.min())

    # VaR historique 95%
    var_95   = -np.percentile(r, 5) if len(r) > 20 else np.nan

    return {
        "returns_series"   : port_returns,
        "return_ann"       : float(ret_ann),
        "volatility_ann"   : float(vol_ann),
        "sharpe_ratio"     : float(sharpe) if not np.isnan(sharpe) else np.nan,
        "sortino_ratio"    : float(sortino) if not np.isnan(sortino) else np.nan,
        "max_drawdown"     : float(max_dd),
        "var_95"           : float(var_95) if not np.isnan(var_95) else np.nan,
        "turnover"         : float(turnover),
        "transaction_cost" : float(transaction_cost),
        "n_days"           : len(r),
    }



# ─────────────────────────────────────────────
# 5. PIPELINE WALK-FORWARD COMPLET
# ─────────────────────────────────────────────

def run_walk_forward(
    returns: pd.DataFrame,
    method: str = "markowitz",
    train_years: int = None,
    test_years: int = None,
) -> tuple:
    """
    Exécute le backtest walk-forward complet.

    Pour chaque fenêtre :
      1. Optimise sur train → obtient les poids
      2. Évalue sur test → obtient les rendements et métriques
      3. Stocke les résultats

    Concatène ensuite tous les rendements de test pour obtenir
    la série de performance complète hors-sample.

    Retourne : (all_returns, all_weights, window_metrics)
      all_returns    : pd.Series  — rendements hors-sample concaténés
      all_weights    : pd.DataFrame — poids à chaque fenêtre
      window_metrics : pd.DataFrame — métriques par fenêtre

    TODO :
      - Génère les fenêtres avec generate_walk_forward_windows
      - Boucle sur les fenêtres :
          weights = optimize_on_window(window["train_returns"], method)
          metrics = evaluate_on_window(window["test_returns"], weights, prev_weights)
          prev_weights = weights  (pour calcul turnover fenêtre suivante)
      - Concatène les rendements de test : pd.concat(all_returns_list)
      - Assemble window_metrics en DataFrame
      - Affiche un résumé : rendement total, Sharpe moyen, nb fenêtres
    """
    windows = generate_walk_forward_windows(returns, train_years, test_years)

    all_returns_list = []
    all_weights_list = []
    window_records   = []
    prev_weights     = None

    print(f"\n[WALK-FORWARD] Exécution — méthode : {method.upper()}")
    print(f"{'─'*70}")

    for w in windows:
        wid = w["window_id"]

        # ── Optimisation sur train ────────────────────────────────────────
        weights = optimize_on_window(w["train_returns"], method=method)

        # ── Évaluation sur test ───────────────────────────────────────────
        metrics = evaluate_on_window(
            w["test_returns"],
            weights,
            prev_weights=prev_weights,
        )

        # Stockage
        all_returns_list.append(metrics["returns_series"])

        weights_row        = weights.copy()
        weights_row.name   = w["test_start"]
        all_weights_list.append(weights_row)

        window_records.append({
            "window_id"        : wid,
            "test_start"       : w["test_start"].date(),
            "test_end"         : w["test_end"].date(),
            "return_ann"       : metrics["return_ann"],
            "volatility_ann"   : metrics["volatility_ann"],
            "sharpe_ratio"     : metrics["sharpe_ratio"],
            "sortino_ratio"    : metrics["sortino_ratio"],
            "max_drawdown"     : metrics["max_drawdown"],
            "var_95"           : metrics["var_95"],
            "turnover"         : metrics["turnover"],
            "transaction_cost" : metrics["transaction_cost"],
        })

        prev_weights = weights

        print(f"  Fenêtre {wid} | "
              f"Test {w['test_start'].date()}→{w['test_end'].date()} | "
              f"Rend {metrics['return_ann']*100:>+6.1f}% | "
              f"Sharpe {metrics['sharpe_ratio']:>5.2f} | "
              f"MaxDD {metrics['max_drawdown']*100:>6.1f}% | "
              f"Turnover {metrics['turnover']*100:.0f}%")

    # Concatène tous les rendements de test → série hors-sample complète
    all_returns    = pd.concat(all_returns_list).sort_index()
    all_returns.name = f"strategy_{method}"

    all_weights    = pd.DataFrame(all_weights_list).sort_index()
    window_metrics = pd.DataFrame(window_records).set_index("window_id")

    # Résumé global
    total_ret  = (1 + all_returns).prod() - 1
    ann_ret    = all_returns.mean() * TRADING_DAYS
    ann_vol    = all_returns.std()  * np.sqrt(TRADING_DAYS)
    avg_sharpe = window_metrics["sharpe_ratio"].mean()

    print(f"{'─'*70}")
    print(f"  RÉSUMÉ : Rendement total {total_ret*100:>+.1f}% | "
          f"Rend. ann. {ann_ret*100:>+.1f}% | "
          f"Vol {ann_vol*100:.1f}% | "
          f"Sharpe moy {avg_sharpe:.2f}")

    return all_returns, all_weights, window_metrics


# ─────────────────────────────────────────────
# 6. MÉTRIQUES GLOBALES DE PERFORMANCE
# ─────────────────────────────────────────────

def compute_global_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.03,
    method_label: str = "Strategy",
) -> pd.DataFrame:
    """
    Calcule les métriques de performance globales sur toute la période
    de backtest hors-sample.

    Métriques additionnelles vs compare_performance de hedging.py :

    Hit ratio :
      Fraction des mois (ou jours) où la stratégie est positive.
      Hit ratio > 50% → la stratégie génère plus de jours positifs que négatifs.
      Resample mensuel : returns.resample("ME").apply(lambda x: (1+x).prod()-1)

    Turnover moyen :
      Coût de rotation moyen par rebalancing → indicateur d'implémentabilité.
      Un turnover élevé signifie des coûts de transaction importants en pratique.

    Alpha & Beta vs benchmark (CAPM) :
      Régression OLS : r_strategy = alpha + beta × r_benchmark + epsilon
      alpha → surperformance ajustée du risque systématique (annualisée × 252)
      beta  → exposition au risque de marché
      Si alpha > 0 et statistiquement significatif → vraie valeur ajoutée

    Information Ratio :
      IR = mean(r_strategy - r_benchmark) / std(r_strategy - r_benchmark) × √252
      Mesure la régularité de la surperformance vs benchmark.
      IR > 0.5 → stratégie consistante, IR > 1.0 → excellente

    TODO :
      - Calcule toutes les métriques listées ci-dessus
      - Régression CAPM avec scipy_stats.linregress
      - Assemble en DataFrame avec colonnes [Strategy, Benchmark]
      - Affiche un tableau formaté complet
    """
    def _base_metrics(r: pd.Series) -> dict:
        """Métriques de base annualisées pour une série de rendements."""
        r = r.dropna()

        ret_ann  = r.mean() * TRADING_DAYS
        vol_ann  = r.std()  * np.sqrt(TRADING_DAYS)
        sharpe   = (ret_ann - risk_free_rate) / vol_ann if vol_ann > 0 else np.nan

        # Sortino
        downside = r[r < 0]
        dd_ann   = downside.std() * np.sqrt(TRADING_DAYS) if len(downside) > 1 else np.nan
        sortino  = (ret_ann - risk_free_rate) / dd_ann if dd_ann and dd_ann > 0 else np.nan

        # Max drawdown
        cumul    = (1 + r).cumprod()
        peak     = cumul.cummax()
        drawdown = (cumul / peak) - 1
        max_dd   = drawdown.min()

        # Calmar
        calmar   = ret_ann / abs(max_dd) if max_dd != 0 else np.nan

        # VaR & CVaR 95%
        var_95  = -np.percentile(r, 5)
        tail    = r[r <= -var_95]
        cvar_95 = -tail.mean() if len(tail) > 0 else var_95

        # Hit ratio mensuel
        # Resample → rendement mensuel → fraction de mois positifs
        monthly     = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        hit_ratio   = (monthly > 0).mean()

        # Rendement total cumulé
        total_return = (1 + r).prod() - 1

        return {
            "total_return"   : total_return,
            "return_ann"     : ret_ann,
            "volatility_ann" : vol_ann,
            "sharpe_ratio"   : sharpe,
            "sortino_ratio"  : sortino,
            "max_drawdown"   : max_dd,
            "calmar_ratio"   : calmar,
            "var_95"         : var_95,
            "cvar_95"        : cvar_95,
            "hit_ratio"      : hit_ratio,
            "n_obs"          : len(r),
        }

    # ── Métriques de base ─────────────────────────────────────────────────
    s_metrics = _base_metrics(strategy_returns)
    b_metrics = _base_metrics(benchmark_returns)

    # ── Régression CAPM : r_strategy = alpha + beta × r_benchmark ─────────
    # Aligne les deux séries sur l'intersection des dates
    aligned   = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    r_s       = aligned.iloc[:, 0].values
    r_b       = aligned.iloc[:, 1].values

    # OLS avec scipy — slope=beta, intercept=alpha journalier
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(r_b, r_s)

    beta_capm  = float(slope)
    alpha_daily = float(intercept)
    alpha_ann  = alpha_daily * TRADING_DAYS   # annualise l'alpha
    r_squared  = float(r_value ** 2)
    alpha_pval = float(p_value)

    # ── Information Ratio ─────────────────────────────────────────────────
    # IR = mean(active_return) / std(active_return) × √252
    # active_return = r_strategy - r_benchmark (tracking difference)
    active_ret = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    ir = (active_ret.mean() / active_ret.std() * np.sqrt(TRADING_DAYS)
          if active_ret.std() > 0 else np.nan)

    # Tracking error annualisée
    tracking_error = active_ret.std() * np.sqrt(TRADING_DAYS)

    # Turnover moyen (si disponible dans les window_metrics — passé en attrs)
    avg_turnover = s_metrics.get("turnover", np.nan)

    # ── Assemblage ────────────────────────────────────────────────────────
    metrics_dict = {
        method_label : {
            **s_metrics,
            "alpha_ann"      : alpha_ann,
            "beta_capm"      : beta_capm,
            "alpha_pvalue"   : alpha_pval,
            "r_squared"      : r_squared,
            "info_ratio"     : ir,
            "tracking_error" : tracking_error,
        },
        "Benchmark" : {
            **b_metrics,
            "alpha_ann"      : 0.0,
            "beta_capm"      : 1.0,
            "alpha_pvalue"   : np.nan,
            "r_squared"      : 1.0,
            "info_ratio"     : np.nan,
            "tracking_error" : 0.0,
        },
    }

    df = pd.DataFrame(metrics_dict)

    # ── Affichage formaté ─────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  MÉTRIQUES GLOBALES — {method_label.upper()} vs BENCHMARK")
    print(f"{'─'*65}")
    print(f"  {'Métrique':<24} {method_label:>16} {'Benchmark':>14}")
    print(f"  {'─'*58}")

    rows = [
        ("Rendement total",    "total_return",   True),
        ("Rendement ann.",     "return_ann",     True),
        ("Volatilité ann.",    "volatility_ann", True),
        ("Sharpe ratio",       "sharpe_ratio",   False),
        ("Sortino ratio",      "sortino_ratio",  False),
        ("Calmar ratio",       "calmar_ratio",   False),
        ("Max Drawdown",       "max_drawdown",   True),
        ("VaR 95%",            "var_95",         True),
        ("CVaR 95%",           "cvar_95",        True),
        ("Hit ratio (mensuel)","hit_ratio",      True),
        ("Alpha ann. (CAPM)",  "alpha_ann",      True),
        ("Beta CAPM",          "beta_capm",      False),
        ("Alpha p-value",      "alpha_pvalue",   False),
        ("R²",                 "r_squared",      False),
        ("Information Ratio",  "info_ratio",     False),
        ("Tracking Error",     "tracking_error", True),
    ]

    for label, key, is_pct in rows:
        s_val = df.loc[key, method_label]
        b_val = df.loc[key, "Benchmark"]

        if is_pct:
            s_str = f"{s_val*100:>+.2f}%" if not np.isnan(s_val) else "   N/A"
            b_str = f"{b_val*100:>+.2f}%" if not np.isnan(b_val) else "   N/A"
        else:
            s_str = f"{s_val:>+.3f}" if not np.isnan(s_val) else "  N/A"
            b_str = f"{b_val:>+.3f}" if not np.isnan(b_val) else "  N/A"

        # Indicateur visuel pour alpha
        flag = ""
        if key == "alpha_ann" and not np.isnan(s_val):
            flag = " ✓" if s_val > 0 and df.loc["alpha_pvalue", method_label] < 0.05 else ""
        if key == "alpha_pvalue" and not np.isnan(s_val):
            flag = " (sig.)" if s_val < 0.05 else " (non sig.)"

        print(f"  {label:<24} {s_str:>16} {b_str:>14}{flag}")

    print(f"{'─'*65}")

    return df



# ─────────────────────────────────────────────
# 7. COMPARAISON DES STRATÉGIES
# ─────────────────────────────────────────────

def compare_strategies(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """
    Compare les trois stratégies en walk-forward :
      1. Markowitz (Sharpe max)
      2. CVaR (minimisation CVaR)
      3. Equal Weight (1/N — benchmark naïf)

    Pour chaque stratégie :
      - Lance run_walk_forward()
      - Calcule compute_global_metrics()
      - Stocke dans un DataFrame comparatif

    Retourne un DataFrame avec les trois stratégies en colonnes
    et les métriques en lignes.

    TODO :
      - Boucle sur les trois méthodes
      - Assemble les résultats en DataFrame comparatif
      - Identifie la meilleure stratégie par Sharpe et par CVaR
      - Affiche le tableau final
    """
    strategies = ["markowitz", "cvar", "equal_weight"]
    strategies = {
        "markowitz"    : "Markowitz",
        "cvar"         : "CVaR",
        "equal_weight" : "1/N",
    }

    all_results     = {}
    comparison_rows = {}

    for method, label in strategies.items():
        print(f"\n{'='*60}")
        print(f"  Stratégie : {label}")
        print(f"{'='*60}")

        try:
            strategy_ret, all_weights, window_metrics = run_walk_forward(
                returns, method=method
            )
            global_metrics = compute_global_metrics(
                strategy_ret, benchmark_returns, method_label=label
            )
            drawdowns = analyze_drawdowns(strategy_ret)

            all_results[method] = {
                "returns"        : strategy_ret,
                "weights"        : all_weights,
                "window_metrics" : window_metrics,
                "global_metrics" : global_metrics,
                "drawdowns"      : drawdowns,
                "label"          : label,
            }

            # Collecte les métriques clés pour le tableau comparatif
            comparison_rows[label] = {
                "return_ann"     : global_metrics.loc["return_ann",     label],
                "volatility_ann" : global_metrics.loc["volatility_ann", label],
                "sharpe_ratio"   : global_metrics.loc["sharpe_ratio",   label],
                "sortino_ratio"  : global_metrics.loc["sortino_ratio",  label],
                "max_drawdown"   : global_metrics.loc["max_drawdown",   label],
                "alpha_ann"      : global_metrics.loc["alpha_ann",      label],
                "info_ratio"     : global_metrics.loc["info_ratio",     label],
                "hit_ratio"      : global_metrics.loc["hit_ratio",      label],
            }

            # Sauvegarde individuelle
            save_backtest_results(
                strategy_ret, window_metrics, global_metrics,
                drawdowns, method
            )

        except Exception as e:
            print(f"[ERROR] Stratégie '{label}' échouée : {e}")
            continue

    # ── Tableau comparatif final ──────────────────────────────────────────
    comparison_df = pd.DataFrame(comparison_rows)

    print(f"\n{'═'*70}")
    print(f"  TABLEAU COMPARATIF FINAL — TOUTES STRATÉGIES")
    print(f"{'═'*70}")
    print(f"  {'Métrique':<22}", end="")
    for label in strategies.values():
        print(f" {label:>14}", end="")
    print()
    print(f"  {'─'*65}")

    display = [
        ("Rendement ann.",  "return_ann",     True),
        ("Volatilité ann.", "volatility_ann", True),
        ("Sharpe ratio",    "sharpe_ratio",   False),
        ("Sortino ratio",   "sortino_ratio",  False),
        ("Max Drawdown",    "max_drawdown",   True),
        ("Alpha ann.",      "alpha_ann",      True),
        ("Info. Ratio",     "info_ratio",     False),
        ("Hit ratio",       "hit_ratio",      True),
    ]

    for label, key, is_pct in display:
        print(f"  {label:<22}", end="")
        for strat_label in strategies.values():
            if strat_label in comparison_df.columns:
                val = comparison_df.loc[key, strat_label]
                if is_pct:
                    print(f" {val*100:>+13.2f}%", end="")
                else:
                    print(f" {val:>14.3f}", end="")
        print()

    # Meilleure stratégie par Sharpe
    best_sharpe = comparison_df.loc["sharpe_ratio"].idxmax()
    best_alpha  = comparison_df.loc["alpha_ann"].idxmax()
    print(f"{'─'*65}")
    print(f"  Meilleur Sharpe : {best_sharpe}")
    print(f"  Meilleur Alpha  : {best_alpha}")
    print(f"{'═'*70}")

    all_results["comparison"] = comparison_df
    return all_results


# ─────────────────────────────────────────────
# 8. DRAWDOWN ANALYSIS
# ─────────────────────────────────────────────

def analyze_drawdowns(
    strategy_returns: pd.Series,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Identifie et analyse les N pires drawdowns de la stratégie.

    Pour chaque drawdown :
      - Date de début (pic)
      - Date du creux (trough)
      - Date de récupération (si elle existe)
      - Profondeur (magnitude en %)
      - Durée jusqu'au creux (jours)
      - Durée de récupération (jours, NaN si pas encore récupéré)

    Utile pour identifier les périodes de crise dans ton univers
    énergie : crise Covid mars 2020, crise gaz 2022, etc.

    TODO :
      - Calcule la série de drawdown : (cumulative / peak) - 1
      - Identifie les périodes de drawdown (drawdown < 0)
      - Pour chaque période : début, creux, récupération, profondeur
      - Retourne un DataFrame trié par profondeur décroissante
      - Affiche les top_n pires drawdowns
    """
    r         = strategy_returns.dropna()
    cumul     = (1 + r).cumprod()
    peak      = cumul.cummax()
    drawdown  = (cumul / peak) - 1

    # Identification des périodes de drawdown
    # Un drawdown commence quand drawdown < 0 et finit quand il revient à 0
    in_drawdown = drawdown < -1e-6   # tolérance numérique

    periods   = []
    dd_start  = None

    for date, is_dd in in_drawdown.items():
        if is_dd and dd_start is None:
            dd_start = date   # début du drawdown
        elif not is_dd and dd_start is not None:
            # Fin du drawdown (récupération)
            dd_period  = drawdown[dd_start:date]
            trough_idx = dd_period.idxmin()
            depth      = float(dd_period.min())
            duration   = (trough_idx - dd_start).days
            recovery   = (date - trough_idx).days

            periods.append({
                "start"             : dd_start.date(),
                "trough"            : trough_idx.date(),
                "recovery"          : date.date(),
                "depth"             : depth,
                "days_to_trough"    : duration,
                "days_to_recovery"  : recovery,
                "total_days"        : (date - dd_start).days,
            })
            dd_start = None

    # Drawdown encore en cours à la fin de la série
    if dd_start is not None:
        dd_period  = drawdown[dd_start:]
        trough_idx = dd_period.idxmin()
        depth      = float(dd_period.min())
        duration   = (trough_idx - dd_start).days

        periods.append({
            "start"             : dd_start.date(),
            "trough"            : trough_idx.date(),
            "recovery"          : None,   # pas encore récupéré
            "depth"             : depth,
            "days_to_trough"    : duration,
            "days_to_recovery"  : None,
            "total_days"        : (r.index[-1] - dd_start).days,
        })

    if not periods:
        print("[DRAWDOWN] Aucun drawdown significatif détecté")
        return pd.DataFrame()

    df = pd.DataFrame(periods)
    df = df.sort_values("depth").head(top_n).reset_index(drop=True)

    print(f"\n[DRAWDOWN] Top {min(top_n, len(df))} pires drawdowns :")
    print(f"  {'#':<3} {'Début':<12} {'Creux':<12} {'Récup.':<12} "
          f"{'Profondeur':>12} {'J→Creux':>9} {'J→Récup.':>10}")
    print(f"  {'─'*72}")

    for i, row in df.iterrows():
        recov_str = str(row["recovery"]) if row["recovery"] else "En cours"
        recov_d   = str(int(row["days_to_recovery"])) if row["days_to_recovery"] else "  -"
        print(f"  {i+1:<3} {str(row['start']):<12} {str(row['trough']):<12} "
              f"{recov_str:<12} {row['depth']*100:>11.2f}% "
              f"{int(row['days_to_trough']):>9} {recov_d:>10}")

    return df


# ─────────────────────────────────────────────
# 9. EXPORT
# ─────────────────────────────────────────────

def save_backtest_results(
    all_returns: pd.Series,
    window_metrics: pd.DataFrame,
    global_metrics: pd.DataFrame,
    drawdowns: pd.DataFrame,
    method: str,
) -> None:
    """
    Sauvegarde tous les outputs dans data/processed/.

    TODO :
      - all_returns    → backtest_returns_{method}_{dates}.csv
      - window_metrics → backtest_windows_{method}_{dates}.csv
      - global_metrics → backtest_metrics_{method}_{dates}.csv
      - drawdowns      → backtest_drawdowns_{method}_{dates}.csv
      - Affiche les filepaths
    """
    output_dir = PATHS["processed"]
    os.makedirs(output_dir, exist_ok=True)

    suffix = f"{method}_{START_DATE}_{END_DATE}"

    fp_ret  = os.path.join(output_dir, f"backtest_returns_{suffix}.csv")
    fp_win  = os.path.join(output_dir, f"backtest_windows_{suffix}.csv")
    fp_met  = os.path.join(output_dir, f"backtest_metrics_{suffix}.csv")
    fp_dd   = os.path.join(output_dir, f"backtest_drawdowns_{suffix}.csv")

    all_returns.to_csv(fp_ret)
    window_metrics.to_csv(fp_win)
    global_metrics.to_csv(fp_met)
    if not drawdowns.empty:
        drawdowns.to_csv(fp_dd, index=False)

    print(f"\n[SAVE] Returns      → {fp_ret}")
    print(f"[SAVE] Windows      → {fp_win}")
    print(f"[SAVE] Metrics      → {fp_met}")
    print(f"[SAVE] Drawdowns    → {fp_dd}")


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run_backtest(method: str = "markowitz") -> dict:
    """
    Pipeline complet du backtest walk-forward.

    Paramètre
    ---------
    method : str — "markowitz", "cvar", ou "equal_weight"
    """
    print("=" * 60)
    print(f"BACKTEST WALK-FORWARD — {method.upper()}")
    print("=" * 60)

    returns, benchmark = load_data()

    all_returns, all_weights, window_metrics = run_walk_forward(
        returns, method=method
    )
    global_metrics = compute_global_metrics(
        all_returns, benchmark, method_label=method
    )
    drawdowns = analyze_drawdowns(all_returns)

    save_backtest_results(
        all_returns, window_metrics, global_metrics,
        drawdowns, method
    )

    print("\n" + "=" * 60)
    print(f"BACKTEST TERMINÉ — {method.upper()}")
    print("=" * 60)

    return {
        "returns"        : all_returns,
        "weights"        : all_weights,
        "window_metrics" : window_metrics,
        "global_metrics" : global_metrics,
        "drawdowns"      : drawdowns,
        "benchmark"      : benchmark,
    }


def run_full_comparison() -> dict:
    """
    Lance les backtests des trois stratégies et les compare.
    Point d'entrée principal pour visualize.py.
    """
    print("=" * 60)
    print("BACKTEST COMPLET — 3 STRATÉGIES")
    print("=" * 60)

    returns, benchmark = load_data()
    results            = compare_strategies(returns, benchmark)
    results["returns"] = returns
    results["benchmark"] = benchmark

    return results
# ─────────────────────────────────────────────
# TEST STANDALONE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Test d'une seule stratégie d'abord
    results = run_backtest(method="markowitz")

    print("\nMétriques globales :")
    print(results["global_metrics"].to_string())

    print("\nTop drawdowns :")
    print(results["drawdowns"].to_string())

    print("\nMétriques par fenêtre :")
    print(results["window_metrics"][
        ["return_ann", "sharpe_ratio", "max_drawdown", "turnover"]
    ].to_string())

    # Comparaison complète (décommenter quand les 3 stratégies tournent)
    # full = run_full_comparison()
    # print(full["comparison"].to_string())