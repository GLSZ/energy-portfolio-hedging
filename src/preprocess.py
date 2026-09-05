# src/preprocess.py

"""
preprocess.py — Nettoyage, alignement et calcul des rendements
à partir des données brutes produites par fetch_data.py.

Rôle :
  1. Charger les prix bruts depuis data/raw/
  2. Aligner toutes les séries sur le même index de dates de trading
  3. Gérer les valeurs manquantes (NaN) de façon cohérente
  4. Calculer les rendements logarithmiques
  5. Calculer les statistiques descriptives (vol, skew, kurtosis, corrélations)
  6. Exporter le DataFrame propre pour portfolio.py et risk.py

Pourquoi les rendements logarithmiques ?
  log(P_t / P_{t-1}) = log(P_t) - log(P_{t-1})
  → Additivité dans le temps : le rendement sur N jours = somme des rendements journaliers
  → Symétrie : +50% puis -50% ≠ 0 en rendements simples, = 0 en log-rendements
  → Hypothèse de normalité mieux vérifiée qu'avec les rendements simples
  → Standard en finance quantitative et en risk management
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TICKERS, START_DATE, END_DATE,
    RISK, PATHS, BACKTEST
)

# ─────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES BRUTES
# ─────────────────────────────────────────────

def load_prices(data_dir: str = None) -> pd.DataFrame:
    """
    Charge le CSV des prix produit par fetch_data.py.

    TODO :
      - Construis le filepath avec PATHS["raw"] et le pattern
        f"prices_{START_DATE}_{END_DATE}.csv"
      - Lis le CSV avec pd.read_csv(), parse l'index en datetime
      - Affiche le nombre de lignes, colonnes et la période chargée
      - Lève une FileNotFoundError explicite si le fichier est absent
        avec un message indiquant de lancer fetch_data.py d'abord
    """

    data_dir = data_dir or PATHS["raw"]
    filepath = os.path.join(data_dir, f"prices_{START_DATE}_{END_DATE}.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Fichier introuvable : {filepath}\n"
            "Lance d'abord : python src/fetch_data.py"
        )

    df = pd.read_csv(filepath, index_col = 0, parse_dates = True)

    # Nettoyage du MultiIndex de colonnes que yfinance peut générer
    # yfinance retourne parfois des colonnes sous forme de tuples ("Close", "ENGI.PA")
    # → on aplatit pour n'avoir que le nom du ticker

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    print(f"[LOAD] Prix — {len(df)} jours × {len(df.columns)} tickers")
    print(f"       Période : {df.index[0].date()} → {df.index[-1].date()}")

    return df


def load_risk_free(data_dir: str = None) -> pd.Series:
    """
    Charge la série du taux sans risque produite par fetch_data.py.

    TODO :
      - Même logique que load_prices
      - Pattern : f"risk_free_{START_DATE}_{END_DATE}.csv"
      - Retourne une pd.Series (pas un DataFrame)
      - L'index doit être en datetime
    """

    data_dir = data_dir or PATHS["raw"]
    filepath = os.path.join(data_dir, f"risk_free_{START_DATE}_{END_DATE}.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Fichier introuvable : {filepath}\n"
            "Lance d'abord : python src/fetch_data.py"
        )

    df = pd.read_csv(filepath, index_col = 0, parse_dates=True)
    series = df.iloc[:, 0]   # première (et seule) colonne → Series
    series.name = "risk_free"

    print(f"[LOAD] Taux sans risque — {len(series)} observations")

    return series


def load_eia(data_dir: str = None) -> pd.DataFrame:
    """
    Charge les prix EIA (WTI, Brent) produits par fetch_data.py.

    TODO :
      - Pattern : f"eia_{START_DATE}_{END_DATE}.csv"
      - Même logique que load_prices
    """
    data_dir = data_dir or PATHS["raw"]
    filepath = os.path.join(data_dir, f"eia_{START_DATE}_{END_DATE}.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Fichier introuvable : {filepath}\n"
            "Lance d'abord : python src/fetch_data.py"
        )

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    print(f"[LOAD] EIA — {len(df)} observations × {len(df.columns)} séries")

    return df


# ─────────────────────────────────────────────
# 2. ALIGNEMENT TEMPOREL
# ─────────────────────────────────────────────

def align_trading_calendar(
    prices: pd.DataFrame,
    risk_free: pd.Series,
    eia: pd.DataFrame,
) -> tuple:
    """
    Aligne les trois sources de données sur un index commun
    de jours de trading.

    Problème : les sources ont des calendriers différents
      - yfinance : jours où au moins un marché est ouvert
        (NYSE, Euronext, LSE ont des jours fériés différents)
      - FRED     : jours calendaires (inclut week-ends → déjà ffill)
      - EIA      : jours ouvrés US uniquement

    Stratégie :
      - Index de référence = index des prix yfinance
        (le plus restrictif = intersection des marchés)
      - On réindexe FRED et EIA sur cet index avec forward fill
        pour combler les jours manquants

    TODO :
      - Définis l'index commun = prices.index
      - Réindexe risk_free sur cet index avec .reindex() + .ffill()
      - Réindexe eia sur cet index avec .reindex() + .ffill()
      - Affiche le nombre de jours dans l'index commun
      - Retourne (prices, risk_free, eia) alignés
    """

    trading_index = prices.index

    # Réindexe le taux sans risque sur l'index de trading
    # ffill(limit=5) : on propage max 5 jours (une semaine de trading)
    # pour ne pas propager trop loin si une série s'arrête

    risk_free_aligned = (
        risk_free
        .reindex(trading_index)
        .ffill(limit = 5)
    )

    # Même logique pour EIA
    eia_aligned = (
        eia
        .reindex(trading_index)
        .ffill(limit=5)
    )

    n_rf_nan  = risk_free_aligned.isnull().sum()
    n_eia_nan = eia_aligned.isnull().any(axis=1).sum()


    print(f"\n[ALIGN] Index commun : {len(trading_index)} jours de trading")
    print(f"        Risk-free NaN résiduels : {n_rf_nan}")
    print(f"        EIA jours incomplets    : {n_eia_nan}")

    return prices, risk_free_aligned, eia_aligned
    


# ─────────────────────────────────────────────
# 3. GESTION DES VALEURS MANQUANTES
# ─────────────────────────────────────────────

def handle_missing_values(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Traite les NaN dans le DataFrame de prix de façon cohérente.

    Stratégie par type de NaN :
      - NaN en début de série (ticker listé après START_DATE)
        → on garde : le ticker n'existait pas encore
      - NaN isolés au milieu (jour férié local, suspension de cotation)
        → forward fill (le prix de la veille est maintenu)
      - NaN en fin de série (ticker suspendu avant END_DATE)
        → on garde : on arrête d'utiliser ce ticker
      - Tickers avec > SEUIL % de NaN
        → exclusion complète avec warning

    TODO :
      - Calcule le % de NaN par ticker
      - Exclut les tickers avec > 20% de NaN (SEUIL configurable)
      - Applique forward fill sur les NaN isolés (limit=5 jours max)
      - Affiche un rapport : tickers exclus, NaN comblés, NaN restants
      - Retourne le DataFrame nettoyé
    """

    SEUIL_NAN_PCT = 20.0   # exclut les tickers avec plus de 20% de NaN
    print(f"\n[MISSING] Analyse des valeurs manquantes (seuil exclusion : {SEUIL_NAN_PCT}%)")

    # ── Étape 1 : exclusion des tickers trop incomplets ──────────────────
    pct_nan    = prices.isnull().mean() * 100
    to_exclude = pct_nan[pct_nan > SEUIL_NAN_PCT].index.tolist()

    if to_exclude:
        print(f"[MISSING] Tickers exclus (> {SEUIL_NAN_PCT}% NaN) :")
        for t in to_exclude:
            print(f"          {t:<12} : {pct_nan[t]:.1f}% NaN")
        prices = prices.drop(columns=to_exclude)
    else:
        print(f"[MISSING] Aucun ticker exclu")

    # ── Étape 2 : forward fill sur les NaN isolés ────────────────────────
    n_nan_before = prices.isnull().sum().sum()
    prices_clean = prices.ffill(limit=5)
    n_nan_after  = prices_clean.isnull().sum().sum()
    n_combles    = n_nan_before - n_nan_after

    print(f"[MISSING] NaN avant ffill : {n_nan_before}")
    print(f"[MISSING] NaN comblés     : {n_combles}")
    print(f"[MISSING] NaN résiduels   : {n_nan_after}")

    # ── Rapport par ticker ────────────────────────────────────────────────
    print(f"\n[MISSING] Rapport par ticker :")
    for col in prices_clean.columns:
        n = prices_clean[col].isnull().sum()
        p = n / len(prices_clean) * 100
        flag = " ⚠" if p > 2 else ""
        print(f"          {col:<12} : {n:>4} NaN ({p:.1f}%){flag}")

    return prices_clean


# ─────────────────────────────────────────────
# 4. CALCUL DES RENDEMENTS LOGARITHMIQUES
# ─────────────────────────────────────────────

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les rendements logarithmiques journaliers.

    Formule : r_t = log(P_t / P_{t-1}) = log(P_t) - log(P_{t-1})

    En pandas : np.log(prices).diff()
    → .diff() calcule la différence entre la ligne t et t-1
    → Appliqué sur log(prix), ça donne exactement le log-rendement

    La première ligne est NaN par construction (pas de t-1 pour t=0)
    → on la supprime avec .dropna()

    TODO :
      - Calcule les log-rendements avec np.log(prices).diff()
      - Supprime la première ligne NaN
      - Vérifie l'absence de NaN résiduels
      - Affiche min/max rendement pour détecter les outliers
        (un rendement > 50% en un jour est suspect sur des indices)
      - Retourne le DataFrame de rendements
    """
    log_returns = np.log(prices).diff().dropna()

    # ── Contrôle des outliers ─────────────────────────────────────────────
    # Un rendement journalier > 50% sur un indice/ETF est suspect
    # (acceptable sur une petite cap ou une commodity en crise)
    THRESHOLD = 0.50

    for col in log_returns.columns:
        extreme = log_returns[col].abs() > THRESHOLD
        if extreme.any():
            dates_extreme = log_returns.index[extreme].tolist()
            print(f"[WARN] {col} — {extreme.sum()} rendement(s) > {THRESHOLD*100:.0f}% :")
            for d in dates_extreme[:3]:   # affiche max 3
                val = log_returns.loc[d, col]
                print(f"       {d.date()} : {val*100:.1f}%")

    # ── Contrôle NaN résiduels ────────────────────────────────────────────
    n_nan = log_returns.isnull().sum().sum()
    if n_nan > 0:
        raise ValueError(
            f"{n_nan} NaN résiduels dans les rendements — "
            "vérifie handle_missing_values()"
        )

    print(f"\n[RETURNS] Log-rendements calculés — {len(log_returns)} jours")
    print(f"[RETURNS] Résumé (rendements journaliers) :")
    for col in log_returns.columns:
        mu  = log_returns[col].mean() * 100
        sig = log_returns[col].std() * 100
        print(f"          {col:<12} : moy {mu:+.3f}%  vol {sig:.3f}%")

    return log_returns


def compute_excess_returns(
    returns: pd.DataFrame,
    risk_free: pd.Series,
) -> pd.DataFrame:
    """
    Calcule les rendements en excès du taux sans risque.

    Formule : excess_return(t) = r_asset(t) - r_free(t)

    Utilisé pour :
      - Le calcul du Sharpe ratio (moyenne / écart-type des excès de rendement)
      - La régression beta (sensibilité au marché nette du taux sans risque)

    TODO :
      - Aligne risk_free sur l'index de returns (même approche que align_trading_calendar)
      - Soustrait risk_free de chaque colonne de returns
      - Retourne le DataFrame des rendements en excès
    """

    # Aligne risk_free sur l'index des rendements
    # (risk_free peut avoir des dates supplémentaires → reindex)
    rf_aligned = risk_free.reindex(returns.index).ffill()

    # Soustrait le taux sans risque de chaque colonne
    # La soustraction broadcast automatiquement sur toutes les colonnes
    excess = returns.subtract(rf_aligned, axis=0)
    excess.columns = returns.columns   # conserve les noms de colonnes

    print(f"\n[EXCESS] Rendements en excès calculés")
    print(f"         Taux sans risque moy (journalier) : "
          f"{rf_aligned.mean()*100:.4f}% "
          f"({rf_aligned.mean()*TRADING_DAYS*100:.2f}% annualisé)")



# ─────────────────────────────────────────────
# 5. STATISTIQUES DESCRIPTIVES
# ─────────────────────────────────────────────

def compute_descriptive_stats(
        returns: pd.DataFrame,
        risk_free: pd.Series = None,
) -> pd.DataFrame:
    """
    Calcule les statistiques clés pour caractériser chaque actif.

    Métriques calculées (toutes annualisées sauf indication) :
    ┌──────────────────────────────────────────────────────────┐
    │ mean_return      : rendement moyen annualisé (× 252)     │
    │ volatility       : écart-type annualisé (× √252)         │
    │ sharpe_ratio     : mean / vol (simplifié, sans risk-free) │
    │ skewness         : asymétrie de la distribution          │
    │                    > 0 = queue droite (gains extrêmes)   │
    │                    < 0 = queue gauche (pertes extrêmes)  │
    │ kurtosis         : épaisseur des queues                  │
    │                    > 3 = leptokurtique (queues épaisses) │
    │                    Normale = 3 (kurtosis de Fisher = 0)  │
    │ max_drawdown     : perte max depuis un pic (en %)        │
    │ var_95           : VaR historique à 95% (journalière)    │
    └──────────────────────────────────────────────────────────┘

    Contexte marché énergie :
      Les commodités énergie ont typiquement une skewness négative
      (krachs rapides, hausses lentes) et une kurtosis élevée
      (queues épaisses = événements extrêmes plus fréquents que
      ce que prédit une loi normale). C'est exactement ce que
      la CVaR capture mieux que la VaR classique.

    TODO :
      - Calcule chaque métrique pour chaque ticker
      - max_drawdown : perte max depuis le pic cumulé
        Indice : (1 + returns).cumprod() donne la valeur cumulée
                 .cummax() donne le pic glissant
                 drawdown = (cumulative / peak) - 1
      - Retourne un DataFrame avec les tickers en lignes
        et les métriques en colonnes
    """

    rf_daily = risk_free.reindex(returns.index).ffill() if risk_free is not None \
               else pd.Series(0.0, index=returns.index)

    records = []

    for col in returns.columns:
        r = returns[col].dropna()
        rf = rf_daily.reindex(r.index).ffill()

        # ── Rendement et volatilité annualisés ───────────────────────────
        mean_ann = r.mean() * TRADING_DAYS
        vol_ann  = r.std()  * np.sqrt(TRADING_DAYS)

        # ── Sharpe ratio annualisé ────────────────────────────────────────
        # On utilise les excès de rendement sur le risk-free
        excess_r   = r - rf
        sharpe     = (excess_r.mean() / r.std()) * np.sqrt(TRADING_DAYS) \
                     if r.std() > 0 else np.nan

        # ── Skewness & Kurtosis ───────────────────────────────────────────
        # scipy_stats.skew / kurtosis → calcul exact (Fisher par défaut)
        # kurtosis de Fisher : normale = 0 (pandas utilise excess kurtosis)
        # kurtosis de Pearson : normale = 3 (convention finance)
        skewness = scipy_stats.skew(r)
        kurtosis = scipy_stats.kurtosis(r) + 3   # → convention Pearson

        # ── Max Drawdown ──────────────────────────────────────────────────
        # Logique :
        #   1. Valeur cumulée du portefeuille (base 1 au départ)
        #   2. Peak glissant = maximum atteint jusqu'à chaque date
        #   3. Drawdown = (valeur actuelle / peak) - 1  (toujours ≤ 0)
        #   4. Max drawdown = minimum du drawdown (pire perte depuis un pic)
        cumulative = (1 + r).cumprod()
        peak       = cumulative.cummax()
        drawdown   = (cumulative / peak) - 1
        max_dd     = drawdown.min()    # valeur négative → ex: -0.42 = -42%

        # ── VaR historique ────────────────────────────────────────────────
        # VaR 95% = percentile 5% des rendements (pertes dans 5% des cas)
        # VaR 99% = percentile 1% des rendements (pertes dans 1% des cas)
        # Signe conventionnel : VaR exprimée en positif (perte)
        var_95 = -np.percentile(r, 5)    # ex: 0.023 = perte de 2.3%
        var_99 = -np.percentile(r, 1)    # ex: 0.041 = perte de 4.1%

        # ── Calmar Ratio ──────────────────────────────────────────────────
        # Rendement annualisé / valeur absolue du max drawdown
        # Ratio > 1 = le rendement compense largement le drawdown
        calmar = mean_ann / abs(max_dd) if max_dd != 0 else np.nan

        records.append({
            "ticker"          : col,
            "mean_return_ann" : round(mean_ann, 4),
            "volatility_ann"  : round(vol_ann, 4),
            "sharpe_ratio"    : round(sharpe, 3),
            "skewness"        : round(skewness, 3),
            "kurtosis"        : round(kurtosis, 3),
            "max_drawdown"    : round(max_dd, 4),
            "var_95_daily"    : round(var_95, 4),
            "var_99_daily"    : round(var_99, 4),
            "calmar_ratio"    : round(calmar, 3),
            "n_obs"           : len(r),
        })

    stats = pd.DataFrame(records).set_index("ticker")

    # ── Affichage formaté ─────────────────────────────────────────────────
    print(f"\n[STATS] Statistiques descriptives — {len(returns.columns)} actifs\n")
    print(f"{'Ticker':<12} {'Rend.Ann':>9} {'Vol.Ann':>8} {'Sharpe':>7} "
          f"{'Skew':>7} {'Kurt':>7} {'MaxDD':>8} {'VaR95':>7}")
    print("-" * 75)
    for _, row in stats.iterrows():
        print(f"{row.name:<12} "
              f"{row['mean_return_ann']*100:>8.1f}% "
              f"{row['volatility_ann']*100:>7.1f}% "
              f"{row['sharpe_ratio']:>7.2f} "
              f"{row['skewness']:>7.2f} "
              f"{row['kurtosis']:>7.2f} "
              f"{row['max_drawdown']*100:>7.1f}% "
              f"{row['var_95_daily']*100:>6.2f}%")

    return stats



def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la matrice de corrélation des rendements.

    La corrélation est centrale en gestion de portefeuille :
      → Des actifs peu corrélés réduisent le risque (diversification)
      → En période de crise, les corrélations tendent vers 1
        (effet "fly to quality" — tout chute ensemble)

    TODO :
      - Calcule returns.corr() (corrélation de Pearson sur les rendements)
      - Affiche les paires les plus corrélées (> 0.7) comme warning
        → trop de corrélation = faible diversification
      - Retourne la matrice de corrélation
    """
    corr = returns.corr()

    # ── Paires fortement corrélées ────────────────────────────────────────
    SEUIL_CORR = 0.70
    print(f"\n[CORR] Paires avec corrélation > {SEUIL_CORR} (risque de concentration) :")

    found = False
    tickers = corr.columns.tolist()
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            c = corr.iloc[i, j]
            if abs(c) > SEUIL_CORR:
                sign = "+" if c > 0 else "-"
                print(f"  {tickers[i]:<12} ↔ {tickers[j]:<12} : {c:+.2f}")
                found = True

    if not found:
        print(f"  Aucune paire au-dessus du seuil — bonne diversification")

    return corr


# ─────────────────────────────────────────────
# 6. EXPORT
# ─────────────────────────────────────────────

def save_processed(df: pd.DataFrame, filename: str) -> str:
    """
    Sauvegarde un DataFrame dans data/processed/ au format CSV.

    TODO :
      - Crée le dossier avec os.makedirs(exist_ok=True)
      - Sauvegarde avec df.to_csv()
      - Affiche le filepath et le nombre de lignes
      - Retourne le filepath
    """
    output_dir = PATHS["processed"]
    os.makedirs(output_dir, exist_ok=True)
    filepath   = os.path.join(output_dir, filename)
    df.to_csv(filepath)
    print(f"[SAVE] {filepath} — {len(df)} lignes × {len(df.columns)} colonnes")
    return filepath



# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run_preprocessing() -> dict:
    """
    Enchaîne toutes les étapes dans l'ordre.

    Retourne un dictionnaire avec tout ce dont portfolio.py a besoin :
    {
      "prices"          : pd.DataFrame — prix alignés et nettoyés
      "returns"         : pd.DataFrame — log-rendements journaliers
      "excess_returns"  : pd.DataFrame — rendements en excès du risk-free
      "risk_free"       : pd.Series    — taux journalier sans risque
      "eia"             : pd.DataFrame — prix WTI et Brent alignés
      "stats"           : pd.DataFrame — statistiques descriptives
      "correlation"     : pd.DataFrame — matrice de corrélation
    }

    TODO :
      - Charge les trois sources avec load_prices / load_risk_free / load_eia
      - Aligne avec align_trading_calendar
      - Nettoie avec handle_missing_values
      - Calcule les rendements avec compute_log_returns
      - Calcule les excès de rendement avec compute_excess_returns
      - Calcule les stats avec compute_descriptive_stats
      - Calcule la matrice de corrélation avec compute_correlation_matrix
      - Sauvegarde returns et stats dans data/processed/
      - Retourne le dictionnaire complet
    """
    print("=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    # ── Chargement ────────────────────────────────────────────────────────
    prices    = load_prices()
    risk_free = load_risk_free()
    eia       = load_eia()

    # ── Alignement temporel ───────────────────────────────────────────────
    prices, risk_free, eia = align_trading_calendar(prices, risk_free, eia)

    # ── Nettoyage NaN ─────────────────────────────────────────────────────
    prices = handle_missing_values(prices)

    # ── Rendements ────────────────────────────────────────────────────────
    returns        = compute_log_returns(prices)
    excess_returns = compute_excess_returns(returns, risk_free)

    # ── Statistiques ──────────────────────────────────────────────────────
    stats       = compute_descriptive_stats(returns, risk_free)
    correlation = compute_correlation_matrix(returns)

    # ── Sauvegarde ────────────────────────────────────────────────────────
    save_processed(returns,        f"returns_{START_DATE}_{END_DATE}.csv")
    save_processed(excess_returns, f"excess_returns_{START_DATE}_{END_DATE}.csv")
    save_processed(stats,          f"stats_{START_DATE}_{END_DATE}.csv")
    save_processed(correlation,    f"correlation_{START_DATE}_{END_DATE}.csv")
    save_processed(prices,         f"prices_clean_{START_DATE}_{END_DATE}.csv")

    print("\n" + "=" * 60)
    print("PREPROCESSING TERMINÉ")
    print(f"  Actifs retenus  : {len(returns.columns)}")
    print(f"  Jours trading   : {len(returns)}")
    print(f"  Période         : {returns.index[0].date()} → {returns.index[-1].date()}")
    print("=" * 60)

    return {
        "prices"         : prices,
        "returns"        : returns,
        "excess_returns" : excess_returns,
        "risk_free"      : risk_free,
        "eia"            : eia,
        "stats"          : stats,
        "correlation"    : correlation,
    }

# ─────────────────────────────────────────────
# TEST STANDALONE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    data = run_preprocessing()

    print("\nStatistiques descriptives :")
    print(data["stats"].to_string())

    print("\nMatrice de corrélation :")
    print(data["correlation"].round(2).to_string())

    print("\nRendements — 5 premières lignes :")
    print(data["returns"].head())