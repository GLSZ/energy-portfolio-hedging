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
    pass


def load_risk_free(data_dir: str = None) -> pd.Series:
    """
    Charge la série du taux sans risque produite par fetch_data.py.

    TODO :
      - Même logique que load_prices
      - Pattern : f"risk_free_{START_DATE}_{END_DATE}.csv"
      - Retourne une pd.Series (pas un DataFrame)
      - L'index doit être en datetime
    """
    pass


def load_eia(data_dir: str = None) -> pd.DataFrame:
    """
    Charge les prix EIA (WTI, Brent) produits par fetch_data.py.

    TODO :
      - Pattern : f"eia_{START_DATE}_{END_DATE}.csv"
      - Même logique que load_prices
    """
    pass


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
    pass


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

    pass


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
    pass


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
    pass


# ─────────────────────────────────────────────
# 5. STATISTIQUES DESCRIPTIVES
# ─────────────────────────────────────────────

def compute_descriptive_stats(returns: pd.DataFrame) -> pd.DataFrame:
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

    TRADING_DAYS = 252

    pass


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
    pass


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
    pass


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
    pass


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