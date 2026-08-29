# config.py

"""
Configuration centralisée du projet Portfolio Management & Hedging sur Commodités Énergie — Marché Européen.

Univers : proxy marché énergie européen avec instruments cotés publiquement
          (en l'absence d'accès aux données EEX/TTF propriétaires)
"""

from datetime import date

# ─────────────────────────────────────────────
# PÉRIODE D'ANALYSE
# ─────────────────────────────────────────────

START_DATE = "2019-01-01"   # couvre pré-Covid, crise 2022, normalisation 2023-2024
END_DATE   = "2024-12-31"   # à ajuster

# Fenêtre d'entraînement / test pour le backtest walk-forward
TRAIN_YEARS = 3   # années d'historique pour calibrer le modèle
TEST_YEARS  = 1   # années sur lesquelles on évalue la stratégie

# ─────────────────────────────────────────────
# UNIVERS D'INVESTISSEMENT
# ─────────────────────────────────────────────

TICKERS = {
    # ── Utilities intégrées européennes ──────────────────────────────────
    # Ces sociétés sont exposées aux prix électricité, gaz et carbone EU
    # → proxy de l'exposition d'un portefeuille de production européen

    "ENGI.PA"  : "Engie — intégré électricité/gaz/ENR (France)",
    "RWE.DE"   : "RWE — mix thermique + ENR croissant (Allemagne)",
    "EOAN.DE"  : "E.ON — distribution + utilities EU",
    "IBE.MC"   : "Iberdrola — utilities + ENR (Espagne/UK)",

    # ── Proxy gaz européen ───────────────────────────────────────────────
    # TTF non disponible gratuitement → Henry Hub corrélé mais imparfait
    # À documenter comme limitation dans le README

    "NG=F"     : "Henry Hub Natural Gas Futures (proxy TTF)",

    # ── Proxy carbone européen ───────────────────────────────────────────
    # EUA (EU Allowances) non disponibles gratuitement en continu
    # → CARB.L est un ETP coté sur le LSE qui réplique le prix EUA

    "CARB.L"   : "WisdomTree Carbon ETP (proxy EUA CO2)",

    # ── Proxy pétrole ────────────────────────────────────────────────────
    # Driver structurel du prix du gaz (indexation contractuelle)

    "BZ=F"     : "Brent Crude Futures (ICE)",

    # ── ENR pur ──────────────────────────────────────────────────────────

    "ICLN"     : "iShares Global Clean Energy ETF",
    "EDPR.LS"  : "EDP Renewables — pure player ENR EU",
}

# ─────────────────────────────────────────────
# INSTRUMENTS DE COUVERTURE & BENCHMARK
# ─────────────────────────────────────────────

HEDGE_INSTRUMENT = "BZ=F"     # instrument utilisé pour le hedging (Brent)
BENCHMARK        = "ENGI.PA"  # benchmark sectoriel de référence
RISK_FREE_SERIES = "DTB3"     # série FRED : US 3-month T-Bill

# ─────────────────────────────────────────────
# PARAMÈTRES DE PORTEFEUILLE
# ─────────────────────────────────────────────

PORTFOLIO = {
    # Allocation initiale (équipondérée — sera optimisée dans portfolio.py)
    # Somme des poids = 1.0
    "initial_weights" : {
        "ENGI.PA"  : 0.15,
        "RWE.DE"   : 0.15,
        "EOAN.DE"  : 0.10,
        "IBE.MC"   : 0.15,
        "NG=F"     : 0.10,
        "CARB.L"   : 0.10,
        "BZ=F"     : 0.10,
        "ICLN"     : 0.10,
        "EDPR.LS"  : 0.05,
    },

    "initial_capital"   : 1_000_000,   # € — capital initial simulé
    "currency"          : "EUR",

    # Contraintes d'optimisation (Markowitz / CVaR)
    "weight_min"        : 0.02,    # poids minimum par actif (évite les positions quasi-nulles)
    "weight_max"        : 0.35,    # poids maximum par actif (diversification forcée)
    "long_only"         : True,    # pas de positions short (portefeuille long-only)
}

# ─────────────────────────────────────────────
# PARAMÈTRES DE RISQUE
# ─────────────────────────────────────────────

RISK = {
    # VaR & CVaR
    "confidence_levels" : [0.95, 0.99],   # niveaux de confiance standard industrie
    "var_window_days"   : 252,             # fenêtre historique pour VaR historique (1 an trading)
    "n_simulations"     : 10_000,          # nombre de scénarios Monte Carlo

    # Annualisation
    "trading_days"      : 252,             # jours de trading par an

    # Stress tests
    "stress_scenarios" : {
        "choc_prix_gaz_+50%"   : {"NG=F"  : +0.50, "BZ=F" : +0.20},
        "choc_prix_gaz_-30%"   : {"NG=F"  : -0.30, "BZ=F" : -0.15},
        "crise_carbone_+40%"   : {"CARB.L": +0.40},
        "crash_ENR_-30%"       : {"ICLN"  : -0.30, "EDPR.LS": -0.30},
        "crise_2022_energy"    : {          # réplique le choc d'août 2022
            "NG=F"   : +2.00,              # +200% (choc TTF historique)
            "BZ=F"   : +0.60,
            "CARB.L" : +0.30,
            "ENGI.PA": -0.35,
            "RWE.DE" : +0.20,
        },
    },
}

# ─────────────────────────────────────────────
# PARAMÈTRES DE HEDGING
# ─────────────────────────────────────────────

HEDGING = {
    "rolling_window_days"    : 60,     # fenêtre pour calcul du rolling beta (60j trading)
    "rebalancing_frequency"  : "ME",   # fréquence de rebalancing — "ME" = Month End (pandas)
    "target_net_exposure"    : 0.0,    # exposition nette cible après hedge (0 = market neutral)
    "hedge_instrument"       : HEDGE_INSTRUMENT,
    "max_hedge_ratio"        : 2.0,    # ratio de couverture maximum (évite le sur-hedge)
}

# ─────────────────────────────────────────────
# PARAMÈTRES DE BACKTEST
# ─────────────────────────────────────────────

BACKTEST = {
    "start_date"             : START_DATE,
    "end_date"               : END_DATE,
    "rebalancing_frequency"  : "ME",     # rebalancing mensuel
    "transaction_costs_bps"  : 5,        # coûts de transaction en basis points (0.05%)
    "slippage_bps"           : 2,        # slippage estimé en basis points
}

# ─────────────────────────────────────────────
# CHEMINS & FICHIERS
# ─────────────────────────────────────────────

PATHS = {
    "raw"       : "data/raw",
    "processed" : "data/processed",
    "outputs"   : "outputs/figures",
    "reports"   : "outputs/reports",
}

# ─────────────────────────────────────────────
# APIs EXTERNES
# ─────────────────────────────────────────────

APIS = {
    # FRED (Federal Reserve) — clé gratuite sur fred.stlouisfed.org
    # À stocker dans .env : FRED_API_KEY=ta_clé_ici
    "fred_base_url" : "https://api.stlouisfed.org/fred/series/observations",

    # EIA — clé gratuite sur eia.gov/opendata
    # À stocker dans .env : EIA_API_KEY=ta_clé_ici
    "eia_base_url"  : "https://api.eia.gov/v2/",
}

EIA_SERIES = {
    "WTI"   : "PET.RWTC.D",     # prix WTI quotidien (USD/baril)
    "BRENT" : "PET.RBRTE.D",    # prix Brent quotidien (USD/baril)
}