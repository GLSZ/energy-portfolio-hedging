# src/fetch_data.py

"""
fetch_data.py — Extraction des données de marché depuis trois sources :

  1. yfinance  → prix OHLCV des tickers (actions, ETFs, futures)
  2. FRED      → taux sans risque (T-Bill 3 mois)
  3. EIA       → prix pétrole et gaz US (séries de référence)

Chaque fonction retourne une pd.DataFrame ou pd.Series propre,
prête à être consommée par preprocess.py.
"""

import os
import sys
import pandas as pd
import yfinance as yf
import requests
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TICKERS, START_DATE, END_DATE,
    RISK_FREE_SERIES, APIS, PATHS
)

# ─────────────────────────────────────────────
# CHARGEMENT DES CLÉS API
# ─────────────────────────────────────────────

env_path = Path(__file__).parent.parent / "logs.env"
load_dotenv(env_path)
print(f"Path to .env : {env_path}")


def _get_api_key(name : str) -> str:
    key = os.getenv(name)
    if not key:
        raise EnvironmentError(
            f"Clé API manquante : {name} \n"
        )
    print(f"[AUTH] {name} chargée - {key[0:5]}{'*' * (len(key) - 5)}")
    return key

# ─────────────────────────────────────────────
# 1. YFINANCE — Prix des tickers
# ─────────────────────────────────────────────

def fetch_prices_yf(
    tickers: dict = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """
    Télécharge les prix de clôture ajustés (adjusted close) pour
    tous les tickers de l'univers via yfinance.

    Pourquoi "adjusted close" ?
    → Les prix sont ajustés des dividendes et splits.
      Sans ajustement, un dividende versé crée une chute artificielle
      du prix le jour ex-dividende, ce qui fausse le calcul des rendements.

    Retourne un DataFrame :
      - index   : dates de trading (pd.DatetimeIndex)
      - colonnes : un ticker par colonne (ex: "ENGI.PA", "RWE.DE", ...)
      - valeurs  : prix de clôture ajusté en devise locale du ticker

    TODO :
      - Télécharge tous les tickers en un seul appel (yf.download accepte une liste)
      - Extrais uniquement la colonne "Close" (ou "Adj Close" selon la version yfinance)
      - Renomme les colonnes avec les noms de tickers proprement
      - Gère le cas où un ticker retourne des données vides (warning + exclusion)
      - Affiche un résumé : nb de tickers, période, nb de NaN par colonne
    """

    ticker_list = list(tickers.keys())
    print(f"\n[yfinance] Téléchargement de {len(ticker_list)} tickers")
    print(f"           Période : {start} → {end}")
    print(f"           Tickers : {ticker_list}")

    raw = yf.download(
        tickers = ticker_list,
        start = start,
        end = end,
        auto_adjust = True,
        group_by="ticker",
        progress = False,
    )

    if raw.empty:
            raise RuntimeError("yfinance n'a retourné aucune donnée — vérifie les tickers et la période")

    # Extraction de la colonne "Close" pour chaque ticker
    prices = pd.DataFrame()

    for ticker in ticker_list:
        try:
            if len(ticker_list) == 1:
                series = raw["Close"] #un seul ticker
            else:
                series = raw[ticker]["Close"]

            if series.dropna().empty:
                print(f"[WARN] {ticker} — aucune donnée retournée → exclu de l'univers")
                continue

            prices[ticker] = series

        except KeyError:
            print(f"[WARN] {ticker} — ticker introuvable dans la réponse yfinance → exclu")

    #supprime lignes avec tous les tickers en NA (week end, fréié)
    prices = prices.dropna(how='all')

    print(f"\n[yfinance] Résultat :")
    print(f"  Tickers récupérés : {len(prices.columns)}/{len(ticker_list)}")
    print(f"  Période effective : {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"  Nombre de jours   : {len(prices)}")
    print(f"\n  NaN par ticker :")
    for col in prices.columns:
        n_nan = prices[col].isnull().sum()
        pct   = n_nan / len(prices) * 100
        flag  = " ⚠" if pct > 5 else ""
        print(f"    {col:<12} : {n_nan:>4} NaN ({pct:.1f}%){flag}")

    return prices

def fetch_single_ticker(
        ticker: str, 
        start : str = START_DATE,
        end: str = END_DATE,
) -> pd.Series:
    """
    Télécharge un ticker individuel et retourne uniquement le prix de clôture.
    Utile pour récupérer séparément le benchmark ou l'instrument de hedge.
    """

    print(f"[yfinance] Téléchargement de {ticker} ({start} → {end})")

    raw = yf.download(
        tickers = ticker,
        start = start, 
        end = end,
        auto_adjust = True,
        progress = False,
    )

    if raw.empty:
        raise RuntimeError(f"Aucune donnée retournée pour le ticker '{ticker}'")

    series = raw["Close"].squeeze() # squeeze() → pd.Series si un seul ticker
    series.name = ticker
    series = series.dropna()

    print(f"[yfinance] {ticker} — {len(series)} jours récupérés "
        f"({series.index[0].date()} → {series.index[-1].date()})")

    return series

# ─────────────────────────────────────────────
# 2. FRED — Taux sans risque
# ─────────────────────────────────────────────

def fetch_risk_free_rate(
    series_id: str = RISK_FREE_SERIES,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.Series:
    """
    Récupère le taux sans risque depuis FRED (Federal Reserve St. Louis).

    Série utilisée : DTB3 = US Treasury Bill 3 mois (annualisé, en %)
    Conversion nécessaire : % annuel → rendement journalier
      r_daily = (1 + r_annual/100) ^ (1/252) - 1

    URL d'appel FRED :
      https://api.stlouisfed.org/fred/series/observations
      ?series_id=DTB3
      &observation_start=2019-01-01
      &observation_end=2024-12-31
      &api_key=...
      &file_type=json

    Retourne une pd.Series :
      - index  : dates
      - valeurs : taux journalier (float, ex: 0.000198 pour ~5% annuel)

    TODO :
      - Construis l'URL avec les bons paramètres (utilise APIS["fred_base_url"])
      - Parse la réponse JSON : les données sont dans response["observations"]
        chaque observation a les clés "date" et "value"
      - Attention : les valeurs manquantes sont encodées "." dans FRED → à convertir en NaN
      - Interpole les NaN (week-ends, jours fériés) par forward fill
      - Convertis le taux annuel % en taux journalier
      - Affiche le taux moyen sur la période pour vérification
    """

    api_key = _get_api_key("FRED_API_KEY")

    params = {
        "series_id" : series_id,
        "observation_start" : start,
        "observation_end" : end, 
        "api_key" : api_key,
        "file_type" : "json",
    }

    print(f"\n[FRED] Récupération de '{series_id}' ({start} → {end})")

    try : 
        response = requests.get(
            APIS["fred_base_url"],
            params = params,
            timeout = 40,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Échec appel FRED : {e}")

    data = response.json()

    if "observations" not in data:
        raise ValueError(
            f"Réponse FRED inattendue — clés disponibles : {list(data.keys())}"
        )

    # Parse les observations
    # FRED encode les valeurs manquantes avec "." → à remplacer par NaN
    records = []
    for obs in data["observations"]:
        val = obs["value"]
        records.append({
            "date"  : pd.to_datetime(obs["date"]),
            "value" : float(val) if val != "." else float("nan"),
        })

    series = pd.DataFrame(records).set_index("date")["value"]
    series.name = f"risk_free_{series_id}"

    # Forward fill sur les NaN (week-ends, jours fériés)
    # Le taux du vendredi est maintenu sur le week-end jusqu'au lundi
    n_nan_before = series.isnull().sum()
    series = series.ffill()
    n_nan_after  = series.isnull().sum()

    if n_nan_after > 0:
        print(f"[WARN] {n_nan_after} NaN restants après ffill — possible début de série vide")

    # Conversion % annuel → taux journalier
    # Ex : 5.0 → 0.019% par jour
    TRADING_DAYS = 252
    series_daily = (1 + series / 100) ** (1 / TRADING_DAYS) - 1

    print(f"[FRED] OK — {len(series_daily)} observations récupérées")
    print(f"       {n_nan_before} NaN comblés par forward fill")
    print(f"       Taux annuel moyen : {series.mean():.2f}%")
    print(f"       Taux journalier moyen : {series_daily.mean():.6f} "
          f"({series_daily.mean() * TRADING_DAYS * 100:.2f}% annualisé)")

    return series_daily

    # Structure JSON retournée par FRED :
    # {
    #   "observations": [
    #     {"date": "2019-01-02", "value": "2.40"},
    #     {"date": "2019-01-03", "value": "."},   ← manquant
    #     ...
    #   ]
    # }


# ─────────────────────────────────────────────
# 3. EIA — Prix pétrole & gaz US
# ─────────────────────────────────────────────

def fetch_eia_series(
    series_id: str,
    start: str = START_DATE,
    end: str = END_DATE,
    label: str = "value",
) -> pd.Series:
    """
    Récupère une série de prix depuis l'API EIA v2.

    L'API EIA v2 fonctionne différemment de v1 :
      URL : https://api.eia.gov/v2/seriesid/{series_id}
            ?api_key=...
            &data[]=value
            &start=2019-01-01
            &end=2024-12-31
            &sort[0][column]=period
            &sort[0][direction]=asc

    Réponse JSON structure :
    {
      "response": {
        "data": [
          {"period": "2019-01-02", "value": 46.54},
          ...
        ]
      }
    }

    Paramètres
    ----------
    series_id : str — identifiant EIA (ex: "PET.RWTC.D" pour WTI)
    label     : str — nom de la colonne dans la Series retournée

    TODO :
      - Construis l'URL correctement (v2 est différente de v1 — attention aux exemples en ligne)
      - Parse la réponse JSON correctement
      - Gère les erreurs HTTP (status != 200) et les réponses vides
      - Retourne une pd.Series indexée par date
    """
    api_key = _get_api_key("EIA_API_KEY")

    # L'URL de l'API EIA v2 intègre le series_id dans le chemin
    # (pas en paramètre comme en v1)
    url = f"{APIS['eia_base_url']}seriesid/{series_id}"

    params = {
        "api_key"              : api_key,
        "data[]"               : "value",
        "start"                : start,
        "end"                  : end,
        "sort[0][column]"      : "period",
        "sort[0][direction]"   : "asc",
        "length"               : 5000,   # max observations par appel
    }

    print(f"\n[EIA] Récupération de '{series_id}' ({start} → {end})")

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Échec appel EIA pour '{series_id}' : {e}")

    payload = response.json()

    # Navigation dans la réponse JSON EIA v2
    try:
        records = payload["response"]["data"]
    except KeyError:
        raise ValueError(
            f"Structure JSON EIA inattendue pour '{series_id}'\n"
            f"Clés disponibles : {list(payload.keys())}"
        )

    if not records:
        raise ValueError(f"Aucune donnée retournée pour la série EIA '{series_id}'")

    # Construction de la Series
    df = pd.DataFrame(records)

    # La colonne de date s'appelle "period" dans l'API EIA v2
    df["period"] = pd.to_datetime(df["period"])
    df = df.set_index("period")

    # La colonne de valeur s'appelle "value"
    # On la renomme avec le label passé en paramètre pour identifier la série
    series = pd.to_numeric(df["value"], errors="coerce")
    series.name = label
    series = series.dropna()
    series = series.sort_index()

    print(f"[EIA] '{series_id}' — {len(series)} observations")
    print(f"      Période : {series.index[0].date()} → {series.index[-1].date()}")
    print(f"      Min : {series.min():.2f}  |  Max : {series.max():.2f}  "
          f"|  Moy : {series.mean():.2f}")

    return series


def fetch_all_eia(
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """
    Récupère toutes les séries EIA définies dans config.py
    et les assemble dans un DataFrame.

    TODO :
      - Boucle sur EIA_SERIES (à ajouter dans config.py si absent)
      - Appelle fetch_eia_series() pour chaque série
      - Assemble en DataFrame avec pd.concat() ou pd.DataFrame()
      - Affiche un résumé des séries récupérées
    """
    # Import local pour ne pas casser si EIA_SERIES absent de config
    try:
        from config import EIA_SERIES
    except ImportError:
        print("[WARN] EIA_SERIES absent de config.py → utilisation des séries par défaut")
        EIA_SERIES = {
            "WTI"   : "PET.RWTC.D",
            "BRENT" : "PET.RBRTE.D",
        }

    print(f"\n[EIA] Récupération de {len(EIA_SERIES)} séries : {list(EIA_SERIES.keys())}")

    series_list = []
    for label, series_id in EIA_SERIES.items():
        try:
            s = fetch_eia_series(series_id, start, end, label=label)
            series_list.append(s)
        except Exception as e:
            print(f"[WARN] Échec EIA '{label}' ({series_id}) : {e}")

    if not series_list:
        raise RuntimeError("Aucune série EIA récupérée avec succès")

    df = pd.concat(series_list, axis=1)

    print(f"\n[EIA] Résumé :")
    print(f"  Séries récupérées : {len(df.columns)}/{len(EIA_SERIES)}")
    print(f"  Période commune   : {df.dropna().index[0].date()} → "
          f"{df.dropna().index[-1].date()}")

    return df
# ─────────────────────────────────────────────
# 4. SAUVEGARDE
# ─────────────────────────────────────────────

def save_raw(df: pd.DataFrame, filename: str) -> str:
    """Sauvegarde un DataFrame dans data/raw/ au format CSV."""
    output_dir = PATHS["raw"]
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath)
    print(f"[SAVE] {filepath} — {len(df)} lignes × {len(df.columns)} colonnes")
    return filepath


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run_fetch(
    start: str = START_DATE,
    end: str = END_DATE,
) -> dict:
    """
    Enchaîne tous les appels dans l'ordre et retourne un dictionnaire
    de DataFrames prêts pour preprocess.py.

    Retourne :
    {
      "prices"    : pd.DataFrame  — prix de clôture ajustés (tous tickers)
      "risk_free" : pd.Series     — taux journalier sans risque
      "eia"       : pd.DataFrame  — prix WTI et Brent EIA
    }

    TODO :
      - Appelle fetch_prices_yf(), fetch_risk_free_rate(), fetch_all_eia()
      - Sauvegarde chaque dataset dans data/raw/ via save_raw()
      - Affiche un résumé global : période, nb de tickers, nb de jours
      - Retourne le dictionnaire
    """
    print("=" * 60)
    print("FETCH DATA")
    print("=" * 60)

    # 1. Prix des tickers (yfinance)
    prices = fetch_prices_yf(TICKERS, start, end)
    save_raw(prices, f"prices_{start}_{end}.csv")

    # 2. Taux sans risque (FRED)
    risk_free = fetch_risk_free_rate(RISK_FREE_SERIES, start, end)
    save_raw(risk_free.to_frame(), f"risk_free_{start}_{end}.csv")

    # 3. Prix EIA (WTI, Brent)
    eia = fetch_all_eia(start, end)
    save_raw(eia, f"eia_{start}_{end}.csv")

    print("\n" + "=" * 60)
    print("FETCH TERMINÉ")
    print(f"  Univers         : {len(prices.columns)} tickers")
    print(f"  Période         : {start} → {end}")
    print(f"  Jours trading   : {len(prices)}")
    print("=" * 60)

    return {
        "prices"    : prices,
        "risk_free" : risk_free,
        "eia"       : eia,
    }
    


# ─────────────────────────────────────────────
# TEST STANDALONE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    data = run_fetch()

    print("\nPrix de clôture — 5 premières lignes :")
    print(data["prices"].head())

    print("\nTaux sans risque — 5 premières valeurs :")
    print(data["risk_free"].head())

    print("\nDonnées EIA — 5 premières lignes :")
    print(data["eia"].head())