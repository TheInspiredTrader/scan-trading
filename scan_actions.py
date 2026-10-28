"""Scan des actions selon le « Plan de trading swing — règles universelles ».

Lancé chaque matin par GitHub Actions. Écrit :
  results/scan_actions.json  (lu par la tâche Claude « Scan du matin »)
  results/scan_actions.md    (résumé lisible)
  results/fondamentaux.json  (cache des données fondamentales, rafraîchi par roulement)

Approximations (pas d'order blocks, FVG, BOS/CHoCH ni notation TradingView via Yahoo) :
  - pivots et zones = sommets et creux de swing journaliers (fractales de 2 bougies) ;
  - structure hebdo = sens de la dernière cassure d'un sommet ou d'un creux de swing hebdomadaire ;
  - notation mensuelle Buy = MACD mensuel au-dessus de son signal ET clôture > SMA10 mensuelle ;
    Sell = les deux inversés ; Neutral sinon.
Éducatif, pas un conseil en investissement.
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

# --------------------------------------------------------------------------- devises
SUFFIX_CCY = {"L": "GBp", "SW": "CHF", "ST": "SEK", "CO": "DKK", "OL": "NOK", "WA": "PLN"}
FX_CCYS = ["USD", "GBP", "CHF", "SEK", "DKK", "NOK", "PLN", "JPY", "CNY", "HKD", "CAD", "AUD"]


def price_currency(symbol: str) -> str:
    if "." not in symbol:
        return "USD"
    return SUFFIX_CCY.get(symbol.rsplit(".", 1)[1], "EUR")


def to_eur(value: float, ccy: str, fx: dict) -> float | None:
    """Convertit un montant en euros. fx[X] = nombre de X pour 1 EUR."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if ccy in ("GBp", "GBX"):
        value, ccy = value / 100.0, "GBP"
    if ccy == "EUR":
        return float(value)
    rate = fx.get(ccy)
    return float(value) / rate if rate else None


# --------------------------------------------------------------------------- indicateurs
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def macd(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    line = ema(s, 12) - ema(s, 26)
    return line, ema(line, 9)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def swings(high: pd.Series, low: pd.Series, k: int = 2) -> tuple[list[int], list[int]]:
    """Indices des sommets et creux de swing (fractales : k bougies de chaque côté)."""
    h, l = high.to_numpy(), low.to_numpy()
    sh, sl = [], []
    for i in range(k, len(h) - k):
        if h[i] > h[i - k:i].max() and h[i] >= h[i + 1:i + 1 + k].max():
            sh.append(i)
        if l[i] < l[i - k:i].min() and l[i] <= l[i + 1:i + 1 + k].min():
            sl.append(i)
    return sh, sl


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample(rule).agg(agg).dropna(subset=["Close"])


def weekly_structure_bullish(w: pd.DataFrame) -> bool | None:
    """Sens de la dernière cassure (en clôture) d'un sommet ou d'un creux de swing hebdo confirmé."""
    if len(w) < 12:
        return None
    sh, sl = swings(w["High"], w["Low"])
    c = w["Close"].to_numpy()
    last = None
    for i in range(len(w)):
        highs = [j for j in sh if j + 2 < i]
        lows = [j for j in sl if j + 2 < i]
        if highs and c[i] > w["High"].iloc[highs[-1]] and (i == 0 or c[i - 1] <= w["High"].iloc[highs[-1]]):
            last = True
        if lows and c[i] < w["Low"].iloc[lows[-1]] and (i == 0 or c[i - 1] >= w["Low"].iloc[lows[-1]]):
            last = False
    if last is None:  # aucune cassure (tendance sans repli) : creux ou sommets des 4 dernières semaines
        lo, hi = w["Low"].iloc[-4:].to_numpy(), w["High"].iloc[-4:].to_numpy()
        if (np.diff(lo) > 0).all():
            last = True
        elif (np.diff(hi) < 0).all():
            last = False
    return last


def monthly_rating(m: pd.DataFrame) -> str:
    if len(m) < 36:
        return "Inconnu"
    line, sig = macd(m["Close"])
    sma10 = m["Close"].rolling(10).mean()
    up = line.iloc[-1] > sig.iloc[-1]
    above = m["Close"].iloc[-1] > sma10.iloc[-1]
    if up and above:
        return "Buy"
    if not up and not above:
        return "Sell"
    return "Neutral"


def major_levels_above(d: pd.DataFrame, w: pd.DataFrame, price: float) -> list[float]:
    """Résistances majeures au-dessus du prix : sommets de swing hebdo (52 sem.) + plus haut 52 semaines."""
    w52 = w.iloc[-53:]
    sh, _ = swings(w52["High"], w52["Low"])
    levels = {round(float(w52["High"].iloc[i]), 6) for i in sh}
    levels.add(round(float(d["High"].iloc[-252:].max()), 6))
    return sorted(x for x in levels if x > price * 1.0001)


# --------------------------------------------------------------------------- setups
def setup_a(d: pd.DataFrame, a: float) -> dict | None:
    """Cassure : clôture d'hier au-dessus du dernier sommet de swing journalier non encore cassé."""
    win = d.iloc[-80:]
    sh, sl = swings(win["High"], win["Low"])
    c = win["Close"].to_numpy()
    n = len(win) - 1  # bougie du signal
    piv = None
    for j in reversed([j for j in sh if j + 2 < n]):
        level = win["High"].iloc[j]
        if (c[j + 1:n] <= level).all():  # jamais clôturé au-dessus avant le signal
            piv = (j, float(level))
            break
    if piv is None:
        return None
    j, level = piv
    lows_before = [i for i in sl if i < n]
    base_low = float(win["Low"].iloc[lows_before[-1]]) if lows_before else float(win["Low"].iloc[-10:].min())
    out = {"setup": "A", "pivot": level, "stop_tech": base_low - CONFIG["marge_stop_atr"] * a,
           "limite_si_gap": level + 0.25 * a, "distance_atr": (level - c[n]) / a}
    out["valide"] = bool(c[n] > level and c[n - 1] <= level)
    return out


def setup_b(d: pd.DataFrame, a: float) -> dict | None:
    """Rebond sur une zone de demande (creux de swing journaliers récents) avec bougie de retournement."""
    win = d.iloc[-60:]
    _, sl = swings(win["High"], win["Low"])
    n = len(win) - 1
    c, lo, hi = win["Close"].to_numpy(), win["Low"].to_numpy(), win["High"].to_numpy()
    best = None
    for j in reversed([j for j in sl if j + 2 < n][-3:]):
        z_low = float(lo[j])
        z_high = z_low + 0.5 * a
        if (c[j + 1:n + 1] < z_low).any():  # zone cassée en clôture : annulée
            continue
        touched = lo[n] <= z_high and c[n] >= z_low
        reversal = c[n] > hi[n - 1]
        cand = {"setup": "B", "zone": [z_low, z_high], "stop_tech": z_low - CONFIG["marge_stop_atr"] * a,
                "distance_atr": max(0.0, (lo[n] - z_high) / a), "valide": bool(touched and reversal)}
        if cand["valide"]:
            return cand
        if best is None or cand["distance_atr"] < best["distance_atr"]:
            best = cand
    return best


# --------------------------------------------------------------------------- données (Yahoo)
def _yf():
    import yfinance as yf  # importé ici pour permettre les tests sans yfinance
    return yf


def download_prices(symbols: list[str]) -> dict[str, pd.DataFrame]:
    yf = _yf()
    out: dict[str, pd.DataFrame] = {}
    chunk = CONFIG["taille_lot_telechargement"]
    for k in range(0, len(symbols), chunk):
        part = symbols[k:k + chunk]
        for attempt in range(4):
            try:
                raw = yf.download(part, period="5y", interval="1d", group_by="ticker", auto_adjust=True,
                                  progress=False, threads=False)
                break
            except Exception as e:  # limite de débit Yahoo, coupure réseau…
                print(f"téléchargement {k}: {e!r}, nouvel essai", file=sys.stderr)
                time.sleep(60 * (attempt + 1))
        else:
            continue
        for s in part:
            try:
                df = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                if len(df):
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    out[s] = df
            except Exception:
                pass
        time.sleep(CONFIG["pause_entre_lots_s"])
    return out


def fetch_fundamentals(symbol: str) -> dict:
    yf = _yf()
    t = yf.Ticker(symbol)
    info = t.info or {}
    f = {k: info.get(k) for k in ("sharesOutstanding", "totalDebt", "totalCash", "ebitda", "freeCashflow",
                                  "trailingPE", "targetMeanPrice", "sector", "financialCurrency", "currency",
                                  "earningsTimestampStart", "earningsTimestamp")}
    try:
        cf = t.cashflow
        if cf is not None and "Free Cash Flow" in cf.index:
            f["fcf_annuels"] = [float(x) for x in cf.loc["Free Cash Flow"].dropna().iloc[:3]]
    except Exception:
        pass
    ts = f.get("earningsTimestampStart") or f.get("earningsTimestamp")
    if ts:
        f["prochains_resultats"] = datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
    else:
        try:
            cal = t.calendar
            dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
            if dates:
                f["prochains_resultats"] = pd.Timestamp(dates[0]).date().isoformat()
        except Exception:
            pass
    return f


# --------------------------------------------------------------------------- analyse
def analyse(sym: str, d: pd.DataFrame, fx: dict, today: pd.Timestamp) -> dict:
    r: dict = {"ticker": sym, "statut": "exclu", "raisons": [], "a_verifier": []}
    d = d[d.index < today]  # uniquement des bougies clôturées
    if len(d) < 260:
        r["raisons"].append("historique < 260 séances")
        return r
    ccy = price_currency(sym)
    w, m = resample(d, "W-FRI"), resample(d, "ME")
    a = float(atr(d).iloc[-1])
    close = float(d["Close"].iloc[-1])
    r.update(devise=ccy, seance=d.index[-1].date().isoformat(), cloture=close, atr=a)

    # Filtres sur les prix
    value_eur = to_eur(float((d["Close"] * d["Volume"]).iloc[-30:].mean()), ccy, fx)
    r["volume_eur_30j"] = value_eur
    if value_eur is None:
        r["a_verifier"].append("liquidité non mesurée (taux de change manquant)")
    if value_eur is not None and value_eur < CONFIG["liquidite_min_eur"]:
        r["raisons"].append("liquidité < 2 M€/jour")
    atr_pct = a / close
    r["atr_pct"] = atr_pct
    if not CONFIG["atr_min_pct"] <= atr_pct <= CONFIG["atr_max_pct"]:
        r["raisons"].append(f"ATR {atr_pct:.1%} hors 1–5 %")
    rating = monthly_rating(m)
    r["notation_mensuelle"] = rating
    if rating == "Sell":
        r["raisons"].append("tendance mensuelle Sell")
    if r["raisons"]:
        return r

    sa, sb = setup_a(d, a), setup_b(d, a)
    wk_bull = weekly_structure_bullish(w.iloc[:-1] if w.index[-1] >= today else w)
    if sb and not wk_bull:  # le setup B exige une structure hebdo haussière
        sb = None
    setup = next((s for s in (sa, sb) if s and s["valide"]), None)
    near = [s for s in (sa, sb) if s and not s["valide"] and 0 <= s["distance_atr"] <= 1]
    r["structure_hebdo_haussiere"] = wk_bull
    if setup is None:
        if near:
            s = min(near, key=lambda x: x["distance_atr"])
            r.update(statut="surveillance", setup_vise=s["setup"], distance_atr=round(s["distance_atr"], 2),
                     niveau=s.get("pivot") or s["zone"][1])
        else:
            r["raisons"].append("aucun setup")
        return r

    # Stop, objectifs, R:R
    entry = close
    stop = setup["stop_tech"]
    dist = entry - stop
    if dist < CONFIG["stop_min_atr"] * a:
        stop, dist = entry - CONFIG["stop_min_atr"] * a, CONFIG["stop_min_atr"] * a
    r.update(setup=setup["setup"], entree=entry, stop=stop, stop_atr=dist / a, details_setup=setup)
    if dist > CONFIG["stop_max_atr"] * a:
        r["raisons"].append(f"stop à {dist / a:.1f} ATR (> 3)")
        return r
    levels = major_levels_above(d, w, entry)
    if levels:
        t1 = levels[0]
        t2 = levels[1] if len(levels) > 1 else max(entry + CONFIG["t2_defaut_r"] * dist, t1 + dist)
    else:
        t1, t2 = entry + CONFIG["rr_min"] * dist, entry + CONFIG["t2_defaut_r"] * dist
        r["a_verifier"].append("aucune résistance au-dessus : T1 = 2R, T2 = 3R")
    rr = (t1 - entry) / dist
    r.update(t1=t1, t2=t2, rr_t1=rr, point_mort=entry + dist)
    if rr < CONFIG["rr_min"] - 1e-9:
        r["raisons"].append(f"R:R vers T1 {rr:.2f} < 2")
        return r

    # Score (sans les fondamentaux : complété plus bas)
    line_w, sig_w = macd(w["Close"])
    line_d, sig_d = macd(d["Close"])
    vol_ratio = float(d["Volume"].iloc[-1] / d["Volume"].iloc[-31:-1].mean()) if d["Volume"].iloc[-31:-1].mean() else 0
    crit = {
        "1 structure hebdo": int(bool(wk_bull)),
        "2 MACD hebdo": int(line_w.iloc[-1] > sig_w.iloc[-1]),
        "3 notation mensuelle Buy": int(rating == "Buy"),
        "4 MACD journalier": int(line_d.iloc[-1] > sig_d.iloc[-1]),
        "5 déclencheur": 2,
        "6 volume ≥ 1,2×": int(vol_ratio >= CONFIG["volume_min_ratio"]),
        "7 espace 1,5 ATR": int(not any(x - entry < CONFIG["espace_atr"] * a for x in levels)),
        "8 valorisation": 0,
        "9 catalyseur": 0,
    }
    r.update(criteres=crit, ratio_volume=vol_ratio, statut="candidat")
    return r


def apply_fundamentals(r: dict, f: dict | None, fx: dict, sector_pe: dict, today: pd.Timestamp) -> None:
    if not f:
        r["a_verifier"] += ["santé financière non mesurée", "date des résultats non mesurée"]
        return
    ccy = r["devise"]
    fin_ccy = f.get("financialCurrency") or ("GBP" if ccy == "GBp" else ccy)
    shares = f.get("sharesOutstanding")
    cap_eur = to_eur(shares * r["cloture"], ccy, fx) if shares else None
    r["capitalisation_eur"] = cap_eur
    if cap_eur is not None and cap_eur < CONFIG["capitalisation_min_eur"]:
        r["raisons"].append("capitalisation < 1 Md€")
    # Agenda des résultats : aucune publication entre la séance du signal et les 5 prochaines séances
    # (≈ 7 jours calendaires). Une publication après la clôture du signal bloque aussi : l'entrée se ferait en gap.
    nxt = f.get("prochains_resultats")
    if nxt:
        r["prochains_resultats"] = nxt
        if pd.Timestamp(r["seance"]) <= pd.Timestamp(nxt) <= today + pd.Timedelta(days=7):
            r["raisons"].append(f"résultats le {nxt}")
    else:
        r["a_verifier"].append("date des résultats non mesurée")
    # Santé financière (hors services financiers)
    if f.get("sector") != "Financial Services":
        fcfs = f.get("fcf_annuels") or []
        if len(fcfs) >= 2:
            if sum(x > 0 for x in fcfs[:3]) < 2:
                r["raisons"].append("FCF négatif sur 2 des 3 derniers exercices")
        else:
            r["a_verifier"].append("historique de FCF non mesuré")
        ebitda, debt, cash = f.get("ebitda"), f.get("totalDebt"), f.get("totalCash") or 0
        if ebitda and ebitda > 0 and debt is not None:
            nd = (debt - cash) / ebitda
            r["dette_nette_ebitda"] = nd
            if nd >= CONFIG["dette_ebitda_max"]:
                r["raisons"].append(f"dette nette/EBITDA {nd:.1f} ≥ 3")
        else:
            r["a_verifier"].append("dette nette/EBITDA non mesurée")
    # Critère 8 : P/E ≤ médiane du secteur OU FCF yield ≥ 5 %
    pe, med = f.get("trailingPE"), sector_pe.get(f.get("sector"))
    fcf_eur = to_eur(f.get("freeCashflow"), fin_ccy, fx) if f.get("freeCashflow") is not None else None
    fcf_yield = fcf_eur / cap_eur if (fcf_eur is not None and cap_eur) else None
    r["fcf_yield"] = fcf_yield
    r["criteres"]["8 valorisation"] = int(bool((pe and med and pe <= med) or (fcf_yield and fcf_yield >= 0.05)))
    # Critère 9 : objectif moyen des analystes ≥ +15 % (même devise que le cours)
    tgt = f.get("targetMeanPrice")
    if tgt:
        r["objectif_analystes"] = tgt
        r["criteres"]["9 catalyseur"] = int(tgt >= r["cloture"] * 1.15)
    else:
        r["a_verifier"].append("objectif des analystes non mesuré")


def size(r: dict, fx: dict) -> None:
    cap = CONFIG["capital_eur"]
    risk_eur = cap * CONFIG["risque_par_trade"]
    dist_eur = to_eur(r["entree"] - r["stop"], r["devise"], fx)
    price_eur = to_eur(r["entree"], r["devise"], fx)
    if not dist_eur or not price_eur:
        r["a_verifier"].append("taille non calculée (taux de change manquant)")
        return
    qty = min(math.floor(risk_eur / dist_eur), math.floor(cap * CONFIG["plafond_position"] / price_eur))
    r.update(quantite=qty, valeur_position_eur=qty * price_eur, risque_effectif_eur=qty * dist_eur)


# --------------------------------------------------------------------------- programme
# --------------------------------------------------------------------------- vérification des symboles
# Remplacements probables pour les symboles qui ne renvoient pas de données sur Yahoo.
# Un candidat n'est retenu que s'il renvoie des cours ; sinon le suivant est essayé.
CANDIDATS = {
    "PHG.AS": ["PHIA.AS"], "ABNA.AS": ["ABN.AS"], "STM.AS": ["STMPA.PA", "STMMI.MI"],
    "NDA.HE": ["NDA-FI.HE", "NDA-SE.ST"], "NOK.HE": ["NOKIA.HE"], "KGP.IR": ["KRX.IR", "KGP.L"],
    "KYGA.IR": ["KRZ.IR", "KYGA.L"], "BDEV.L": ["BTRW.L"], "SIG.SW": ["SIGN.SW"],
    "FI-N.SW": ["GF.SW", "FIN.SW"], "ROG.SW": ["RO.SW", "ROG.SW"], "SLET.L": ["SLA.L", "ABDN.L"],
    "FLU.CO": ["FLS.CO"], "BK": ["BNY"],
}


def has_data(symbol: str) -> bool:
    yf = _yf()
    try:
        raw = yf.download(symbol, period="1mo", interval="1d", auto_adjust=True, progress=False, threads=False)
        if raw is None or raw.empty:
            return False
        cols = raw.columns
        if isinstance(cols, pd.MultiIndex):  # colonnes (champ, ticker) ou (ticker, champ) selon la version
            level = next((i for i in range(cols.nlevels) if "Close" in cols.get_level_values(i)), None)
            if level is None:
                return False
            close = raw.xs("Close", axis=1, level=level)
        else:
            close = raw.get("Close")
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close is not None and close.dropna().shape[0] >= 5
    except Exception:
        return False


def company_name(symbol: str) -> str | None:
    """Nom de la société, pour contrôler qu'un remplacement désigne bien la même entreprise."""
    try:
        info = _yf().Ticker(symbol).info or {}
        return info.get("longName") or info.get("shortName")
    except Exception:
        return None


def search_candidates(symbol: str) -> list[dict]:
    """Recherche Yahoo par la racine du symbole, sur la même place de cotation (même suffixe)."""
    yf = _yf()
    base, suffix = (symbol.rsplit(".", 1) + [""])[:2] if "." in symbol else (symbol, "")
    try:
        quotes = yf.Search(base.split("-")[0], max_results=10, news_count=0).quotes
    except Exception:
        return []
    out = []
    for q in quotes:
        s = q.get("symbol", "")
        same_place = s.endswith("." + suffix) if suffix else "." not in s
        if q.get("quoteType") == "EQUITY" and same_place and s != symbol:
            out.append({"symbole": s, "nom": q.get("longname") or q.get("shortname")})
    return out


def verifier_symboles(tickers: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Teste des remplacements pour les symboles restés sans données au scan précédent.

    Correspondance connue qui renvoie des cours → remplacement dans tickers.csv (colonne Yahoo).
    Résultat de recherche → simple proposition, à valider à la main (risque de confusion de société).
    """
    prev_path, log_path = RESULTS / "scan_actions.json", RESULTS / "symboles.json"
    if not prev_path.exists():
        return tickers
    missing = json.loads(prev_path.read_text(encoding="utf-8")).get("couverture", {}).get("sans_donnees", [])
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    for sym in missing:
        if sym not in set(tickers["Yahoo"]):
            continue
        entry = {"verifie_le": now.isoformat(timespec="minutes"), "statut": "introuvable", "essais": []}
        for cand in CANDIDATS.get(sym, []):
            ok = has_data(cand)
            entry["essais"].append({"symbole": cand, "donnees": ok})
            if ok and cand != sym:
                tickers.loc[tickers["Yahoo"] == sym, "Yahoo"] = cand
                entry.update(statut="remplace", nouveau=cand, nom=company_name(cand))
                break
            if ok and cand == sym:  # le symbole d'origine répond de nouveau : refus passager
                entry["statut"] = "de_nouveau_disponible"
                break
            time.sleep(1)
        if entry["statut"] == "introuvable":
            props = [p for p in search_candidates(sym) if has_data(p["symbole"])][:3]
            if props:
                entry.update(statut="propositions", propositions=props)
        log[sym] = entry
        print(f"symbole {sym}: {entry['statut']} {entry.get('nouveau', '')}")
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    tickers = tickers.drop_duplicates("Yahoo")
    tickers.to_csv(ROOT / "tickers.csv", index=False)
    return tickers


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc)
    today = pd.Timestamp(now.date())
    tickers = verifier_symboles(pd.read_csv(ROOT / "tickers.csv"), now)
    symbols = tickers["Yahoo"].tolist()
    sector_of = {}

    fx_syms = [f"EUR{c}=X" for c in FX_CCYS]
    prices = download_prices(symbols + fx_syms)
    missing = [s for s in symbols + fx_syms if s not in prices]
    if missing:  # seconde tentative, par petits lots, après une pause (refus passagers de Yahoo)
        time.sleep(30)
        CONFIG["taille_lot_telechargement"] = 5
        prices.update(download_prices(missing))
    fx = {c: float(prices[f"EUR{c}=X"]["Close"].iloc[-1]) for c in FX_CCYS if f"EUR{c}=X" in prices}

    results = []
    for s in symbols:
        if s not in prices:
            results.append({"ticker": s, "statut": "exclu", "raisons": ["pas de données Yahoo"], "a_verifier": []})
            continue
        try:
            results.append(analyse(s, prices[s], fx, today))
        except Exception as e:
            results.append({"ticker": s, "statut": "exclu", "raisons": [f"erreur d'analyse : {e!r}"], "a_verifier": []})

    passed_price = sum(1 for r in results if r["statut"] in ("candidat", "surveillance") or r["raisons"] == ["aucun setup"])

    # Fondamentaux : d'abord les candidats, puis rafraîchissement par roulement
    cache_path = RESULTS / "fondamentaux.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    stale = lambda s: s not in cache or (now - datetime.fromisoformat(cache[s]["date"])).days >= CONFIG["fondamentaux_age_max_j"]
    cands = [r["ticker"] for r in results if r["statut"] == "candidat"]
    budget = CONFIG["fondamentaux_par_jour"]
    queue = [s for s in cands if stale(s)] + sorted((s for s in symbols if s not in cands and stale(s)),
                                                    key=lambda s: cache.get(s, {}).get("date", ""))
    fetched = 0
    for s in queue[:budget]:
        try:
            cache[s] = {"date": now.isoformat(), "data": fetch_fundamentals(s)}
            fetched += 1
        except Exception as e:
            print(f"fondamentaux {s}: {e!r}", file=sys.stderr)
            time.sleep(30)
        time.sleep(CONFIG["pause_fondamentaux_s"])
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    by_sector: dict[str, list[float]] = {}
    for s, v in cache.items():
        pe, sec = v["data"].get("trailingPE"), v["data"].get("sector")
        if sec and pe and 0 < pe < 200:
            by_sector.setdefault(sec, []).append(pe)
    sector_pe = {k: float(np.median(v)) for k, v in by_sector.items() if len(v) >= 3}

    for r in results:
        if r["statut"] != "candidat":
            continue
        apply_fundamentals(r, cache.get(r["ticker"], {}).get("data"), fx, sector_pe, today)
        r["score"] = sum(r["criteres"].values())
        size(r, fx)
        if any(x.startswith("résultats le") for x in r["raisons"]):
            r["statut"] = "bloque_resultats"
        elif r["raisons"]:
            r["statut"] = "exclu"
        elif r["score"] >= CONFIG["score_reel"]:
            r["statut"] = "valide"
        elif r["score"] >= CONFIG["score_papier"]:
            r["statut"] = "papier"
        else:
            r["statut"] = "exclu"
            r["raisons"].append(f"score {r['score']}/10")

    pick = lambda st: sorted((r for r in results if r["statut"] == st), key=lambda r: -r.get("score", 0))
    watch = sorted((r for r in results if r["statut"] == "surveillance"), key=lambda r: r["distance_atr"])
    out = {
        "genere_le": now.isoformat(timespec="minutes"),
        "regles": "Plan de trading swing — règles universelles (actions, long uniquement)",
        "capital_eur": CONFIG["capital_eur"], "risque_par_trade": CONFIG["risque_par_trade"],
        "valides": pick("valide"), "papier": pick("papier"), "bloques_resultats": pick("bloque_resultats"),
        "surveillance": watch[:CONFIG["surveillance_max"]],
        "couverture": {
            "tickers": len(symbols), "avec_donnees": sum(1 for s in symbols if s in prices),
            "passent_filtres_prix": passed_price,
            "candidats_setup": len(cands), "fondamentaux_rafraichis": fetched,
            "fondamentaux_en_cache": len(cache), "taux_change": fx,
            "sans_donnees": [s for s in symbols if s not in prices],
        },
        "approximations": "Pivots, zones et structure calculés avec les swings journaliers et hebdomadaires ; "
                          "notation mensuelle = MACD mensuel + SMA10 ; pas d'order blocks, FVG, BOS/CHoCH ni notation TradingView.",
    }
    (RESULTS / "scan_actions.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=_json),
                                              encoding="utf-8")
    (RESULTS / "scan_actions.md").write_text(render_md(out), encoding="utf-8")
    print(f"{len(out['valides'])} validées, {len(out['papier'])} papier, {len(watch)} en surveillance")


def _json(o):
    return o.item() if hasattr(o, "item") else str(o)


def fmt(x: float | None, nd: int = 2) -> str:
    return "—" if x is None else f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def render_md(o: dict) -> str:
    L = [f"# Scan des actions — {o['genere_le'][:10]}", "",
         f"{len(o['valides'])} entrée(s) validée(s), {len(o['papier'])} trade(s) papier, "
         f"{len(o['bloques_resultats'])} bloquée(s) par les résultats.", "", "## Entrées validées", ""]
    rows = o["valides"] or []
    if rows:
        L += ["| Action | Setup | Score | Entrée | Stop | T1 | T2 | R:R T1 | Quantité | À vérifier |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in rows:
            L.append(f"| {r['ticker']} | {r['setup']} | {r['score']}/10 | {fmt(r['entree'])} {r['devise']} | {fmt(r['stop'])} | "
                     f"{fmt(r['t1'])} | {fmt(r['t2'])} | {fmt(r['rr_t1'])} | {r.get('quantite', '—')} | {'; '.join(r['a_verifier']) or '—'} |")
    else:
        L.append("Aucune entrée validée aujourd'hui.")
    for title, key in (("Trades papier", "papier"), ("Bloquées par les résultats", "bloques_resultats")):
        L += ["", f"## {title}", ""]
        L += [f"- {r['ticker']} : setup {r.get('setup')}, score {r.get('score')}/10, entrée {fmt(r.get('entree'))}, "
              f"stop {fmt(r.get('stop'))}" + (f", résultats le {r.get('prochains_resultats')}" if key == "bloques_resultats" else "")
              for r in o[key]] or ["Aucun."]
    L += ["", "## À surveiller", ""]
    L += [f"- {r['ticker']} : setup {r['setup_vise']}, niveau {fmt(r['niveau'])} {r['devise']}, à {fmt(r['distance_atr'])} ATR"
          for r in o["surveillance"]] or ["Aucune."]
    c = o["couverture"]
    L += ["", f"Couverture : {c['avec_donnees']}/{c['tickers']} actions avec données, {c['candidats_setup']} setups, "
          f"{c['fondamentaux_en_cache']} fiches fondamentales en cache.", "", f"*{o['approximations']}*", "",
          "*Éducatif, pas un conseil en investissement.*", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
