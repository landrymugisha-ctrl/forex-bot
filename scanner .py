#!/usr/bin/env python3
"""
Forex swing alert bot -- Landry's 9-step strategy, top-down.

How it works
  1. Direction: W1 + D1 or D1 + H4 must agree (a weekly structure pointing the other way vetoes the trade).
  2. Areas of interest (3+ touches, 5-60 pips, wicks ignored) are searched W1 first, then D1, then H4.
     The area must sit between the last HL and HH (buy) or the last LH and LL (sell).
  3. The entry (break + retest + engulfing candle + EMA 50) is looked for on the H4, 2H, 1H and 30M charts.

Two kinds of Telegram messages:
  * WATCH  : 1 to 3 of the 8 checklist points are missing -> tells you what it is waiting for.
  * SIGNAL : the 4 MAIN points are met (trend, area, break+retest, engulfing) -> entry, SL, TP + H4 chart.

Usage:
  python scanner.py --once          scan all pairs one time
  python scanner.py --loop          scan every 15 minutes, forever
  python scanner.py --backtest      test the rules on past data (full signals only)
  python scanner.py --backtest --tf H4,1H   choose which entry charts to test (default H4)
  python scanner.py --get-chat-id   print your Telegram chat id (send your bot a message first)
  Add --demo to use fake data (no internet needed), --verbose to see every score,
  and --pairs EURUSD,GBPUSD to use only some pairs.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# SETTINGS  (everything you may want to tune is here)
# --------------------------------------------------------------------------
CFG = dict(
    pairs=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
           "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "EURGBP", "EURCHF", "EURAUD", "EURNZD",
           "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD", "AUDCAD", "AUDCHF", "AUDNZD",
           "NZDCAD", "NZDCHF"],
    swing_n={"W1": 2, "D1": 3, "H4": 3},  # candles each side that define a swing high/low
    weekly_veto=True,        # top-down: skip trades that go against the weekly structure
    entry_timeframes=["H4", "2H", "1H", "30M"],  # charts where the entry is looked for (no 15M)
    watch_timeframes=["H4"], # WATCH messages only for these charts (lower charts would be noisy)
    watch_max_missing=3,     # send a WATCH when 1 up to this many of the 8 points are missing
    aoi_timeframes=["W1", "D1", "H4"],   # charts where areas of interest are searched.
                                         # Remove "H4" to only trade areas from the daily/weekly charts.
    zone_lookback={"W1": 104, "D1": 120, "H4": 120},  # candles to look back on each chart
    break_search_days=90,    # how far back to look for the break of an area
    ema_period=50,
    ema_lookback=5,          # last N closes (entry chart) must be on the right side of EMA 50
    atr_period=14,
    min_touches=3,           # Step 3: touches needed for an area of interest
    zone_tol_atr=0.4,        # swing points closer than this (x ATR) are grouped into one area
    zone_min_pips=5,         # area of interest size: at least 5 pips ...
    zone_max_pips=60,        # ... and at most 60 pips
    retest_tol_atr=0.3,      # how close (x ATR) the retest must come to the area edge
    main_points=["timeframes", "zone", "retest", "engulf"],   # these can never be missing for a SIGNAL
    count_both_pivots=True,  # count swing highs AND swing lows as touches
    retest_window=6,         # the retest must happen within the last N candles of the entry chart
    sl_buffer_atr=0.15,      # small buffer beyond the structure for the stop loss
    min_rr=2.0,              # Step 8
    pref_rr=3.0,
    scan_every_minutes=15,
)

# Yahoo Finance names for symbols that are not normal forex pairs
YAHOO_SYMBOL = {"XAUUSD": "GC=F"}      # gold: Yahoo gold futures (close to spot, small difference)

STATE_FILE = Path("state.json")
CHART_DIR = Path("charts")

TF_RANK = {"W1": 3, "D1": 2, "H4": 1}
TF_BAR = {"W1": pd.Timedelta(days=7), "D1": pd.Timedelta(days=1), "H4": pd.Timedelta(hours=4),
          "2H": pd.Timedelta(hours=2), "1H": pd.Timedelta(hours=1), "30M": pd.Timedelta(minutes=30)}

CHECK_LABELS = {
    "timeframes": "1) MAIN: Trend - timeframes agree (W1+D1 or D1+H4), weekly not against",
    "structure":  "2) H4 market structure supports the direction",
    "zone":       "3) MAIN: Area of interest (3+ touches, 5-60 pips, between HL-HH / LH-LL)",
    "ema":        "4) EMA 50 on the correct side",
    "retest":     "5) MAIN: Break of the area and retest from the right side",
    "engulf":     "6) MAIN: Engulfing candle closed",
    "sl":         "7) Stop loss beyond structure",
    "rr":         "8) Risk/reward at least 1:2",
}
WAIT_FOR = {
    "timeframes": "Timeframes do not agree yet (or the weekly is against it). No trade until they do.",
    "structure":  "H4 has not formed the right structure (HH/HL for buy, LH/LL for sell) yet.",
    "zone":       "No area with 3+ touches between the last HL and HH (or LH and LL).",
    "ema":        "EMA 50 is not on the right side of price yet.",
    "retest":     "Wait for price to break the area and come back to retest it (from above for a buy, from below for a sell).",
    "engulf":     "Wait for an engulfing candle to CLOSE at the retest.",
    "sl":         "No valid structural stop loss yet.",
    "rr":         "Risk/reward is below 1:2 or there is no structural target. Skip unless it improves.",
}


# --------------------------------------------------------------------------
# INDICATORS AND STRUCTURE
# --------------------------------------------------------------------------
def mirror(df):
    """Flip prices upside-down so SELL logic can reuse the BUY logic."""
    m = pd.DataFrame(index=df.index)
    m["open"], m["close"] = -df["open"], -df["close"]
    m["high"], m["low"] = -df["low"], -df["high"]
    return m


def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def pivots(df, n, bodies=False):
    """Swing highs and lows: (position, price). A swing needs n candles on each side.
    bodies=True ignores the wicks and uses candle bodies (open/close) only."""
    if bodies:
        h = np.maximum(df["open"].to_numpy(), df["close"].to_numpy())
        l = np.minimum(df["open"].to_numpy(), df["close"].to_numpy())
    else:
        h, l = df["high"].to_numpy(), df["low"].to_numpy()
    hi, lo = [], []
    for i in range(n, len(df) - n):
        if h[i] >= h[i - n:i + n + 1].max():
            hi.append((i, h[i]))
        if l[i] <= l[i - n:i + n + 1].min():
            lo.append((i, l[i]))
    return hi, lo


def structure(df, n):
    """'bull' if last swing high and low are both higher, 'bear' if both lower."""
    hi, lo = pivots(df, n)
    if len(hi) < 2 or len(lo) < 2:
        return "none"
    if hi[-1][1] > hi[-2][1] and lo[-1][1] > lo[-2][1]:
        return "bull"
    if hi[-1][1] < hi[-2][1] and lo[-1][1] < lo[-2][1]:
        return "bear"
    return "none"


def build_weekly(d1):
    w = d1.resample("W-FRI").agg({"open": "first", "high": "max",
                                  "low": "min", "close": "last"}).dropna()
    if len(w) and d1.index[-1] < w.index[-1]:   # drop the week still forming
        w = w.iloc[:-1]
    return w


# --------------------------------------------------------------------------
# AREAS OF INTEREST
# --------------------------------------------------------------------------
def find_zones(df, tf, cfg, pip=0.0001):
    """Areas of interest on one chart. Wicks are ignored (candle bodies only).
    3+ touches, size 5-60 pips, lying between the last HL and the last HH (mirrored for sells)."""
    win = df.iloc[-cfg["zone_lookback"][tf]:]
    if len(win) < 30:
        return []
    a = atr(df, cfg["atr_period"]).iloc[-1]
    if a is None or np.isnan(a) or a <= 0:
        return []
    hi, lo = pivots(win, cfg["swing_n"][tf], bodies=True)
    if not hi or not lo:
        return []
    rng_low, rng_high = lo[-1][1], hi[-1][1]      # last HL and last HH (body levels)
    if rng_low >= rng_high:
        return []
    min_sz, max_sz = cfg["zone_min_pips"] * pip, cfg["zone_max_pips"] * pip
    pts = sorted(hi + lo if cfg["count_both_pivots"] else hi, key=lambda p: p[1])
    tolz = min(cfg["zone_tol_atr"] * a, max_sz)
    groups, cur = [], [pts[0]]
    for p in pts[1:]:
        if p[1] - cur[0][1] <= tolz:
            cur.append(p)
        else:
            groups.append(cur)
            cur = [p]
    groups.append(cur)
    zones = []
    eps = 0.5 * pip
    for g in groups:
        if len(g) < cfg["min_touches"]:
            continue
        bot, top = min(p[1] for p in g), max(p[1] for p in g)
        if bot < rng_low - eps or top > rng_high + eps:      # whole area must sit between HL and HH
            continue
        if top - bot < min_sz:                               # too thin: widen to the minimum size
            mid = (top + bot) / 2
            bot, top = mid - min_sz / 2, mid + min_sz / 2
        if top - bot > max_sz + eps:
            continue
        last_time = win.index[max(p[0] for p in g)] + TF_BAR[tf]   # when the last touch closed
        zones.append(dict(bot=bot, top=top, touches=len(g), tf=tf, last_time=last_time))
    return zones


def pip_size(pair):
    return 0.01 if "JPY" in pair else 0.0001


def search_bars(tf, cfg):
    return int(cfg["break_search_days"] * pd.Timedelta(days=1) / TF_BAR[tf])


# --------------------------------------------------------------------------
# THE STRATEGY (steps 3-8, written for a BUY; SELL runs on mirrored prices)
# --------------------------------------------------------------------------
def evaluate_buy(entry, entry_tf, h4, d1, w1, cfg, only_complete=False, pip=0.0001):
    out = dict(zone=False, ema=False, retest=False, engulf=False, sl=False, rr=False)
    det = {}
    a = atr(entry.iloc[-100:], cfg["atr_period"]).iloc[-1]
    if a is None or np.isnan(a) or a <= 0:
        return out, det
    arr = entry.iloc[-search_bars(entry_tf, cfg):]
    o, h, l, c = (arr[k].to_numpy() for k in ("open", "high", "low", "close"))
    L = len(arr) - 1
    ema = entry["close"].ewm(span=cfg["ema_period"], adjust=False).mean().iloc[-len(arr):].to_numpy()

    # Step 4: EMA 50 must be below price (touching it with a wick is fine)
    el = cfg["ema_lookback"]
    out["ema"] = bool(ema[L] < c[L] and (c[L - el + 1:L + 1] > ema[L - el + 1:L + 1]).all())

    # Step 6: bullish engulfing candle that has closed
    tol = 0.05 * a
    out["engulf"] = bool(c[L - 1] < o[L - 1] and c[L] > o[L]
                         and c[L] >= o[L - 1] and o[L] <= c[L - 1] + tol)

    if only_complete and not out["engulf"]:
        return out, det          # backtest shortcut: a signal needs an engulfing candle

    # Step 3: areas of interest. Search order W1 -> D1 -> H4 (a lower chart is used only when the
    # higher one has no usable area, or its areas have no break and retest).
    frames = {"W1": w1, "D1": d1, "H4": h4}
    zones = []
    for tf in cfg["aoi_timeframes"]:
        zones += find_zones(frames[tf], tf, cfg, pip)
    if not zones:
        return out, det
    out["zone"] = True

    # Step 5: break ABOVE the area, then a retest (checked on the entry chart).
    #  - a candle closes above the area (the break)
    #  - price comes back down to the area within the last N candles (the retest);
    #    it may go into or through the area
    #  - the engulfing candle must CLOSE above the top of the area (below it for a sell)
    rw = cfg["retest_window"]
    rtol = cfg["retest_tol_atr"] * a
    zones.sort(key=lambda z: (-TF_RANK[z["tf"]], -z["last_time"].value))
    best = None
    for z in zones:
        k0 = int(arr.index.searchsorted(z["last_time"], side="left"))
        z["ok"] = False
        above = np.nonzero(c[k0:] > z["top"])[0]
        if len(above):
            brk = k0 + int(above[0])
            if brk < L:
                start = max(brk + 1, L - rw + 1)
                closes_above = bool(c[L] > z["top"])
                touched = bool((l[start:L + 1] <= z["top"] + rtol).any())
                z["ok"] = closes_above and touched
                z["start"] = start
        if z["ok"]:
            best = z
            break
    if best is None:
        best = zones[0]
    out["retest"] = best["ok"]

    # Step 7: stop loss beyond the retest low / area bottom
    start = best.get("start", max(0, L - rw + 1))
    price = c[L]
    sl = min(l[start:L + 1].min(), best["bot"]) - cfg["sl_buffer_atr"] * a
    risk = price - sl
    out["sl"] = bool(risk > 0)
    det.update(zone=best, entry=price, sl=sl, atr=a, time=arr.index[L])

    # Step 8: take profit from structure (H4 and D1 swing highs), min 1:2, prefer 1:3
    if risk > 0:
        h4_hi, _ = pivots(h4.iloc[-cfg["zone_lookback"]["H4"]:], cfg["swing_n"]["H4"])
        d_hi, _ = pivots(d1.tail(250), cfg["swing_n"]["D1"])
        cands = [p for _, p in h4_hi if p > price] + [p for _, p in d_hi if p > price]
        if cands:
            target = min(cands)
            avail = (target - price) / risk
            det.update(target_level=target, rr_avail=avail)
            if avail >= cfg["pref_rr"]:
                det.update(tp=price + cfg["pref_rr"] * risk, rr=cfg["pref_rr"])
                out["rr"] = True
            elif avail >= cfg["min_rr"]:
                det.update(tp=price + cfg["min_rr"] * risk, rr=cfg["min_rr"])
                out["rr"] = True
        if "tp" not in det:          # no good structural target: fixed 1:2 and the 8th point stays missing
            det.update(tp=price + cfg["min_rr"] * risk, rr=cfg["min_rr"], tp_fixed=True)
    return out, det


def analyze(entry, entry_tf, h4, d1, w1, cfg=CFG, only_complete=False, pip=0.0001):
    """Returns one result per direction with the 8-point checklist for this entry chart."""
    if len(entry) < 120 or len(h4) < 150 or len(d1) < 40 or len(w1) < 8:
        return []
    sn = cfg["swing_n"]
    st = {"W1": structure(w1, sn["W1"]), "D1": structure(d1, sn["D1"]),
          "H4": structure(h4, sn["H4"])}
    results = []
    for direction, tag in (("BUY", "bull"), ("SELL", "bear")):
        agree = (st["W1"] == tag and st["D1"] == tag) or (st["D1"] == tag and st["H4"] == tag)
        # Top-down: the higher time frame has priority. A weekly structure pointing the
        # OPPOSITE way vetoes the trade (an unclear weekly does not veto).
        opposite = "bear" if tag == "bull" else "bull"
        if cfg.get("weekly_veto", True) and st["W1"] == opposite:
            agree = False
        if only_complete and not agree:
            continue                 # backtest shortcut: this direction cannot give a full signal
        if direction == "BUY":
            checks, det = evaluate_buy(entry, entry_tf, h4, d1, w1, cfg, only_complete, pip)
        else:
            checks, det = evaluate_buy(mirror(entry.iloc[-search_bars(entry_tf, cfg):]), entry_tf,
                                       mirror(h4), mirror(d1), mirror(w1), cfg, only_complete, pip)
            for k in ("entry", "sl", "tp", "target_level"):
                if k in det:
                    det[k] = -det[k]
            if "zone" in det:
                z = dict(det["zone"])
                z["bot"], z["top"] = -z["top"], -z["bot"]
                det["zone"] = z
        full = {"timeframes": agree, "structure": st["H4"] == tag, **checks}
        missing = [k for k, v in full.items() if not v]
        main_ok = all(full[k] for k in cfg["main_points"])
        complete = main_ok and full["sl"] and "tp" in det        # the 4 main points decide the signal
        results.append(dict(dir=direction, tf=entry_tf, checks=full, missing=missing,
                            score=8 - len(missing), complete=complete, det=det, st=st))
    return results


# --------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------
AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}


def _clean(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].astype(float).dropna()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx
    return df


def _closed_only(df, tf, now):
    return df[df.index + TF_BAR[tf] <= now]


def fetch_real(pair):
    """Returns {'D1','H4','2H','1H','30M'} of CLOSED candles (30M only if available)."""
    import yfinance as yf
    t = YAHOO_SYMBOL.get(pair, f"{pair}=X")
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    d1 = _clean(yf.download(t, period="5y", interval="1d", auto_adjust=False, progress=False))
    h1 = _clean(yf.download(t, period="730d", interval="1h", auto_adjust=False, progress=False))
    if d1.empty or h1.empty:
        raise RuntimeError("no price data returned (internet or Yahoo problem)")
    d1.index = d1.index.normalize()
    frames = {"D1": _closed_only(d1, "D1", now),
              "H4": _closed_only(h1.resample("4h").agg(AGG).dropna(), "H4", now),
              "2H": _closed_only(h1.resample("2h").agg(AGG).dropna(), "2H", now),
              "1H": _closed_only(h1, "1H", now)}
    try:
        m30 = _clean(yf.download(t, period="60d", interval="30m", auto_adjust=False, progress=False))
        if not m30.empty:
            frames["30M"] = _closed_only(m30, "30M", now)
    except Exception:
        pass
    time.sleep(0.3)
    return frames


def fetch_demo(pair, days=730):
    seed = sum(map(ord, pair))
    rng = np.random.default_rng(seed)
    n = days * 48
    drift = np.repeat(rng.normal(0, 0.00002, n // 480 + 2), 480)[:n]
    price = 1.1 * np.exp(np.cumsum(drift + rng.normal(0, 0.0005, n)))
    opn = np.r_[price[0], price[:-1]]
    high = np.maximum(opn, price) * (1 + np.abs(rng.normal(0, 0.0002, n)))
    low = np.minimum(opn, price) * (1 - np.abs(rng.normal(0, 0.0002, n)))
    idx = pd.date_range("2024-10-01", periods=n, freq="30min")
    m30 = pd.DataFrame({"open": opn, "high": high, "low": low, "close": price}, index=idx)
    return {"30M": m30, "1H": m30.resample("1h").agg(AGG).dropna(),
            "2H": m30.resample("2h").agg(AGG).dropna(), "H4": m30.resample("4h").agg(AGG).dropna(),
            "D1": m30.resample("1D").agg(AGG).dropna()}


# --------------------------------------------------------------------------
# TELEGRAM + CHART
# --------------------------------------------------------------------------
def fmt(pair, x):
    if pair == "XAUUSD":
        return f"{x:.2f}"
    return f"{x:.3f}" if "JPY" in pair else f"{x:.5f}"


def tg_send(text, photo=None):
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[no Telegram keys set, printing instead]\n" + text + "\n")
        return False
    import requests
    base = f"https://api.telegram.org/bot{token}"
    try:
        if photo:
            with open(photo, "rb") as f:
                r = requests.post(base + "/sendPhoto", data={"chat_id": chat, "caption": text[:1024]},
                                  files={"photo": f}, timeout=30)
        else:
            r = requests.post(base + "/sendMessage", data={"chat_id": chat, "text": text}, timeout=30)
        if not r.ok:
            print("Telegram error:", r.text)
        return r.ok
    except Exception as e:
        print("Telegram failed:", e)
        return False


def make_chart(pair, df_entry, r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = r["det"]
    df = df_entry.iloc[-90:]
    ema = df_entry["close"].ewm(span=CFG["ema_period"], adjust=False).mean().iloc[-90:]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (_, row) in enumerate(df.iterrows()):
        col = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        ax.vlines(i, row["low"], row["high"], color=col, lw=1)
        ax.add_patch(plt.Rectangle((i - 0.3, min(row["open"], row["close"])), 0.6,
                                   abs(row["close"] - row["open"]) or 1e-9, color=col))
    ax.plot(range(len(df)), ema.values, color="#ffb300", lw=1.3, label="EMA 50")
    z = d.get("zone")
    if z:
        ax.axhspan(z["bot"], z["top"], color="#5c6bc0", alpha=0.25,
                   label=f"Area of interest ({z['tf']})")
    if r["complete"]:
        ax.axhline(d["entry"], color="white", ls="--", lw=1, label="Entry")
        ax.axhline(d["sl"], color="#ef5350", ls="--", lw=1.2, label="Stop loss")
        ax.axhline(d["tp"], color="#26a69a", ls="--", lw=1.2, label="Take profit")
    ax.set_xlim(-1, len(df) + 1)
    ax.autoscale(axis="y")
    ax.set_facecolor("#131722")
    fig.patch.set_facecolor("#131722")
    ax.tick_params(colors="#b2b5be")
    ax.set_title(f"{pair} H4 chart - {r['dir']} (entry on {r['tf']})", color="white")
    ax.legend(facecolor="#1e222d", labelcolor="white", loc="upper left", fontsize=8)
    ax.set_xticks([])
    CHART_DIR.mkdir(exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def msg_signal(pair, r):
    d = r["det"]
    z = d["zone"]
    extra = ""
    if r["missing"]:
        extra += "Minor points not met: " + ", ".join(r["missing"]) + "\n"
    if d.get("tp_fixed"):
        extra += "Take profit is a fixed 1:2 (no structural target found).\n"
    return (f"🚨 {pair} - {r['dir']} SIGNAL (entry on the {r['tf']} chart)\n"
            f"All 4 main points met: trend, area of interest, break+retest, engulfing. ({r['score']}/8 in total)\n\n"
            f"Entry: {fmt(pair, d['entry'])}\n"
            f"Stop loss: {fmt(pair, d['sl'])}\n"
            f"Take profit: {fmt(pair, d['tp'])}  (RR 1:{d['rr']:.0f})\n"
            f"Area ({z['tf']} chart): {fmt(pair, z['bot'])} - {fmt(pair, z['top'])} ({z['touches']} touches)\n"
            f"{r['tf']} candle closed: {d['time']} UTC\n{extra}\n"
            f"Check the chart yourself before entering.")


def msg_watch(pair, r):
    lines = "\n".join(f"• {CHECK_LABELS[k]}\n   -> {WAIT_FOR[k]}" for k in r["missing"])
    z = r["det"].get("zone")
    area = (f"Area of interest: {z['tf']} chart, {fmt(pair, z['bot'])} - {fmt(pair, z['top'])} "
            f"({z['touches']} touches)\n\n") if z else ""
    return (f"👀 {pair} - {r['dir']} WATCH on the {r['tf']} chart ({r['score']}/8)\n\n{area}"
            f"Still missing:\n{lines}\n\n"
            f"This is NOT a signal yet. I'll message you when the 4 MAIN points are met.")


# --------------------------------------------------------------------------
# SCAN
# --------------------------------------------------------------------------
def load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def scan_once(demo=False, verbose=False):
    state = load_state()
    for pair in CFG["pairs"]:
        try:
            frames = fetch_demo(pair) if demo else fetch_real(pair)
            w1 = build_weekly(frames["D1"])
        except Exception as e:
            print(f"{pair}: could not scan ({e})")
            continue
        for tf in CFG["entry_timeframes"]:
            if tf not in frames:
                continue
            try:
                results = analyze(frames[tf], tf, frames["H4"], frames["D1"], w1, pip=pip_size(pair))
            except Exception as e:
                print(f"{pair} {tf}: could not analyze ({e})")
                continue
            candle = str(frames[tf].index[-1])
            for r in results:
                key = f"{pair}|{r['dir']}|{tf}"
                prev = state.get(key, {})
                if verbose:
                    print(f"{pair:7s} {tf:3s} {r['dir']:4s} {r['score']}/8  missing: {r['missing']}")
                if r["complete"]:
                    if prev.get("full") != candle:
                        img = make_chart(pair, frames["H4"], r, CHART_DIR / f"{pair}_{tf}_{r['dir']}.png")
                        tg_send(msg_signal(pair, r), photo=img)
                        state[key] = {**prev, "full": candle}
                elif tf in CFG["watch_timeframes"] and 1 <= len(r["missing"]) <= CFG["watch_max_missing"]:
                    sig = ",".join(r["missing"])
                    if prev.get("missing_sig") != sig:
                        img = make_chart(pair, frames["H4"], r, CHART_DIR / f"{pair}_{tf}_{r['dir']}_watch.png")
                        tg_send(msg_watch(pair, r), photo=img)
                        state[key] = {**prev, "missing_sig": sig}
                elif "missing_sig" in prev:
                    state[key] = {k: v for k, v in prev.items() if k != "missing_sig"}
    STATE_FILE.write_text(json.dumps(state, indent=1))


# --------------------------------------------------------------------------
# BACKTEST  (full signals only; no spread or slippage; SL wins if both hit in one candle)
# Only candles that could be an engulfing candle are checked, which makes it much faster.
# --------------------------------------------------------------------------
def backtest(demo=False, tfs=("H4",)):
    trades = []
    for pair in CFG["pairs"]:
        try:
            frames = fetch_demo(pair) if demo else fetch_real(pair)
        except Exception as e:
            print(f"{pair}: no data ({e})")
            continue
        h4, d1 = frames["H4"], frames["D1"]
        h4_close = (h4.index + TF_BAR["H4"]).to_numpy()
        d1_close = (d1.index + TF_BAR["D1"]).to_numpy()
        for tf in tfs:
            if tf not in frames:
                print(f"{pair}: no {tf} data, skipped")
                continue
            E = frames[tf]
            o, h, l, c = (E[k].to_numpy() for k in ("open", "high", "low", "close"))
            e_close = (E.index + TF_BAR[tf]).to_numpy()
            ema_all = E["close"].ewm(span=CFG["ema_period"], adjust=False).mean().to_numpy()
            el = CFG["ema_lookback"]
            above = pd.Series(c > ema_all).rolling(el).min().to_numpy() == 1     # last N closes above EMA
            below = pd.Series(c < ema_all).rolling(el).min().to_numpy() == 1
            pre = np.zeros(len(E), dtype=bool)          # could this candle give a full signal?
            pre[1:] = ((c[:-1] < o[:-1]) & (c[1:] > o[1:]) & (c[1:] >= o[:-1])) | \
                      ((c[:-1] > o[:-1]) & (c[1:] < o[1:]) & (c[1:] <= o[:-1]))
            sb = search_bars(tf, CFG)
            wk_key, wk = None, None
            next_ok = 0
            for i in np.nonzero(pre)[0]:
                if i < max(next_ok, 150):
                    continue
                ih = int(np.searchsorted(h4_close, e_close[i], side="right"))
                idd = int(np.searchsorted(d1_close, e_close[i], side="right"))
                d1s = d1.iloc[max(0, idd - 800):idd]
                if len(d1s) < 40 or ih < 150:
                    continue
                if d1s.index[-1] != wk_key:             # weekly candles only change once a day
                    wk_key, wk = d1s.index[-1], build_weekly(d1s)
                res = analyze(E.iloc[max(0, i - sb + 1):i + 1], tf, h4.iloc[max(0, ih - 400):ih], d1s, wk,
                              only_complete=True, pip=pip_size(pair))
                sig = next((r for r in res if r["complete"]), None)
                if not sig:
                    continue
                d = sig["det"]
                buy = sig["dir"] == "BUY"
                outcome, j = None, i + 1
                while j < len(E):
                    hit_sl = l[j] <= d["sl"] if buy else h[j] >= d["sl"]
                    hit_tp = h[j] >= d["tp"] if buy else l[j] <= d["tp"]
                    if hit_sl:
                        outcome = -1.0
                        break
                    if hit_tp:
                        outcome = d["rr"]
                        break
                    j += 1
                if outcome is None:
                    break
                trades.append(dict(pair=pair, dir=sig["dir"], entry_tf=tf, time=str(E.index[i]),
                                   aoi_tf=d["zone"]["tf"], R=outcome))
                next_ok = j + 1
            print(f"{pair} {tf} done", flush=True)
    if not trades:
        print("No trades found. Your rules are strict; that is expected on a short sample.")
        return
    df = pd.DataFrame(trades)
    df.to_csv("backtest_trades.csv", index=False)
    wins = (df["R"] > 0).mean() * 100
    print(f"\nTrades: {len(df)} | Win rate: {wins:.0f}% | Avg R: {df['R'].mean():.2f} | "
          f"Total R: {df['R'].sum():.1f}")
    rename = {"mean": "avg_R", "sum": "total_R"}
    print("\nBy pair:")
    print(df.groupby("pair")["R"].agg(["count", "sum"]).rename(columns=rename))
    print("\nBy entry chart:")
    print(df.groupby("entry_tf")["R"].agg(["count", "mean", "sum"]).rename(columns=rename))
    print("\nBy chart of the area of interest:")
    print(df.groupby("aoi_tf")["R"].agg(["count", "mean", "sum"]).rename(columns=rename))
    print("\nSaved backtest_trades.csv")


def get_chat_id():
    import requests
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        sys.exit("Set TELEGRAM_BOT_TOKEN first.")
    data = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=30).json()
    chats = {u["message"]["chat"]["id"] for u in data.get("result", []) if "message" in u}
    print("Your chat id(s):", chats or "none yet - send your bot a message, then run again")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--get-chat-id", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--pairs", help="comma separated, e.g. EURUSD,GBPUSD")
    ap.add_argument("--tf", default="H4", help="entry charts for the backtest, e.g. H4,1H")
    a = ap.parse_args()
    if a.pairs:
        CFG["pairs"] = [p.strip().upper() for p in a.pairs.split(",")]
    if a.get_chat_id:
        get_chat_id()
    elif a.backtest:
        backtest(demo=a.demo, tfs=[t.strip().upper() for t in a.tf.split(",")])
    elif a.loop:
        while True:
            scan_once(a.demo, a.verbose)
            time.sleep(CFG["scan_every_minutes"] * 60)
    else:
        scan_once(a.demo, a.verbose)
