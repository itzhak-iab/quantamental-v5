#!/usr/bin/env python3
"""
Q5 Macro Agent — Daily Pipeline
Pulls real market data (Yahoo Finance) + Gemini AI analysis → dashboard_data.json

Runs daily via GitHub Actions at 02:00 UTC (Mon-Fri)
"""

import json
import os
import sys
from datetime import datetime, timezone


def get_market_data(tickers):
    """Fetch real market data from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed")
        sys.exit(1)

    results = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="5d")

            if hist.empty:
                print(f"WARNING: No data for {ticker}")
                continue

            current_price = round(hist['Close'].iloc[-1], 2)
            prev_close = round(hist['Close'].iloc[-2], 2) if len(hist) >= 2 else current_price
            change_pct = round(((current_price - prev_close) / prev_close) * 100, 2)
            avg_volume = int(hist['Volume'].mean())
            last_volume = int(hist['Volume'].iloc[-1])
            volume_ratio = round(last_volume / avg_volume, 2) if avg_volume > 0 else 1.0

            pre_market_price = info.get('preMarketPrice', None)
            pre_market_change = None
            if pre_market_price and prev_close:
                pre_market_change = round(((pre_market_price - prev_close) / prev_close) * 100, 2)

            results[ticker] = {
                "price": current_price,
                "prev_close": prev_close,
                "change_pct": change_pct,
                "volume": last_volume,
                "avg_volume": avg_volume,
                "volume_ratio": volume_ratio,
                "pre_market_price": pre_market_price,
                "pre_market_change_pct": pre_market_change,
                "market_cap": info.get("marketCap", None),
                "pe_ratio": info.get("trailingPE", None),
                "dividend_yield": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else None,
                "52w_high": info.get("fiftyTwoWeekHigh", None),
                "52w_low": info.get("fiftyTwoWeekLow", None),
                "sector": info.get("sector", ""),
                "name": info.get("shortName", ticker),
            }
            print(f"  OK {ticker}: ${current_price} ({change_pct:+.2f}%)")

        except Exception as e:
            print(f"  FAIL {ticker}: {e}")
            results[ticker] = {"error": str(e)}

    return results


def get_vix():
    """Fetch current VIX index."""
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="1d")
        if not hist.empty:
            return round(hist['Close'].iloc[-1], 2)
    except Exception as e:
        print(f"  FAIL VIX: {e}")
    return None

GEMINI_PROMPT = """You are Q5, an institutional trading intelligence system.
You specialize in event-driven arbitrage (mean reversion on ETF dump effects)
and macro-anchored stock analysis.

IRON RULES (MUST appear in every recommendation):
- Stop Loss: 1.5% (hard, non-negotiable)
- Target: 3% profit
- No overnight positions
- Forced close at 15:55 market time (MOC)
- Maximum exposure per trade: user-defined

WATCHLIST (Macro Anchor Stocks):
- CCJ: Uranium / Nuclear Energy (Western mining monopoly, AI energy demand)
- GOGL: Dry Bulk Shipping (aging fleet, zero new orders, ESG bottleneck)
- ICL: Potash / Fertilizers (India contracts, food security)

Based on the following REAL market data from today, generate a JSON analysis.

MARKET DATA:
{market_data}

VIX: {vix}
Date: {date}

RESPOND WITH VALID JSON ONLY (no markdown, no code blocks):
{{
  "arbitrage_opportunities": [
    {{
      "ticker": "SYMBOL",
      "title": "Hebrew title of the opportunity",
      "subtitle": "Hebrew one-line explanation",
      "trigger_type": "etf_dump | earnings_miss | sector_rotation | gap_down",
      "entry_price": 0.00,
      "stop_loss": 0.00,
      "target": 0.00,
      "recommendation": "BUY LIMIT | SELL SHORT | NO ACTION",
      "entry_window": "09:45 - 10:30",
      "quantity_note": "based on max exposure",
      "news_trigger": "Hebrew description of the news/event that triggered this",
      "news_source": "Source name, HH:MM AM",
      "ai_reasoning": [
        "Hebrew bullet point 1",
        "Hebrew bullet point 2",
        "Hebrew bullet point 3"
      ],
      "confidence": "high | medium | low"
    }}
  ],
  "macro_watchlist": [
    {{
      "ticker": "SYMBOL",
      "title": "Hebrew sector title",
      "subtitle": "Hebrew thesis subtitle",
      "pe_ratio": 0.0,
      "dividend_yield": "X.X%",
      "key_metric_label": "Hebrew label",
      "key_metric_value": "value",
      "status": "APPROVED HOLD | UNDER REVIEW | REMOVED",
      "macro_reasoning": [
        "Hebrew bullet point 1",
        "Hebrew bullet point 2"
      ]
    }}
  ],
  "market_summary": {{
    "vix": {vix},
    "exposure_pct": 15,
    "market_regime": "risk_on | risk_off | neutral",
    "daily_note": "Hebrew one-line market note"
  }}
}}

IMPORTANT:
- All Hebrew text must be in Hebrew.
- If no good arbitrage opportunity exists today, return an empty array.
- Be conservative. Only recommend trades with clear edge.
- Always include the 3 macro watchlist stocks with updated analysis.
"""

def run_gemini_analysis(market_data, vix):
    """Send market data to Gemini and get structured JSON analysis."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        return None

    try:
        import google.generativeai as genai
    except ImportError:
        print("ERROR: google-generativeai not installed")
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = GEMINI_PROMPT.format(
        market_data=json.dumps(market_data, indent=2, ensure_ascii=False),
        vix=vix or "N/A",
        date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.3,
            ),
        )
        result = json.loads(response.text)
        print("  OK Gemini analysis received")
        return result
    except json.JSONDecodeError as e:
        print(f"  FAIL Gemini returned invalid JSON: {e}")
        print(f"    Raw: {response.text[:500]}")
        return None
    except Exception as e:
        print(f"  FAIL Gemini error: {e}")
        return None


def main():
    print("=" * 60)
    print(f"Q5 MACRO AGENT — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    watchlist = ["CCJ", "GOGL", "ICL"]
    print("\n[1/3] Fetching market data...")
    market_data = get_market_data(watchlist)

    print("\n[2/3] Fetching VIX...")
    vix = get_vix()
    print(f"  VIX: {vix}")

    print("\n[3/3] Running Gemini analysis...")
    analysis = run_gemini_analysis(market_data, vix)

    if not analysis:
        print("\nPipeline failed — no analysis generated")
        sys.exit(1)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market_data_raw": market_data,
        **analysis,
    }

    output_path = "dashboard_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {output_path}")
    print(f"  Arbitrage opportunities: {len(analysis.get('arbitrage_opportunities', []))}")
    print(f"  Watchlist stocks: {len(analysis.get('macro_watchlist', []))}")
    print("=" * 60)


if __name__ == "__main__":
    main()
