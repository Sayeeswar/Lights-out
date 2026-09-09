"""
Derive the common valuation multiples (P/E, forward P/E, P/S, P/B,
EV/EBITDA, PEG) from a yfinance ``info`` dict.

yfinance already exposes most of these as pre-computed fields; we prefer
those and fall back to computing them from raw components so that one
missing field doesn't blank out the whole metric. Everything is
None-safe: a metric we can't build comes back as ``None`` with a short
note explaining why, rather than raising.

The pipeline attaches the result to ``yahoo_data["valuation_metrics"]`` so
the answer model reads finished numbers instead of doing arithmetic.
"""

from collections.abc import Mapping

# Substrings that mark a question as "about valuation". The pipeline uses
# these to force an ``info`` fetch even when the intent router forgot to
# ask for it.
VALUATION_KEYWORDS = (
    "valuation", "overvalued", "undervalued", "fairly priced", "fair value",
    "p/e", "pe ratio", "price to earnings", "price-to-earnings",
    "p/s", "price to sales", "price-to-sales",
    "p/b", "price to book", "price-to-book",
    "ev/ebitda", "ebitda", "enterprise value", "peg",
)

_METRIC_NAMES = (
    "pe_ratio", "forward_pe_ratio", "ps_ratio",
    "pb_ratio", "ev_to_ebitda", "peg_ratio",
)


def _first_number(info, *keys):
    """Return the first key whose value is a finite real number, else None."""
    for key in keys:
        value = info.get(key)
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            number = float(value)
            if number == number and number not in (float("inf"), float("-inf")):
                return number
    return None


def _ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _metric(value, formula, inputs, note=None):
    return {
        "value": None if value is None else round(value, 4),
        "formula": formula,
        "inputs": inputs,
        "note": note,
    }


def relevant_multiples(info):
    """
    Which of the six multiples are actually meaningful for THIS company,
    given its sector / industry. Returns ``{metric_name: bool}`` covering:
    pe_ratio, forward_pe_ratio, ps_ratio, pb_ratio, ev_to_ebitda, peg_ratio.

    Domain note: for banks / insurers / capital-markets names (yfinance
    ``sector == "Financial Services"``), EV/EBITDA and P/S are not
    meaningful — there is no clean EBITDA and "revenue" mixes net interest
    and fee income — so P/E, forward P/E, P/B and PEG carry the analysis.
    For every other sector, all six apply.
    """
    sector = (info.get("sector") or "").strip()

    # Banks / insurers / capital-markets firms: EV/EBITDA and P/S don't
    # translate (no clean EBITDA, "revenue" is net interest + fee income).
    if sector in ("Financial Services", "Financial", "Financials"):
        applicable = {name: True for name in _METRIC_NAMES}
        applicable["ev_to_ebitda"] = False
        applicable["ps_ratio"] = False
        return applicable

    # Everyone else: all six multiples are fair to show.
    return {name: True for name in _METRIC_NAMES}


def compute_valuation_metrics(info):
    """Build the valuation-multiple block from a yfinance ``info`` dict."""
    if not isinstance(info, Mapping):
        return {}

    price = _first_number(info, "currentPrice", "regularMarketPrice", "previousClose")
    eps_ttm = _first_number(info, "trailingEps")
    eps_fwd = _first_number(info, "forwardEps")
    market_cap = _first_number(info, "marketCap")
    revenue = _first_number(info, "totalRevenue")
    book_per_share = _first_number(info, "bookValue")
    ebitda = _first_number(info, "ebitda")
    total_debt = _first_number(info, "totalDebt")
    total_cash = _first_number(info, "totalCash")

    enterprise_value = _first_number(info, "enterpriseValue")
    if enterprise_value is None and market_cap is not None:
        enterprise_value = market_cap + (total_debt or 0.0) - (total_cash or 0.0)

    pe = _first_number(info, "trailingPE") or _ratio(price, eps_ttm)
    forward_pe = _first_number(info, "forwardPE") or _ratio(price, eps_fwd)
    ps = _first_number(info, "priceToSalesTrailing12Months") or _ratio(market_cap, revenue)
    pb = _first_number(info, "priceToBook") or _ratio(price, book_per_share)
    ev_ebitda = _first_number(info, "enterpriseToEbitda") or _ratio(enterprise_value, ebitda)
    peg = _first_number(info, "trailingPegRatio", "pegRatio")

    metrics = {
        "pe_ratio": _metric(
            pe, "Current Stock Price / Trailing EPS",
            {"current_price": price, "trailing_eps": eps_ttm},
            None if pe is not None else "Needs current price and trailing EPS.",
        ),
        "forward_pe_ratio": _metric(
            forward_pe, "Current Stock Price / Forward EPS",
            {"current_price": price, "forward_eps": eps_fwd},
            None if forward_pe is not None else "Needs current price and a forward EPS estimate.",
        ),
        "ps_ratio": _metric(
            ps, "Market Capitalisation / Total Revenue (TTM)",
            {"market_cap": market_cap, "total_revenue": revenue},
            None if ps is not None else "Needs market cap and total revenue.",
        ),
        "pb_ratio": _metric(
            pb, "Current Stock Price / Book Value Per Share",
            {"current_price": price, "book_value_per_share": book_per_share},
            None if pb is not None else "Needs current price and book value per share.",
        ),
        "ev_to_ebitda": _metric(
            ev_ebitda, "(Market Cap + Total Debt - Cash) / EBITDA",
            {
                "enterprise_value": enterprise_value,
                "market_cap": market_cap,
                "total_debt": total_debt,
                "total_cash": total_cash,
                "ebitda": ebitda,
            },
            None if ev_ebitda is not None else "Needs enterprise value and EBITDA.",
        ),
        "peg_ratio": _metric(
            peg, "P/E Ratio / annual EPS growth rate (%)",
            {"reported_peg": peg},
            None if peg is not None else "yfinance did not supply a PEG ratio.",
        ),
    }

    for name, applicable in relevant_multiples(info).items():
        if name in metrics:
            metrics[name]["applicable_for_sector"] = applicable

    metrics["context"] = {
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "currency": info.get("financialCurrency") or info.get("currency"),
    }
    return metrics
