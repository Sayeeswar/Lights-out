"""
Order financial-statement periods newest-first before handing them to the
model, so a question about "last year" resolves to a predictable column.
"""

from datetime import date, datetime

# Submodules whose cleaned shape is {row_label: {period_str: value}} and
# therefore need their periods ordered before we hand them to the model.
STATEMENT_SUBMODULES = {
    "income_stmt", "quarterly_income_stmt",
    "balance_sheet", "quarterly_balance_sheet",
    "cashflow", "quarterly_cashflow",
}


def order_statement_periods(period_labels, today=None, annual=True, keep=6):
    """
    Given the period labels from a financial statement (ISO date strings
    such as "2025-03-31", occasionally a non-date like "TTM"), return them
    ordered so the most recent relevant period comes first.

    generate_answer() uses this so a question about "last year" resolves to
    a predictable column instead of whichever one yfinance happened to list
    first.

    Policy:
    - Labels that don't parse as a date (e.g. "TTM") are kept, but placed
      after every real dated period so they never win "latest".
    - Periods whose end date is after `today` are dropped: yfinance
      sometimes carries a column before results are actually reported.
    - For annual statements, a fiscal year that ends inside the *current*
      calendar year is treated as "this year's" report, not "last year's",
      so it is excluded from the resolved list - unless doing so would
      leave nothing, in which case the filter is skipped.
    - At most `keep` dated periods are returned, newest first.
    """
    if today is None:
        today = date.today()

    reported = []   # (end_date, label) for periods already reported
    undated = []    # labels we couldn't parse, e.g. "TTM"

    for label in period_labels:
        try:
            end = datetime.strptime(str(label)[:10], "%Y-%m").date()
        except ValueError:
            undated.append(label)
            continue
        if end <= today:
            reported.append((end, label))

    before_this_year = [pair for pair in reported if pair[0].year < today.year]
    chosen = before_this_year if (annual and before_this_year) else reported

    chosen.sort(key=lambda pair: pair[0], reverse=True)
    return [label for _, label in chosen[:keep]] + undated


def reorder_statement_data(yahoo_data, today=None, annual=True):
    """
    Apply order_statement_periods() to every row of a cleaned statement so
    each row's periods share one consistent, newest-first order.
    Non-dict payloads are returned untouched.
    """
    if not isinstance(yahoo_data, dict) or not yahoo_data:
        return yahoo_data

    all_periods = set()
    for row in yahoo_data.values():
        if isinstance(row, dict):
            all_periods.update(row.keys())

    ordered = order_statement_periods(list(all_periods), today, annual=annual)

    result = {}
    for row_label, row in yahoo_data.items():
        if isinstance(row, dict):
            result[row_label] = {p: row[p] for p in ordered if p in row}
        else:
            result[row_label] = row
    return result
