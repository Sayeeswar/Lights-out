"""
personal_loan_tools.py - personal loans only, no loan_type parameter.

Data comes from Sayeeswar_Loan_Excel.xlsx. One function per area (column),
ready to use as agent tools. The original loan_tools.py is unchanged.

Plain functions are defined first so you can test them directly.
PERSONAL_LOAN_TOOLS at the bottom wraps them with function_tool for the Agents SDK.
"""


NOT_SPECIFIED = "Not specified - please confirm with a loan officer or the bank's website."

# ---------------------------------------------------------------
# 1) The data (copied from your Excel file)
# ---------------------------------------------------------------
PERSONAL_LOAN = {
    "eligibility": {
        "employment": "Salaried employees and pensioners of Central/State Government or Armed Forces",
        "minimum_income": NOT_SPECIFIED,
        "age": "Usually around 21-60/65 years, depending on the lender",
        "credit_score": (
            "A CIBIL score of 750+ generally improves your chances and can help "
            "with better interest rates, though lower scores may still qualify"
        ),
    },
    "documents": [
        "PAN/Aadhaar or other KYC",
        "Address proof",
        "Income/salary proof",
        "Bank statements",
    ],
    "interest_rate": {
        "starting_rate": "10.00% p.a.",
        "rate_range": "10.00%-15.00% p.a.",
        "note": (
            "The exact rate depends on the applicable scheme and the customer's "
            "risk/credit profile"
        ),
        "mclr": "2-year MCLR is 8.75%, with a spread of 1.25%-6.25%",
        "how_interest_is_calculated": (
            "Daily reducing balance with monthly rests. Interest is charged on the "
            "outstanding principal, not the original amount, so the interest part "
            "falls as you repay. Example: on Rs 5,00,000 at 10% p.a., the first-year "
            "interest on the full amount would be Rs 50,000, but it reduces as the "
            "balance reduces."
        ),
    },
    "charges": {
        "processing_fee": (
            "Usually up to 1.50% of the loan amount + GST, subject to "
            "minimum/maximum limits depending on the scheme"
        ),
        "prepayment_part_payment": (
            "Often nil for many personal-loan schemes, but depends on the specific product"
        ),
        "foreclosure": "May be nil for certain schemes; product-specific conditions apply",
        "late_payment_penal_interest": "Additional penal charges are applied on overdue EMIs",
        "cheque_ecs_bounce": "Applicable if EMI payment fails",
    },
    "loan_amount": {
        "minimum": "Rs 1 lakh",
        "maximum": "Rs 50 lakh",
        "rule": (
            "The amount you qualify for is not automatically the product maximum. "
            "It is limited by income and repayment capacity: EMI/NMI ratio up to 65% "
            "and a maximum of 30 times Net Monthly Income (NMI). Whichever gives the "
            "lower amount applies."
        ),
        "example": "If NMI is Rs 50,000, then 30x NMI is Rs 15 lakh, not Rs 50 lakh.",
    },
    "repayment_tenure": {
        "minimum": "6 months",
        "maximum": "84 months (7 years)",
        "note": (
            "The loan must generally be closed by the applicable retirement date, "
            "end of contract, or age limit."
        ),
    },
}


def _answer(area: str) -> dict:
    """Wrap one area of the personal-loan data with the bank name."""
    return {area: PERSONAL_LOAN[area]}


# ---------------------------------------------------------------
# 2) One function per area (the columns in your Excel file)
# ---------------------------------------------------------------
def get_eligibility() -> dict:
    """Retrieve SBI personal-loan eligibility details.

    Use for questions about eligible employment categories, age limits, minimum
    income, or CIBIL score guidance. Returns the available eligibility fields;
    it does not assess or guarantee an individual application.
    """
    return _answer("eligibility")


def get_documents() -> dict:
    """Retrieve the documents listed for an SBI personal-loan application.

    Use when asked what identity/KYC, address, income, or bank-statement
    documents may be required. Returns the available document list.
    """
    return _answer("documents")


def get_interest_rate() -> dict:
    """Retrieve SBI personal-loan interest-rate and calculation information.

    Use for questions about the starting rate, rate range, MCLR and spread,
    factors affecting the rate, or how interest is calculated. Returns the
    available rate details; the final rate depends on the applicable scheme
    and the customer's profile.
    """
    return _answer("interest_rate")


def get_charges() -> dict:
    """Retrieve the listed SBI personal-loan charges and fee conditions.

    Use for questions about processing fees, part-payment or prepayment,
    foreclosure, overdue-EMI penal charges, or cheque/ECS bounces. Returns
    available charge details, which may vary by scheme or product.
    """
    return _answer("charges")


def get_loan_amount() -> dict:
    """Retrieve SBI personal-loan amount limits and income-based constraints.

    Use for questions about minimum or maximum loan amounts, repayment-capacity
    limits, EMI/NMI ratio, or the 30-times-NMI rule. Returns the listed limits
    and an example; an individual eligible amount is not guaranteed.
    """
    return _answer("loan_amount")


def get_repayment_tenure() -> dict:
    """Retrieve SBI personal-loan repayment-tenure information.

    Use for questions about the minimum or maximum repayment period and
    applicable closure constraints. Returns the available tenure details.
    """
    return _answer("repayment_tenure")


# ---------------------------------------------------------------
# 3) Wrap as agent tools (only this part needs the openai-agents package)
# ---------------------------------------------------------------
from agents import function_tool  # noqa: E402

PERSONAL_LOAN_TOOLS = [
    function_tool(f)
    for f in (
        get_eligibility,
        get_documents,
        get_interest_rate,
        get_charges,
        get_loan_amount,
        get_repayment_tenure,
    )
]


if __name__ == "__main__":
    # Quick self-test: python personal_loan_tools.py
    print(get_eligibility())
    print(get_loan_amount())
    print(get_repayment_tenure())