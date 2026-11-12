import re


def detect_scope(question, documents):
    """
    Work out which company/year a question is about.

    `documents` is [(company, year), ...] from the database.
    Returns (company, year); either may be None when the question
    doesn't name one (None means "don't filter on it").
    """
    q = question.lower()

    companies = sorted({c for c, _ in documents})
    mentioned = [
        c for c in companies
        if re.search(rf"\b{re.escape(c.lower())}\b", q)
    ]

    # Exactly one company named -> filter to it. Zero or several -> no filter.
    company = mentioned[0] if len(mentioned) == 1 else None

    year = None
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", q)]
    available = {
        y for c, y in documents
        if company is None or c == company
    }
    candidates = [y for y in years if y in available]
    if len(set(candidates)) == 1:
        year = candidates[0]

    return company, year