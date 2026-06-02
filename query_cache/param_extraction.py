"""
Turn a natural-language question + session context into the named parameters a
matched :class:`~query_cache.sql_patterns.SQLPattern` needs.

Only *values* are produced here; they are always handed to the database driver
as bound parameters, never concatenated into SQL.
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Dict, List, Optional, Tuple

from .sql_patterns import ParamSpec

_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_LEAVE_TYPES = {
    "casual": "%casual%",
    "sick": "%sick%",
    "earned": "%earned%",
    "privilege": "%privilege%",
    "annual": "%annual%",
    "maternity": "%maternity%",
    "paternity": "%paternity%",
    "comp off": "%comp%",
    "compensatory": "%comp%",
    "bereavement": "%bereavement%",
}

_TICKET_RE = re.compile(r"(?:tkt[-\s]?|ticket\s+#?|#)\s*([a-z]*-?\d+)", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def _extract_month(text: str, now: _dt.date) -> int:
    low = text.lower()
    for name, num in _MONTHS.items():
        if re.search(rf"\b{name}\b", low):
            return num
    if "last month" in low:
        return 12 if now.month == 1 else now.month - 1
    return now.month  # "this month" / unspecified -> current


def _extract_year(text: str, now: _dt.date) -> int:
    low = text.lower()
    m = _YEAR_RE.search(low)
    if m:
        return int(m.group(1))
    if "last year" in low:
        return now.year - 1
    return now.year


def _extract_leave_type(text: str) -> Optional[str]:
    low = text.lower()
    for kw, like in _LEAVE_TYPES.items():
        if kw in low:
            return like
    return None


def _extract_ticket_id(text: str) -> Optional[str]:
    m = _TICKET_RE.search(text)
    return m.group(1).upper() if m else None


def extract_params(
    specs: List[ParamSpec],
    query: str,
    context: Optional[Dict[str, object]] = None,
    now: Optional[_dt.date] = None,
) -> Tuple[Dict[str, object], List[str]]:
    """Resolve every spec into a value.

    Returns ``(params, missing)`` where ``missing`` lists the names of *required*
    parameters that could not be resolved - the resolver uses that to decide
    whether the cached pattern is safe to run.
    """
    context = context or {}
    now = now or _dt.date.today()
    params: Dict[str, object] = {}
    missing: List[str] = []

    for spec in specs:
        value: object = None

        if spec.source == "session":
            value = context.get(spec.name)

        elif spec.source == "current":
            if spec.type == "month":
                value = _extract_month(query, now)
            elif spec.type == "year":
                value = _extract_year(query, now)
            else:
                value = context.get(spec.name)

        elif spec.source == "query":
            if spec.name == "leave_type":
                value = _extract_leave_type(query)
            elif spec.name == "ticket_id":
                value = _extract_ticket_id(query)
            elif spec.type == "month":
                value = _extract_month(query, now)
            elif spec.type == "year":
                value = _extract_year(query, now)
            else:
                value = context.get(spec.name)

        if value is None and spec.required:
            missing.append(spec.name)
        if value is not None:
            params[spec.name] = value

    return params, missing
