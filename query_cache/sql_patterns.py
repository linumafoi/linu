"""
The HR SQL pattern library.

Each :class:`SQLPattern` couples a parameterized, **read-only** ``SELECT`` with a
set of natural-language example phrasings. Those phrasings are what get embedded
into pgvector; an incoming question that is semantically close to any of them
resolves to the pattern's SQL without an LLM call.

SQL safety
----------
* Every template is a single ``SELECT`` (enforced again at execution time).
* All user-derived values are passed as ``psycopg2`` named parameters
  (``%(name)s``) - they are *never* string-formatted into the SQL, so the cache
  cannot introduce SQL injection.

Assumed schema (edit the SQL here to match your real ``KB`` database)
---------------------------------------------------------------------
* ``employees(employee_id, name, email, designation, department, date_of_joining)``
* ``salaries(employee_id, month, year, basic, hra, allowances, gross,
             pf, esi, tds, other_deductions, lop_days, net_pay)``  (1 row / month)
* ``leave_balances(employee_id, leave_type, allotted, used, balance, year)``
* ``attendance(employee_id, month, year, present_days, absent_days, lop_days)``
* ``tickets(ticket_id, employee_id, user_email, user_query, category, status,
            priority, resolution, created_at)``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

ParamType = Literal["int", "str", "month", "year"]
ParamSource = Literal["session", "query", "current"]


@dataclass(frozen=True)
class ParamSpec:
    """A single named parameter consumed by a pattern's SQL template."""

    name: str
    type: ParamType
    #: ``session``  -> taken from the authenticated employee context
    #: ``query``    -> extracted from the natural-language text
    #: ``current``  -> defaults to "now" (current month / year) when absent
    source: ParamSource
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class SQLPattern:
    """A reusable, parameterized SQL query keyed by natural-language intent."""

    intent: str
    category: str
    description: str
    examples: List[str]
    sql: str
    params: List[ParamSpec] = field(default_factory=list)
    #: Hint for the frontend: render as a table vs. a single prose answer.
    multi_row: bool = False

    @property
    def param_names(self) -> List[str]:
        return [p.name for p in self.params]


# Shared parameter specs ------------------------------------------------------
_EMP = ParamSpec(
    "employee_id", "str", "session", required=True,
    description="Authenticated employee id from the chat session.",
)
_MONTH = ParamSpec(
    "month", "month", "current", required=False,
    description="Calendar month 1-12; defaults to the current month.",
)
_YEAR = ParamSpec(
    "year", "year", "current", required=False,
    description="Four digit year; defaults to the current year.",
)


PATTERNS: List[SQLPattern] = [
    # ── Payroll ──────────────────────────────────────────────────────────────
    SQLPattern(
        intent="net_salary_for_month",
        category="payroll",
        description="Net (take-home) salary for a given month/year.",
        examples=[
            "what is my net salary",
            "how much is my take home this month",
            "my net pay",
            "in-hand salary",
            "what will i get credited this month",
            "show my take-home pay",
        ],
        sql=(
            "SELECT month, year, net_pay "
            "FROM salaries "
            "WHERE employee_id = %(employee_id)s AND month = %(month)s AND year = %(year)s"
        ),
        params=[_EMP, _MONTH, _YEAR],
    ),
    SQLPattern(
        intent="gross_salary_for_month",
        category="payroll",
        description="Gross salary (before deductions) for a month.",
        examples=[
            "what is my gross salary",
            "my gross pay this month",
            "salary before deductions",
            "ctc breakup gross",
        ],
        sql=(
            "SELECT month, year, gross "
            "FROM salaries "
            "WHERE employee_id = %(employee_id)s AND month = %(month)s AND year = %(year)s"
        ),
        params=[_EMP, _MONTH, _YEAR],
    ),
    SQLPattern(
        intent="salary_breakup_for_month",
        category="payroll",
        description="Full salary component break-up for a month.",
        examples=[
            "show my salary breakup",
            "components of my salary",
            "break down my pay",
            "what are my salary components",
            "detailed payslip",
            "earnings and deductions split",
        ],
        sql=(
            "SELECT basic, hra, allowances, gross, pf, esi, tds, "
            "other_deductions, net_pay "
            "FROM salaries "
            "WHERE employee_id = %(employee_id)s AND month = %(month)s AND year = %(year)s"
        ),
        params=[_EMP, _MONTH, _YEAR],
    ),
    SQLPattern(
        intent="ytd_earnings",
        category="payroll",
        description="Year-to-date total gross and net earnings.",
        examples=[
            "total earnings this year",
            "year to date salary",
            "how much have i earned this year",
            "ytd income",
            "my cumulative pay this year",
        ],
        sql=(
            "SELECT year, SUM(gross) AS total_gross, SUM(net_pay) AS total_net "
            "FROM salaries "
            "WHERE employee_id = %(employee_id)s AND year = %(year)s "
            "GROUP BY year"
        ),
        params=[_EMP, _YEAR],
    ),

    # ── Statutory deductions ──────────────────────────────────────────────────
    SQLPattern(
        intent="pf_contribution_for_month",
        category="statutory",
        description="Provident Fund contribution for a month.",
        examples=[
            "my pf contribution",
            "how much provident fund was deducted",
            "pf deducted this month",
            "epf amount",
            "provident fund this month",
        ],
        sql=(
            "SELECT month, year, pf "
            "FROM salaries "
            "WHERE employee_id = %(employee_id)s AND month = %(month)s AND year = %(year)s"
        ),
        params=[_EMP, _MONTH, _YEAR],
    ),
    SQLPattern(
        intent="esi_deduction_for_month",
        category="statutory",
        description="ESI deduction for a month.",
        examples=[
            "my esi deduction",
            "how much esi was deducted",
            "esi this month",
            "employee state insurance amount",
        ],
        sql=(
            "SELECT month, year, esi "
            "FROM salaries "
            "WHERE employee_id = %(employee_id)s AND month = %(month)s AND year = %(year)s"
        ),
        params=[_EMP, _MONTH, _YEAR],
    ),
    SQLPattern(
        intent="tds_for_year",
        category="statutory",
        description="Total tax (TDS) deducted in a year.",
        examples=[
            "how much tax was deducted",
            "my tds this year",
            "total income tax deducted",
            "tax deducted at source",
            "ytd tds",
        ],
        sql=(
            "SELECT year, SUM(tds) AS total_tds "
            "FROM salaries "
            "WHERE employee_id = %(employee_id)s AND year = %(year)s "
            "GROUP BY year"
        ),
        params=[_EMP, _YEAR],
    ),
    SQLPattern(
        intent="total_deductions_for_month",
        category="statutory",
        description="All deductions (pf + esi + tds + other) for a month.",
        examples=[
            "my total deductions",
            "how much was deducted this month",
            "sum of all deductions",
            "total cuts from my salary",
        ],
        sql=(
            "SELECT month, year, "
            "(pf + esi + tds + other_deductions) AS total_deductions "
            "FROM salaries "
            "WHERE employee_id = %(employee_id)s AND month = %(month)s AND year = %(year)s"
        ),
        params=[_EMP, _MONTH, _YEAR],
    ),

    # ── Leave ──────────────────────────────────────────────────────────────────
    SQLPattern(
        intent="leave_balance_all",
        category="leave",
        description="Remaining balance across every leave type.",
        examples=[
            "what is my leave balance",
            "how many leaves do i have left",
            "show my leave balances",
            "remaining leaves",
            "leaves available",
            "how much leave is left",
        ],
        sql=(
            "SELECT leave_type, allotted, used, balance "
            "FROM leave_balances "
            "WHERE employee_id = %(employee_id)s AND year = %(year)s "
            "ORDER BY leave_type"
        ),
        params=[_EMP, _YEAR],
        multi_row=True,
    ),
    SQLPattern(
        intent="leave_balance_by_type",
        category="leave",
        description="Remaining balance for one specific leave type.",
        examples=[
            "casual leave balance",
            "how many sick leaves do i have",
            "earned leave left",
            "privilege leave balance",
            "remaining casual leave",
        ],
        sql=(
            "SELECT leave_type, allotted, used, balance "
            "FROM leave_balances "
            "WHERE employee_id = %(employee_id)s AND year = %(year)s "
            "AND leave_type ILIKE %(leave_type)s"
        ),
        params=[
            _EMP, _YEAR,
            ParamSpec("leave_type", "str", "query", required=True,
                      description="Leave type keyword, e.g. casual / sick / earned."),
        ],
    ),
    SQLPattern(
        intent="leaves_used_this_year",
        category="leave",
        description="Total leave days consumed this year.",
        examples=[
            "how many leaves have i taken this year",
            "leaves used so far",
            "total leave days consumed",
            "how much leave have i used",
        ],
        sql=(
            "SELECT year, SUM(used) AS total_used "
            "FROM leave_balances "
            "WHERE employee_id = %(employee_id)s AND year = %(year)s "
            "GROUP BY year"
        ),
        params=[_EMP, _YEAR],
    ),

    # ── Attendance / LOP ───────────────────────────────────────────────────────
    SQLPattern(
        intent="lop_days_for_month",
        category="attendance",
        description="Loss-of-pay days for a month.",
        examples=[
            "my loss of pay days",
            "how many lop days this month",
            "loss of pay this month",
            "lop count",
            "unpaid leave days",
        ],
        sql=(
            "SELECT month, year, lop_days "
            "FROM attendance "
            "WHERE employee_id = %(employee_id)s AND month = %(month)s AND year = %(year)s"
        ),
        params=[_EMP, _MONTH, _YEAR],
    ),
    SQLPattern(
        intent="attendance_summary_for_month",
        category="attendance",
        description="Present / absent / LOP summary for a month.",
        examples=[
            "my attendance this month",
            "show my attendance summary",
            "how many days was i present",
            "attendance record",
            "present and absent days",
        ],
        sql=(
            "SELECT month, year, present_days, absent_days, lop_days "
            "FROM attendance "
            "WHERE employee_id = %(employee_id)s AND month = %(month)s AND year = %(year)s"
        ),
        params=[_EMP, _MONTH, _YEAR],
    ),

    # ── Employee profile ───────────────────────────────────────────────────────
    SQLPattern(
        intent="my_profile",
        category="profile",
        description="Basic profile: name, designation, department, email.",
        examples=[
            "show my details",
            "what is my designation",
            "which department am i in",
            "my profile",
            "my employee details",
            "what is my role",
        ],
        sql=(
            "SELECT employee_id, name, email, designation, department "
            "FROM employees "
            "WHERE employee_id = %(employee_id)s"
        ),
        params=[_EMP],
    ),
    SQLPattern(
        intent="date_of_joining",
        category="profile",
        description="The employee's date of joining and tenure.",
        examples=[
            "when did i join",
            "my joining date",
            "date of joining",
            "how long have i been here",
            "my tenure",
        ],
        sql=(
            "SELECT date_of_joining, "
            "AGE(CURRENT_DATE, date_of_joining) AS tenure "
            "FROM employees "
            "WHERE employee_id = %(employee_id)s"
        ),
        params=[_EMP],
    ),

    # ── Tickets ────────────────────────────────────────────────────────────────
    SQLPattern(
        intent="my_open_tickets",
        category="tickets",
        description="Open / in-progress support tickets for the employee.",
        examples=[
            "my open tickets",
            "do i have any pending tickets",
            "show my unresolved tickets",
            "tickets still open",
            "my pending support requests",
        ],
        sql=(
            "SELECT ticket_id, category, status, priority, created_at "
            "FROM tickets "
            "WHERE employee_id = %(employee_id)s "
            "AND status IN ('OPEN', 'IN_PROGRESS') "
            "ORDER BY created_at DESC"
        ),
        params=[_EMP],
        multi_row=True,
    ),
    SQLPattern(
        intent="all_my_tickets",
        category="tickets",
        description="Every ticket raised by the employee.",
        examples=[
            "all my tickets",
            "show my ticket history",
            "list every ticket i raised",
            "my complete ticket list",
        ],
        sql=(
            "SELECT ticket_id, category, status, resolution, created_at "
            "FROM tickets "
            "WHERE employee_id = %(employee_id)s "
            "ORDER BY created_at DESC"
        ),
        params=[_EMP],
        multi_row=True,
    ),
    SQLPattern(
        intent="ticket_status_by_id",
        category="tickets",
        description="Status and resolution of one ticket by id.",
        examples=[
            "status of ticket",
            "what is the status of my ticket",
            "track ticket",
            "is my ticket resolved",
            "ticket update",
        ],
        sql=(
            "SELECT ticket_id, status, priority, resolution, created_at "
            "FROM tickets "
            "WHERE employee_id = %(employee_id)s AND ticket_id = %(ticket_id)s"
        ),
        params=[
            _EMP,
            ParamSpec("ticket_id", "str", "query", required=True,
                      description="Ticket reference, e.g. TKT-1042."),
        ],
    ),
]


def pattern_by_intent(intent: str) -> Optional[SQLPattern]:
    """Look up a pattern by its unique intent key."""
    for p in PATTERNS:
        if p.intent == intent:
            return p
    return None
