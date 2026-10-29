#!/usr/bin/env python3
"""Calendar date helpers shared by the trait pipeline.

BEAST reports node heights and transition times in the time unit of the
phylogeny (years, for every tree this pipeline accepts).  A height is measured
from the most recent tip, so a transition at height ``h`` happened
``h`` years before the most recent sampling date (MRSD).

Calendar dates are handled as *decimal years* (``2021.4``) so that arithmetic
stays linear, and converted to ISO strings only for reporting.
"""
from __future__ import annotations

import datetime as _dt
import re

__all__ = [
    "parse_date",
    "decimal_year",
    "format_date",
    "height_to_decimal_year",
    "label_date",
    "infer_mrsd",
]

#: the last ISO date written in a taxon label, e.g. ``...|India|B.1.617.2|2021-05-05``
_LABEL_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")

#: a dated tree puts ``date + height`` on the MRSD up to the day its label records
_MRSD_TOLERANCE_DAYS = 3.0



def parse_date(value) -> _dt.date:
    """Parse ``YYYY-MM-DD`` (or a ``date``/``datetime``) into a ``date``."""
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if value is None:
        raise ValueError("no MRSD given")
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%d-%m-%Y"):
        try:
            return _dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"cannot parse date {value!r}, expected YYYY-MM-DD")


def decimal_year(value) -> float:
    """``2021-07-02`` -> ``2021.5`` (day resolution, leap years included)."""
    date = parse_date(value)
    year_start = _dt.date(date.year, 1, 1)
    year_end = _dt.date(date.year + 1, 1, 1)
    return date.year + (date - year_start).days / (year_end - year_start).days


def format_date(value: float) -> str:
    """Decimal year -> ``YYYY-MM-DD``."""
    year = int(value)
    year_start = _dt.date(year, 1, 1)
    days = (_dt.date(year + 1, 1, 1) - year_start).days
    offset = min(max(round((value - year) * days), 0), days - 1)
    return (year_start + _dt.timedelta(days=offset)).isoformat()


def height_to_decimal_year(height: float, mrsd) -> float:
    """Node height (years before the most recent tip) -> decimal calendar year."""
    return decimal_year(mrsd) - float(height)


def label_date(label: str | None):
    """The last ``YYYY-MM-DD`` written in a taxon label, or ``None``."""
    found = _LABEL_DATE.findall(label or "")
    return found[-1] if found else None


def infer_mrsd(trees_path):
    """The calendar present of a tree log, read off the dates in its tip labels.

    The trees are in years before their most recent tip, so every tip satisfies
    ``date + height == mrsd``; the median of those offsets is the MRSD.  Returns
    ``None`` when the labels carry no dates, or when the offsets disagree — the
    trees are then not a clock in years and no calendar axis exists.
    """
    from trait_trees import iter_nodes, node_heights, read_trees

    offsets: list[float] = []
    for _, root in read_trees(trees_path, indices={0}):
        heights = node_heights(root)
        for node in iter_nodes(root):
            if not node.is_tip:
                continue
            date = label_date(node.name)
            if date is None:
                continue
            try:
                offsets.append(decimal_year(date) + heights[id(node)])
            except ValueError:
                continue
    if len(offsets) < 2:
        return None
    offsets.sort()
    median = offsets[len(offsets) // 2]
    tolerance = _MRSD_TOLERANCE_DAYS / 365.25
    agreeing = sum(1 for value in offsets if abs(value - median) <= tolerance)
    if agreeing < 0.95 * len(offsets):
        return None
    return format_date(median)
