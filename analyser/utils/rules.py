"""
Rules engine for determining eligible tasks.
"""

from typing import List, Tuple, Dict, Any, Callable

from .types import Predicates


class TaskRulesEngine:
    """Engine for evaluating rules to determine eligible tasks."""

    def __init__(self):
        self._rules: List[Tuple[str, Callable[[Predicates], bool], List[str]]] = []

    def add_rule(
        self, name: str, predicate_fn: Callable[[Predicates], bool], tasks: List[str]
    ):
        """Add a rule to the engine."""
        self._rules.append((name, predicate_fn, tasks))

    def evaluate(
        self, predicates: Predicates
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        """Evaluate rules against predicates to determine eligible tasks."""
        eligible: List[str] = []
        rationale: Dict[str, List[str]] = {}

        for name, pred_fn, tasks in self._rules:
            try:
                if pred_fn(predicates):
                    for t in tasks:
                        if t not in eligible:
                            eligible.append(t)
                    rationale[name] = tasks
            except Exception:
                continue

        return eligible, rationale


def default_rules() -> TaskRulesEngine:
    """Create default task rules."""
    engine = TaskRulesEngine()

    engine.add_rule(
        "has_table_rules",
        lambda p: p.has_table,
        [
            "convertToPDF",
            "convertToWordDoc",
            "copyAsMarkdown",
            "exportToCSV",
            "exportToXLSX",
        ],
    )

    engine.add_rule(
        "has_text_rules",
        lambda p: (p.has_text and not p.has_table),
        ["summarise", "aiSearchWithInput", "searchOnGoogle", "copyAsText"],
    )

    return engine
