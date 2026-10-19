"""Decides how a confirmed event is routed, using severity_rules.yaml."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    severity: str   # critical | moderate | info
    routing: str    # incident | digest | inventory
    note: str = ""


def classify(cfg, *args, **kwargs) -> Decision:
    """Classify an event according to module rules.

    Supports both signatures:
      classify(cfg, module, cls, best_conf, area_frac=None)
      classify(cfg, cls, best_conf) [backward compatible, assumes module='track']
    """
    if len(args) == 2:
        module = "track"
        cls = str(args[0])
        best_conf = float(args[1])
        area_frac = kwargs.get("area_frac")
    elif len(args) >= 3:
        module = str(args[0])
        cls = str(args[1])
        best_conf = float(args[2])
        area_frac = kwargs.get("area_frac", args[3] if len(args) > 3 else None)
    else:
        raise ValueError("classify requires at least (cls, best_conf) or (module, cls, best_conf)")

    rule = cfg.rule(module, cls)
    severity = rule.get("severity", "moderate")

    if severity == "info":
        return Decision("info", "inventory", "Logged as inventory only.")

    if severity == "critical":
        floor = float(rule.get("critical_min_confidence", 0.0))
        if best_conf < floor:
            return Decision(
                "critical", "digest",
                f"Critical class but best confidence {best_conf:.0%} is below the "
                f"{floor:.0%} incident threshold. Routed to the daily digest for review.",
            )

        min_area = rule.get("critical_min_area_frac")
        if min_area is not None and area_frac is not None and area_frac < float(min_area):
            return Decision(
                "critical", "digest",
                f"Critical class but bbox area fraction {area_frac:.1%} is below the "
                f"{float(min_area):.1%} threshold. Routed to the daily digest for review.",
            )

        return Decision("critical", "incident")

    return Decision("moderate", "digest")
