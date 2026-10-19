"""Loads reporting/config/severity_rules.yaml with multi-module support and safe defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
DEFAULT_CONFIG_PATH = PKG_DIR / "config" / "severity_rules.yaml"

_DEFAULTS = {
    "aggregation": {
        "min_detection_confidence": 0.25,
        "confirm_hits": 3,
        "confirm_window": 6,
        "max_gap_frames": 15,
        "match_iou": 0.10,
        "match_center_factor": 1.5,
        "last_image_every": 3,
        "queue_size": 128,
    },
    "evidence": {"jpeg_quality": 90, "crop_padding": 0.25, "max_dim": 1280},
    "reports": {
        "output_dir": "reports",
        "project_name": "Railway Infrastructure & Defect Inspection Suite",
        "team": "HEDGE-Z",
        "system_version": "1.0",
        "operator": "",
    },
    "email": {
        "smtp_host": "localhost",
        "smtp_port": 587,
        "use_tls": True,
        "username": "",
        "password_env": "SMTP_PASSWORD",
        "from_addr": "Railway Inspection Suite <inspections@railway-infra.org>",
        "default_recipients": ["maintenance@railway-infra.org"],
    },
}


class _ClassOrderList(list):
    """List of class names for backward compatibility; callable to get per-module order."""
    def __init__(self, cfg: Config, default_module: str = "track"):
        self._cfg = cfg
        super().__init__(cfg.classes_for(default_module).keys())

    def __call__(self, module: str = "track") -> list[str]:
        return list(self._cfg.classes_for(module).keys())


class Config:
    def __init__(self, data: dict):
        self.data = data or {}
        for section, defaults in _DEFAULTS.items():
            merged = dict(defaults)
            merged.update(self.data.get(section) or {})
            setattr(self, section if section != "reports" else "reports_cfg", merged)
        self.agg = self.aggregation
        self.evidence_cfg = self.evidence
        self.email_cfg = getattr(self, "email", _DEFAULTS["email"])
        self.models: dict = self.data.get("models") or {}

        # Multi-module restructuring with backward-compatibility for flat YAML
        if "modules" in self.data and isinstance(self.data["modules"], dict):
            self.modules_data = self.data["modules"]
        else:
            # Legacy flat schema: synthesize a track module
            self.modules_data = {
                "track": {
                    "display_name": "Railway Track Defect Inspection",
                    "inspection_label": "Railway track defect inspection",
                    "class_aliases": self.data.get("class_aliases") or {},
                    "classes": self.data.get("classes") or {},
                    "aggregation": self.data.get("aggregation") or {},
                    "models": self.data.get("models") or {},
                }
            }

        # Build alias maps per module
        self._aliases: dict[str, dict[str, str]] = {}
        for mod, mod_cfg in self.modules_data.items():
            aliases: dict[str, str] = {}
            for canon, names in (mod_cfg.get("class_aliases") or {}).items():
                aliases[str(canon).strip().lower()] = canon
                for n in names or []:
                    aliases[str(n).strip().lower()] = canon
            # Also register the canonical class names themselves
            for canon in (mod_cfg.get("classes") or {}).keys():
                aliases[str(canon).strip().lower()] = canon
            self._aliases[mod] = aliases

        # Legacy backward-compatibility attributes
        self.classes: dict = self.classes_for("track")
        self.class_order = _ClassOrderList(self, "track")

    # --- modules & classes -------------------------------------------------
    def module_names(self) -> list[str]:
        return list(self.modules_data.keys())

    def module_cfg(self, module: str = "track") -> dict:
        return self.modules_data.get(module, {})

    def classes_for(self, module: str = "track") -> dict:
        return self.modules_data.get(module, {}).get("classes") or {}

    def canonical(self, module_or_raw: str, raw_name: str | None = None) -> str | None:
        """Map a raw class string to canonical key: canonical(raw) or canonical(module, raw)."""
        if raw_name is None:
            raw_str = str(module_or_raw).strip().lower()
            # Default to track
            canon = self._aliases.get("track", {}).get(raw_str)
            if canon:
                return canon
            # Fallback: search across all modules
            for aliases in self._aliases.values():
                if raw_str in aliases:
                    return aliases[raw_str]
            return None
        else:
            module = str(module_or_raw).strip().lower()
            raw_str = str(raw_name).strip().lower()
            return self._aliases.get(module, {}).get(raw_str)

    def rule(self, module_or_cls: str, cls: str | None = None) -> dict:
        """Return rule dict for class: rule(cls) or rule(module, cls)."""
        if cls is None:
            cls_key = str(module_or_cls)
            r = self.classes_for("track").get(cls_key)
            if r is not None:
                return r
            for mod in self.module_names():
                r = self.classes_for(mod).get(cls_key)
                if r is not None:
                    return r
            return {}
        else:
            module = str(module_or_cls).strip().lower()
            return self.classes_for(module).get(str(cls), {})

    def aggregation_cfg(self, module: str = "track") -> dict:
        """Returns effective aggregation config merging global with module overrides."""
        base = dict(self.agg)
        mod_agg = (self.modules_data.get(module) or {}).get("aggregation") or {}
        base.update(mod_agg)
        return base

    def get_model_info(self, module: str, model_type: str) -> dict:
        mod_models = (self.modules_data.get(module) or {}).get("models") or {}
        if model_type in mod_models:
            return mod_models[model_type]
        return self.models.get(model_type, {})

    # --- output ------------------------------------------------------------
    @property
    def output_dir(self) -> Path:
        p = Path(self.reports_cfg["output_dir"])
        return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config(data)
