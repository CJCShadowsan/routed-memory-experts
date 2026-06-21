from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    served_model: str
    source: str
    domains: tuple[str, ...]
    status: str
    notes: str = ""


@dataclass(frozen=True)
class AdapterManifest:
    base_model: str
    adapters: tuple[AdapterSpec, ...]

    def model_for_domain(self, domain: str, default_model: str | None = None) -> str:
        for adapter in self.adapters:
            if domain in adapter.domains and adapter.status in {"available", "proven", "candidate"}:
                return adapter.served_model
        return default_model or self.base_model

    def route_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for adapter in self.adapters:
            for domain in adapter.domains:
                mapping[domain] = adapter.served_model
        return mapping


def load_adapter_manifest(path: str | Path | None) -> AdapterManifest:
    if path is None:
        raise ValueError("adapter manifest path is required")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    adapters = tuple(
        AdapterSpec(
            name=str(item["name"]),
            served_model=str(item["served_model"]),
            source=str(item["source"]),
            domains=tuple(str(domain) for domain in item.get("domains", [])),
            status=str(item.get("status", "candidate")),
            notes=str(item.get("notes", "")),
        )
        for item in data.get("adapters", [])
    )
    return AdapterManifest(base_model=str(data["base_model"]), adapters=adapters)


def manifest_to_dict(manifest: AdapterManifest) -> dict:
    return {
        "base_model": manifest.base_model,
        "adapters": [
            {
                "name": adapter.name,
                "served_model": adapter.served_model,
                "source": adapter.source,
                "domains": list(adapter.domains),
                "status": adapter.status,
                "notes": adapter.notes,
            }
            for adapter in manifest.adapters
        ],
        "route_map": manifest.route_map(),
    }
