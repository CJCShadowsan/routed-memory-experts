import json

from routed_memory_experts.adapter_manifest import load_adapter_manifest, manifest_to_dict


def test_adapter_manifest_routes_domains(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "base_model": "base",
        "adapters": [{"name": "a", "served_model": "adapter-a", "source": "repo/a", "domains": ["python"], "status": "proven"}],
    }))
    manifest = load_adapter_manifest(path)
    assert manifest.model_for_domain("python") == "adapter-a"
    assert manifest.model_for_domain("security") == "base"
    assert manifest_to_dict(manifest)["route_map"] == {"python": "adapter-a"}
