from __future__ import annotations
import hashlib, importlib, json
from pathlib import Path
from .contracts import ModelPlugin

class ModelRegistry:
    def __init__(self, root: str | Path = "models/registry"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_plugin(spec: str, path: str | Path, device: str = "cpu") -> ModelPlugin:
        module_name, class_name = spec.split(":", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        plugin: ModelPlugin = cls()
        plugin.load(Path(path), device=device)
        return plugin

    def register(self, model_dir: str | Path, manifest: dict) -> Path:
        model_dir = Path(model_dir)
        files = sorted(p for p in model_dir.rglob("*") if p.is_file() and p.name != "manifest.json")
        digest = hashlib.sha256()
        for p in files:
            digest.update(p.relative_to(model_dir).as_posix().encode())
            digest.update(p.read_bytes())
        manifest = {**manifest, "sha256": digest.hexdigest(), "files": [p.relative_to(model_dir).as_posix() for p in files]}
        out = model_dir / "manifest.json"
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return out

    def validate_for_activation(self, model_dir: str | Path, allow_unvalidated: bool = False) -> dict:
        p = Path(model_dir) / "manifest.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing model manifest: {p}")
        m = json.loads(p.read_text(encoding="utf-8"))
        if not allow_unvalidated and m.get("validation_status") != "approved":
            raise RuntimeError("Model is not approved for activation")
        return m
