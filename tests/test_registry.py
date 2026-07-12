from app.ai.registry import ModelRegistry
def test_registry_creates_root(tmp_path):assert ModelRegistry(tmp_path).root.exists()
