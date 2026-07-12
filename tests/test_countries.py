from pathlib import Path
import yaml
def test_11_country_profiles():
 d=yaml.safe_load(Path("configs/countries.yaml").read_text());assert len(d["countries"])==11
