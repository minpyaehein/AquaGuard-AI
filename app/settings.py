from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    app_name: str = "FloodWatch AI Geo — ASEAN Command Center"
    demo_mode: bool = True
    model_registry_dir: str = "models/registry"
    risk_model_plugin: str = "app.ai.plugins.sklearn_risk:SklearnRiskModel"
    risk_model_path: str = "models/registry/risk_demo/model.joblib"
    authorized_reviewer_token: str = "change-me"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
settings = Settings()
