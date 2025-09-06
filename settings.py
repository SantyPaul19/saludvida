from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    postgres_user: str = Field(default="appuser", alias="POSTGRES_USER")
    postgres_password: str = Field(default="apppass", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="diabetes_risk", alias="POSTGRES_DB")
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    database_url: str | None = Field(default=None, alias="DATABASE_URL")
    risk_threshold: float = Field(default=0.5, alias="RISK_THRESHOLD")

    # Ruta al modelo entrenado
    model_path: str = "models/best_model.pkl"

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def sql_url(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

settings = Settings()
