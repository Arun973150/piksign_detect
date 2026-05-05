"""Application configuration loader.

For PikSign Detect, the single source of environment configuration is:
  piksign_detect/.env

The loader intentionally overrides any same-named process environment values
so an old terminal variable or the sibling piksign/.env cannot silently change
this app's behavior.
"""

from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = APP_ROOT / ".env"


def load_app_env() -> Path:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return ENV_PATH

    if ENV_PATH.is_file():
        load_dotenv(ENV_PATH, override=True)
    return ENV_PATH
