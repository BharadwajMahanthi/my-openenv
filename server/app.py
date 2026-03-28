"""ASGI entrypoint used by OpenEnv CLI during validation/deployment."""
from support_triage_env.server import app, run_server

__all__ = ["app", "main"]


def main() -> None:
    """Invoke the packaged uvicorn server."""
    run_server()


if __name__ == "__main__":
    main()
