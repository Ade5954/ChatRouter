"""Command line entry point: ``python -m chatrouter``."""

from __future__ import annotations

import argparse
import sys

from .config.loader import ConfigError, load_config


def main() -> int:
    parser = argparse.ArgumentParser(prog="chatrouter", description="LLM traffic gateway")
    parser.add_argument("-c", "--config", default=None, help="path to config.yaml")
    parser.add_argument("--host", default=None, help="override the bind address")
    parser.add_argument("--port", type=int, default=None, help="override the bind port")
    parser.add_argument("--reload", action="store_true", help="enable auto-reload (development)")
    parser.add_argument(
        "--check", action="store_true", help="validate the configuration and exit"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if args.check:
        print(
            f"configuration OK: {len(config.providers)} provider(s), "
            f"{len(config.models)} model(s), {len(config.tenants)} tenant(s)"
        )
        return 0

    import uvicorn

    from .app import create_app

    host = args.host or config.server.host
    port = args.port or config.server.port

    if args.reload:
        uvicorn.run("chatrouter.app:create_app", host=host, port=port, reload=True, factory=True)
    else:
        uvicorn.run(
            # Pass the path rather than the parsed config so the app knows
            # where to persist admin config edits (create_app records it).
            create_app(config_path=args.config),
            host=host,
            port=port,
            workers=1,
            log_config=None,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
