"""Compatibility entrypoint for tooling that imports `app:app`."""

from server.app import app, main


if __name__ == "__main__":
    main()
