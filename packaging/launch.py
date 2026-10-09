"""Entry point for the packaged app (PyInstaller needs a script, not a console-script name)."""
from firstlight.app import main

if __name__ == "__main__":
    raise SystemExit(main())
