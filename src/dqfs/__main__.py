"""Allow ``python -m dqfs`` execution."""

from dqfs.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
