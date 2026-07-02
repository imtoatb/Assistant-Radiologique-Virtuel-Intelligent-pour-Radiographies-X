from __future__ import annotations

import argparse
import getpass
from pathlib import Path


def write_access_token(token: str, token_path: Path) -> None:
    token = token.strip()
    if not token:
        raise ValueError("Kaggle token cannot be empty")
    if not token.startswith("KGAT_"):
        raise ValueError("This does not look like a Kaggle API token")

    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(token, encoding="utf-8")
    print(f"Kaggle token saved to: {token_path}")


def check_auth() -> None:
    try:
        from kagglehub.auth import whoami
    except ImportError as exc:
        raise RuntimeError("kagglehub is not installed") from exc

    try:
        user = whoami()
    except Exception as exc:
        raise RuntimeError(
            "Kaggle authentication could not be validated. "
            "Check that the token is fresh, that your network can reach Kaggle, "
            "and that the RSNA challenge rules are accepted."
        ) from exc
    print(f"Kaggle authentication OK: {user['username']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token-path",
        type=Path,
        default=Path.home() / ".kaggle" / "access_token",
        help="Where kagglehub reads the KGAT access token.",
    )
    parser.add_argument("--check", action="store_true", help="Validate Kaggle authentication after saving.")
    args = parser.parse_args()

    token = getpass.getpass("Paste your Kaggle API token: ")
    write_access_token(token, args.token_path)

    if args.check:
        check_auth()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc
