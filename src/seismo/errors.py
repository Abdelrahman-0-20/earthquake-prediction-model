"""Shared exceptions and exit codes for the seismo CLI.

Exit codes are an API — scripts and CI depend on them:

    0  success
    2  usage / configuration error (also argparse's default for bad flags)
    3  validation failed — data did not pass the quality gates
    4  API / network failure
    5  missing required artifact
"""

from __future__ import annotations

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_VALIDATION = 3
EXIT_API = 4
EXIT_MISSING = 5


class SeismoError(Exception):
    """Base class for expected, user-facing failures.

    Carries an `exit_code` and a `hint` describing the fix — errors must state
    what went wrong and the exact next step, never just the failure.
    """

    exit_code = EXIT_USAGE

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint

    def render(self) -> str:
        lines = [f"Error: {self}"]
        if self.hint:
            lines.append(f"  Fix: {self.hint}")
        return "\n".join(lines)


class ConfigError(SeismoError):
    exit_code = EXIT_USAGE


class ValidationError(SeismoError):
    exit_code = EXIT_VALIDATION


class ApiError(SeismoError):
    exit_code = EXIT_API


class MissingArtifact(SeismoError):
    exit_code = EXIT_MISSING
