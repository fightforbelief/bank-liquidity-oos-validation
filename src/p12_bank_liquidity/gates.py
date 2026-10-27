"""Fail-closed gate errors. Downstream commands exit non-zero with a reason."""


class GateBlocked(RuntimeError):
    """A requested action is blocked until a human gate is approved."""

    def __init__(self, reason: str, exit_code: int = 2):
        super().__init__(reason)
        self.reason = reason
        self.exit_code = exit_code
