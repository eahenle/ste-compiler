from enum import StrEnum

from pydantic import BaseModel


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class TextSpan(BaseModel):
    sentence: int
    start: int
    end: int


class Diagnostic(BaseModel):
    code: str
    severity: Severity
    message: str
    span: TextSpan | None = None
    suggestions: list[str] = []
    ir_node_id: str | None = None


class ValidationReport(BaseModel):
    status: str
    violations: list[Diagnostic]

    @classmethod
    def from_diagnostics(cls, items: list[Diagnostic]) -> "ValidationReport":
        rejected = any(x.severity in {Severity.ERROR, Severity.CRITICAL} for x in items)
        return cls(status="rejected" if rejected else "accepted", violations=items)
