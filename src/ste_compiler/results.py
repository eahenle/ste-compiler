"""Versioned machine-readable results for end-to-end compiler workflows."""

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ste_compiler.diagnostics import ValidationReport
from ste_compiler.ir.models import Document
from ste_compiler.realizer.base import RealizationResult

COMPILE_SOURCE_SCHEMA_VERSION: Literal["compile-source-v1"] = "compile-source-v1"


class StrictResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceIdentity(StrictResultModel):
    id: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CompiledSentenceMapping(StrictResultModel):
    sentence: int = Field(ge=1)
    text: str
    ir_node_ids: tuple[str, ...]
    features: dict[str, object]


class CompileSourceResult(StrictResultModel):
    schema_version: Literal["compile-source-v1"]
    source: SourceIdentity
    text: str
    mappings: tuple[CompiledSentenceMapping, ...]
    validation: ValidationReport
    metadata: dict[str, str]
    ir: Document

    @classmethod
    def from_compilation(
        cls,
        *,
        source_bytes: bytes,
        source_id: str,
        document: Document,
        realization: RealizationResult,
        validation: ValidationReport,
    ) -> "CompileSourceResult":
        return cls(
            schema_version=COMPILE_SOURCE_SCHEMA_VERSION,
            source=SourceIdentity(
                id=source_id,
                sha256=hashlib.sha256(source_bytes).hexdigest(),
            ),
            text=realization.text,
            mappings=tuple(
                CompiledSentenceMapping(
                    sentence=mapping.sentence,
                    text=mapping.text,
                    ir_node_ids=mapping.ir_node_ids,
                    features=mapping.features,
                )
                for mapping in realization.mappings
            ),
            validation=validation,
            metadata=realization.metadata,
            ir=document,
        )
