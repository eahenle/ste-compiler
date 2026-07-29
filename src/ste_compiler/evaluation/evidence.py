"""Hash-bound, recomputable benchmark evidence without implicit quality claims."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import secrets
import stat
import sys
from collections import Counter
from pathlib import Path
from typing import Annotated, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    model_validator,
)

from ste_compiler.training import CorpusSelectionV1, read_training_release

SHA256_PATTERN: Final = r"^[0-9a-f]{64}$"
IDENTIFIER_PATTERN: Final = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
BENCHMARK_SPEC_SCHEMA_VERSION: Final = "ste-benchmark-spec-v1"
FAILURE_TAXONOMY_SCHEMA_VERSION: Final = "ste-benchmark-failure-taxonomy-v1"
PREDICTION_SCHEMA_VERSION: Final = "ste-benchmark-prediction-v1"
PREDICTION_MANIFEST_SCHEMA_VERSION: Final = "ste-benchmark-prediction-manifest-v1"
METRICS_SCHEMA_VERSION: Final = "ste-benchmark-metrics-v1"
REPORT_MANIFEST_SCHEMA_VERSION: Final = "ste-benchmark-report-manifest-v1"
MAX_INPUT_BYTES: Final = 64 * 1024 * 1024
SPLITS: Final = ("train", "validation", "test", "adversarial")
SUPPORTED_METRICS: Final = (
    "complete_success_rate",
    "frontend.schema_valid_rate",
    "frontend.required_field_precision",
    "frontend.required_field_recall",
    "frontend.required_field_f1",
    "frontend.hallucinated_node_rate",
    "frontend.ambiguity_preservation_rate",
    "frontend.source_span_exact_rate",
    "frontend.source_span_overlap_rate",
    "realizer.exact_symbolic_plan_rate",
    "realizer.grammar_valid_rate",
    "realizer.eos_completion_rate",
    "realizer.unauthorized_symbol_rate",
    "realizer.constraint_rejection_rate",
    "validator.accepted_rate",
    "validator.rejected_rate",
    "validator.false_accept_rate",
    "provenance_coverage_rate",
    "deterministic_repeatability_rate",
)
LEXICAL_DIAGNOSTIC_CODES: Final = frozenset(
    {
        "TERMINOLOGY_ALIAS",
        "UNAUTHORIZED_WORD",
    }
)
STRUCTURAL_DIAGNOSTIC_CODES: Final = frozenset(
    {
        "PARAGRAPH_TOO_LONG",
        "SENTENCE_TOO_LONG",
    }
)
SEMANTIC_DIAGNOSTIC_CODES: Final = frozenset(
    {
        "CAUSAL_RELATION_NOT_PRESERVED",
        "CONDITION_NOT_PRESERVED",
        "HAZARD_NOT_PRESERVED",
        "NEGATION_NOT_PRESERVED",
        "QUANTITY_NOT_PRESERVED",
        "REQUIRED_NODE_OMITTED",
        "TEMPORAL_RELATION_NOT_PRESERVED",
        "UNSUPPORTED_SEMANTIC_CHANGE",
    }
)
WARNING_DIAGNOSTIC_CODES: Final = frozenset(
    {
        "AMBIGUOUS_PRONOUN",
        "PASSIVE_VOICE",
    }
)
REJECTING_DIAGNOSTIC_CODES: Final = (
    LEXICAL_DIAGNOSTIC_CODES | STRUCTURAL_DIAGNOSTIC_CODES | SEMANTIC_DIAGNOSTIC_CODES
)
VALIDATOR_DIAGNOSTIC_CODES: Final = REJECTING_DIAGNOSTIC_CODES | WARNING_DIAGNOSTIC_CODES
EvidenceKind = Literal["deterministic_fixture", "external_measured"]
FailureStage = Literal["none", "frontend", "realizer", "validator"]
FailureTaxonomyStage = Literal["frontend", "realizer", "validator"]
FailureCode = Literal[
    "frontend.schema_invalid",
    "frontend.required_field_omission",
    "frontend.source_span_mismatch",
    "frontend.hallucinated_node",
    "frontend.ambiguity_not_preserved",
    "realizer.unauthorized_symbol",
    "realizer.incomplete_eos",
    "realizer.grammar_invalid",
    "realizer.constraint_rejection",
    "validator.semantic_rejection",
    "validator.lexical_rejection",
    "validator.structural_rejection",
    "validator.false_accept",
]
FAILURE_TAXONOMY_V1: Final[tuple[tuple[FailureCode, FailureTaxonomyStage], ...]] = (
    ("frontend.schema_invalid", "frontend"),
    ("frontend.required_field_omission", "frontend"),
    ("frontend.source_span_mismatch", "frontend"),
    ("frontend.hallucinated_node", "frontend"),
    ("frontend.ambiguity_not_preserved", "frontend"),
    ("realizer.unauthorized_symbol", "realizer"),
    ("realizer.incomplete_eos", "realizer"),
    ("realizer.grammar_invalid", "realizer"),
    ("realizer.constraint_rejection", "realizer"),
    ("validator.semantic_rejection", "validator"),
    ("validator.lexical_rejection", "validator"),
    ("validator.structural_rejection", "validator"),
    ("validator.false_accept", "validator"),
)
NonnegativeCount = Annotated[StrictInt, Field(ge=0)]
StrictFiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class StrictEvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FileIdentityV1(StrictEvidenceModel):
    path: StrictStr = Field(min_length=1, pattern=r"\S")
    sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    bytes: StrictInt = Field(ge=0)


class DatasetIdentityV1(StrictEvidenceModel):
    dataset_version: StrictStr = Field(min_length=1, pattern=IDENTIFIER_PATTERN)
    manifest_sha256: StrictStr = Field(pattern=SHA256_PATTERN)


class BenchmarkCaseV1(StrictEvidenceModel):
    case_id: StrictStr = Field(min_length=1, pattern=IDENTIFIER_PATTERN)
    split: Literal["test", "adversarial"]
    source_sha256: StrictStr = Field(pattern=SHA256_PATTERN)


class BenchmarkSystemV1(StrictEvidenceModel):
    system_id: StrictStr = Field(min_length=1, pattern=IDENTIFIER_PATTERN)
    evidence_kind: EvidenceKind
    artifact_manifest_sha256: StrictStr | None = Field(default=None, pattern=SHA256_PATTERN)
    description: StrictStr = Field(min_length=1)

    @model_validator(mode="after")
    def measured_system_requires_artifact_identity(self) -> BenchmarkSystemV1:
        if self.evidence_kind == "external_measured" and self.artifact_manifest_sha256 is None:
            raise ValueError("external measured systems require an artifact manifest SHA-256")
        if self.evidence_kind == "deterministic_fixture" and self.artifact_manifest_sha256:
            raise ValueError("deterministic fixtures must not claim a model artifact identity")
        return self


class BenchmarkSpecV1(StrictEvidenceModel):
    schema_version: Literal["ste-benchmark-spec-v1"]
    benchmark_id: StrictStr = Field(min_length=1, pattern=IDENTIFIER_PATTERN)
    version: StrictStr = Field(min_length=1, pattern=IDENTIFIER_PATTERN)
    evidence_label: Literal["deterministic_fixture_only", "external_measured"]
    claim_scope: StrictStr = Field(min_length=1)
    non_certification_notice: StrictStr = Field(min_length=1)
    dataset: DatasetIdentityV1
    cases: tuple[BenchmarkCaseV1, ...] = Field(min_length=1)
    systems: tuple[BenchmarkSystemV1, ...] = Field(min_length=1)
    metrics: tuple[StrictStr, ...] = Field(min_length=1)
    failure_taxonomy_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    confidence_interval: Literal["wilson-score-95"]
    seed: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def frozen_contract(self) -> BenchmarkSpecV1:
        case_ids = [case.case_id for case in self.cases]
        system_ids = [system.system_id for system in self.systems]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("benchmark case IDs must be unique")
        if len(set(system_ids)) != len(system_ids):
            raise ValueError("benchmark system IDs must be unique")
        if tuple(self.metrics) != SUPPORTED_METRICS:
            raise ValueError("benchmark metrics must equal the frozen v1 metric inventory")
        kinds = {system.evidence_kind for system in self.systems}
        if len(kinds) != 1:
            raise ValueError("benchmark systems must not mix fixture and measured evidence")
        expected_label = (
            "deterministic_fixture_only"
            if kinds == {"deterministic_fixture"}
            else "external_measured"
        )
        if self.evidence_label != expected_label:
            raise ValueError("benchmark evidence label does not match its systems")
        return self


class FailureCodeV1(StrictEvidenceModel):
    code: StrictStr = Field(
        min_length=1,
        pattern=r"^(frontend|realizer|validator)\.[a-z0-9_]+$",
    )
    stage: FailureTaxonomyStage
    description: StrictStr = Field(min_length=1)

    @model_validator(mode="after")
    def stage_matches_code(self) -> FailureCodeV1:
        if not self.code.startswith(self.stage + "."):
            raise ValueError("failure code prefix must match its stage")
        return self


class FailureTaxonomyV1(StrictEvidenceModel):
    schema_version: Literal["ste-benchmark-failure-taxonomy-v1"]
    version: StrictStr = Field(min_length=1, pattern=IDENTIFIER_PATTERN)
    codes: tuple[FailureCodeV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def frozen_inventory(self) -> FailureTaxonomyV1:
        inventory = tuple((item.code, item.stage) for item in self.codes)
        if inventory != FAILURE_TAXONOMY_V1:
            raise ValueError(
                "failure taxonomy must equal the frozen v1 code, order, and stage inventory"
            )
        return self


class FrontendObservationV1(StrictEvidenceModel):
    status: Literal["succeeded", "failed"]
    schema_valid: StrictBool
    required_fields_gold: StrictInt = Field(ge=0)
    required_fields_predicted: StrictInt = Field(ge=0)
    required_fields_matched: StrictInt = Field(ge=0)
    predicted_nodes: StrictInt = Field(ge=0)
    hallucinated_nodes: StrictInt = Field(ge=0)
    ambiguities_gold: StrictInt = Field(ge=0)
    ambiguities_preserved: StrictInt = Field(ge=0)
    source_spans_gold: StrictInt = Field(ge=0)
    source_spans_exact: StrictInt = Field(ge=0)
    source_spans_overlapping: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def bounded_counts(self) -> FrontendObservationV1:
        if self.required_fields_matched > min(
            self.required_fields_gold,
            self.required_fields_predicted,
        ):
            raise ValueError("matched required fields exceed gold or predicted counts")
        if self.hallucinated_nodes > self.predicted_nodes:
            raise ValueError("hallucinated nodes exceed predicted nodes")
        if self.ambiguities_preserved > self.ambiguities_gold:
            raise ValueError("preserved ambiguities exceed gold ambiguities")
        if self.source_spans_exact > self.source_spans_gold:
            raise ValueError("exact source spans exceed gold source spans")
        if self.source_spans_overlapping > self.source_spans_gold:
            raise ValueError("overlapping source spans exceed gold source spans")
        if self.source_spans_exact > self.source_spans_overlapping:
            raise ValueError("exact source spans exceed overlapping source spans")
        if self.status == "succeeded" and not self.schema_valid:
            raise ValueError("a successful frontend observation must be schema-valid")
        if self.status == "succeeded" and (
            self.required_fields_matched != self.required_fields_gold
            or self.required_fields_predicted != self.required_fields_gold
            or self.hallucinated_nodes != 0
            or self.ambiguities_preserved != self.ambiguities_gold
            or self.source_spans_exact != self.source_spans_gold
        ):
            raise ValueError(
                "a successful frontend observation must satisfy every frozen pass boundary"
            )
        return self


class RealizerObservationV1(StrictEvidenceModel):
    status: Literal["succeeded", "failed", "not_run"]
    exact_symbolic_plan: StrictBool | None
    grammar_valid: StrictBool | None
    eos_completed: StrictBool | None
    predicted_symbol_count: StrictInt | None = Field(default=None, ge=0)
    unauthorized_symbol_count: StrictInt | None = Field(default=None, ge=0)
    constraint_rejected: StrictBool | None

    @model_validator(mode="after")
    def status_controls_measurements(self) -> RealizerObservationV1:
        values = (
            self.exact_symbolic_plan,
            self.grammar_valid,
            self.eos_completed,
            self.predicted_symbol_count,
            self.unauthorized_symbol_count,
            self.constraint_rejected,
        )
        if self.status == "not_run" and any(value is not None for value in values):
            raise ValueError("a realizer that did not run cannot have measurements")
        if self.status != "not_run" and any(value is None for value in values):
            raise ValueError("a realizer attempt requires complete measurements")
        if (
            self.predicted_symbol_count is not None
            and self.unauthorized_symbol_count is not None
            and self.unauthorized_symbol_count > self.predicted_symbol_count
        ):
            raise ValueError("unauthorized symbols exceed predicted symbols")
        if self.status == "succeeded" and (
            not self.grammar_valid
            or not self.eos_completed
            or self.unauthorized_symbol_count != 0
            or self.constraint_rejected
        ):
            raise ValueError("a successful realizer observation violates its success boundary")
        if self.status == "failed" and (
            self.grammar_valid
            and self.eos_completed
            and self.unauthorized_symbol_count == 0
            and not self.constraint_rejected
        ):
            raise ValueError("a failed realizer observation has no recorded failure")
        if self.exact_symbolic_plan and (
            not self.grammar_valid or not self.eos_completed or self.unauthorized_symbol_count != 0
        ):
            raise ValueError(
                "an exact symbolic plan violates grammar, EOS, or symbol authorization"
            )
        return self


class ValidatorObservationV1(StrictEvidenceModel):
    status: Literal["accepted", "rejected", "not_run"]
    gold_should_accept: StrictBool
    diagnostic_codes: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def not_run_has_no_diagnostics(self) -> ValidatorObservationV1:
        if self.status == "not_run" and self.diagnostic_codes:
            raise ValueError("a validator that did not run cannot have diagnostics")
        unknown_codes = set(self.diagnostic_codes) - VALIDATOR_DIAGNOSTIC_CODES
        if unknown_codes:
            raise ValueError(
                "validator diagnostic codes must come from the frozen validation pipeline"
            )
        rejecting_diagnostic = any(
            code in REJECTING_DIAGNOSTIC_CODES for code in self.diagnostic_codes
        )
        if self.status == "rejected" and not rejecting_diagnostic:
            raise ValueError("a rejected validator observation requires a rejecting diagnostic")
        if self.status == "accepted" and rejecting_diagnostic:
            raise ValueError(
                "an accepted validator observation cannot contain rejecting diagnostics"
            )
        return self


class PredictionRecordV1(StrictEvidenceModel):
    schema_version: Literal["ste-benchmark-prediction-v1"]
    benchmark_id: StrictStr = Field(min_length=1, pattern=IDENTIFIER_PATTERN)
    case_id: StrictStr = Field(min_length=1, pattern=IDENTIFIER_PATTERN)
    system_id: StrictStr = Field(min_length=1, pattern=IDENTIFIER_PATTERN)
    evidence_kind: EvidenceKind
    dataset: DatasetIdentityV1
    source_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    frontend: FrontendObservationV1
    realizer: RealizerObservationV1
    validator: ValidatorObservationV1
    failure_stage: FailureStage
    failure_code: FailureCode | None
    provenance_complete: StrictBool
    deterministic_repeat: StrictBool
    raw_output: StrictStr | None
    notes: StrictStr = Field(min_length=1)

    @model_validator(mode="after")
    def causal_failure_stage(self) -> PredictionRecordV1:
        if self.frontend.status == "failed":
            expected_stage: FailureStage = "frontend"
            if self.realizer.status != "not_run" or self.validator.status != "not_run":
                raise ValueError("downstream stages must not run after a frontend failure")
        elif self.realizer.status == "not_run":
            raise ValueError("the realizer must run after a successful frontend")
        elif self.realizer.status == "failed":
            expected_stage = "realizer"
            if self.validator.status != "not_run":
                raise ValueError("the validator must not run after a realizer failure")
        elif self.validator.status == "not_run":
            raise ValueError("the validator must run after successful upstream stages")
        elif (self.validator.status == "rejected" and self.validator.gold_should_accept) or (
            self.validator.status == "accepted" and not self.validator.gold_should_accept
        ):
            expected_stage = "validator"
        else:
            expected_stage = "none"
        if self.failure_stage != expected_stage:
            raise ValueError("failure stage does not match stage observations")
        if expected_stage == "none" and self.failure_code is not None:
            raise ValueError("successful predictions cannot have a failure code")
        if expected_stage != "none" and (
            self.failure_code is None or not self.failure_code.startswith(expected_stage + ".")
        ):
            raise ValueError("failure code must match the observed failure stage")
        if self.failure_code == "frontend.schema_invalid" and self.frontend.schema_valid:
            raise ValueError("schema-invalid code requires an invalid frontend schema")
        if self.failure_code == "frontend.required_field_omission" and not (
            self.frontend.required_fields_matched < self.frontend.required_fields_gold
        ):
            raise ValueError("required-field-omission code requires an omitted gold field")
        if self.failure_code == "frontend.source_span_mismatch" and not (
            self.frontend.source_spans_exact < self.frontend.source_spans_gold
        ):
            raise ValueError("source-span-mismatch code requires a non-exact gold span")
        if (
            self.failure_code == "frontend.hallucinated_node"
            and self.frontend.hallucinated_nodes == 0
        ):
            raise ValueError("hallucinated-node code requires a hallucinated node")
        if self.failure_code == "frontend.ambiguity_not_preserved" and not (
            self.frontend.ambiguities_preserved < self.frontend.ambiguities_gold
        ):
            raise ValueError("ambiguity-not-preserved code requires a lost ambiguity")
        if self.failure_code == "realizer.unauthorized_symbol" and not (
            self.realizer.unauthorized_symbol_count and self.realizer.unauthorized_symbol_count > 0
        ):
            raise ValueError("unauthorized-symbol code requires an unauthorized symbol")
        if (
            self.failure_code == "realizer.incomplete_eos"
            and self.realizer.eos_completed is not False
        ):
            raise ValueError("incomplete-EOS code requires incomplete generation")
        if (
            self.failure_code == "realizer.grammar_invalid"
            and self.realizer.grammar_valid is not False
        ):
            raise ValueError("grammar-invalid code requires invalid grammar")
        standalone_constraint_rejection = (
            self.realizer.status == "failed"
            and self.realizer.constraint_rejected is True
            and self.realizer.grammar_valid is True
            and self.realizer.eos_completed is True
            and self.realizer.unauthorized_symbol_count == 0
        )
        if (
            self.failure_code == "realizer.constraint_rejection"
            and not standalone_constraint_rejection
        ):
            raise ValueError("constraint-rejection code requires a standalone constraint rejection")
        if standalone_constraint_rejection and self.failure_code != "realizer.constraint_rejection":
            raise ValueError(
                "a standalone constraint rejection requires the realizer.constraint_rejection code"
            )
        if self.failure_code == "validator.false_accept" and not (
            self.validator.status == "accepted" and not self.validator.gold_should_accept
        ):
            raise ValueError("false-accept code requires acceptance of a gold rejection")
        if (
            self.failure_stage == "validator"
            and (self.validator.status == "accepted" and not self.validator.gold_should_accept)
            and self.failure_code != "validator.false_accept"
        ):
            raise ValueError("a false acceptance requires the validator.false_accept code")
        if self.failure_code in {
            "validator.semantic_rejection",
            "validator.lexical_rejection",
            "validator.structural_rejection",
        } and not (self.validator.status == "rejected" and self.validator.gold_should_accept):
            raise ValueError("validator-rejection codes require rejection of a gold acceptance")
        lexical_diagnostic = any(
            code in LEXICAL_DIAGNOSTIC_CODES for code in self.validator.diagnostic_codes
        )
        structural_diagnostic = any(
            code in STRUCTURAL_DIAGNOSTIC_CODES for code in self.validator.diagnostic_codes
        )
        semantic_diagnostic = any(
            code in SEMANTIC_DIAGNOSTIC_CODES for code in self.validator.diagnostic_codes
        )
        if self.failure_code == "validator.lexical_rejection" and not lexical_diagnostic:
            raise ValueError("lexical-rejection code requires a lexical diagnostic")
        if self.failure_code == "validator.structural_rejection" and not structural_diagnostic:
            raise ValueError("structural-rejection code requires a structural diagnostic")
        if self.failure_code == "validator.semantic_rejection" and not semantic_diagnostic:
            raise ValueError("semantic-rejection code requires a semantic diagnostic")
        return self


class PredictionManifestV1(StrictEvidenceModel):
    schema_version: Literal["ste-benchmark-prediction-manifest-v1"]
    benchmark_id: StrictStr = Field(min_length=1, pattern=IDENTIFIER_PATTERN)
    benchmark_spec_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    dataset: DatasetIdentityV1
    systems: tuple[BenchmarkSystemV1, ...] = Field(min_length=1)
    predictions: FileIdentityV1
    record_count: StrictInt = Field(gt=0)


class ConfidenceIntervalV1(StrictEvidenceModel):
    method: Literal["wilson-score-95", "none-derived"]
    lower: StrictFiniteFloat | None = Field(default=None, ge=0, le=1)
    upper: StrictFiniteFloat | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> ConfidenceIntervalV1:
        if self.method == "none-derived":
            if self.lower is not None or self.upper is not None:
                raise ValueError("derived metrics must not claim confidence bounds")
            return self
        if (self.lower is None) != (self.upper is None):
            raise ValueError("Wilson confidence bounds must both be present or both be null")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("confidence interval lower bound must not exceed its upper bound")
        return self


class MetricEstimateV1(StrictEvidenceModel):
    numerator: StrictInt | None = Field(default=None, ge=0)
    denominator: StrictInt | None = Field(default=None, ge=0)
    value: StrictFiniteFloat | None = Field(default=None, ge=0, le=1)
    confidence_interval: ConfidenceIntervalV1

    @model_validator(mode="after")
    def validate_estimate(self) -> MetricEstimateV1:
        interval = self.confidence_interval
        if interval.method == "none-derived":
            if self.numerator is not None or self.denominator is not None:
                raise ValueError("derived metrics must not claim a numerator or denominator")
            return self
        if self.numerator is None or self.denominator is None:
            raise ValueError("Wilson metrics require a numerator and denominator")
        if self.numerator > self.denominator:
            raise ValueError("metric numerator must not exceed its denominator")
        if self.denominator == 0:
            if self.value is not None:
                raise ValueError("zero-denominator metrics must have a null value")
            if interval.lower is not None or interval.upper is not None:
                raise ValueError("zero-denominator metrics must have null confidence bounds")
            return self
        expected_value = self.numerator / self.denominator
        if self.value is None or not math.isclose(
            self.value,
            expected_value,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("metric value must equal numerator divided by denominator")
        if interval.lower is None or interval.upper is None:
            raise ValueError("nonempty Wilson metrics require confidence bounds")
        z = 1.959963984540054
        scale = 1 + z * z / self.denominator
        center = (expected_value + z * z / (2 * self.denominator)) / scale
        margin = (
            z
            * math.sqrt(
                expected_value * (1 - expected_value) / self.denominator
                + z * z / (4 * self.denominator * self.denominator)
            )
            / scale
        )
        expected_lower = max(0.0, center - margin)
        expected_upper = min(1.0, center + margin)
        if not math.isclose(
            interval.lower,
            expected_lower,
            rel_tol=0.0,
            abs_tol=1e-15,
        ) or not math.isclose(
            interval.upper,
            expected_upper,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("Wilson confidence bounds do not match the metric counts")
        return self


class BenchmarkMetricsV1(StrictEvidenceModel):
    schema_version: Literal["ste-benchmark-metrics-v1"]
    benchmark_id: StrictStr
    evidence_label: Literal["deterministic_fixture_only", "external_measured"]
    claim_scope: StrictStr
    benchmark_spec_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    prediction_manifest_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    predictions_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    record_count: StrictInt = Field(gt=0)
    systems: dict[str, SystemMetricsV1] = Field(min_length=1)

    @model_validator(mode="after")
    def record_count_matches_systems(self) -> BenchmarkMetricsV1:
        if self.record_count != sum(system.record_count for system in self.systems.values()):
            raise ValueError("benchmark record count must equal the sum of system record counts")
        return self


class SystemMetricsV1(StrictEvidenceModel):
    record_count: StrictInt = Field(gt=0)
    metrics: dict[str, MetricEstimateV1]
    failure_stage_counts: dict[FailureStage, NonnegativeCount]
    failure_code_counts: dict[FailureCode, NonnegativeCount]

    @model_validator(mode="after")
    def frozen_metric_inventory(self) -> SystemMetricsV1:
        if set(self.metrics) != set(SUPPORTED_METRICS):
            raise ValueError("system metrics must equal the frozen v1 metric inventory")
        if sum(self.failure_stage_counts.values()) != self.record_count:
            raise ValueError("failure stage counts must sum to the system record count")
        failed_record_count = self.record_count - self.failure_stage_counts.get("none", 0)
        if sum(self.failure_code_counts.values()) != failed_record_count:
            raise ValueError("failure code counts must equal the failed record count")
        for stage in ("frontend", "realizer", "validator"):
            stage_count = self.failure_stage_counts.get(stage, 0)
            code_count = sum(
                count
                for code, count in self.failure_code_counts.items()
                if code.startswith(stage + ".")
            )
            if code_count != stage_count:
                raise ValueError(f"{stage} failure code counts must equal its failure stage count")
        f1_name = "frontend.required_field_f1"
        for name, estimate in self.metrics.items():
            expected_method = "none-derived" if name == f1_name else "wilson-score-95"
            if estimate.confidence_interval.method != expected_method:
                raise ValueError(f"{name} must use the {expected_method} confidence method")

        frontend_failures = self.failure_stage_counts.get("frontend", 0)
        realizer_failures = self.failure_stage_counts.get("realizer", 0)
        validator_failures = self.failure_stage_counts.get("validator", 0)
        complete_successes = self.failure_stage_counts.get("none", 0)
        realizer_population = self.record_count - frontend_failures
        validator_population = realizer_population - realizer_failures

        self._require_metric_counts(
            "complete_success_rate",
            numerator=complete_successes,
            denominator=self.record_count,
        )
        for name in (
            "frontend.schema_valid_rate",
            "provenance_coverage_rate",
            "deterministic_repeatability_rate",
        ):
            self._require_metric_denominator(name, self.record_count)
        for name in (
            "realizer.exact_symbolic_plan_rate",
            "realizer.grammar_valid_rate",
            "realizer.eos_completion_rate",
            "realizer.constraint_rejection_rate",
        ):
            self._require_metric_denominator(name, realizer_population)
        for name in ("validator.accepted_rate", "validator.rejected_rate"):
            self._require_metric_denominator(name, validator_population)

        schema_valid = self._metric_counts("frontend.schema_valid_rate")
        if schema_valid[0] < realizer_population:
            raise ValueError(
                "frontend.schema_valid_rate numerator must include every successful frontend"
            )

        exact_plan = self._metric_counts("realizer.exact_symbolic_plan_rate")
        grammar_valid = self._metric_counts("realizer.grammar_valid_rate")
        eos_complete = self._metric_counts("realizer.eos_completion_rate")
        constraint_rejections = self._metric_counts("realizer.constraint_rejection_rate")
        for name, numerator in (
            ("realizer.grammar_valid_rate", grammar_valid[0]),
            ("realizer.eos_completion_rate", eos_complete[0]),
        ):
            if numerator < validator_population:
                raise ValueError(f"{name} numerator must include every successful realizer")
        if exact_plan[0] > min(grammar_valid[0], eos_complete[0]):
            raise ValueError(
                "realizer.exact_symbolic_plan_rate numerator must not exceed grammar-valid "
                "or EOS-complete attempts"
            )
        if constraint_rejections[0] > realizer_failures:
            raise ValueError(
                "realizer.constraint_rejection_rate numerator must not exceed realizer failures"
            )
        unauthorized_symbols = self._metric_counts("realizer.unauthorized_symbol_rate")
        if realizer_failures == 0 and unauthorized_symbols[0] != 0:
            raise ValueError(
                "realizer.unauthorized_symbol_rate numerator must be zero without realizer failures"
            )

        accepted = self._metric_counts("validator.accepted_rate")
        rejected = self._metric_counts("validator.rejected_rate")
        if accepted[0] + rejected[0] != validator_population:
            raise ValueError(
                "validator accepted and rejected numerators must sum to the validator population"
            )
        false_accepts = self._metric_counts("validator.false_accept_rate")
        if false_accepts[1] > validator_population:
            raise ValueError(
                "validator.false_accept_rate denominator must not exceed the validator population"
            )
        if false_accepts[0] > min(accepted[0], validator_failures):
            raise ValueError(
                "validator.false_accept_rate numerator must not exceed accepted validator failures"
            )
        false_rejections = validator_failures - false_accepts[0]
        correct_rejections = false_accepts[1] - false_accepts[0]
        expected_rejections = false_rejections + correct_rejections
        expected_acceptances = validator_population - expected_rejections
        if (accepted[0], rejected[0]) != (expected_acceptances, expected_rejections):
            raise ValueError(
                "validator accepted and rejected numerators must reconcile with validator "
                "failures and false-accept counts"
            )

        precision_counts = self._metric_counts("frontend.required_field_precision")
        recall_counts = self._metric_counts("frontend.required_field_recall")
        if precision_counts[0] != recall_counts[0]:
            raise ValueError("frontend required-field precision and recall must share a numerator")
        exact_spans = self._metric_counts("frontend.source_span_exact_rate")
        overlapping_spans = self._metric_counts("frontend.source_span_overlap_rate")
        if exact_spans[1] != overlapping_spans[1]:
            raise ValueError("frontend source-span rates must share a denominator")
        if exact_spans[0] > overlapping_spans[0]:
            raise ValueError(
                "frontend exact source-span numerator must not exceed overlapping source spans"
            )

        if frontend_failures == 0:
            if not (
                precision_counts[0] == precision_counts[1] == recall_counts[1] == recall_counts[0]
            ):
                raise ValueError(
                    "frontend required-field counts must be exact without frontend failures"
                )
            for name in (
                "frontend.ambiguity_preservation_rate",
                "frontend.source_span_exact_rate",
                "frontend.source_span_overlap_rate",
            ):
                numerator, denominator = self._metric_counts(name)
                if numerator != denominator:
                    raise ValueError(f"{name} must be exact without frontend failures")
            if self._metric_counts("frontend.hallucinated_node_rate")[0] != 0:
                raise ValueError(
                    "frontend.hallucinated_node_rate numerator must be zero without "
                    "frontend failures"
                )

        precision = self.metrics["frontend.required_field_precision"].value
        recall = self.metrics["frontend.required_field_recall"].value
        expected_f1 = _f1_value(precision, recall)
        actual_f1 = self.metrics[f1_name].value
        if expected_f1 is None:
            if actual_f1 is not None:
                raise ValueError(
                    "frontend.required_field_f1 must be null when precision or recall is null"
                )
        elif actual_f1 is None or not math.isclose(
            actual_f1,
            expected_f1,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "frontend.required_field_f1 must equal the harmonic mean of precision and recall"
            )
        return self

    def _metric_counts(self, name: str) -> tuple[int, int]:
        estimate = self.metrics[name]
        if estimate.numerator is None or estimate.denominator is None:
            raise ValueError(f"{name} must expose Wilson metric counts")
        return estimate.numerator, estimate.denominator

    def _require_metric_denominator(self, name: str, expected: int) -> None:
        denominator = self._metric_counts(name)[1]
        if denominator != expected:
            raise ValueError(f"{name} denominator must equal its measured population ({expected})")

    def _require_metric_counts(self, name: str, *, numerator: int, denominator: int) -> None:
        actual = self._metric_counts(name)
        if actual != (numerator, denominator):
            raise ValueError(
                f"{name} counts must equal the measured population result "
                f"({numerator}/{denominator})"
            )


class ReportManifestV1(StrictEvidenceModel):
    schema_version: Literal["ste-benchmark-report-manifest-v1"]
    benchmark_id: StrictStr
    evidence_label: Literal["deterministic_fixture_only", "external_measured"]
    benchmark_spec_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    failure_taxonomy_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    prediction_manifest_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    predictions_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    system_artifact_manifest_sha256s: tuple[
        Annotated[StrictStr, Field(pattern=SHA256_PATTERN)],
        ...,
    ]
    artifacts: tuple[FileIdentityV1, ...] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def frozen_artifact_inventory(self) -> ReportManifestV1:
        system_artifacts = self.system_artifact_manifest_sha256s
        if self.evidence_label == "deterministic_fixture_only":
            if system_artifacts:
                raise ValueError(
                    "deterministic fixture reports must not claim system artifact manifests"
                )
        elif not system_artifacts:
            raise ValueError("external measured reports require system artifact manifest SHA-256s")
        elif system_artifacts != tuple(sorted(set(system_artifacts))):
            raise ValueError(
                "system artifact manifest SHA-256s must be unique and in canonical order"
            )
        artifact_paths = tuple(artifact.path for artifact in self.artifacts)
        if artifact_paths != ("metrics.json", "report.md"):
            raise ValueError(
                "report manifest artifacts must be exactly metrics.json and report.md "
                "in canonical order"
            )
        return self


def _canonical_json(value: BaseModel | dict[str, object]) -> bytes:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_regular(path: Path) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        file_fd = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot open benchmark input {path}: {error}") from error
    try:
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"benchmark input must be a single-link regular file: {path}")
        if before.st_size > MAX_INPUT_BYTES:
            raise ValueError(f"benchmark input exceeds its size limit: {path}")
        chunks: list[bytes] = []
        byte_count = 0
        while chunk := os.read(
            file_fd,
            min(1024 * 1024, MAX_INPUT_BYTES + 1 - byte_count),
        ):
            chunks.append(chunk)
            byte_count += len(chunk)
            if byte_count > MAX_INPUT_BYTES:
                raise ValueError(f"benchmark input exceeds its size limit: {path}")
        after = os.fstat(file_fd)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        data = b"".join(chunks)
        if identity_before != identity_after or len(data) != before.st_size:
            raise ValueError(f"benchmark input changed while it was read: {path}")
        return data
    except OSError as error:
        raise ValueError(f"cannot read benchmark input {path}: {error}") from error
    finally:
        os.close(file_fd)


def _model_file(path: Path, model: type[StrictEvidenceModel]) -> tuple[StrictEvidenceModel, bytes]:
    data = _read_regular(path)
    try:
        return model.model_validate_json(data), data
    except ValueError as error:
        raise ValueError(f"invalid benchmark input {path}: {error}") from error


def _prediction_lines(data: bytes) -> tuple[PredictionRecordV1, ...]:
    records: list[PredictionRecordV1] = []
    for number, line in enumerate(data.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"benchmark predictions contain a blank line at {number}")
        try:
            records.append(PredictionRecordV1.model_validate_json(line))
        except ValueError as error:
            raise ValueError(f"invalid benchmark prediction line {number}: {error}") from error
    if not records:
        raise ValueError("benchmark predictions must not be empty")
    return tuple(records)


def _validate_case_gold_contract(records: tuple[PredictionRecordV1, ...]) -> None:
    gold_contract_by_case: dict[str, tuple[int, int, int, bool]] = {}
    field_names = (
        "frontend.required_fields_gold",
        "frontend.ambiguities_gold",
        "frontend.source_spans_gold",
        "validator.gold_should_accept",
    )
    for record in records:
        contract = (
            record.frontend.required_fields_gold,
            record.frontend.ambiguities_gold,
            record.frontend.source_spans_gold,
            record.validator.gold_should_accept,
        )
        expected = gold_contract_by_case.setdefault(record.case_id, contract)
        if contract == expected:
            continue
        mismatched_fields = ", ".join(
            field_name
            for field_name, expected_value, actual_value in zip(
                field_names,
                expected,
                contract,
                strict=True,
            )
            if actual_value != expected_value
        )
        raise ValueError(
            "prediction gold contract disagrees across systems for case "
            f"{record.case_id}: {mismatched_fields}"
        )


def _wilson(numerator: int, denominator: int) -> MetricEstimateV1:
    if denominator == 0:
        return MetricEstimateV1(
            numerator=numerator,
            denominator=denominator,
            value=None,
            confidence_interval=ConfidenceIntervalV1(method="wilson-score-95"),
        )
    value = numerator / denominator
    z = 1.959963984540054
    scale = 1 + z * z / denominator
    center = (value + z * z / (2 * denominator)) / scale
    margin = (
        z
        * math.sqrt(value * (1 - value) / denominator + z * z / (4 * denominator * denominator))
        / scale
    )
    return MetricEstimateV1(
        numerator=numerator,
        denominator=denominator,
        value=value,
        confidence_interval=ConfidenceIntervalV1(
            method="wilson-score-95",
            lower=max(0.0, center - margin),
            upper=min(1.0, center + margin),
        ),
    )


def _f1_value(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _derived_f1(precision: MetricEstimateV1, recall: MetricEstimateV1) -> MetricEstimateV1:
    value = _f1_value(precision.value, recall.value)
    return MetricEstimateV1(
        numerator=None,
        denominator=None,
        value=value,
        confidence_interval=ConfidenceIntervalV1(method="none-derived"),
    )


def _system_metrics(records: tuple[PredictionRecordV1, ...]) -> SystemMetricsV1:
    frontends = [record.frontend for record in records]
    realizers = [record.realizer for record in records if record.realizer.status != "not_run"]
    validators = [record.validator for record in records if record.validator.status != "not_run"]
    required_gold = sum(item.required_fields_gold for item in frontends)
    required_predicted = sum(item.required_fields_predicted for item in frontends)
    required_matched = sum(item.required_fields_matched for item in frontends)
    precision = _wilson(required_matched, required_predicted)
    recall = _wilson(required_matched, required_gold)
    predicted_nodes = sum(item.predicted_nodes for item in frontends)
    predicted_symbols = sum(item.predicted_symbol_count or 0 for item in realizers)
    metric_values = {
        "complete_success_rate": _wilson(
            sum(record.failure_stage == "none" for record in records),
            len(records),
        ),
        "frontend.schema_valid_rate": _wilson(
            sum(item.schema_valid for item in frontends),
            len(frontends),
        ),
        "frontend.required_field_precision": precision,
        "frontend.required_field_recall": recall,
        "frontend.required_field_f1": _derived_f1(precision, recall),
        "frontend.hallucinated_node_rate": _wilson(
            sum(item.hallucinated_nodes for item in frontends),
            predicted_nodes,
        ),
        "frontend.ambiguity_preservation_rate": _wilson(
            sum(item.ambiguities_preserved for item in frontends),
            sum(item.ambiguities_gold for item in frontends),
        ),
        "frontend.source_span_exact_rate": _wilson(
            sum(item.source_spans_exact for item in frontends),
            sum(item.source_spans_gold for item in frontends),
        ),
        "frontend.source_span_overlap_rate": _wilson(
            sum(item.source_spans_overlapping for item in frontends),
            sum(item.source_spans_gold for item in frontends),
        ),
        "realizer.exact_symbolic_plan_rate": _wilson(
            sum(item.exact_symbolic_plan is True for item in realizers),
            len(realizers),
        ),
        "realizer.grammar_valid_rate": _wilson(
            sum(item.grammar_valid is True for item in realizers),
            len(realizers),
        ),
        "realizer.eos_completion_rate": _wilson(
            sum(item.eos_completed is True for item in realizers),
            len(realizers),
        ),
        "realizer.unauthorized_symbol_rate": _wilson(
            sum(item.unauthorized_symbol_count or 0 for item in realizers),
            predicted_symbols,
        ),
        "realizer.constraint_rejection_rate": _wilson(
            sum(item.constraint_rejected is True for item in realizers),
            len(realizers),
        ),
        "validator.accepted_rate": _wilson(
            sum(item.status == "accepted" for item in validators),
            len(validators),
        ),
        "validator.rejected_rate": _wilson(
            sum(item.status == "rejected" for item in validators),
            len(validators),
        ),
        "validator.false_accept_rate": _wilson(
            sum(item.status == "accepted" and not item.gold_should_accept for item in validators),
            sum(not item.gold_should_accept for item in validators),
        ),
        "provenance_coverage_rate": _wilson(
            sum(record.provenance_complete for record in records),
            len(records),
        ),
        "deterministic_repeatability_rate": _wilson(
            sum(record.deterministic_repeat for record in records),
            len(records),
        ),
    }
    if tuple(metric_values) != SUPPORTED_METRICS:
        raise RuntimeError("computed metric order does not match the frozen benchmark inventory")
    failure_stages = Counter(record.failure_stage for record in records)
    failure_codes = Counter(
        record.failure_code for record in records if record.failure_code is not None
    )
    return SystemMetricsV1(
        record_count=len(records),
        metrics=metric_values,
        failure_stage_counts=dict(sorted(failure_stages.items())),
        failure_code_counts=dict(sorted(failure_codes.items())),
    )


def recompute_metrics(
    spec: BenchmarkSpecV1,
    records: tuple[PredictionRecordV1, ...],
    *,
    spec_sha256: str,
    prediction_manifest_sha256: str,
    predictions_sha256: str,
) -> BenchmarkMetricsV1:
    """Recompute every reported value independently for each frozen system."""

    systems = {
        system.system_id: _system_metrics(
            tuple(record for record in records if record.system_id == system.system_id)
        )
        for system in spec.systems
    }
    return BenchmarkMetricsV1(
        schema_version=METRICS_SCHEMA_VERSION,
        benchmark_id=spec.benchmark_id,
        evidence_label=spec.evidence_label,
        claim_scope=spec.claim_scope,
        benchmark_spec_sha256=spec_sha256,
        prediction_manifest_sha256=prediction_manifest_sha256,
        predictions_sha256=predictions_sha256,
        record_count=len(records),
        systems=systems,
    )


def _validate_release_cases(spec: BenchmarkSpecV1, release: Path) -> None:
    manifest_bytes = _read_regular(release / "manifest.json")
    if _sha256(manifest_bytes) != spec.dataset.manifest_sha256:
        raise ValueError("benchmark dataset manifest SHA-256 does not match the specification")
    try:
        manifest = json.loads(manifest_bytes)
        artifacts = {item["path"]: item for item in manifest["artifacts"]}
        selection = CorpusSelectionV1(
            dataset_version=spec.dataset.dataset_version,
            manifest_sha256=spec.dataset.manifest_sha256,
            train_sha256=artifacts["train.jsonl"]["sha256"],
            validation_sha256=artifacts["validation.jsonl"]["sha256"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"benchmark dataset manifest is invalid: {error}") from error
    snapshot = read_training_release(release, selection)
    released = {
        record.record_id: (split, record.source_sha256)
        for split in SPLITS
        for record in getattr(snapshot, split)
    }
    for case in spec.cases:
        if released.get(case.case_id) != (case.split, case.source_sha256):
            raise ValueError(f"benchmark case does not match the frozen dataset: {case.case_id}")


def _indented_markdown(value: str) -> list[str]:
    lines = value.splitlines() or [""]
    return ["    " + line for line in lines]


def _markdown(
    spec: BenchmarkSpecV1,
    metrics: BenchmarkMetricsV1,
    records: tuple[PredictionRecordV1, ...],
) -> bytes:
    if spec.evidence_label == "deterministic_fixture_only":
        evidence_notice = (
            "> Deterministic fixture evidence only. These values test the evidence pipeline; they "
            "are not model-quality measurements, certification evidence, or external benchmark "
            "results."
        )
        external_section = (
            "No external measured runs are included. The v1 generator fails closed for measured "
            "evidence until a future schema binds recomputable raw stage artifacts and an "
            "evaluator manifest."
        )
    else:
        evidence_notice = (
            "> External measured evidence, limited to the frozen benchmark and claim scope below. "
            "It is not certification evidence or a claim of standards compliance."
        )
        external_section = (
            "This report includes external measured systems cross-bound to the artifact-manifest "
            "SHA-256 identities in the benchmark specification."
        )
    lines = [
        "# Benchmark evidence report",
        "",
        evidence_notice,
        "",
        "Specification non-certification notice:",
        "",
        *_indented_markdown(spec.non_certification_notice),
        "",
        f"Benchmark: `{spec.benchmark_id}`",
        "",
        "Claim scope:",
        "",
        *_indented_markdown(spec.claim_scope),
        "",
    ]
    for system in spec.systems:
        system_metrics = metrics.systems[system.system_id]
        lines.extend(
            [
                f"## System `{system.system_id}`",
                "",
                "### Recomputed metrics",
                "",
                "| Metric | Value | Numerator | Denominator | 95% interval |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for name in SUPPORTED_METRICS:
            estimate = system_metrics.metrics[name]
            interval = estimate.confidence_interval
            value = "n/a" if estimate.value is None else f"{estimate.value:.6f}"
            interval_text = (
                "n/a (derived)"
                if interval.method == "none-derived"
                else (
                    "n/a"
                    if interval.lower is None
                    else f"[{interval.lower:.6f}, {interval.upper:.6f}]"
                )
            )
            lines.append(
                f"| `{name}` | {value} | "
                f"{'n/a' if estimate.numerator is None else f'{estimate.numerator:g}'} | "
                f"{'n/a' if estimate.denominator is None else f'{estimate.denominator:g}'} | "
                f"{interval_text} |"
            )
        lines.extend(
            [
                "",
                "### Failure taxonomy counts",
                "",
                "| Stage | Count |",
                "| --- | ---: |",
            ]
        )
        for stage, count in system_metrics.failure_stage_counts.items():
            lines.append(f"| `{stage}` | {count} |")
        lines.extend(
            [
                "",
                "| Failure code | Count |",
                "| --- | ---: |",
            ]
        )
        if system_metrics.failure_code_counts:
            for code, count in system_metrics.failure_code_counts.items():
                lines.append(f"| `{code}` | {count} |")
        else:
            lines.append("| _(none observed)_ | 0 |")
        lines.append("")
    failure_heading = (
        "## Uncensored deterministic failure fixtures"
        if spec.evidence_label == "deterministic_fixture_only"
        else "## Uncensored measured failure examples"
    )
    lines.extend([failure_heading, ""])
    for record in records:
        if record.failure_stage == "none":
            continue
        lines.extend(
            [
                (
                    f"### System `{record.system_id}` — case `{record.case_id}`"
                    f" — `{record.failure_code}`"
                ),
                "",
                f"System: `{record.system_id}`",
                "",
                f"Case: `{record.case_id}`",
                "",
                f"Stage: `{record.failure_stage}`",
                "",
                "Raw output:",
                "",
                *_indented_markdown(record.raw_output or "<no output>"),
                "",
                "Notes:",
                "",
                *_indented_markdown(record.notes),
                "",
            ]
        )
    lines.extend(
        [
            "## External measured runs",
            "",
            external_section,
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _rename_no_replace(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source_name)
    destination_bytes = os.fsencode(destination_name)
    if sys.platform == "darwin":
        try:
            rename = libc.renameatx_np
        except AttributeError as error:
            raise ValueError(
                "atomic no-replace benchmark report publication is unavailable"
            ) from error
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            0x00000004,
        )
    elif sys.platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError as error:
            raise ValueError(
                "atomic no-replace benchmark report publication is unavailable"
            ) from error
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            1,
        )
    else:
        raise ValueError(
            f"atomic no-replace benchmark report publication is unsupported on {sys.platform}"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ValueError(f"benchmark report output was created concurrently: {destination_name}")
    raise ValueError(f"cannot publish benchmark report atomically: {os.strerror(error_number)}")


def _write_stage_file(stage_descriptor: int, name: str, content: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
        dir_fd=stage_descriptor,
    )
    try:
        remaining = memoryview(content)
        while remaining:
            written = os.write(descriptor, remaining)
            if written == 0:
                raise OSError(errno.EIO, "zero-byte write while staging benchmark report")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_stage(parent_descriptor: int) -> tuple[str, int]:
    for _ in range(100):
        stage_name = f".ste-benchmark-report-{secrets.token_hex(16)}"
        try:
            os.mkdir(stage_name, 0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        try:
            stage_descriptor = os.open(
                stage_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_descriptor,
            )
        except BaseException:
            os.rmdir(stage_name, dir_fd=parent_descriptor)
            raise
        return stage_name, stage_descriptor
    raise ValueError("cannot allocate a private benchmark report staging directory")


def _cleanup_stage(
    parent_descriptor: int,
    stage_name: str,
    stage_descriptor: int,
    artifact_names: tuple[str, ...],
) -> None:
    for artifact_name in artifact_names:
        try:
            os.unlink(artifact_name, dir_fd=stage_descriptor)
        except FileNotFoundError:
            pass
    try:
        named_stage = os.stat(
            stage_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    opened_stage = os.fstat(stage_descriptor)
    if (
        stat.S_ISDIR(named_stage.st_mode)
        and named_stage.st_dev == opened_stage.st_dev
        and named_stage.st_ino == opened_stage.st_ino
    ):
        os.rmdir(stage_name, dir_fd=parent_descriptor)


def _publish_report(
    parent_descriptor: int,
    output_name: str,
    artifacts: tuple[tuple[str, bytes], ...],
) -> None:
    stage_name, stage_descriptor = _create_stage(parent_descriptor)
    published = False
    artifact_names = tuple(name for name, _ in artifacts)
    try:
        for name, content in artifacts:
            _write_stage_file(stage_descriptor, name, content)
        os.fsync(stage_descriptor)
        _rename_no_replace(parent_descriptor, stage_name, output_name)
        published = True
        os.fsync(parent_descriptor)
    except OSError as error:
        raise ValueError(f"cannot publish benchmark report atomically: {error}") from error
    finally:
        try:
            if not published:
                _cleanup_stage(
                    parent_descriptor,
                    stage_name,
                    stage_descriptor,
                    artifact_names,
                )
        finally:
            os.close(stage_descriptor)


def _require_hardened_publication_platform() -> None:
    if (
        os.name != "posix"
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_CLOEXEC")
        or not all(
            operation in os.supports_dir_fd
            for operation in (os.open, os.mkdir, os.stat, os.unlink, os.rmdir)
        )
        or os.stat not in os.supports_follow_symlinks
        or not (sys.platform == "darwin" or sys.platform.startswith("linux"))
    ):
        raise ValueError(
            "benchmark report publication requires hardened POSIX directory handles and "
            "atomic no-replace publication"
        )


def generate_evidence_report(
    *,
    specification_path: Path,
    taxonomy_path: Path,
    prediction_manifest_path: Path,
    predictions_path: Path,
    dataset_release: Path,
    output: Path,
    system_manifest_paths: tuple[Path, ...] = (),
) -> ReportManifestV1:
    """Validate all cross-bindings and write one deterministic evidence report."""

    if os.path.lexists(output):
        raise ValueError(f"benchmark report output path must not exist: {output}")
    _require_hardened_publication_platform()
    spec_model, spec_bytes = _model_file(specification_path, BenchmarkSpecV1)
    taxonomy_model, taxonomy_bytes = _model_file(taxonomy_path, FailureTaxonomyV1)
    prediction_manifest_model, prediction_manifest_bytes = _model_file(
        prediction_manifest_path,
        PredictionManifestV1,
    )
    assert isinstance(spec_model, BenchmarkSpecV1)
    assert isinstance(taxonomy_model, FailureTaxonomyV1)
    assert isinstance(prediction_manifest_model, PredictionManifestV1)
    if spec_model.evidence_label == "external_measured":
        raise ValueError(
            "external measured reports require a future schema that binds raw stage artifacts "
            "and an evaluator manifest; v1 accepts deterministic fixture evidence only"
        )
    spec_sha256 = _sha256(spec_bytes)
    taxonomy_sha256 = _sha256(taxonomy_bytes)
    prediction_manifest_sha256 = _sha256(prediction_manifest_bytes)
    predictions_bytes = _read_regular(predictions_path)
    predictions_sha256 = _sha256(predictions_bytes)

    if spec_model.failure_taxonomy_sha256 != taxonomy_sha256:
        raise ValueError("failure taxonomy SHA-256 does not match the benchmark specification")
    if prediction_manifest_model.benchmark_spec_sha256 != spec_sha256:
        raise ValueError("prediction manifest does not bind the benchmark specification")
    if prediction_manifest_model.benchmark_id != spec_model.benchmark_id:
        raise ValueError("prediction manifest benchmark ID does not match the specification")
    if prediction_manifest_model.dataset != spec_model.dataset:
        raise ValueError("prediction manifest dataset does not match the specification")
    if prediction_manifest_model.systems != spec_model.systems:
        raise ValueError("prediction manifest systems do not match the specification")
    expected_system_manifest_sha256s = tuple(
        sorted(
            system.artifact_manifest_sha256
            for system in spec_model.systems
            if system.artifact_manifest_sha256 is not None
        )
    )
    actual_system_manifest_sha256s = tuple(
        sorted(_sha256(_read_regular(path)) for path in system_manifest_paths)
    )
    if actual_system_manifest_sha256s != expected_system_manifest_sha256s:
        raise ValueError("system artifact manifests do not match the benchmark specification")
    if (
        prediction_manifest_model.predictions.path != predictions_path.name
        or prediction_manifest_model.predictions.sha256 != predictions_sha256
        or prediction_manifest_model.predictions.bytes != len(predictions_bytes)
    ):
        raise ValueError("prediction artifact identity does not match the prediction manifest")
    records = _prediction_lines(predictions_bytes)
    if prediction_manifest_model.record_count != len(records):
        raise ValueError("prediction count does not match the prediction manifest")

    taxonomy_codes = {item.code for item in taxonomy_model.codes}
    case_by_id = {case.case_id: case for case in spec_model.cases}
    system_by_id = {system.system_id: system for system in spec_model.systems}
    expected_keys = [
        (case.case_id, system.system_id)
        for case in spec_model.cases
        for system in spec_model.systems
    ]
    actual_keys = [(record.case_id, record.system_id) for record in records]
    if actual_keys != expected_keys:
        raise ValueError("predictions do not match the frozen case/system order")
    for record in records:
        case = case_by_id[record.case_id]
        system = system_by_id[record.system_id]
        if (
            record.benchmark_id != spec_model.benchmark_id
            or record.dataset != spec_model.dataset
            or record.source_sha256 != case.source_sha256
            or record.evidence_kind != system.evidence_kind
        ):
            raise ValueError(f"prediction cross-binding is invalid: {record.case_id}")
        if record.failure_code is not None and record.failure_code not in taxonomy_codes:
            raise ValueError(f"prediction uses an unknown failure code: {record.failure_code}")
    _validate_case_gold_contract(records)
    _validate_release_cases(spec_model, dataset_release)

    metrics = recompute_metrics(
        spec_model,
        records,
        spec_sha256=spec_sha256,
        prediction_manifest_sha256=prediction_manifest_sha256,
        predictions_sha256=predictions_sha256,
    )
    metrics_bytes = _canonical_json(metrics)
    report_bytes = _markdown(spec_model, metrics, records)
    artifacts = (
        FileIdentityV1(
            path="metrics.json",
            sha256=_sha256(metrics_bytes),
            bytes=len(metrics_bytes),
        ),
        FileIdentityV1(
            path="report.md",
            sha256=_sha256(report_bytes),
            bytes=len(report_bytes),
        ),
    )
    report_manifest = ReportManifestV1(
        schema_version=REPORT_MANIFEST_SCHEMA_VERSION,
        benchmark_id=spec_model.benchmark_id,
        evidence_label=spec_model.evidence_label,
        benchmark_spec_sha256=spec_sha256,
        failure_taxonomy_sha256=taxonomy_sha256,
        prediction_manifest_sha256=prediction_manifest_sha256,
        predictions_sha256=predictions_sha256,
        system_artifact_manifest_sha256s=expected_system_manifest_sha256s,
        artifacts=artifacts,
    )
    if output.name in {"", ".", ".."}:
        raise ValueError("benchmark report output must name a directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent_descriptor = os.open(
            output.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as error:
        raise ValueError(
            f"benchmark report parent must be a real directory: {output.parent}"
        ) from error
    try:
        if not stat.S_ISDIR(os.fstat(parent_descriptor).st_mode):
            raise ValueError(f"benchmark report parent must be a real directory: {output.parent}")
        try:
            os.stat(
                output.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise ValueError(f"benchmark report output path must not exist: {output}")
        _publish_report(
            parent_descriptor,
            output.name,
            (
                ("metrics.json", metrics_bytes),
                ("report.md", report_bytes),
                ("report-manifest.json", _canonical_json(report_manifest)),
            ),
        )
    finally:
        os.close(parent_descriptor)
    return report_manifest
