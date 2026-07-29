import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from ste_compiler.evaluation import (
    BenchmarkMetricsV1,
    BenchmarkSpecV1,
    FailureTaxonomyV1,
    MetricEstimateV1,
    PredictionRecordV1,
    ReportManifestV1,
    generate_evidence_report,
)
from ste_compiler.evaluation import evidence as evidence_module

ROOT = Path(__file__).parents[2]
FIXTURE = ROOT / "data/benchmark/v1"
RELEASE = ROOT / "datasets/demonstration-corpus-2"
SPECIFICATION = FIXTURE / "benchmark-spec.json"
TAXONOMY = FIXTURE / "failure-taxonomy.json"
PREDICTION_MANIFEST = FIXTURE / "prediction-manifest.json"
PREDICTIONS = FIXTURE / "predictions.jsonl"
EXPECTED = FIXTURE / "expected-report"


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _generate(output: Path):
    return generate_evidence_report(
        specification_path=SPECIFICATION,
        taxonomy_path=TAXONOMY,
        prediction_manifest_path=PREDICTION_MANIFEST,
        predictions_path=PREDICTIONS,
        dataset_release=RELEASE,
        output=output,
    )


def _replace_metric_counts(system, name: str, numerator: int, denominator: int) -> None:
    system["metrics"][name] = evidence_module._wilson(numerator, denominator).model_dump(
        mode="json"
    )


def _resign_predictions(tmp_path: Path, mutate) -> tuple[Path, Path]:
    predictions = [json.loads(line) for line in PREDICTIONS.read_text().splitlines()]
    mutate(predictions)
    prediction_bytes = b"".join(
        (
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode()
        for record in predictions
    )
    prediction_path = tmp_path / "predictions.jsonl"
    prediction_path.write_bytes(prediction_bytes)
    manifest = json.loads(PREDICTION_MANIFEST.read_text())
    manifest["predictions"] = {
        "path": prediction_path.name,
        "sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "bytes": len(prediction_bytes),
    }
    manifest_path = tmp_path / "prediction-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, prediction_path


@pytest.mark.parametrize(
    ("model", "payload", "path", "coerced_value"),
    [
        (
            BenchmarkSpecV1,
            json.loads(SPECIFICATION.read_text()),
            ("seed",),
            "1729",
        ),
        (
            PredictionRecordV1,
            json.loads(PREDICTIONS.read_text().splitlines()[0]),
            ("frontend", "schema_valid"),
            "yes",
        ),
        (
            BenchmarkMetricsV1,
            json.loads((EXPECTED / "metrics.json").read_text()),
            ("record_count",),
            "4",
        ),
    ],
)
def test_evidence_models_reject_coercible_noncanonical_json_primitives(
    model,
    payload,
    path,
    coerced_value,
):
    target = payload
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = coerced_value

    with pytest.raises(ValidationError):
        model.model_validate_json(json.dumps(payload))


def test_fixture_report_reconstructs_byte_for_byte(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = _generate(first)
    second_manifest = _generate(second)

    assert first_manifest == second_manifest
    assert first_manifest.evidence_label == "deterministic_fixture_only"
    assert _files(first) == _files(second) == _files(EXPECTED)
    metrics = json.loads((first / "metrics.json").read_text())
    assert metrics["record_count"] == 4
    system = metrics["systems"]["failure-taxonomy-fixture-v1"]
    assert system["record_count"] == 4
    assert system["failure_stage_counts"] == {
        "frontend": 1,
        "none": 1,
        "realizer": 1,
        "validator": 1,
    }
    assert system["failure_code_counts"] == {
        "frontend.schema_invalid": 1,
        "realizer.unauthorized_symbol": 1,
        "validator.semantic_rejection": 1,
    }
    assert system["metrics"]["complete_success_rate"]["numerator"] == 1
    assert system["metrics"]["complete_success_rate"]["denominator"] == 4
    assert system["metrics"]["complete_success_rate"]["value"] == 0.25
    report = (first / "report.md").read_text()
    assert "Deterministic fixture evidence only" in report
    assert "not model-quality measurements" in report
    assert (
        "Specification non-certification notice:\n\n"
        "    This fixture is not model-quality evidence, external benchmark evidence, "
        "certification evidence, or a claim of ASD-STE100 compliance."
    ) in report
    assert "No external measured runs are included" in report
    assert (
        "| Failure code | Count |\n"
        "| --- | ---: |\n"
        "| `frontend.schema_invalid` | 1 |\n"
        "| `realizer.unauthorized_symbol` | 1 |\n"
        "| `validator.semantic_rejection` | 1 |"
    ) in report


def test_golden_report_manifest_satisfies_standalone_schema():
    payload = json.loads((EXPECTED / "report-manifest.json").read_text())

    manifest = ReportManifestV1.model_validate(payload)

    assert manifest.evidence_label == "deterministic_fixture_only"
    assert manifest.system_artifact_manifest_sha256s == ()
    assert tuple(artifact.path for artifact in manifest.artifacts) == (
        "metrics.json",
        "report.md",
    )


def _taxonomy_payload() -> dict:
    return json.loads(TAXONOMY.read_text())


def test_failure_taxonomy_fixture_matches_frozen_v1_inventory():
    taxonomy = FailureTaxonomyV1.model_validate(_taxonomy_payload())

    assert tuple((item.code, item.stage) for item in taxonomy.codes) == (
        evidence_module.FAILURE_TAXONOMY_V1
    )
    assert get_args(evidence_module.FailureCode) == tuple(
        code for code, _stage in evidence_module.FAILURE_TAXONOMY_V1
    )


@pytest.mark.parametrize(
    "index",
    range(len(evidence_module.FAILURE_TAXONOMY_V1)),
)
def test_failure_taxonomy_rejects_each_possible_omission(index):
    payload = _taxonomy_payload()
    payload["codes"].pop(index)

    with pytest.raises(
        ValidationError,
        match="failure taxonomy must equal the frozen v1 code, order, and stage inventory",
    ):
        FailureTaxonomyV1.model_validate(payload)


@pytest.mark.parametrize(
    "index",
    range(len(evidence_module.FAILURE_TAXONOMY_V1) - 1),
)
def test_failure_taxonomy_rejects_each_adjacent_reordering(index):
    payload = _taxonomy_payload()
    payload["codes"][index], payload["codes"][index + 1] = (
        payload["codes"][index + 1],
        payload["codes"][index],
    )

    with pytest.raises(
        ValidationError,
        match="failure taxonomy must equal the frozen v1 code, order, and stage inventory",
    ):
        FailureTaxonomyV1.model_validate(payload)


@pytest.mark.parametrize(
    ("code", "stage"),
    (
        ("frontend.custom_failure", "frontend"),
        ("realizer.custom_failure", "realizer"),
        ("validator.custom_failure", "validator"),
    ),
)
def test_failure_taxonomy_rejects_additions_in_each_stage(code, stage):
    payload = _taxonomy_payload()
    payload["codes"].append(
        {
            "code": code,
            "stage": stage,
            "description": "A custom extension that v1 does not permit.",
        }
    )

    with pytest.raises(
        ValidationError,
        match="failure taxonomy must equal the frozen v1 code, order, and stage inventory",
    ):
        FailureTaxonomyV1.model_validate(payload)


@pytest.mark.parametrize(
    "index",
    range(len(evidence_module.FAILURE_TAXONOMY_V1)),
)
def test_failure_taxonomy_rejects_each_incorrect_stage_mapping(index):
    payload = _taxonomy_payload()
    stages = ("frontend", "realizer", "validator")
    expected_stage = payload["codes"][index]["stage"]
    payload["codes"][index]["stage"] = next(stage for stage in stages if stage != expected_stage)

    with pytest.raises(ValidationError):
        FailureTaxonomyV1.model_validate(payload)


def test_report_manifest_rejects_malformed_system_artifact_hash():
    payload = json.loads((EXPECTED / "report-manifest.json").read_text())
    payload["evidence_label"] = "external_measured"
    payload["system_artifact_manifest_sha256s"] = ["not-a-sha256"]

    with pytest.raises(ValidationError, match="String should match pattern"):
        ReportManifestV1.model_validate(payload)


def test_fixture_report_manifest_rejects_system_artifact_inventory():
    payload = json.loads((EXPECTED / "report-manifest.json").read_text())
    payload["system_artifact_manifest_sha256s"] = ["0" * 64]

    with pytest.raises(
        ValidationError,
        match="deterministic fixture reports must not claim system artifact manifests",
    ):
        ReportManifestV1.model_validate(payload)


def test_external_report_manifest_requires_system_artifact_inventory():
    payload = json.loads((EXPECTED / "report-manifest.json").read_text())
    payload["evidence_label"] = "external_measured"

    with pytest.raises(
        ValidationError,
        match="external measured reports require system artifact manifest SHA-256s",
    ):
        ReportManifestV1.model_validate(payload)


def test_external_report_manifest_rejects_duplicate_system_artifact_hashes():
    payload = json.loads((EXPECTED / "report-manifest.json").read_text())
    payload["evidence_label"] = "external_measured"
    payload["system_artifact_manifest_sha256s"] = ["0" * 64, "0" * 64]

    with pytest.raises(
        ValidationError,
        match="system artifact manifest SHA-256s must be unique and in canonical order",
    ):
        ReportManifestV1.model_validate(payload)


def test_external_report_manifest_rejects_unsorted_system_artifact_hashes():
    payload = json.loads((EXPECTED / "report-manifest.json").read_text())
    payload["evidence_label"] = "external_measured"
    payload["system_artifact_manifest_sha256s"] = ["1" * 64, "0" * 64]

    with pytest.raises(
        ValidationError,
        match="system artifact manifest SHA-256s must be unique and in canonical order",
    ):
        ReportManifestV1.model_validate(payload)


def test_external_report_manifest_accepts_canonical_system_artifact_inventory():
    payload = json.loads((EXPECTED / "report-manifest.json").read_text())
    payload["evidence_label"] = "external_measured"
    payload["system_artifact_manifest_sha256s"] = ["0" * 64, "1" * 64]

    manifest = ReportManifestV1.model_validate(payload)

    assert manifest.system_artifact_manifest_sha256s == ("0" * 64, "1" * 64)


@pytest.mark.parametrize(
    "artifact_paths",
    [
        (),
        ("metrics.json",),
        ("report.md",),
        ("metrics.json", "metrics.json"),
        ("report.md", "report.md"),
        ("metrics.json", "report.md", "extra.json"),
        ("metrics.json", "arbitrary.md"),
        ("arbitrary.json", "report.md"),
        ("report.md", "metrics.json"),
    ],
    ids=[
        "empty",
        "missing-report",
        "missing-metrics",
        "duplicate-metrics",
        "duplicate-report",
        "additional-artifact",
        "arbitrary-second-path",
        "arbitrary-first-path",
        "noncanonical-order",
    ],
)
def test_report_manifest_schema_rejects_noncanonical_artifact_inventory(artifact_paths):
    payload = json.loads((EXPECTED / "report-manifest.json").read_text())
    identities_by_path = {artifact["path"]: artifact for artifact in payload["artifacts"]}
    payload["artifacts"] = [
        identities_by_path.get(
            path,
            {
                "path": path,
                "sha256": "0" * 64,
                "bytes": 0,
            },
        )
        for path in artifact_paths
    ]

    with pytest.raises(
        ValidationError,
        match=(
            "report manifest artifacts must be exactly metrics.json and report.md "
            "in canonical order|Tuple should have at (?:least|most)"
        ),
    ):
        ReportManifestV1.model_validate(payload)


@pytest.mark.parametrize(
    ("estimate", "message"),
    [
        (
            {
                "numerator": 2,
                "denominator": 1,
                "value": 0.25,
                "confidence_interval": {
                    "method": "wilson-score-95",
                    "lower": 0.9,
                    "upper": 0.1,
                },
            },
            "lower bound must not exceed",
        ),
        (
            {
                "numerator": 1,
                "denominator": 2,
                "value": 0.25,
                "confidence_interval": {
                    "method": "wilson-score-95",
                    "lower": 0.09453120573423068,
                    "upper": 0.9054687942657693,
                },
            },
            "value must equal numerator divided by denominator",
        ),
        (
            {
                "numerator": None,
                "denominator": None,
                "value": 0.5,
                "confidence_interval": {
                    "method": "wilson-score-95",
                    "lower": None,
                    "upper": None,
                },
            },
            "Wilson metrics require a numerator and denominator",
        ),
        (
            {
                "numerator": 1,
                "denominator": 2,
                "value": 0.5,
                "confidence_interval": {
                    "method": "none-derived",
                    "lower": None,
                    "upper": None,
                },
            },
            "derived metrics must not claim a numerator or denominator",
        ),
        (
            {
                "numerator": 0.5,
                "denominator": 1.5,
                "value": 1 / 3,
                "confidence_interval": {
                    "method": "wilson-score-95",
                    "lower": 0.1,
                    "upper": 0.9,
                },
            },
            "Input should be a valid integer",
        ),
    ],
)
def test_metric_schema_rejects_relationally_invalid_estimates(estimate, message):
    with pytest.raises(ValidationError, match=message):
        MetricEstimateV1.model_validate(estimate)


def test_metric_schema_rejects_tampered_wilson_bounds():
    estimate = evidence_module._wilson(1, 2).model_dump(mode="json")
    estimate["confidence_interval"]["lower"] += 0.01

    with pytest.raises(
        ValidationError,
        match="Wilson confidence bounds do not match the metric counts",
    ):
        MetricEstimateV1.model_validate(estimate)


def test_benchmark_metrics_schema_accepts_exact_frozen_metric_inventory():
    payload = json.loads((EXPECTED / "metrics.json").read_text())

    metrics = BenchmarkMetricsV1.model_validate(payload)

    system = next(iter(metrics.systems.values()))
    assert set(system.metrics) == set(evidence_module.SUPPORTED_METRICS)


def test_benchmark_metrics_schema_rejects_empty_system_inventory():
    payload = json.loads((EXPECTED / "metrics.json").read_text())
    payload["systems"] = {}

    with pytest.raises(ValidationError, match="Dictionary should have at least 1 item"):
        BenchmarkMetricsV1.model_validate(payload)


def test_benchmark_metrics_schema_rejects_unknown_failure_stage():
    payload = json.loads((EXPECTED / "metrics.json").read_text())
    system = next(iter(payload["systems"].values()))
    system["failure_stage_counts"]["postprocessor"] = 0

    with pytest.raises(ValidationError, match="Input should be"):
        BenchmarkMetricsV1.model_validate(payload)


def test_benchmark_metrics_schema_rejects_unknown_failure_code_namespace():
    payload = json.loads((EXPECTED / "metrics.json").read_text())
    system = next(iter(payload["systems"].values()))
    system["failure_code_counts"]["postprocessor.unknown_failure"] = 0

    with pytest.raises(ValidationError, match="Input should be"):
        BenchmarkMetricsV1.model_validate(payload)


@pytest.mark.parametrize(
    ("inventory_name", "key"),
    [
        ("failure_stage_counts", "frontend"),
        ("failure_code_counts", "frontend.schema_invalid"),
    ],
)
def test_benchmark_metrics_schema_rejects_negative_failure_counts(inventory_name, key):
    payload = json.loads((EXPECTED / "metrics.json").read_text())
    system = next(iter(payload["systems"].values()))
    system[inventory_name][key] = -1

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        BenchmarkMetricsV1.model_validate(payload)


def test_benchmark_metrics_schema_rejects_failure_stage_total_mismatch():
    payload = json.loads((EXPECTED / "metrics.json").read_text())
    system = next(iter(payload["systems"].values()))
    system["failure_stage_counts"]["frontend"] += 1

    with pytest.raises(
        ValidationError,
        match="failure stage counts must sum to the system record count",
    ):
        BenchmarkMetricsV1.model_validate(payload)


def test_benchmark_metrics_schema_rejects_failure_code_total_mismatch():
    payload = json.loads((EXPECTED / "metrics.json").read_text())
    system = next(iter(payload["systems"].values()))
    system["failure_code_counts"]["frontend.schema_invalid"] += 1

    with pytest.raises(
        ValidationError,
        match="failure code counts must equal the failed record count",
    ):
        BenchmarkMetricsV1.model_validate(payload)


def test_benchmark_metrics_schema_rejects_failure_code_stage_mismatch():
    payload = json.loads((EXPECTED / "metrics.json").read_text())
    system = next(iter(payload["systems"].values()))
    system["failure_code_counts"]["frontend.schema_invalid"] = 0
    system["failure_code_counts"]["validator.semantic_rejection"] = 2

    with pytest.raises(
        ValidationError,
        match="frontend failure code counts must equal its failure stage count",
    ):
        BenchmarkMetricsV1.model_validate(payload)


@pytest.mark.parametrize("explicit_zero_counts", (False, True))
def test_benchmark_metrics_schema_accepts_omitted_or_explicit_zero_failure_stages(
    explicit_zero_counts,
):
    successful_record = PredictionRecordV1.model_validate_json(
        PREDICTIONS.read_bytes().splitlines()[0]
    )
    system = evidence_module._system_metrics((successful_record,)).model_dump(mode="json")
    if explicit_zero_counts:
        system["failure_stage_counts"].update(
            {
                "frontend": 0,
                "realizer": 0,
                "validator": 0,
            }
        )
        system["failure_code_counts"].update(
            {
                "frontend.schema_invalid": 0,
                "realizer.unauthorized_symbol": 0,
                "validator.semantic_rejection": 0,
            }
        )

    metrics = evidence_module.SystemMetricsV1.model_validate(system)

    assert metrics.failure_stage_counts.get("frontend", 0) == 0
    assert sum(metrics.failure_code_counts.values()) == 0
    assert metrics.metrics["complete_success_rate"].numerator == 1


@pytest.mark.parametrize(
    ("metric_name", "numerator", "denominator", "message"),
    [
        ("complete_success_rate", 2, 4, "counts must equal the measured population result"),
        ("complete_success_rate", 1, 5, "counts must equal the measured population result"),
        (
            "frontend.schema_valid_rate",
            3,
            5,
            "denominator must equal its measured population",
        ),
        ("provenance_coverage_rate", 4, 5, "denominator must equal its measured population"),
        (
            "deterministic_repeatability_rate",
            4,
            5,
            "denominator must equal its measured population",
        ),
        (
            "realizer.exact_symbolic_plan_rate",
            1,
            4,
            "denominator must equal its measured population",
        ),
        (
            "realizer.grammar_valid_rate",
            2,
            4,
            "denominator must equal its measured population",
        ),
        (
            "realizer.eos_completion_rate",
            3,
            4,
            "denominator must equal its measured population",
        ),
        (
            "realizer.constraint_rejection_rate",
            1,
            4,
            "denominator must equal its measured population",
        ),
        (
            "validator.accepted_rate",
            1,
            3,
            "denominator must equal its measured population",
        ),
        (
            "validator.rejected_rate",
            1,
            3,
            "denominator must equal its measured population",
        ),
    ],
)
def test_system_metrics_schema_binds_record_and_stage_metric_populations(
    metric_name,
    numerator,
    denominator,
    message,
):
    payload = json.loads((EXPECTED / "metrics.json").read_text())
    system = next(iter(payload["systems"].values()))
    _replace_metric_counts(system, metric_name, numerator, denominator)

    with pytest.raises(ValidationError, match=message):
        evidence_module.SystemMetricsV1.model_validate(system)


@pytest.mark.parametrize(
    ("metric_name", "numerator", "denominator", "message"),
    [
        (
            "frontend.schema_valid_rate",
            2,
            4,
            "numerator must include every successful frontend",
        ),
        (
            "realizer.grammar_valid_rate",
            1,
            3,
            "numerator must include every successful realizer",
        ),
        (
            "realizer.eos_completion_rate",
            1,
            3,
            "numerator must include every successful realizer",
        ),
        (
            "realizer.constraint_rejection_rate",
            2,
            3,
            "numerator must not exceed realizer failures",
        ),
    ],
)
def test_system_metrics_schema_binds_stage_failure_numerator_bounds(
    metric_name,
    numerator,
    denominator,
    message,
):
    payload = json.loads((EXPECTED / "metrics.json").read_text())
    system = next(iter(payload["systems"].values()))
    _replace_metric_counts(system, metric_name, numerator, denominator)

    with pytest.raises(ValidationError, match=message):
        evidence_module.SystemMetricsV1.model_validate(system)


def test_system_metrics_schema_reconciles_validator_outcome_population():
    payload = json.loads((EXPECTED / "metrics.json").read_text())
    system = next(iter(payload["systems"].values()))
    _replace_metric_counts(system, "validator.accepted_rate", 0, 2)

    with pytest.raises(
        ValidationError,
        match="accepted and rejected numerators must sum to the validator population",
    ):
        evidence_module.SystemMetricsV1.model_validate(system)


def test_system_metrics_schema_bounds_false_accept_population():
    payload = json.loads((EXPECTED / "metrics.json").read_text())
    system = next(iter(payload["systems"].values()))
    _replace_metric_counts(system, "validator.false_accept_rate", 0, 3)

    with pytest.raises(
        ValidationError,
        match="false_accept_rate denominator must not exceed the validator population",
    ):
        evidence_module.SystemMetricsV1.model_validate(system)


def test_system_metrics_schema_bounds_false_accepts_by_accepted_failures():
    payload = json.loads((EXPECTED / "metrics.json").read_text())
    system = next(iter(payload["systems"].values()))
    _replace_metric_counts(system, "validator.accepted_rate", 2, 2)
    _replace_metric_counts(system, "validator.rejected_rate", 0, 2)
    _replace_metric_counts(system, "validator.false_accept_rate", 2, 2)

    with pytest.raises(
        ValidationError,
        match="false_accept_rate numerator must not exceed accepted validator failures",
    ):
        evidence_module.SystemMetricsV1.model_validate(system)


def test_system_metrics_schema_rejects_validator_outcome_swap_preserving_population():
    success = PredictionRecordV1.model_validate_json(PREDICTIONS.read_text().splitlines()[0])
    false_rejection = PredictionRecordV1.model_validate_json(
        PREDICTIONS.read_text().splitlines()[3]
    )
    correct_rejection_payload = json.loads(PREDICTIONS.read_text().splitlines()[3])
    correct_rejection_payload["validator"]["gold_should_accept"] = False
    correct_rejection_payload["failure_stage"] = "none"
    correct_rejection_payload["failure_code"] = None
    correct_rejection = PredictionRecordV1.model_validate(correct_rejection_payload)
    system = evidence_module._system_metrics(
        (success, false_rejection, correct_rejection)
    ).model_dump(mode="json")
    assert system["metrics"]["validator.accepted_rate"]["numerator"] == 1
    assert system["metrics"]["validator.rejected_rate"]["numerator"] == 2
    _replace_metric_counts(system, "validator.accepted_rate", 2, 3)
    _replace_metric_counts(system, "validator.rejected_rate", 1, 3)

    with pytest.raises(
        ValidationError,
        match="must reconcile with validator failures and false-accept counts",
    ):
        evidence_module.SystemMetricsV1.model_validate(system)


def test_system_metrics_schema_rejects_false_accept_tamper_preserving_population():
    payload = json.loads((EXPECTED / "metrics.json").read_text())
    system = next(iter(payload["systems"].values()))
    _replace_metric_counts(system, "validator.false_accept_rate", 1, 1)

    with pytest.raises(
        ValidationError,
        match="must reconcile with validator failures and false-accept counts",
    ):
        evidence_module.SystemMetricsV1.model_validate(system)


def test_system_metrics_schema_reconciles_a_correct_rejection_with_no_validator_failure():
    correct_rejection = json.loads(PREDICTIONS.read_text().splitlines()[3])
    correct_rejection["validator"]["gold_should_accept"] = False
    correct_rejection["failure_stage"] = "none"
    correct_rejection["failure_code"] = None

    system = evidence_module._system_metrics(
        (PredictionRecordV1.model_validate(correct_rejection),)
    )

    assert system.failure_stage_counts == {"none": 1}
    assert system.metrics["validator.accepted_rate"].numerator == 0
    assert system.metrics["validator.rejected_rate"].numerator == 1
    assert system.metrics["validator.false_accept_rate"].numerator == 0
    assert system.metrics["validator.false_accept_rate"].denominator == 1


def test_system_metrics_schema_reconciles_zero_validator_failures_and_gold_rejections():
    success = PredictionRecordV1.model_validate_json(PREDICTIONS.read_text().splitlines()[0])

    system = evidence_module._system_metrics((success,))

    assert system.failure_stage_counts == {"none": 1}
    assert system.metrics["validator.accepted_rate"].numerator == 1
    assert system.metrics["validator.rejected_rate"].numerator == 0
    assert system.metrics["validator.false_accept_rate"].numerator == 0
    assert system.metrics["validator.false_accept_rate"].denominator == 0


def test_system_metrics_schema_binds_exact_plan_to_success_measurements():
    payload = json.loads((EXPECTED / "metrics.json").read_text())
    system = next(iter(payload["systems"].values()))
    _replace_metric_counts(system, "realizer.exact_symbolic_plan_rate", 3, 3)

    with pytest.raises(
        ValidationError,
        match="exact_symbolic_plan_rate numerator must not exceed grammar-valid",
    ):
        evidence_module.SystemMetricsV1.model_validate(system)


def test_system_metrics_schema_binds_required_field_shared_numerator():
    payload = json.loads((EXPECTED / "metrics.json").read_text())
    system = next(iter(payload["systems"].values()))
    _replace_metric_counts(system, "frontend.required_field_precision", 13, 14)
    precision = evidence_module.MetricEstimateV1.model_validate(
        system["metrics"]["frontend.required_field_precision"]
    )
    recall = evidence_module.MetricEstimateV1.model_validate(
        system["metrics"]["frontend.required_field_recall"]
    )
    system["metrics"]["frontend.required_field_f1"] = evidence_module._derived_f1(
        precision, recall
    ).model_dump(mode="json")

    with pytest.raises(
        ValidationError,
        match="required-field precision and recall must share a numerator",
    ):
        evidence_module.SystemMetricsV1.model_validate(system)


@pytest.mark.parametrize(
    ("numerator", "denominator", "message"),
    [
        (3, 5, "source-span rates must share a denominator"),
        (2, 4, "exact source-span numerator must not exceed overlapping source spans"),
    ],
)
def test_system_metrics_schema_binds_source_span_counts(numerator, denominator, message):
    payload = json.loads((EXPECTED / "metrics.json").read_text())
    system = next(iter(payload["systems"].values()))
    _replace_metric_counts(
        system,
        "frontend.source_span_overlap_rate",
        numerator,
        denominator,
    )

    with pytest.raises(ValidationError, match=message):
        evidence_module.SystemMetricsV1.model_validate(system)


def test_benchmark_metrics_schema_rejects_aggregate_record_count_mismatch():
    payload = json.loads((EXPECTED / "metrics.json").read_text())
    payload["record_count"] += 1

    with pytest.raises(
        ValidationError,
        match="benchmark record count must equal the sum of system record counts",
    ):
        BenchmarkMetricsV1.model_validate(payload)


@pytest.mark.parametrize(
    ("metric_name", "replacement_name", "expected_method"),
    [
        (
            "complete_success_rate",
            "frontend.required_field_f1",
            "wilson-score-95",
        ),
        (
            "frontend.required_field_f1",
            "complete_success_rate",
            "none-derived",
        ),
    ],
)
def test_system_metrics_schema_binds_confidence_methods_to_metric_names(
    metric_name,
    replacement_name,
    expected_method,
):
    payload = json.loads((EXPECTED / "metrics.json").read_text())
    system = next(iter(payload["systems"].values()))
    system["metrics"][metric_name] = system["metrics"][replacement_name]

    with pytest.raises(
        ValidationError,
        match=rf"{metric_name} must use the {expected_method} confidence method",
    ):
        evidence_module.SystemMetricsV1.model_validate(system)


def test_system_metrics_schema_rejects_mismatched_derived_f1():
    payload = json.loads((EXPECTED / "metrics.json").read_text())
    system = next(iter(payload["systems"].values()))
    system["metrics"]["frontend.required_field_f1"]["value"] = 0.5

    with pytest.raises(
        ValidationError,
        match="must equal the harmonic mean of precision and recall",
    ):
        evidence_module.SystemMetricsV1.model_validate(system)


@pytest.mark.parametrize(
    ("precision", "recall", "expected_f1"),
    [
        ((0, 3), (0, 4), 0.0),
        ((0, 0), (0, 4), None),
        ((3, 6), (3, 4), 0.6),
    ],
)
def test_system_metrics_schema_accepts_recomputed_f1_for_zero_and_nonzero_cases(
    precision,
    recall,
    expected_f1,
):
    payload = json.loads((EXPECTED / "metrics.json").read_text())
    system = next(iter(payload["systems"].values()))
    precision_estimate = evidence_module._wilson(*precision)
    recall_estimate = evidence_module._wilson(*recall)
    f1_estimate = evidence_module._derived_f1(precision_estimate, recall_estimate)
    system["metrics"]["frontend.required_field_precision"] = precision_estimate.model_dump(
        mode="json"
    )
    system["metrics"]["frontend.required_field_recall"] = recall_estimate.model_dump(mode="json")
    system["metrics"]["frontend.required_field_f1"] = f1_estimate.model_dump(mode="json")

    metrics = evidence_module.SystemMetricsV1.model_validate(system)

    assert metrics.metrics["frontend.required_field_f1"].value == expected_f1


@pytest.mark.parametrize("mutation", ("missing", "additional", "empty"))
def test_benchmark_metrics_schema_rejects_non_frozen_metric_inventory(mutation):
    payload = json.loads((EXPECTED / "metrics.json").read_text())
    system_metrics = next(iter(payload["systems"].values()))["metrics"]
    if mutation == "missing":
        system_metrics.pop(evidence_module.SUPPORTED_METRICS[0])
    elif mutation == "additional":
        system_metrics["invented_metric"] = system_metrics[evidence_module.SUPPORTED_METRICS[0]]
    else:
        system_metrics.clear()

    with pytest.raises(
        ValidationError,
        match="system metrics must equal the frozen v1 metric inventory",
    ):
        BenchmarkMetricsV1.model_validate(payload)


def test_fixture_stage_observations_are_causally_distinct():
    records = [
        PredictionRecordV1.model_validate_json(line)
        for line in PREDICTIONS.read_bytes().splitlines()
    ]

    assert [record.failure_stage for record in records] == [
        "none",
        "frontend",
        "realizer",
        "validator",
    ]
    assert records[1].realizer.status == "not_run"
    assert records[1].validator.status == "not_run"
    assert records[2].realizer.status == "failed"
    assert records[2].validator.status == "not_run"
    assert records[3].realizer.status == "succeeded"
    assert records[3].validator.status == "rejected"


def test_prediction_schema_rejects_downstream_execution_after_frontend_failure():
    record = json.loads(PREDICTIONS.read_text().splitlines()[1])
    record["realizer"] = json.loads(PREDICTIONS.read_text().splitlines()[0])["realizer"]

    with pytest.raises(
        ValidationError,
        match="downstream stages must not run after a frontend failure",
    ):
        PredictionRecordV1.model_validate(record)


def test_schema_invalid_failure_code_requires_invalid_schema():
    record = json.loads(PREDICTIONS.read_text().splitlines()[1])
    record["frontend"]["schema_valid"] = True

    with pytest.raises(
        ValidationError,
        match="schema-invalid code requires an invalid frontend schema",
    ):
        PredictionRecordV1.model_validate(record)


@pytest.mark.parametrize(
    ("record_index", "failure_code"),
    (
        (1, "frontend.custom_failure"),
        (2, "realizer.custom_failure"),
        (3, "validator.custom_failure"),
    ),
)
def test_prediction_schema_rejects_custom_failure_codes_standalone(
    record_index,
    failure_code,
):
    record = json.loads(PREDICTIONS.read_text().splitlines()[record_index])
    record["failure_code"] = failure_code

    with pytest.raises(ValidationError, match="Input should be"):
        PredictionRecordV1.model_validate(record)


def _standalone_constraint_rejection() -> dict:
    record = json.loads(PREDICTIONS.read_text().splitlines()[2])
    record["realizer"].update(
        {
            "constraint_rejected": True,
            "eos_completed": True,
            "exact_symbolic_plan": False,
            "grammar_valid": True,
            "unauthorized_symbol_count": 0,
        }
    )
    record["failure_code"] = "realizer.constraint_rejection"
    return record


def test_standalone_constraint_rejection_has_a_frozen_failure_code():
    taxonomy = FailureTaxonomyV1.model_validate_json(TAXONOMY.read_bytes())
    record = PredictionRecordV1.model_validate(_standalone_constraint_rejection())

    assert "realizer.constraint_rejection" in {item.code for item in taxonomy.codes}
    assert record.failure_code == "realizer.constraint_rejection"
    assert record.realizer.constraint_rejected is True


def test_standalone_constraint_rejection_requires_its_specific_failure_code():
    record = _standalone_constraint_rejection()
    record["failure_code"] = "realizer.other_failure"

    with pytest.raises(ValidationError, match="Input should be"):
        PredictionRecordV1.model_validate(record)


def test_constraint_rejection_code_requires_standalone_constraint_observations():
    record = _standalone_constraint_rejection()
    record["realizer"]["unauthorized_symbol_count"] = 1

    with pytest.raises(
        ValidationError,
        match="constraint-rejection code requires a standalone",
    ):
        PredictionRecordV1.model_validate(record)


def test_prediction_schema_rejects_incomplete_success_path():
    record = json.loads(PREDICTIONS.read_text().splitlines()[0])
    record["realizer"] = json.loads(PREDICTIONS.read_text().splitlines()[1])["realizer"]
    record["validator"] = json.loads(PREDICTIONS.read_text().splitlines()[1])["validator"]

    with pytest.raises(
        ValidationError,
        match="realizer must run after a successful frontend",
    ):
        PredictionRecordV1.model_validate(record)


def test_complete_success_requires_frontend_recall_and_exact_spans():
    record = json.loads(PREDICTIONS.read_text().splitlines()[0])
    record["frontend"]["required_fields_matched"] = 0
    record["frontend"]["source_spans_exact"] = 0
    record["frontend"]["source_spans_overlapping"] = 0

    with pytest.raises(
        ValidationError,
        match="must satisfy every frozen pass boundary",
    ):
        PredictionRecordV1.model_validate(record)


def test_validator_failure_stage_distinguishes_correct_rejection_and_false_accept():
    rejected = json.loads(PREDICTIONS.read_text().splitlines()[3])
    rejected["validator"]["gold_should_accept"] = False
    rejected["failure_stage"] = "none"
    rejected["failure_code"] = None
    correct_rejection = PredictionRecordV1.model_validate(rejected)

    accepted = json.loads(PREDICTIONS.read_text().splitlines()[0])
    accepted["validator"]["gold_should_accept"] = False
    accepted["failure_stage"] = "validator"
    accepted["failure_code"] = "validator.false_accept"
    false_accept = PredictionRecordV1.model_validate(accepted)

    assert correct_rejection.failure_stage == "none"
    assert false_accept.failure_stage == "validator"


@pytest.mark.parametrize(
    "diagnostic_code",
    tuple(sorted(evidence_module.REJECTING_DIAGNOSTIC_CODES)),
)
def test_accepted_validator_observation_rejects_fatal_diagnostics(diagnostic_code):
    accepted = json.loads(PREDICTIONS.read_text().splitlines()[0])
    accepted["validator"]["diagnostic_codes"] = [diagnostic_code]

    with pytest.raises(
        ValidationError,
        match="accepted validator observation cannot contain rejecting diagnostics",
    ):
        PredictionRecordV1.model_validate(accepted)


@pytest.mark.parametrize("warning_code", ("AMBIGUOUS_PRONOUN", "PASSIVE_VOICE"))
def test_accepted_validator_observation_allows_warning_diagnostics(warning_code):
    accepted = json.loads(PREDICTIONS.read_text().splitlines()[0])
    accepted["validator"]["diagnostic_codes"] = [warning_code]

    prediction = PredictionRecordV1.model_validate(accepted)

    assert prediction.validator.status == "accepted"
    assert prediction.validator.diagnostic_codes == (warning_code,)


@pytest.mark.parametrize(
    "diagnostic_code",
    ("UNAUTHORIZED_BANANA", "LEXICAL_FABRICATED", "UNKNOWN_DIAGNOSTIC"),
)
def test_validator_observation_rejects_diagnostics_outside_frozen_pipeline(
    diagnostic_code,
):
    accepted = json.loads(PREDICTIONS.read_text().splitlines()[0])
    accepted["validator"]["diagnostic_codes"] = [diagnostic_code]

    with pytest.raises(
        ValidationError,
        match="diagnostic codes must come from the frozen validation pipeline",
    ):
        PredictionRecordV1.model_validate(accepted)


@pytest.mark.parametrize(
    "diagnostic_codes",
    ((), ("AMBIGUOUS_PRONOUN",), ("PASSIVE_VOICE",)),
)
def test_rejected_validator_observation_requires_a_rejecting_diagnostic(
    diagnostic_codes,
):
    rejected = json.loads(PREDICTIONS.read_text().splitlines()[3])
    rejected["validator"]["diagnostic_codes"] = list(diagnostic_codes)

    with pytest.raises(
        ValidationError,
        match="rejected validator observation requires a rejecting diagnostic",
    ):
        PredictionRecordV1.model_validate(rejected)


def test_false_accept_cannot_be_mislabeled_as_semantic_rejection():
    accepted = json.loads(PREDICTIONS.read_text().splitlines()[0])
    accepted["validator"]["gold_should_accept"] = False
    accepted["failure_stage"] = "validator"
    accepted["failure_code"] = "validator.semantic_rejection"

    with pytest.raises(
        ValidationError,
        match="false acceptance requires the validator.false_accept code",
    ):
        PredictionRecordV1.model_validate(accepted)


def test_structural_rejection_cannot_be_mislabeled_as_semantic():
    rejected = json.loads(PREDICTIONS.read_text().splitlines()[3])
    rejected["validator"]["diagnostic_codes"] = ["SENTENCE_TOO_LONG"]

    with pytest.raises(
        ValidationError,
        match="semantic-rejection code requires a semantic diagnostic",
    ):
        PredictionRecordV1.model_validate(rejected)

    rejected["failure_code"] = "validator.structural_rejection"
    prediction = PredictionRecordV1.model_validate(rejected)

    assert prediction.failure_code == "validator.structural_rejection"


def test_structural_rejection_code_requires_a_structural_diagnostic():
    rejected = json.loads(PREDICTIONS.read_text().splitlines()[3])
    rejected["failure_code"] = "validator.structural_rejection"

    with pytest.raises(
        ValidationError,
        match="structural-rejection code requires a structural diagnostic",
    ):
        PredictionRecordV1.model_validate(rejected)


@pytest.mark.parametrize("warning_code", ("AMBIGUOUS_PRONOUN", "PASSIVE_VOICE"))
def test_warning_only_structural_diagnostics_cannot_claim_rejection(warning_code):
    rejected = json.loads(PREDICTIONS.read_text().splitlines()[3])
    rejected["validator"]["diagnostic_codes"] = [warning_code]
    rejected["failure_code"] = "validator.structural_rejection"

    with pytest.raises(
        ValidationError,
        match="rejected validator observation requires a rejecting diagnostic",
    ):
        PredictionRecordV1.model_validate(rejected)


def test_external_measured_system_requires_artifact_manifest_identity():
    specification = json.loads(SPECIFICATION.read_text())
    specification["evidence_label"] = "external_measured"
    specification["systems"][0]["evidence_kind"] = "external_measured"

    with pytest.raises(
        ValidationError,
        match="external measured systems require an artifact manifest SHA-256",
    ):
        BenchmarkSpecV1.model_validate(specification)


def test_benchmark_spec_rejects_mixed_fixture_and_measured_systems():
    specification = json.loads(SPECIFICATION.read_text())
    measured = {
        **specification["systems"][0],
        "system_id": "external-system",
        "evidence_kind": "external_measured",
        "artifact_manifest_sha256": "0" * 64,
    }
    specification["systems"].append(measured)
    specification["evidence_label"] = "external_measured"

    with pytest.raises(
        ValidationError,
        match="must not mix fixture and measured evidence",
    ):
        BenchmarkSpecV1.model_validate(specification)


def test_report_generation_fails_closed_for_external_measured_attestations(tmp_path):
    specification = json.loads(SPECIFICATION.read_text())
    specification["evidence_label"] = "external_measured"
    specification["systems"][0]["evidence_kind"] = "external_measured"
    specification["systems"][0]["artifact_manifest_sha256"] = "0" * 64
    external_specification = tmp_path / "external-spec.json"
    external_specification.write_text(json.dumps(specification))

    with pytest.raises(
        ValueError,
        match="external measured reports require a future schema",
    ):
        generate_evidence_report(
            specification_path=external_specification,
            taxonomy_path=TAXONOMY,
            prediction_manifest_path=PREDICTION_MANIFEST,
            predictions_path=PREDICTIONS,
            dataset_release=RELEASE,
            output=tmp_path / "report",
        )


def test_benchmark_identifiers_reject_markdown_structure():
    specification = json.loads(SPECIFICATION.read_text())
    specification["benchmark_id"] = "benchmark\n## Fabricated result"

    with pytest.raises(ValidationError, match="String should match pattern"):
        BenchmarkSpecV1.model_validate(specification)


def test_report_rejects_prediction_byte_tampering(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_bytes(PREDICTIONS.read_bytes() + b" ")

    with pytest.raises(ValueError, match="prediction artifact identity does not match"):
        generate_evidence_report(
            specification_path=SPECIFICATION,
            taxonomy_path=TAXONOMY,
            prediction_manifest_path=PREDICTION_MANIFEST,
            predictions_path=predictions,
            dataset_release=RELEASE,
            output=tmp_path / "report",
        )


def test_report_rejects_symlinked_prediction_input(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    predictions.symlink_to(PREDICTIONS)

    with pytest.raises(ValueError, match="cannot open benchmark input"):
        generate_evidence_report(
            specification_path=SPECIFICATION,
            taxonomy_path=TAXONOMY,
            prediction_manifest_path=PREDICTION_MANIFEST,
            predictions_path=predictions,
            dataset_release=RELEASE,
            output=tmp_path / "report",
        )


def test_report_rejects_unknown_failure_code_even_when_predictions_are_resigned(tmp_path):
    manifest, predictions = _resign_predictions(
        tmp_path,
        lambda records: records[1].update(
            {
                "failure_code": "frontend.unknown_fixture_failure",
            }
        ),
    )

    with pytest.raises(ValueError, match="Input should be"):
        generate_evidence_report(
            specification_path=SPECIFICATION,
            taxonomy_path=TAXONOMY,
            prediction_manifest_path=manifest,
            predictions_path=predictions,
            dataset_release=RELEASE,
            output=tmp_path / "report",
        )


def test_report_renders_untrusted_notes_as_indented_code(tmp_path):
    manifest, predictions = _resign_predictions(
        tmp_path,
        lambda records: records[1].update(
            {
                "notes": "fixture note\n\n## Fabricated external result\n100%",
            }
        ),
    )
    output = tmp_path / "report"

    generate_evidence_report(
        specification_path=SPECIFICATION,
        taxonomy_path=TAXONOMY,
        prediction_manifest_path=manifest,
        predictions_path=predictions,
        dataset_release=RELEASE,
        output=output,
    )

    report = (output / "report.md").read_text()
    assert "\n## Fabricated external result" not in report
    assert "\n    ## Fabricated external result" in report


def test_report_renders_specification_notice_as_indented_code():
    specification = json.loads(SPECIFICATION.read_text())
    specification["non_certification_notice"] = (
        "Not certification evidence.\n\n"
        "## Fabricated certification\n"
        "> Certified by an external authority\n"
        "<script>alert('certified')</script>\n"
        "```markdown\n"
        "# Fabricated result\n"
        "```"
    )
    spec = BenchmarkSpecV1.model_validate(specification)
    records = tuple(
        PredictionRecordV1.model_validate_json(line)
        for line in PREDICTIONS.read_text().splitlines()
    )
    metrics = evidence_module.recompute_metrics(
        spec,
        records,
        spec_sha256="0" * 64,
        prediction_manifest_sha256="1" * 64,
        predictions_sha256="2" * 64,
    )

    report = evidence_module._markdown(spec, metrics, records).decode()

    assert "\n## Fabricated certification" not in report
    assert "\n> Certified by an external authority" not in report
    assert "\n<script>alert('certified')</script>" not in report
    assert "\n```markdown" not in report
    assert "\n# Fabricated result" not in report
    assert (
        "Specification non-certification notice:\n\n"
        "    Not certification evidence.\n"
        "    \n"
        "    ## Fabricated certification\n"
        "    > Certified by an external authority\n"
        "    <script>alert('certified')</script>\n"
        "    ```markdown\n"
        "    # Fabricated result\n"
        "    ```"
    ) in report


def test_failure_examples_identify_each_system_for_the_same_case():
    specification = json.loads(SPECIFICATION.read_text())
    second_system = {
        **specification["systems"][0],
        "system_id": "second-failure-fixture-v1",
    }
    specification["systems"].append(second_system)
    spec = BenchmarkSpecV1.model_validate(specification)

    first_failure = json.loads(PREDICTIONS.read_text().splitlines()[1])
    second_failure = {
        **first_failure,
        "system_id": second_system["system_id"],
    }
    records = tuple(
        PredictionRecordV1.model_validate(record) for record in (first_failure, second_failure)
    )
    metrics = evidence_module.recompute_metrics(
        spec,
        records,
        spec_sha256="0" * 64,
        prediction_manifest_sha256="1" * 64,
        predictions_sha256="2" * 64,
    )

    report = evidence_module._markdown(spec, metrics, records).decode()

    for system_id in (
        specification["systems"][0]["system_id"],
        second_system["system_id"],
    ):
        heading = (
            f"### System `{system_id}` — case `{first_failure['case_id']}`"
            f" — `{first_failure['failure_code']}`"
        )
        assert heading in report
        assert f"System: `{system_id}`" in report
    assert report.count(f"Case: `{first_failure['case_id']}`") == 2


def test_failure_code_tables_are_sorted_and_attributed_to_each_system():
    specification = json.loads(SPECIFICATION.read_text())
    first_system_id = specification["systems"][0]["system_id"]
    second_system_id = "second-failure-fixture-v1"
    specification["systems"].append(
        {
            **specification["systems"][0],
            "system_id": second_system_id,
        }
    )
    spec = BenchmarkSpecV1.model_validate(specification)
    predictions = [json.loads(line) for line in PREDICTIONS.read_text().splitlines()]
    first_system_records = (
        predictions[2],
        predictions[1],
    )
    second_system_records = tuple(
        {
            **predictions[index],
            "system_id": second_system_id,
        }
        for index in (3, 1)
    )
    records = tuple(
        PredictionRecordV1.model_validate(record)
        for record in (*first_system_records, *second_system_records)
    )
    metrics = evidence_module.recompute_metrics(
        spec,
        records,
        spec_sha256="0" * 64,
        prediction_manifest_sha256="1" * 64,
        predictions_sha256="2" * 64,
    )

    report = evidence_module._markdown(spec, metrics, records).decode()
    first_section = report.split(f"## System `{first_system_id}`", 1)[1].split(
        f"## System `{second_system_id}`",
        1,
    )[0]
    second_section = report.split(f"## System `{second_system_id}`", 1)[1].split(
        "## Uncensored deterministic failure fixtures",
        1,
    )[0]

    assert (
        "| Failure code | Count |\n"
        "| --- | ---: |\n"
        "| `frontend.schema_invalid` | 1 |\n"
        "| `realizer.unauthorized_symbol` | 1 |"
    ) in first_section
    assert "| `validator.semantic_rejection` |" not in first_section
    assert (
        "| Failure code | Count |\n"
        "| --- | ---: |\n"
        "| `frontend.schema_invalid` | 1 |\n"
        "| `validator.semantic_rejection` | 1 |"
    ) in second_section
    assert "| `realizer.unauthorized_symbol` |" not in second_section


def test_failure_code_table_explicitly_reports_an_empty_system():
    specification = json.loads(SPECIFICATION.read_text())
    spec = BenchmarkSpecV1.model_validate(specification)
    success = PredictionRecordV1.model_validate_json(PREDICTIONS.read_text().splitlines()[0])
    records = (success,)
    metrics = evidence_module.recompute_metrics(
        spec,
        records,
        spec_sha256="0" * 64,
        prediction_manifest_sha256="1" * 64,
        predictions_sha256="2" * 64,
    )

    report = evidence_module._markdown(spec, metrics, records).decode()

    assert ("| Failure code | Count |\n| --- | ---: |\n| _(none observed)_ | 0 |") in report


def test_report_rejects_dataset_manifest_mismatch(tmp_path):
    release = tmp_path / "release"
    shutil.copytree(RELEASE, release)
    manifest = release / "manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")

    with pytest.raises(ValueError, match="dataset manifest SHA-256 does not match"):
        generate_evidence_report(
            specification_path=SPECIFICATION,
            taxonomy_path=TAXONOMY,
            prediction_manifest_path=PREDICTION_MANIFEST,
            predictions_path=PREDICTIONS,
            dataset_release=release,
            output=tmp_path / "report",
        )


def test_fixture_report_rejects_unbound_system_manifest(tmp_path):
    system_manifest = tmp_path / "unbound-model-manifest.json"
    system_manifest.write_text("{}\n")

    with pytest.raises(ValueError, match="system artifact manifests do not match"):
        generate_evidence_report(
            specification_path=SPECIFICATION,
            taxonomy_path=TAXONOMY,
            prediction_manifest_path=PREDICTION_MANIFEST,
            predictions_path=PREDICTIONS,
            dataset_release=RELEASE,
            output=tmp_path / "report",
            system_manifest_paths=(system_manifest,),
        )


def test_report_refuses_existing_output_without_modifying_it(tmp_path):
    output = tmp_path / "report"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("preserve")

    with pytest.raises(ValueError, match="output path must not exist"):
        _generate(output)

    assert sentinel.read_text() == "preserve"
    assert set(output.iterdir()) == {sentinel}


def test_report_refuses_existing_empty_output_directory(tmp_path):
    output = tmp_path / "report"
    output.mkdir()

    with pytest.raises(ValueError, match="output path must not exist"):
        _generate(output)

    assert output.is_dir()
    assert not list(output.iterdir())
    assert not list(tmp_path.glob(".ste-benchmark-report-*"))


def test_report_refuses_output_directory_created_during_publication(tmp_path, monkeypatch):
    output = tmp_path / "report"
    real_rename = evidence_module._rename_no_replace

    def create_destination_then_rename(parent_descriptor, source_name, destination_name):
        os.mkdir(destination_name, dir_fd=parent_descriptor)
        real_rename(parent_descriptor, source_name, destination_name)

    monkeypatch.setattr(
        evidence_module,
        "_rename_no_replace",
        create_destination_then_rename,
    )

    with pytest.raises(ValueError, match="output was created concurrently"):
        _generate(output)

    assert output.is_dir()
    assert not list(output.iterdir())
    assert not list(tmp_path.glob(".ste-benchmark-report-*"))


def test_report_publication_failure_leaves_no_partial_output(tmp_path, monkeypatch):
    output = tmp_path / "report"

    def fail_rename(parent_descriptor, source_name, destination_name):
        raise OSError("injected atomic publication failure")

    monkeypatch.setattr(evidence_module, "_rename_no_replace", fail_rename)

    with pytest.raises(ValueError, match="cannot publish benchmark report atomically"):
        _generate(output)

    assert not output.exists()
    assert not list(tmp_path.glob(".ste-benchmark-report-*"))


def test_report_refuses_symlink_parent_without_writing_target(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    parent = tmp_path / "parent"
    parent.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="parent must be a real directory"):
        _generate(parent / "report")

    assert not list(target.iterdir())


def test_report_publication_fails_closed_on_unsupported_platform(monkeypatch):
    monkeypatch.setattr(evidence_module.sys, "platform", "freebsd")

    with pytest.raises(ValueError, match="requires hardened POSIX directory handles"):
        evidence_module._require_hardened_publication_platform()


def test_report_publication_stays_in_verified_parent_after_path_swap(
    tmp_path,
    monkeypatch,
):
    parent = tmp_path / "trusted"
    parent.mkdir()
    relocated_parent = tmp_path / "relocated-trusted"
    attacker_parent = tmp_path / "attacker"
    attacker_parent.mkdir()
    output = parent / "report"
    real_rename = evidence_module._rename_no_replace

    def swap_parent_then_rename(parent_descriptor, source_name, destination_name):
        parent.rename(relocated_parent)
        parent.symlink_to(attacker_parent, target_is_directory=True)
        real_rename(parent_descriptor, source_name, destination_name)

    monkeypatch.setattr(
        evidence_module,
        "_rename_no_replace",
        swap_parent_then_rename,
    )

    _generate(output)

    assert _files(relocated_parent / "report") == _files(EXPECTED)
    assert not (attacker_parent / "report").exists()
    assert not list(relocated_parent.glob(".ste-benchmark-report-*"))
    assert not list(attacker_parent.glob(".ste-benchmark-report-*"))


def test_concurrent_output_and_parent_swap_clean_up_only_verified_parent(
    tmp_path,
    monkeypatch,
):
    parent = tmp_path / "trusted"
    parent.mkdir()
    relocated_parent = tmp_path / "relocated-trusted"
    attacker_parent = tmp_path / "attacker"
    attacker_parent.mkdir()
    output = parent / "report"
    real_rename = evidence_module._rename_no_replace

    def swap_parent_create_output_then_rename(
        parent_descriptor,
        source_name,
        destination_name,
    ):
        parent.rename(relocated_parent)
        parent.symlink_to(attacker_parent, target_is_directory=True)
        os.mkdir(destination_name, dir_fd=parent_descriptor)
        real_rename(parent_descriptor, source_name, destination_name)

    monkeypatch.setattr(
        evidence_module,
        "_rename_no_replace",
        swap_parent_create_output_then_rename,
    )

    with pytest.raises(ValueError, match="output was created concurrently"):
        _generate(output)

    assert (relocated_parent / "report").is_dir()
    assert not list((relocated_parent / "report").iterdir())
    assert not (attacker_parent / "report").exists()
    assert not list(relocated_parent.glob(".ste-benchmark-report-*"))
    assert not list(attacker_parent.glob(".ste-benchmark-report-*"))
