import hashlib
import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from ste_compiler.evaluation import (
    BenchmarkMetricsV1,
    BenchmarkSpecV1,
    FailureTaxonomyV1,
    MetricEstimateV1,
    PredictionRecordV1,
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

    with pytest.raises(ValidationError, match="String should match pattern"):
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
    payload = json.loads((EXPECTED / "metrics.json").read_text())
    system = next(iter(payload["systems"].values()))
    system["failure_stage_counts"] = {"none": system["record_count"]}
    system["failure_code_counts"] = {}
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

    metrics = BenchmarkMetricsV1.model_validate(payload)

    actual = next(iter(metrics.systems.values()))
    assert actual.failure_stage_counts.get("frontend", 0) == 0
    assert sum(actual.failure_code_counts.values()) == 0


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
        ((2, 4), (3, 4), 0.6),
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

    with pytest.raises(
        ValidationError,
        match="standalone constraint rejection requires",
    ):
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
    ("REQUIRED_NODE_OMITTED", "UNAUTHORIZED_WORD", "SENTENCE_TOO_LONG"),
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

    with pytest.raises(ValueError, match="prediction uses an unknown failure code"):
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

    def create_destination_then_rename(source, destination):
        destination.mkdir()
        real_rename(source, destination)

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

    def fail_rename(source, destination):
        raise OSError("injected atomic publication failure")

    monkeypatch.setattr(evidence_module, "_rename_no_replace", fail_rename)

    with pytest.raises(ValueError, match="cannot publish benchmark report atomically"):
        _generate(output)

    assert not output.exists()
    assert not list(tmp_path.glob(".ste-benchmark-report-*"))
