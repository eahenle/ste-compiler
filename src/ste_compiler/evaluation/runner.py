from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import statistics
import re
import yaml
from ste_compiler.ir.serialization import load_document
from ste_compiler.realizer import DeterministicRealizer
from ste_compiler.realizer.base import RealizationResult
from ste_compiler.terminology import TerminologyRegistry, Vocabulary
from ste_compiler.validators.lexical import LexicalValidator
from ste_compiler.validators.structural import StructuralValidator
from ste_compiler.validators.semantic import SemanticValidator


@dataclass
class Metrics:
    vocabulary_compliance_rate: float
    structural_rule_pass_rate: float
    required_field_coverage: float
    negation_preservation: float
    quantity_preservation: float
    temporal_order_preservation: float
    unauthorized_term_rate: float
    average_sentence_length: float
    rejection_rate: float
    determinism: float
    human_clarity_review: None = None
    human_fidelity_review: None = None


def evaluate(
    corpus: Path, vocabulary: Vocabulary, terminology: TerminologyRegistry
) -> dict[str, Metrics]:
    cases = yaml.safe_load((corpus / "baselines.yaml").read_text())["cases"]
    systems = ["direct", "prompted", "deterministic"]
    accum: dict[str, list[dict[str, float]]] = {s: [] for s in systems}
    lexical, structural = LexicalValidator(vocabulary, terminology), StructuralValidator()
    for case in cases:
        doc = load_document(corpus / case["ir"])
        generated = DeterministicRealizer().realize(doc, vocabulary, terminology)
        for system in systems:
            result = (
                generated if system == "deterministic" else RealizationResult(case[system], (), {})
            )
            ld, sd = lexical.validate(result.text), structural.validate(result.text)
            sem = SemanticValidator().validate(doc, result)
            words = re.findall(r"\b[\w-]+\b", result.text)
            codes = {x.code for x in sem}
            errors = ld + sd + sem
            accum[system].append(
                {
                    "vocab": 1.0 if not ld else 0.0,
                    "structure": 1.0 if not sd else 0.0,
                    "coverage": 0.0 if "REQUIRED_NODE_OMITTED" in codes else 1.0,
                    "negation": 0.0
                    if "NEGATION_NOT_PRESERVED" in codes or not result.mappings
                    else 1.0,
                    "quantity": 0.0
                    if "QUANTITY_NOT_PRESERVED" in codes or not result.mappings
                    else 1.0,
                    "temporal": 0.0
                    if "TEMPORAL_RELATION_NOT_PRESERVED" in codes or not result.mappings
                    else 1.0,
                    "unauthorized": len(ld) / max(len(words), 1),
                    "length": len(words),
                    "rejected": 1.0 if errors else 0.0,
                    "determinism": 1.0
                    if system == "deterministic"
                    and result.text
                    == DeterministicRealizer().realize(doc, vocabulary, terminology).text
                    else 0.0,
                }
            )
    return {
        system: Metrics(
            vocabulary_compliance_rate=statistics.mean(x["vocab"] for x in rows),
            structural_rule_pass_rate=statistics.mean(x["structure"] for x in rows),
            required_field_coverage=statistics.mean(x["coverage"] for x in rows),
            negation_preservation=statistics.mean(x["negation"] for x in rows),
            quantity_preservation=statistics.mean(x["quantity"] for x in rows),
            temporal_order_preservation=statistics.mean(x["temporal"] for x in rows),
            unauthorized_term_rate=statistics.mean(x["unauthorized"] for x in rows),
            average_sentence_length=statistics.mean(x["length"] for x in rows),
            rejection_rate=statistics.mean(x["rejected"] for x in rows),
            determinism=statistics.mean(x["determinism"] for x in rows),
        )
        for system, rows in accum.items()
    }


def write_reports(results: dict[str, Metrics], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    raw = {k: asdict(v) for k, v in results.items()}
    (output / "report.json").write_text(json.dumps(raw, indent=2) + "\n")
    headers = list(next(iter(raw.values())))
    rows = ["| System | " + " | ".join(headers) + " |", "|---|" + "---|" * len(headers)]
    rows += [
        f"| {system} | " + " | ".join(str(values[h]) for h in headers) + " |"
        for system, values in raw.items()
    ]
    (output / "report.md").write_text("# Evaluation report\n\n" + "\n".join(rows) + "\n")
