"""Validate vulnerability and license reports against the reviewed dependency policy."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import NoReturn

POLICY_SCHEMA = "ste-dependency-audit-policy-v1"
POLICY_KEYS = frozenset(
    {
        "schema_version",
        "allowed_license_expressions",
        "denied_license_markers",
        "license_exceptions",
        "vulnerability_suppressions",
    }
)
VULNERABILITY_SUPPRESSION_KEYS = frozenset(
    {"package", "version", "vulnerability_id", "reason", "expires"}
)
LICENSE_EXCEPTION_KEYS = frozenset({"package", "version", "license", "reason", "expires"})
AUDIT_REPORT_KEYS = frozenset({"dependencies", "fixes"})
AUDIT_DEPENDENCY_KEYS = frozenset({"name", "version", "vulns"})
AUDIT_VULNERABILITY_KEYS = frozenset({"id", "fix_versions", "aliases", "description"})
LICENSE_RECORD_KEYS = frozenset({"Name", "Version", "License"})
PROFILE_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
CANONICAL_NAME_PATTERN = re.compile(r"[-_.]+")


class DependencyPolicyError(ValueError):
    """An input cannot be interpreted safely."""


class DependencyPolicyViolation(ValueError):
    """A valid input violates reviewed policy."""


@dataclass(frozen=True)
class Suppression:
    package: str
    version: str
    subject: str
    reason: str
    expires: date


@dataclass(frozen=True)
class Dependency:
    name: str
    version: str
    vulnerabilities: tuple[str, ...]


@dataclass(frozen=True)
class LicenseRecord:
    name: str
    version: str
    license: str


@dataclass(frozen=True)
class Policy:
    allowed_licenses: frozenset[str]
    denied_markers: tuple[str, ...]
    license_exceptions: tuple[Suppression, ...]
    vulnerability_suppressions: tuple[Suppression, ...]


def _fail(message: str) -> NoReturn:
    raise DependencyPolicyError(message)


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DependencyPolicyError(f"cannot read JSON {path}: {error}") from error


def _object(value: object, *, location: str) -> dict[str, object]:
    if not isinstance(value, dict):
        _fail(f"{location} must be an object")
    if not all(isinstance(key, str) for key in value):
        _fail(f"{location} keys must be strings")
    return value


def _list(value: object, *, location: str) -> list[object]:
    if not isinstance(value, list):
        _fail(f"{location} must be an array")
    return value


def _string(value: object, *, location: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        _fail(f"{location} must be a nonblank, trimmed string")
    return value


def _exact_keys(value: dict[str, object], expected: frozenset[str], *, location: str) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        _fail(f"{location} keys are invalid; missing={missing!r}; unexpected={unexpected!r}")


def _canonical_name(
    value: object,
    *,
    location: str,
    require_canonical: bool = False,
) -> str:
    name = _string(value, location=location)
    canonical = CANONICAL_NAME_PATTERN.sub("-", name).lower()
    if not canonical:
        _fail(f"{location} must identify a package")
    if require_canonical and name != canonical:
        _fail(f"{location} must use its canonical lowercase package name {canonical!r}")
    return canonical


def _string_array(
    value: object,
    *,
    location: str,
    require_sorted: bool = False,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    items = tuple(
        _string(item, location=f"{location}[{index}]")
        for index, item in enumerate(_list(value, location=location))
    )
    if not allow_empty and not items:
        _fail(f"{location} must not be empty")
    if len(set(items)) != len(items):
        _fail(f"{location} must contain unique strings")
    if require_sorted and tuple(sorted(items)) != items:
        _fail(f"{location} must be sorted")
    return items


def _date(value: object, *, location: str) -> date:
    raw = _string(value, location=location)
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as error:
        raise DependencyPolicyError(f"{location} must be an ISO date") from error
    if parsed.isoformat() != raw:
        _fail(f"{location} must use canonical YYYY-MM-DD form")
    return parsed


def _suppressions(
    value: object,
    *,
    location: str,
    subject_key: str,
    expected_keys: frozenset[str],
) -> tuple[Suppression, ...]:
    result: list[Suppression] = []
    identities: set[tuple[str, str, str]] = set()
    for index, item in enumerate(_list(value, location=location)):
        item_location = f"{location}[{index}]"
        raw = _object(item, location=item_location)
        _exact_keys(raw, expected_keys, location=item_location)
        suppression = Suppression(
            package=_canonical_name(
                raw["package"],
                location=f"{item_location}.package",
                require_canonical=True,
            ),
            version=_string(raw["version"], location=f"{item_location}.version"),
            subject=_string(raw[subject_key], location=f"{item_location}.{subject_key}"),
            reason=_string(raw["reason"], location=f"{item_location}.reason"),
            expires=_date(raw["expires"], location=f"{item_location}.expires"),
        )
        identity = (suppression.package, suppression.version, suppression.subject)
        if identity in identities:
            _fail(f"{location} contains duplicate suppression {identity!r}")
        identities.add(identity)
        result.append(suppression)
    if tuple(sorted(result, key=lambda item: (item.package, item.version, item.subject))) != tuple(
        result
    ):
        _fail(f"{location} must be sorted by package, version, and subject")
    return tuple(result)


def _load_policy(path: Path) -> Policy:
    raw = _object(_load_json(path), location="policy")
    _exact_keys(raw, POLICY_KEYS, location="policy")
    if raw["schema_version"] != POLICY_SCHEMA:
        _fail(f"policy.schema_version must be {POLICY_SCHEMA!r}")
    allowed = frozenset(
        _string_array(
            raw["allowed_license_expressions"],
            location="policy.allowed_license_expressions",
            require_sorted=True,
            allow_empty=False,
        )
    )
    denied = _string_array(
        raw["denied_license_markers"],
        location="policy.denied_license_markers",
        require_sorted=True,
        allow_empty=False,
    )
    license_exceptions = _suppressions(
        raw["license_exceptions"],
        location="policy.license_exceptions",
        subject_key="license",
        expected_keys=LICENSE_EXCEPTION_KEYS,
    )
    for exception in license_exceptions:
        if exception.subject in allowed:
            _fail(
                "policy.license_exceptions must not duplicate an allowed license expression: "
                f"{exception.subject!r}"
            )
    return Policy(
        allowed_licenses=allowed,
        denied_markers=denied,
        license_exceptions=license_exceptions,
        vulnerability_suppressions=_suppressions(
            raw["vulnerability_suppressions"],
            location="policy.vulnerability_suppressions",
            subject_key="vulnerability_id",
            expected_keys=VULNERABILITY_SUPPRESSION_KEYS,
        ),
    )


def _load_audit_report(path: Path) -> tuple[Dependency, ...]:
    raw = _object(_load_json(path), location="audit report")
    _exact_keys(raw, AUDIT_REPORT_KEYS, location="audit report")
    if _list(raw["fixes"], location="audit report.fixes"):
        _fail("audit report.fixes must be empty because CI never runs fix mode")
    dependencies: list[Dependency] = []
    names: set[str] = set()
    for index, item in enumerate(_list(raw["dependencies"], location="audit report.dependencies")):
        location = f"audit report.dependencies[{index}]"
        dependency = _object(item, location=location)
        _exact_keys(dependency, AUDIT_DEPENDENCY_KEYS, location=location)
        name = _canonical_name(dependency["name"], location=f"{location}.name")
        if name in names:
            _fail(f"audit report contains duplicate package {name!r}")
        names.add(name)
        version = _string(dependency["version"], location=f"{location}.version")
        vulnerabilities: list[str] = []
        for vulnerability_index, vulnerability_item in enumerate(
            _list(dependency["vulns"], location=f"{location}.vulns")
        ):
            vulnerability_location = f"{location}.vulns[{vulnerability_index}]"
            vulnerability = _object(vulnerability_item, location=vulnerability_location)
            _exact_keys(
                vulnerability,
                AUDIT_VULNERABILITY_KEYS,
                location=vulnerability_location,
            )
            vulnerability_id = _string(
                vulnerability["id"],
                location=f"{vulnerability_location}.id",
            )
            _string_array(
                vulnerability["fix_versions"],
                location=f"{vulnerability_location}.fix_versions",
            )
            _string_array(
                vulnerability["aliases"],
                location=f"{vulnerability_location}.aliases",
            )
            _string(
                vulnerability["description"],
                location=f"{vulnerability_location}.description",
            )
            vulnerabilities.append(vulnerability_id)
        dependencies.append(Dependency(name, version, tuple(vulnerabilities)))
    if not dependencies:
        _fail("audit report must contain at least one dependency")
    return tuple(dependencies)


def _load_license_report(path: Path) -> tuple[LicenseRecord, ...]:
    records: list[LicenseRecord] = []
    names: set[str] = set()
    for index, item in enumerate(_list(_load_json(path), location="license report")):
        location = f"license report[{index}]"
        raw = _object(item, location=location)
        _exact_keys(raw, LICENSE_RECORD_KEYS, location=location)
        name = _canonical_name(raw["Name"], location=f"{location}.Name")
        if name in names:
            _fail(f"license report contains duplicate package {name!r}")
        names.add(name)
        records.append(
            LicenseRecord(
                name=name,
                version=_string(raw["Version"], location=f"{location}.Version"),
                license=_string(raw["License"], location=f"{location}.License"),
            )
        )
    if not records:
        _fail("license report must contain at least one package")
    return tuple(records)


def _active_match(
    suppressions: tuple[Suppression, ...],
    *,
    package: str,
    version: str,
    subject: str,
    as_of: date,
) -> Suppression | None:
    for suppression in suppressions:
        if (
            suppression.package == package
            and suppression.version == version
            and suppression.subject == subject
        ):
            if suppression.expires < as_of:
                raise DependencyPolicyViolation(
                    f"suppression expired on {suppression.expires}: {package}=={version} {subject}"
                )
            return suppression
    return None


def _check_vulnerabilities(arguments: argparse.Namespace, policy: Policy) -> str:
    dependencies = _load_audit_report(arguments.report)
    vulnerability_count = sum(len(item.vulnerabilities) for item in dependencies)
    expected_scanner_exit = 1 if vulnerability_count else 0
    if arguments.scanner_exit_code != expected_scanner_exit:
        _fail(
            "scanner exit code does not match report findings: "
            f"exit={arguments.scanner_exit_code}; findings={vulnerability_count}"
        )
    used: set[Suppression] = set()
    violations: list[str] = []
    for dependency in dependencies:
        for vulnerability_id in dependency.vulnerabilities:
            try:
                suppression = _active_match(
                    policy.vulnerability_suppressions,
                    package=dependency.name,
                    version=dependency.version,
                    subject=vulnerability_id,
                    as_of=arguments.as_of,
                )
            except DependencyPolicyViolation as error:
                violations.append(str(error))
                continue
            if suppression is None:
                violations.append(
                    f"unsuppressed vulnerability: "
                    f"{dependency.name}=={dependency.version} {vulnerability_id}"
                )
            else:
                used.add(suppression)
    unused = set(policy.vulnerability_suppressions) - used
    violations.extend(
        f"unused vulnerability suppression: {item.package}=={item.version} {item.subject}"
        for item in sorted(unused, key=lambda item: (item.package, item.version, item.subject))
    )
    if violations:
        raise DependencyPolicyViolation("\n".join(violations))
    return (
        f"dependency vulnerability policy passed for {arguments.profile}: "
        f"{len(dependencies)} packages; {vulnerability_count} suppressed findings"
    )


def _check_licenses(arguments: argparse.Namespace, policy: Policy) -> str:
    expected_dependencies = {
        item.name: item.version for item in _load_audit_report(arguments.expected_audit_report)
    }
    licenses = _load_license_report(arguments.report)
    observed_dependencies = {item.name: item.version for item in licenses}
    if observed_dependencies != expected_dependencies:
        missing = sorted(
            f"{name}=={version}"
            for name, version in expected_dependencies.items()
            if observed_dependencies.get(name) != version
        )
        unexpected = sorted(
            f"{name}=={version}"
            for name, version in observed_dependencies.items()
            if expected_dependencies.get(name) != version
        )
        _fail(
            "license inventory does not match audited dependency inventory; "
            f"missing_or_wrong={missing!r}; unexpected_or_wrong={unexpected!r}"
        )
    used: set[Suppression] = set()
    violations: list[str] = []
    for record in licenses:
        try:
            exception = _active_match(
                policy.license_exceptions,
                package=record.name,
                version=record.version,
                subject=record.license,
                as_of=arguments.as_of,
            )
        except DependencyPolicyViolation as error:
            violations.append(str(error))
            continue
        if exception is not None:
            used.add(exception)
            continue
        marker = next(
            (
                denied
                for denied in policy.denied_markers
                if denied.casefold() in record.license.casefold()
            ),
            None,
        )
        if marker is not None:
            violations.append(
                f"denied license marker {marker!r}: "
                f"{record.name}=={record.version} {record.license!r}"
            )
        elif record.license not in policy.allowed_licenses:
            violations.append(
                f"unreviewed license: {record.name}=={record.version} {record.license!r}"
            )
    unused = set(policy.license_exceptions) - used
    violations.extend(
        f"unused license exception: {item.package}=={item.version} {item.subject!r}"
        for item in sorted(unused, key=lambda item: (item.package, item.version, item.subject))
    )
    if violations:
        raise DependencyPolicyViolation("\n".join(violations))
    return (
        f"dependency license policy passed for {arguments.profile}: "
        f"{len(licenses)} packages; {len(used)} exceptions"
    )


def _as_of(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--as-of must be an ISO date") from error
    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError("--as-of must use canonical YYYY-MM-DD form")
    return parsed


def _profile(value: str) -> str:
    if PROFILE_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("profile must be a lowercase kebab-case identifier")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    vulnerability = subparsers.add_parser("vulnerabilities")
    vulnerability.add_argument("--policy", type=Path, required=True)
    vulnerability.add_argument("--report", type=Path, required=True)
    vulnerability.add_argument("--scanner-exit-code", type=int, required=True)
    vulnerability.add_argument("--profile", type=_profile, required=True)
    vulnerability.add_argument("--as-of", type=_as_of, required=True)
    license_parser = subparsers.add_parser("licenses")
    license_parser.add_argument("--policy", type=Path, required=True)
    license_parser.add_argument("--report", type=Path, required=True)
    license_parser.add_argument("--expected-audit-report", type=Path, required=True)
    license_parser.add_argument("--profile", type=_profile, required=True)
    license_parser.add_argument("--as-of", type=_as_of, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        policy = _load_policy(arguments.policy)
        if arguments.command == "vulnerabilities":
            result = _check_vulnerabilities(arguments, policy)
        else:
            result = _check_licenses(arguments, policy)
    except DependencyPolicyViolation as error:
        print(f"dependency policy violation: {error}", file=sys.stderr)
        return 1
    except DependencyPolicyError as error:
        print(f"dependency policy error: {error}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
