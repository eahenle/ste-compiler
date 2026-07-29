"""Deterministic offline CPU smoke training for encoder-decoder models."""

from __future__ import annotations

import ctypes
import errno
import fnmatch
import hashlib
import importlib
import importlib.metadata
import inspect
import math
import os
import platform
import random
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Final, cast

from ste_compiler.artifacts import (
    ARTIFACT_MANIFEST_NAME,
    MAX_ARTIFACT_FILES,
    MAX_ARTIFACT_MANIFEST_BYTES,
    MAX_ARTIFACT_PATH_DEPTH,
    ArtifactBundleManifestV1,
    ArtifactFileV1,
    ArtifactVerificationError,
    VerifiedArtifactBundle,
    artifact_manifest_sha256,
    build_artifact_manifest,
    canonical_artifact_manifest_json,
    open_verified_artifact_bundle,
    parse_canonical_artifact_manifest,
    verify_artifact_bundle,
)

from .config import (
    ArtifactIdentityV1,
    EncoderDecoderTrainingConfigV1,
    canonical_training_config_json,
    training_config_sha256,
)
from .manifest import (
    RUN_MANIFEST_SCHEMA_VERSION,
    VALIDATION_SCHEMA_VERSION,
    CorpusRunIdentityV1,
    EncoderDecoderRunManifestV1,
    FileIdentityV1,
    HardwareProvenanceV1,
    PackageProvenanceV1,
    ParameterCountsV1,
    ValidationMetricsV1,
    canonical_run_manifest_json,
    canonical_validation_metrics_json,
)
from .release_reader import (
    ReleasedTrainingRecordV1,
    TrainingReleaseSnapshot,
    read_training_release,
)

_SAFE_SNAPSHOT_PATTERNS = [
    "*.codes",
    "*.json",
    "*.merges",
    "*.model",
    "*.safetensors",
    "*.spm",
    "*.tiktoken",
    "*.tokenizer",
    "*.txt",
    "*.vocab",
]
_PROHIBITED_MODEL_SUFFIXES = frozenset({".bin", ".ckpt", ".pickle", ".pkl", ".pt", ".pth"})
_REQUIRED_DEPENDENCIES = frozenset(
    {
        "huggingface-hub",
        "numpy",
        "safetensors",
        "ste-compiler",
        "tokenizers",
        "torch",
        "transformers",
    }
)
_EVALUATION_COMMAND: Final = (
    "ste-compiler",
    "evaluate-encoder-decoder-checkpoint",
    "<training-config>",
    "<corpus-release>",
    "<checkpoint>",
    "--run-manifest-sha256",
    "<run-manifest-sha256>",
    "--json",
)
_MAX_TREE_FILE_BYTES = 8 * 1024 * 1024 * 1024
_MAX_TREE_BYTES = 32 * 1024 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_PACKAGE_SUFFIXES = frozenset({".py"})
_MAX_RUN_MANIFEST_BYTES: Final = 2 * 1024 * 1024
_MAX_TRAINING_CONFIG_BYTES: Final = 256 * 1024
_MAX_VALIDATION_METRICS_BYTES: Final = 64 * 1024
_METADATA_SIZE_LIMITS: Final = {
    "run-manifest.json": _MAX_RUN_MANIFEST_BYTES,
    "training-config.json": _MAX_TRAINING_CONFIG_BYTES,
    "validation-metrics.json": _MAX_VALIDATION_METRICS_BYTES,
}


class EncoderDecoderTrainingError(ValueError):
    """Raised when the reproducible training boundary cannot be established."""


@dataclass(frozen=True)
class PreparedTrainingRecord:
    record_id: str
    split: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]


@dataclass(frozen=True)
class CapturedTree:
    path: Path
    artifacts: tuple[FileIdentityV1, ...]


@dataclass(frozen=True)
class EncoderDecoderArtifactPreflight:
    """Validated metadata for one content-bound encoder-decoder artifact bundle."""

    run_manifest: EncoderDecoderRunManifestV1
    artifact_manifest_sha256: str


@dataclass(frozen=True)
class EncoderDecoderTrainingBundleResult:
    """The run manifest and exact staged bundle identity published by one training run."""

    run_manifest: EncoderDecoderRunManifestV1
    artifact_manifest_sha256: str


def _safe_snapshot_artifact(relative_path: str) -> bool:
    path = Path(relative_path)
    return (
        any(fnmatch.fnmatchcase(path.name, pattern) for pattern in _SAFE_SNAPSHOT_PATTERNS)
        and path.suffix.casefold() not in _PROHIBITED_MODEL_SUFFIXES
    )


def _load_neural_runtime() -> tuple[Any, Any, Any]:
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        huggingface_hub = importlib.import_module("huggingface_hub")
    except ImportError as error:
        raise EncoderDecoderTrainingError(
            "encoder-decoder training requires the 'encoder-training' extra"
        ) from error
    return torch, transformers, huggingface_hub


def _stable_regular_file(
    source_fd: int,
    name: str,
    destination: BinaryIO,
    *,
    relative_path: str,
    allow_symlink: bool,
    max_bytes: int = _MAX_TREE_FILE_BYTES,
) -> FileIdentityV1:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if not allow_symlink and hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    entry_before = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
    try:
        handle_fd = os.open(name, flags, dir_fd=source_fd)
    except OSError as error:
        raise EncoderDecoderTrainingError(
            f"cannot safely open artifact: {relative_path}"
        ) from error
    digest = hashlib.sha256()
    byte_count = 0
    try:
        before = os.fstat(handle_fd)
        if not stat.S_ISREG(before.st_mode) or (not allow_symlink and before.st_nlink != 1):
            raise EncoderDecoderTrainingError(
                f"artifact must be a single-link regular file: {relative_path}"
            )
        if before.st_size > max_bytes:
            raise EncoderDecoderTrainingError(f"artifact exceeds size limit: {relative_path}")
        while True:
            chunk = os.read(handle_fd, _COPY_CHUNK_BYTES)
            if not chunk:
                break
            byte_count += len(chunk)
            if byte_count > max_bytes:
                raise EncoderDecoderTrainingError(f"artifact exceeds size limit: {relative_path}")
            digest.update(chunk)
            if destination.write(chunk) != len(chunk):
                raise OSError(f"short write while capturing artifact: {relative_path}")
        after = os.fstat(handle_fd)
    finally:
        os.close(handle_fd)
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    entry_after = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
    if (
        byte_count != before.st_size
        or any(getattr(before, field) != getattr(after, field) for field in stable_fields)
        or any(
            getattr(entry_before, field) != getattr(entry_after, field) for field in stable_fields
        )
    ):
        raise EncoderDecoderTrainingError(f"artifact changed while read: {relative_path}")
    return FileIdentityV1(
        path=relative_path,
        sha256=digest.hexdigest(),
        bytes=byte_count,
    )


def _capture_directory(
    source_fd: int,
    destination: Path,
    *,
    prefix: str = "",
    allow_file_symlinks: bool,
    remaining_bytes: list[int],
    include_file: Callable[[str], bool] | None = None,
) -> tuple[FileIdentityV1, ...]:
    try:
        names = sorted(os.listdir(source_fd))
    except OSError as error:
        raise EncoderDecoderTrainingError("cannot enumerate artifact directory") from error
    identities: list[FileIdentityV1] = []
    for name in names:
        if not name or name in {".", ".."} or "/" in name or "\0" in name:
            raise EncoderDecoderTrainingError("artifact directory contains an unsafe name")
        relative_path = f"{prefix}/{name}" if prefix else name
        try:
            metadata = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        except OSError as error:
            raise EncoderDecoderTrainingError(
                f"cannot inspect artifact: {relative_path}"
            ) from error
        destination_path = destination / name
        if stat.S_ISDIR(metadata.st_mode):
            destination_path.mkdir(mode=0o700)
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                child_fd = os.open(name, flags, dir_fd=source_fd)
            except OSError as error:
                raise EncoderDecoderTrainingError(
                    f"cannot safely open artifact directory: {relative_path}"
                ) from error
            try:
                child_identities = _capture_directory(
                    child_fd,
                    destination_path,
                    prefix=relative_path,
                    allow_file_symlinks=allow_file_symlinks,
                    remaining_bytes=remaining_bytes,
                    include_file=include_file,
                )
            finally:
                os.close(child_fd)
            if child_identities:
                identities.extend(child_identities)
            else:
                destination_path.rmdir()
            continue
        if include_file is not None and not include_file(relative_path):
            continue
        if not stat.S_ISREG(metadata.st_mode) and not (
            allow_file_symlinks and stat.S_ISLNK(metadata.st_mode)
        ):
            raise EncoderDecoderTrainingError(
                f"artifact tree contains a non-regular entry: {relative_path}"
            )
        try:
            with destination_path.open("xb") as destination_handle:
                identity = _stable_regular_file(
                    source_fd,
                    name,
                    destination_handle,
                    relative_path=relative_path,
                    allow_symlink=stat.S_ISLNK(metadata.st_mode),
                    max_bytes=min(_MAX_TREE_FILE_BYTES, remaining_bytes[0]),
                )
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
        except OSError as error:
            raise EncoderDecoderTrainingError(
                f"cannot materialize artifact: {relative_path}"
            ) from error
        identities.append(identity)
        remaining_bytes[0] -= identity.bytes
    return tuple(identities)


def _capture_tree(
    source: Path,
    destination: Path,
    *,
    allow_file_symlinks: bool = False,
    include_file: Callable[[str], bool] | None = None,
) -> CapturedTree:
    if destination.exists() or destination.is_symlink():
        raise EncoderDecoderTrainingError(
            f"private artifact destination must not exist: {destination}"
        )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        source_fd = os.open(source, flags)
    except OSError as error:
        raise EncoderDecoderTrainingError(
            f"artifact root must be a real directory: {source}"
        ) from error
    destination.mkdir(mode=0o700, parents=True)
    try:
        artifacts = _capture_directory(
            source_fd,
            destination,
            allow_file_symlinks=allow_file_symlinks,
            remaining_bytes=[_MAX_TREE_BYTES],
            include_file=include_file,
        )
    finally:
        os.close(source_fd)
    if not artifacts:
        raise EncoderDecoderTrainingError("artifact tree must contain files")
    return CapturedTree(
        path=destination,
        artifacts=tuple(sorted(artifacts, key=lambda artifact: artifact.path)),
    )


def _resolve_snapshot(
    identity: ArtifactIdentityV1,
    huggingface_hub: Any,
    *,
    cache_dir: Path | None,
    require_safetensors: bool,
) -> Path:
    kwargs: dict[str, object] = {
        "repo_id": identity.repo_id,
        "revision": identity.revision,
        "local_files_only": True,
        "allow_patterns": _SAFE_SNAPSHOT_PATTERNS,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    try:
        snapshot = Path(cast(str, huggingface_hub.snapshot_download(**kwargs))).resolve(strict=True)
    except Exception as error:
        raise EncoderDecoderTrainingError(
            f"cached Hub snapshot is unavailable for {identity.repo_id}@{identity.revision}"
        ) from error
    if not snapshot.is_dir() or snapshot.name != identity.revision:
        raise EncoderDecoderTrainingError(
            "Hub snapshot resolver did not return the configured commit directory"
        )
    if not (snapshot / "config.json").is_file():
        raise EncoderDecoderTrainingError("Hub snapshot is missing config.json")
    if require_safetensors and not any(snapshot.glob("*.safetensors")):
        raise EncoderDecoderTrainingError("base-model snapshot is missing safetensors weights")
    return snapshot


def _load_components(
    config: EncoderDecoderTrainingConfigV1,
    *,
    cache_dir: Path | None,
    private_root: Path,
    runtime_modules: tuple[Any, Any, Any],
) -> tuple[Any, Any, Any, Any, CapturedTree]:
    torch, transformers, huggingface_hub = runtime_modules
    base_snapshot = _resolve_snapshot(
        config.base_model,
        huggingface_hub,
        cache_dir=cache_dir,
        require_safetensors=True,
    )
    captured = _capture_tree(
        base_snapshot,
        private_root / "base-model",
        allow_file_symlinks=True,
        include_file=_safe_snapshot_artifact,
    )
    common = {"local_files_only": True, "trust_remote_code": False}
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(captured.path),
        **common,
    )
    loaded = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        str(captured.path),
        **common,
        use_safetensors=True,
        output_loading_info=True,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise EncoderDecoderTrainingError("model loader did not return loading diagnostics")
    model, loading_info = loaded
    if not isinstance(loading_info, dict):
        raise EncoderDecoderTrainingError("model loader returned invalid loading diagnostics")
    failures = {
        key: loading_info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading_info.get(key)
    }
    if failures:
        raise EncoderDecoderTrainingError(
            f"base-model snapshot does not exactly match the model architecture: {failures}"
        )
    _validate_model_tokenizer_compatibility(tokenizer, model, config)
    return torch, transformers, tokenizer, model, captured


def _integer_tokens(value: object, *, context: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EncoderDecoderTrainingError(f"tokenizer returned invalid {context} token IDs")
    tokens: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise EncoderDecoderTrainingError(f"tokenizer returned invalid {context} token IDs")
        tokens.append(item)
    if not tokens:
        raise EncoderDecoderTrainingError(f"tokenizer returned no {context} token IDs")
    return tuple(tokens)


def _encode(
    tokenizer: Any,
    text: str,
    *,
    add_special_tokens: bool,
    context: str,
) -> tuple[int, ...]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    except Exception as error:
        raise EncoderDecoderTrainingError(f"tokenizer could not encode {context}") from error
    return _integer_tokens(encoded, context=context)


def _decode(tokenizer: Any, token_ids: tuple[int, ...], *, context: str) -> str:
    try:
        return cast(
            str,
            tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
        )
    except Exception as error:
        raise EncoderDecoderTrainingError(f"tokenizer could not decode {context}") from error


def _validate_model_tokenizer_compatibility(
    tokenizer: Any,
    model: Any,
    config: EncoderDecoderTrainingConfigV1,
) -> None:
    def embedding_capacity(embedding: Any) -> int:
        for attribute in ("num_embeddings", "out_features"):
            value = getattr(embedding, attribute, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        weight = getattr(embedding, "weight", None)
        shape = getattr(weight, "shape", ())
        if shape and isinstance(shape[0], int):
            return int(shape[0])
        raise AttributeError("embedding vocabulary capacity is unavailable")

    try:
        vocabulary_size = len(tokenizer)
        input_capacity = embedding_capacity(model.get_input_embeddings())
        output_embeddings = model.get_output_embeddings()
        output_capacity = (
            embedding_capacity(output_embeddings)
            if output_embeddings is not None
            else input_capacity
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise EncoderDecoderTrainingError(
            "model and tokenizer did not expose compatible vocabulary capacities"
        ) from error
    configured_vocabulary = int(getattr(model.config, "vocab_size", -1))
    if (
        vocabulary_size <= 0
        or configured_vocabulary != vocabulary_size
        or input_capacity != vocabulary_size
        or output_capacity != vocabulary_size
    ):
        raise EncoderDecoderTrainingError(
            "model and tokenizer vocabulary capacities do not match exactly"
        )
    for name in ("pad_token_id", "eos_token_id"):
        token_id = getattr(tokenizer, name, None)
        model_token_id = getattr(model.config, name, None)
        if token_id is None or isinstance(token_id, bool):
            raise EncoderDecoderTrainingError(f"tokenizer must define {name}")
        if int(token_id) < 0 or int(token_id) >= vocabulary_size:
            raise EncoderDecoderTrainingError(f"tokenizer {name} exceeds model vocabulary")
        if name == "pad_token_id" and (model_token_id is None or isinstance(model_token_id, bool)):
            raise EncoderDecoderTrainingError("model must define pad_token_id")
        if model_token_id is not None and int(model_token_id) != int(token_id):
            raise EncoderDecoderTrainingError(f"model and tokenizer {name} values do not match")
    unknown_token_id = getattr(tokenizer, "unk_token_id", None)
    model_unknown_token_id = getattr(model.config, "unk_token_id", None)
    if unknown_token_id is not None:
        if isinstance(unknown_token_id, bool) or not 0 <= int(unknown_token_id) < vocabulary_size:
            raise EncoderDecoderTrainingError("tokenizer unk_token_id exceeds model vocabulary")
        if model_unknown_token_id is not None and int(model_unknown_token_id) != int(
            unknown_token_id
        ):
            raise EncoderDecoderTrainingError(
                "model and tokenizer unk_token_id values do not match"
            )
    decoder_start_token_id = getattr(model.config, "decoder_start_token_id", None)
    if decoder_start_token_id is None or isinstance(decoder_start_token_id, bool):
        raise EncoderDecoderTrainingError("model must define decoder_start_token_id")
    if not 0 <= int(decoder_start_token_id) < vocabulary_size:
        raise EncoderDecoderTrainingError("model decoder_start_token_id exceeds model vocabulary")
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if (
        isinstance(tokenizer_limit, int)
        and 0 < tokenizer_limit < 10**12
        and max(config.max_source_tokens, config.max_target_tokens) > tokenizer_limit
    ):
        raise EncoderDecoderTrainingError(
            "configured token limits exceed tokenizer model_max_length"
        )
    position_limits = [
        getattr(model.config, name, None) for name in ("max_position_embeddings", "n_positions")
    ]
    finite_position_limits = [
        int(limit)
        for limit in position_limits
        if isinstance(limit, int) and not isinstance(limit, bool) and 0 < limit < 10**12
    ]
    if finite_position_limits and max(
        config.max_source_tokens,
        config.max_target_tokens,
    ) > min(finite_position_limits):
        raise EncoderDecoderTrainingError("configured token limits exceed model position capacity")


def _all_records(release: TrainingReleaseSnapshot) -> tuple[ReleasedTrainingRecordV1, ...]:
    return (*release.train, *release.validation, *release.test, *release.adversarial)


def preflight_encoder_decoder_tokenizer(
    tokenizer: Any,
    release: TrainingReleaseSnapshot,
    config: EncoderDecoderTrainingConfigV1,
) -> tuple[PreparedTrainingRecord, ...]:
    """Validate the full symbol inventory and every source/target length before training."""

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    unknown_token_id = getattr(tokenizer, "unk_token_id", None)
    if eos_token_id is None or pad_token_id is None:
        raise EncoderDecoderTrainingError("tokenizer must define EOS and padding token IDs")
    forbidden_ids = {int(eos_token_id)}
    if unknown_token_id is not None:
        forbidden_ids.add(int(unknown_token_id))
    for symbol in sorted(release.symbol_inventory):
        for form in (symbol, f" {symbol}"):
            token_ids = _encode(
                tokenizer,
                form,
                add_special_tokens=False,
                context=f"symbolic form {form!r}",
            )
            if (
                forbidden_ids.intersection(token_ids)
                or _decode(
                    tokenizer,
                    token_ids,
                    context=f"symbolic form {form!r}",
                )
                != form
            ):
                raise EncoderDecoderTrainingError(
                    f"tokenizer cannot losslessly encode symbolic form: {form!r}"
                )

    prepared: list[PreparedTrainingRecord] = []
    for record in _all_records(release):
        input_ids = _encode(
            tokenizer,
            record.serialized_ir,
            add_special_tokens=True,
            context=f"source for {record.record_id!r}",
        )
        if (
            unknown_token_id is not None
            and int(unknown_token_id) in input_ids
            or _decode(
                tokenizer,
                input_ids,
                context=f"source for {record.record_id!r}",
            )
            != record.serialized_ir
        ):
            raise EncoderDecoderTrainingError(
                f"tokenizer cannot losslessly encode source for {record.record_id!r}"
            )
        target_ids = _encode(
            tokenizer,
            record.symbols,
            add_special_tokens=False,
            context=f"target for {record.record_id!r}",
        )
        if (
            forbidden_ids.intersection(target_ids)
            or _decode(
                tokenizer,
                target_ids,
                context=f"target for {record.record_id!r}",
            )
            != record.symbols
        ):
            raise EncoderDecoderTrainingError(
                f"tokenizer cannot losslessly encode target for {record.record_id!r}"
            )
        labels = (*target_ids, int(eos_token_id))
        if len(input_ids) > config.max_source_tokens:
            raise EncoderDecoderTrainingError(
                f"source for {record.record_id!r} exceeds max_source_tokens "
                f"({len(input_ids)} > {config.max_source_tokens})"
            )
        if len(labels) > config.max_target_tokens:
            raise EncoderDecoderTrainingError(
                f"target for {record.record_id!r} exceeds max_target_tokens "
                f"({len(labels)} > {config.max_target_tokens})"
            )
        prepared.append(
            PreparedTrainingRecord(
                record_id=record.record_id,
                split=record.split,
                input_ids=input_ids,
                labels=labels,
            )
        )
    return tuple(prepared)


def _batch(
    torch: Any, records: Sequence[PreparedTrainingRecord], pad_token_id: int
) -> dict[str, Any]:
    input_tensors = [torch.tensor(record.input_ids, dtype=torch.long) for record in records]
    label_tensors = [torch.tensor(record.labels, dtype=torch.long) for record in records]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_tensors,
        batch_first=True,
        padding_value=pad_token_id,
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        label_tensors,
        batch_first=True,
        padding_value=-100,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(pad_token_id).long(),
        "labels": labels,
    }


def _seeded_records(
    torch: Any,
    records: tuple[PreparedTrainingRecord, ...],
    seed: int,
) -> Iterator[PreparedTrainingRecord]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    while True:
        for position in torch.randperm(len(records), generator=generator).tolist():
            yield records[position]


def _take_batch(
    stream: Iterator[PreparedTrainingRecord],
    size: int,
) -> tuple[PreparedTrainingRecord, ...]:
    return tuple(next(stream) for _ in range(size))


def _finite_loss(loss: Any, *, context: str) -> float:
    value = float(loss.detach().cpu().item())
    if not math.isfinite(value) or value < 0:
        raise EncoderDecoderTrainingError(f"{context} produced a non-finite loss")
    return value


def _train(
    torch: Any,
    model: Any,
    tokenizer: Any,
    prepared: tuple[PreparedTrainingRecord, ...],
    config: EncoderDecoderTrainingConfigV1,
) -> tuple[tuple[str, ...], tuple[float, ...], int]:
    train_records = tuple(record for record in prepared if record.split == "train")
    if not train_records:
        raise EncoderDecoderTrainingError("training release has no train records")
    pad_token_id = int(tokenizer.pad_token_id)
    stream = _seeded_records(torch, train_records, config.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.optimizer.learning_rate),
        weight_decay=float(config.optimizer.weight_decay),
    )
    model.to("cpu")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    record_order: list[str] = []
    optimizer_losses: list[float] = []
    micro_steps = 0
    for _ in range(config.max_steps):
        accumulated_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            batch_records = _take_batch(stream, config.micro_batch_size)
            record_order.extend(record.record_id for record in batch_records)
            outputs = model(**_batch(torch, batch_records, pad_token_id))
            raw_loss = _finite_loss(outputs.loss, context="training")
            accumulated_loss += raw_loss
            (outputs.loss / config.gradient_accumulation_steps).backward()
            micro_steps += 1
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_losses.append(accumulated_loss / config.gradient_accumulation_steps)
    return tuple(record_order), tuple(optimizer_losses), micro_steps


def _evaluate(
    torch: Any,
    model: Any,
    tokenizer: Any,
    prepared: tuple[PreparedTrainingRecord, ...],
) -> ValidationMetricsV1:
    validation = tuple(record for record in prepared if record.split == "validation")
    if not validation:
        raise EncoderDecoderTrainingError("training release has no validation records")
    model.to("cpu")
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for record in validation:
            outputs = model(**_batch(torch, (record,), int(tokenizer.pad_token_id)))
            losses.append(_finite_loss(outputs.loss, context="validation"))
    return ValidationMetricsV1(
        schema_version=VALIDATION_SCHEMA_VERSION,
        record_count=len(validation),
        mean_loss=sum(losses) / len(losses),
    )


def _safe_save_pretrained(model: Any, output: Path) -> None:
    kwargs: dict[str, object] = {}
    if "safe_serialization" in inspect.signature(model.save_pretrained).parameters:
        kwargs["safe_serialization"] = True
    model.save_pretrained(str(output), **kwargs)


def _validate_checkpoint_artifacts(
    artifacts: tuple[FileIdentityV1, ...],
) -> tuple[FileIdentityV1, ...]:
    for identity in artifacts:
        if Path(identity.path).suffix.casefold() in _PROHIBITED_MODEL_SUFFIXES:
            raise EncoderDecoderTrainingError(
                f"checkpoint contains a pickle-capable artifact: {identity.path}"
            )
    if not any(Path(identity.path).suffix.casefold() == ".safetensors" for identity in artifacts):
        raise EncoderDecoderTrainingError("checkpoint does not contain safetensors weights")
    return artifacts


def verify_safe_encoder_decoder_checkpoint(directory: Path) -> tuple[FileIdentityV1, ...]:
    """Stable-read and content-bind a regular, safetensors-only checkpoint tree."""

    with tempfile.TemporaryDirectory(prefix="ste-checkpoint-verify-") as private:
        captured = _capture_tree(directory, Path(private) / "checkpoint")
        return _validate_checkpoint_artifacts(captured.artifacts)


def _artifact_file(identity: FileIdentityV1) -> ArtifactFileV1:
    return ArtifactFileV1(
        path=identity.path,
        sha256=identity.sha256,
        bytes=identity.bytes,
    )


def _run_file(identity: ArtifactFileV1) -> FileIdentityV1:
    return FileIdentityV1(
        path=identity.path,
        sha256=identity.sha256,
        bytes=identity.bytes,
    )


def _read_bounded_regular_bytes(
    root: Path,
    name: str,
    *,
    max_bytes: int,
) -> bytes:
    directory_fd = -1
    file_fd = -1
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
        file_flags |= os.O_NOFOLLOW
    try:
        directory_fd = os.open(root, directory_flags)
        entry_before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        file_fd = os.open(name, file_flags, dir_fd=directory_fd)
        before = os.fstat(file_fd)
        if (
            not stat.S_ISREG(entry_before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or entry_before.st_nlink != 1
            or before.st_nlink != 1
            or (entry_before.st_dev, entry_before.st_ino, entry_before.st_size)
            != (before.st_dev, before.st_ino, before.st_size)
        ):
            raise EncoderDecoderTrainingError(
                f"metadata must be a single-link regular file: {name}"
            )
        if before.st_size > max_bytes:
            raise EncoderDecoderTrainingError(f"metadata exceeds size limit: {name}")
        chunks: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(file_fd, min(_COPY_CHUNK_BYTES, max_bytes + 1 - byte_count))
            if not chunk:
                break
            chunks.append(chunk)
            byte_count += len(chunk)
            if byte_count > max_bytes:
                raise EncoderDecoderTrainingError(f"metadata exceeds size limit: {name}")
        after = os.fstat(file_fd)
        entry_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except EncoderDecoderTrainingError:
        raise
    except FileNotFoundError as error:
        raise EncoderDecoderTrainingError(f"missing {name}") from error
    except OSError as error:
        raise EncoderDecoderTrainingError(f"cannot safely read metadata: {name}") from error
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if directory_fd >= 0:
            os.close(directory_fd)
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if (
        byte_count != before.st_size
        or any(getattr(before, field) != getattr(after, field) for field in stable_fields)
        or any(getattr(before, field) != getattr(entry_after, field) for field in stable_fields)
    ):
        raise EncoderDecoderTrainingError(f"metadata changed while read: {name}")
    return b"".join(chunks)


def _validate_metadata_inventory(manifest: ArtifactBundleManifestV1) -> None:
    identities = {identity.path: identity for identity in manifest.files}
    for name, max_bytes in _METADATA_SIZE_LIMITS.items():
        identity = identities.get(name)
        if identity is None:
            raise EncoderDecoderTrainingError(f"encoder-decoder artifact bundle is missing {name}")
        if identity.bytes > max_bytes:
            raise EncoderDecoderTrainingError(f"metadata exceeds size limit: {name}")


def _preflight_source_metadata_inventory(
    root: Path,
    expected_manifest_sha256: str,
) -> None:
    if len(expected_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_manifest_sha256
    ):
        raise ArtifactVerificationError(
            "artifact manifest SHA-256 must be 64 lowercase hexadecimal characters"
        )
    manifest_bytes = _read_bounded_regular_bytes(
        root,
        ARTIFACT_MANIFEST_NAME,
        max_bytes=MAX_ARTIFACT_MANIFEST_BYTES,
    )
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256:
        raise ArtifactVerificationError("artifact manifest SHA-256 does not match")
    manifest = parse_canonical_artifact_manifest(manifest_bytes)
    if (
        manifest.architecture != "encoder-decoder"
        or manifest.artifact_type != "encoder-decoder-checkpoint"
        or manifest.entrypoint != "."
    ):
        raise EncoderDecoderTrainingError("artifact bundle is not an encoder-decoder checkpoint")
    _validate_metadata_inventory(manifest)


def _parse_run_manifest_bytes(data: bytes) -> EncoderDecoderRunManifestV1:
    try:
        manifest = EncoderDecoderRunManifestV1.model_validate_json(data)
    except ValueError as error:
        raise EncoderDecoderTrainingError(
            f"encoder-decoder run manifest is invalid: {error}"
        ) from error
    if data != canonical_run_manifest_json(manifest):
        raise EncoderDecoderTrainingError("encoder-decoder run manifest is not canonical JSON")
    return manifest


def _preflight_verified_encoder_decoder_artifact_bundle(
    verified: VerifiedArtifactBundle,
) -> EncoderDecoderArtifactPreflight:
    bundle_manifest = verified.manifest
    if (
        bundle_manifest.architecture != "encoder-decoder"
        or bundle_manifest.artifact_type != "encoder-decoder-checkpoint"
        or bundle_manifest.entrypoint != "."
    ):
        raise EncoderDecoderTrainingError("artifact bundle is not an encoder-decoder checkpoint")
    _validate_metadata_inventory(bundle_manifest)
    materialized = verified.path
    run_manifest_bytes = _read_bounded_regular_bytes(
        materialized,
        "run-manifest.json",
        max_bytes=_MAX_RUN_MANIFEST_BYTES,
    )
    training_config_bytes = _read_bounded_regular_bytes(
        materialized,
        "training-config.json",
        max_bytes=_MAX_TRAINING_CONFIG_BYTES,
    )
    validation_bytes = _read_bounded_regular_bytes(
        materialized,
        "validation-metrics.json",
        max_bytes=_MAX_VALIDATION_METRICS_BYTES,
    )
    run_manifest = _parse_run_manifest_bytes(run_manifest_bytes)

    bundle_files = tuple(_run_file(identity) for identity in bundle_manifest.files)
    output_artifacts = tuple(
        identity for identity in bundle_files if identity.path != "run-manifest.json"
    )
    _validate_checkpoint_artifacts(bundle_files)
    output_paths = {identity.path for identity in output_artifacts}
    if "config.json" not in output_paths:
        raise EncoderDecoderTrainingError("encoder-decoder artifact bundle is missing config.json")
    if (
        run_manifest.output_artifacts != output_artifacts
        or training_config_bytes != canonical_training_config_json(run_manifest.training_config)
        or run_manifest.training_config_sha256
        != training_config_sha256(run_manifest.training_config)
        or validation_bytes != canonical_validation_metrics_json(run_manifest.validation)
        or run_manifest.base_model != run_manifest.training_config.base_model
        or run_manifest.tokenizer != run_manifest.training_config.tokenizer
        or run_manifest.base_model_artifacts != run_manifest.tokenizer_artifacts
        or not any(identity.path == "config.json" for identity in run_manifest.base_model_artifacts)
    ):
        raise EncoderDecoderTrainingError(
            "encoder-decoder artifact bundle does not match its run manifest"
        )
    _validate_checkpoint_artifacts(run_manifest.base_model_artifacts)
    with tempfile.TemporaryDirectory(prefix="ste-encoder-artifact-load-") as private:
        _, model, _ = _reload_components(
            _load_neural_runtime()[1],
            materialized,
            Path(private),
        )
        del model
    return EncoderDecoderArtifactPreflight(
        run_manifest=run_manifest,
        artifact_manifest_sha256=verified.manifest_sha256,
    )


def preflight_encoder_decoder_artifact_bundle(
    root: Path,
    expected_manifest_sha256: str,
) -> EncoderDecoderArtifactPreflight:
    """Verify and preflight a content-pinned encoder-decoder artifact bundle."""

    try:
        _preflight_source_metadata_inventory(root, expected_manifest_sha256)
        with open_verified_artifact_bundle(
            root,
            expected_manifest_sha256,
        ) as verified:
            return _preflight_verified_encoder_decoder_artifact_bundle(verified)
    except ArtifactVerificationError as error:
        raise EncoderDecoderTrainingError(
            f"encoder-decoder artifact bundle verification failed: {error}"
        ) from error


def encoder_decoder_artifact_manifest_sha256(root: Path) -> str:
    """Discover and validate the canonical manifest digest for a local bundle."""

    try:
        manifest_bytes = _read_bounded_regular_bytes(
            root,
            ARTIFACT_MANIFEST_NAME,
            max_bytes=MAX_ARTIFACT_MANIFEST_BYTES,
        )
        manifest = parse_canonical_artifact_manifest(manifest_bytes)
    except ArtifactVerificationError as error:
        raise EncoderDecoderTrainingError(
            f"encoder-decoder artifact manifest is invalid: {error}"
        ) from error
    if manifest.architecture != "encoder-decoder":
        raise EncoderDecoderTrainingError("artifact bundle is not an encoder-decoder checkpoint")
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    preflight_encoder_decoder_artifact_bundle(root, digest)
    return digest


def _reload_components(
    transformers: Any,
    checkpoint: Path,
    private_root: Path,
) -> tuple[Any, Any, tuple[FileIdentityV1, ...]]:
    captured = _capture_tree(checkpoint, private_root / "checkpoint")
    artifacts = _validate_checkpoint_artifacts(captured.artifacts)
    common = {"local_files_only": True, "trust_remote_code": False}
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(captured.path), **common)
        loaded = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            str(captured.path),
            **common,
            use_safetensors=True,
            output_loading_info=True,
        )
    except Exception as error:
        raise EncoderDecoderTrainingError(
            "saved checkpoint could not be reloaded through the safe local boundary"
        ) from error
    if not isinstance(loaded, tuple) or len(loaded) != 2 or not isinstance(loaded[1], dict):
        raise EncoderDecoderTrainingError(
            "saved checkpoint loader did not return loading diagnostics"
        )
    model, loading_info = loaded
    if any(
        loading_info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise EncoderDecoderTrainingError(
            "saved checkpoint does not exactly match the model architecture"
        )
    return tokenizer, model, artifacts


def _file_identity(path: Path) -> FileIdentityV1:
    parent = path.parent
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        parent_fd = os.open(parent, flags)
    except OSError as error:
        raise EncoderDecoderTrainingError(f"cannot read provenance file: {path}") from error
    try:
        with tempfile.TemporaryFile() as sink:
            identity = _stable_regular_file(
                parent_fd,
                path.name,
                sink,
                relative_path=str(path.resolve()),
                allow_symlink=False,
            )
    finally:
        os.close(parent_fd)
    return identity


def _package_tree_sha256(package_root: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="ste-package-provenance-") as private:
        captured = _capture_tree(package_root, Path(private) / "package")
    selected = tuple(
        identity
        for identity in captured.artifacts
        if Path(identity.path).suffix in _PACKAGE_SUFFIXES or Path(identity.path).name == "py.typed"
    )
    if not selected:
        raise EncoderDecoderTrainingError(
            f"package tree contains no Python sources: {package_root}"
        )
    digest = hashlib.sha256()
    for identity in selected:
        digest.update(identity.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(identity.sha256.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(identity.bytes).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _git_package_provenance(
    source_root: Path,
    dependency_lock: Path,
) -> PackageProvenanceV1:
    if source_root.is_symlink() or not source_root.is_dir():
        raise EncoderDecoderTrainingError(f"source root must be a real directory: {source_root}")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status_output = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise EncoderDecoderTrainingError(
            "package source commit could not be derived from the source root"
        ) from error
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise EncoderDecoderTrainingError(
            "package source root did not produce a full commit digest"
        )
    if status_output:
        raise EncoderDecoderTrainingError("package source root must be clean")
    checkout_tree = _package_tree_sha256(source_root / "src" / "ste_compiler")
    runtime_tree = _package_tree_sha256(Path(__file__).resolve().parents[1])
    if checkout_tree != runtime_tree:
        raise EncoderDecoderTrainingError(
            "package source tree does not match the executing ste_compiler package"
        )
    installed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if raw_name is None or not raw_name.strip() or not distribution.version:
            raise EncoderDecoderTrainingError(
                "cannot derive a complete installed-distribution inventory"
            )
        name = re.sub(r"[-_.]+", "-", raw_name.strip()).lower()
        existing = installed.get(name)
        if existing is not None and existing != distribution.version:
            raise EncoderDecoderTrainingError(
                f"multiple installed versions prevent reproducible provenance: {name}"
            )
        installed[name] = distribution.version
    missing = sorted(_REQUIRED_DEPENDENCIES - installed.keys())
    if missing:
        raise EncoderDecoderTrainingError(
            "runtime dependency inventory is missing required distributions: " + ", ".join(missing)
        )
    dependencies = tuple(sorted(installed.items()))
    return PackageProvenanceV1(
        distribution="ste-compiler",
        version=installed["ste-compiler"],
        source_commit=commit,
        source_dirty=False,
        source_tree_sha256=runtime_tree,
        dependency_lock=_file_identity(dependency_lock),
        dependencies=dependencies,
    )


def _peak_rss_bytes() -> int | None:
    try:
        resource = importlib.import_module("resource")
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, ImportError, OSError, ValueError):
        return None
    if sys.platform != "darwin":
        peak *= 1024
    return max(peak, 0)


def _hardware(torch: Any) -> HardwareProvenanceV1:
    return HardwareProvenanceV1(
        device="cpu",
        python=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine() or "unknown",
        processor=platform.processor(),
        logical_cpu_count=os.cpu_count(),
        torch_threads=int(torch.get_num_threads()),
        process_peak_rss_bytes=_peak_rss_bytes(),
    )


def _parameter_counts(model: Any) -> ParameterCountsV1:
    parameters = tuple(model.parameters())
    return ParameterCountsV1(
        total=sum(int(parameter.numel()) for parameter in parameters),
        trainable=sum(
            int(parameter.numel()) for parameter in parameters if parameter.requires_grad
        ),
    )


def _fsync_tree(directory: Path) -> None:
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


@dataclass(frozen=True)
class _PinnedOutputDirectory:
    descriptor: int
    device: int
    inode: int


def _open_pinned_output_directory(directory: Path) -> _PinnedOutputDirectory:
    descriptor = -1
    try:
        descriptor = os.open(
            directory,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        metadata = os.fstat(descriptor)
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise EncoderDecoderTrainingError(
            f"cannot pin staged training output: {directory}"
        ) from error
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise EncoderDecoderTrainingError(
            f"staged training output must be a real directory: {directory}"
        )
    return _PinnedOutputDirectory(
        descriptor=descriptor,
        device=metadata.st_dev,
        inode=metadata.st_ino,
    )


def _assert_pinned_output_directory(
    directory: Path,
    pinned: _PinnedOutputDirectory,
    *,
    operation: str,
) -> None:
    try:
        descriptor_metadata = os.fstat(pinned.descriptor)
        path_metadata = os.stat(directory, follow_symlinks=False)
    except OSError as error:
        raise EncoderDecoderTrainingError(
            f"training output changed during {operation}: {directory}"
        ) from error
    expected_identity = (pinned.device, pinned.inode)
    if (
        not stat.S_ISDIR(descriptor_metadata.st_mode)
        or not stat.S_ISDIR(path_metadata.st_mode)
        or (descriptor_metadata.st_dev, descriptor_metadata.st_ino) != expected_identity
        or (path_metadata.st_dev, path_metadata.st_ino) != expected_identity
    ):
        raise EncoderDecoderTrainingError(
            f"training output changed during {operation}: {directory}"
        )


def _clear_pinned_directory(directory_fd: int, remaining_entries: list[int]) -> None:
    """Delete entries relative to an already pinned directory, never through a path."""

    while True:
        try:
            with os.scandir(directory_fd) as entries:
                entry = next(entries, None)
        except OSError as error:
            raise EncoderDecoderTrainingError(
                "cannot enumerate invalid pinned artifact output"
            ) from error
        if entry is None:
            return
        if remaining_entries[0] <= 0:
            raise EncoderDecoderTrainingError(
                "invalid pinned artifact output exceeds the cleanup entry limit"
            )
        remaining_entries[0] -= 1
        name = entry.name
        try:
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                child_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                try:
                    _clear_pinned_directory(child_fd, remaining_entries)
                finally:
                    os.close(child_fd)
                os.rmdir(name, dir_fd=directory_fd)
            else:
                os.unlink(name, dir_fd=directory_fd)
        except OSError as error:
            raise EncoderDecoderTrainingError(
                "cannot remove invalid pinned artifact output"
            ) from error


def _remove_invalid_pinned_output(
    output: Path,
    pinned: _PinnedOutputDirectory,
) -> None:
    """Remove a failed publication only through its pinned directory descriptor."""

    _assert_pinned_output_directory(
        output,
        pinned,
        operation="invalid artifact cleanup",
    )
    remaining_entries = (MAX_ARTIFACT_FILES + 1) * (MAX_ARTIFACT_PATH_DEPTH + 1)
    _clear_pinned_directory(pinned.descriptor, [remaining_entries])
    parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        path_metadata = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(path_metadata.st_mode) or (
            path_metadata.st_dev,
            path_metadata.st_ino,
        ) != (pinned.device, pinned.inode):
            raise EncoderDecoderTrainingError(
                f"training output changed during invalid artifact cleanup: {output}"
            )
        os.rmdir(output.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _write_bytes(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        if handle.write(data) != len(data):
            raise OSError(f"failed to write complete artifact: {path.name}")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def _isolated_deterministic_runtime(torch: Any, seed: int) -> Iterator[None]:
    python_random_state = random.getstate()
    numpy: Any | None = None
    numpy_random_state: Any | None = None
    try:
        numpy = importlib.import_module("numpy")
        numpy_random_state = numpy.random.get_state()
    except (AttributeError, ImportError):
        pass
    torch_random_state = torch.get_rng_state()
    deterministic_enabled = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch_threads = int(torch.get_num_threads())
    try:
        random.seed(seed)
        if numpy is not None:
            numpy.random.seed(seed % (2**32))
        torch.random.default_generator.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        yield
    finally:
        random.setstate(python_random_state)
        if numpy is not None and numpy_random_state is not None:
            numpy.random.set_state(numpy_random_state)
        torch.set_rng_state(torch_random_state)
        torch.use_deterministic_algorithms(
            deterministic_enabled,
            warn_only=deterministic_warn_only,
        )
        torch.set_num_threads(torch_threads)


def _rename_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux"):
        rename = libc.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, 1)
    else:
        raise EncoderDecoderTrainingError(
            f"atomic no-replace publication is unsupported on {sys.platform}"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise EncoderDecoderTrainingError(
            f"training output was created concurrently: {destination}"
        )
    raise EncoderDecoderTrainingError(
        f"cannot atomically publish output {destination}: {os.strerror(error_number)}"
    )


def _require_no_replace_publication() -> None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if sys.platform == "darwin":
            rename = libc.renamex_np
        elif sys.platform.startswith("linux"):
            rename = libc.renameat2
        else:
            raise AttributeError
        del rename
    except (AttributeError, OSError) as error:
        raise EncoderDecoderTrainingError(
            f"atomic no-replace publication is unsupported on {sys.platform}"
        ) from error


def _run_encoder_decoder_training(
    config: EncoderDecoderTrainingConfigV1,
    release_path: Path,
    output: Path,
    *,
    source_root: Path,
    dependency_lock: Path,
    cache_dir: Path | None = None,
    runtime_modules: tuple[Any, Any, Any],
) -> EncoderDecoderTrainingBundleResult:
    started = time.perf_counter()
    release = read_training_release(release_path, config.corpus)
    package = _git_package_provenance(source_root, dependency_lock)

    if not output.parent.is_dir() or output.parent.is_symlink():
        raise EncoderDecoderTrainingError(
            f"training output parent must be a real directory: {output.parent}"
        )
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.stage-",
            dir=output.parent,
        )
    )
    runtime = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.runtime-",
            dir=output.parent,
        )
    )
    pinned_stage: _PinnedOutputDirectory | None = None
    installed = False
    try:
        pinned_stage = _open_pinned_output_directory(stage)
        torch, transformers, tokenizer, model, base_snapshot = _load_components(
            config,
            cache_dir=cache_dir,
            private_root=runtime / "initial",
            runtime_modules=runtime_modules,
        )
        prepared = preflight_encoder_decoder_tokenizer(tokenizer, release, config)
        parameter_counts = _parameter_counts(model)
        record_order, optimizer_losses, micro_steps = _train(
            torch,
            model,
            tokenizer,
            prepared,
            config,
        )
        tokenizer.save_pretrained(str(stage))
        _safe_save_pretrained(model, stage)
        _write_bytes(stage / "training-config.json", canonical_training_config_json(config))
        verify_safe_encoder_decoder_checkpoint(stage)
        reloaded_tokenizer, reloaded_model, _ = _reload_components(
            transformers,
            stage,
            runtime / "saved",
        )
        _validate_model_tokenizer_compatibility(reloaded_tokenizer, reloaded_model, config)
        reloaded_prepared = preflight_encoder_decoder_tokenizer(
            reloaded_tokenizer,
            release,
            config,
        )
        validation = _evaluate(
            torch,
            reloaded_model,
            reloaded_tokenizer,
            reloaded_prepared,
        )
        _write_bytes(
            stage / "validation-metrics.json",
            canonical_validation_metrics_json(validation),
        )
        output_artifacts = verify_safe_encoder_decoder_checkpoint(stage)
        manifest = EncoderDecoderRunManifestV1(
            schema_version=RUN_MANIFEST_SCHEMA_VERSION,
            architecture="encoder-decoder",
            training_config_sha256=training_config_sha256(config),
            training_config=config,
            package=package,
            corpus=CorpusRunIdentityV1(
                dataset_version=release.manifest.dataset_version,
                manifest_sha256=release.manifest_sha256,
                artifacts=release.artifact_sha256,
            ),
            base_model=config.base_model,
            tokenizer=config.tokenizer,
            base_model_artifacts=base_snapshot.artifacts,
            tokenizer_artifacts=base_snapshot.artifacts,
            seed=config.seed,
            parameter_counts=parameter_counts,
            optimizer_steps=config.max_steps,
            micro_steps=micro_steps,
            record_order=record_order,
            optimizer_losses=optimizer_losses,
            validation=validation,
            hardware=_hardware(torch),
            duration_seconds=time.perf_counter() - started,
            output_artifacts=output_artifacts,
            evaluation_command=_EVALUATION_COMMAND,
        )
        _write_bytes(stage / "run-manifest.json", canonical_run_manifest_json(manifest))
        bundle_files = verify_safe_encoder_decoder_checkpoint(stage)
        artifact_manifest = build_artifact_manifest(
            architecture="encoder-decoder",
            artifact_type="encoder-decoder-checkpoint",
            entrypoint=".",
            files=tuple(_artifact_file(identity) for identity in bundle_files),
        )
        artifact_digest = artifact_manifest_sha256(artifact_manifest)
        _write_bytes(
            stage / ARTIFACT_MANIFEST_NAME,
            canonical_artifact_manifest_json(artifact_manifest),
        )
        _fsync_tree(stage)
        _assert_pinned_output_directory(
            stage,
            pinned_stage,
            operation="staged artifact verification",
        )
        preflight = preflight_encoder_decoder_artifact_bundle(stage, artifact_digest)
        if preflight.run_manifest != manifest:
            raise EncoderDecoderTrainingError("staged artifact bundle changed before publication")
        _assert_pinned_output_directory(
            stage,
            pinned_stage,
            operation="staged artifact publication",
        )
        _rename_no_replace(stage, output)
        installed = True
        _assert_pinned_output_directory(
            output,
            pinned_stage,
            operation="atomic artifact publication",
        )
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        try:
            published_manifest = verify_artifact_bundle(output, artifact_digest)
        except ArtifactVerificationError as error:
            raise EncoderDecoderTrainingError(
                f"published artifact bundle does not match the verified stage: {error}"
            ) from error
        if (
            published_manifest.run_manifest_sha256
            != hashlib.sha256(canonical_run_manifest_json(manifest)).hexdigest()
        ):
            raise EncoderDecoderTrainingError(
                "published artifact run manifest does not match the completed training run"
            )
        _assert_pinned_output_directory(
            output,
            pinned_stage,
            operation="published artifact verification",
        )
        return EncoderDecoderTrainingBundleResult(
            run_manifest=manifest,
            artifact_manifest_sha256=preflight.artifact_manifest_sha256,
        )
    except BaseException:
        if installed and pinned_stage is not None and output.exists():
            _remove_invalid_pinned_output(output, pinned_stage)
        raise
    finally:
        if pinned_stage is not None:
            os.close(pinned_stage.descriptor)
        if not installed:
            shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(runtime, ignore_errors=True)


def run_encoder_decoder_training_bundle(
    config: EncoderDecoderTrainingConfigV1,
    release_path: Path,
    output: Path,
    *,
    source_root: Path,
    dependency_lock: Path,
    cache_dir: Path | None = None,
) -> EncoderDecoderTrainingBundleResult:
    """Train and return the run manifest with the exact verified staged bundle digest."""

    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise EncoderDecoderTrainingError(f"training output must not already exist: {output}")
    _require_no_replace_publication()
    runtime_modules = _load_neural_runtime()
    with _isolated_deterministic_runtime(runtime_modules[0], config.seed):
        return _run_encoder_decoder_training(
            config,
            release_path,
            output,
            source_root=source_root,
            dependency_lock=dependency_lock,
            cache_dir=cache_dir,
            runtime_modules=runtime_modules,
        )


def run_encoder_decoder_training(
    config: EncoderDecoderTrainingConfigV1,
    release_path: Path,
    output: Path,
    *,
    source_root: Path,
    dependency_lock: Path,
    cache_dir: Path | None = None,
) -> EncoderDecoderRunManifestV1:
    """Compatibility wrapper returning the encoder-decoder run manifest."""

    return run_encoder_decoder_training_bundle(
        config,
        release_path,
        output,
        source_root=source_root,
        dependency_lock=dependency_lock,
        cache_dir=cache_dir,
    ).run_manifest


def _evaluate_encoder_decoder_checkpoint(
    config: EncoderDecoderTrainingConfigV1,
    checkpoint: Path,
    run_manifest_sha256: str,
    *,
    release: TrainingReleaseSnapshot,
    runtime_modules: tuple[Any, Any, Any],
) -> ValidationMetricsV1:
    torch, transformers, _ = runtime_modules
    for name, max_bytes in _METADATA_SIZE_LIMITS.items():
        _read_bounded_regular_bytes(checkpoint, name, max_bytes=max_bytes)
    with tempfile.TemporaryDirectory(prefix="ste-checkpoint-evaluate-") as private:
        tokenizer, model, checkpoint_artifacts = _reload_components(
            transformers,
            checkpoint,
            Path(private),
        )
        materialized = Path(private) / "checkpoint"
        artifact_map = {identity.path: identity for identity in checkpoint_artifacts}
        manifest_identity = artifact_map.get("run-manifest.json")
        if manifest_identity is None:
            raise EncoderDecoderTrainingError("checkpoint is missing run-manifest.json")
        if manifest_identity.sha256 != run_manifest_sha256:
            raise EncoderDecoderTrainingError("checkpoint run manifest digest does not match")
        config_bytes = _read_bounded_regular_bytes(
            materialized,
            "training-config.json",
            max_bytes=_MAX_TRAINING_CONFIG_BYTES,
        )
        validation_bytes = _read_bounded_regular_bytes(
            materialized,
            "validation-metrics.json",
            max_bytes=_MAX_VALIDATION_METRICS_BYTES,
        )
        manifest_bytes = _read_bounded_regular_bytes(
            materialized,
            "run-manifest.json",
            max_bytes=_MAX_RUN_MANIFEST_BYTES,
        )
        manifest = _parse_run_manifest_bytes(manifest_bytes)
        if config_bytes != canonical_training_config_json(config):
            raise EncoderDecoderTrainingError("checkpoint training configuration does not match")
        actual_output_artifacts = tuple(
            identity
            for identity in checkpoint_artifacts
            if identity.path not in {"run-manifest.json", ARTIFACT_MANIFEST_NAME}
        )
        _validate_model_tokenizer_compatibility(tokenizer, model, config)
        prepared = preflight_encoder_decoder_tokenizer(tokenizer, release, config)
        expected_micro_steps = config.max_steps * config.gradient_accumulation_steps
        expected_order = tuple(
            record.record_id
            for record in _take_batch(
                _seeded_records(
                    torch,
                    tuple(record for record in prepared if record.split == "train"),
                    config.seed,
                ),
                expected_micro_steps * config.micro_batch_size,
            )
        )
        if (
            manifest.training_config_sha256 != training_config_sha256(config)
            or manifest.training_config != config
            or manifest.corpus.dataset_version != release.manifest.dataset_version
            or manifest.corpus.manifest_sha256 != release.manifest_sha256
            or manifest.corpus.artifacts != release.artifact_sha256
            or manifest.base_model != config.base_model
            or manifest.tokenizer != config.tokenizer
            or manifest.base_model_artifacts != manifest.tokenizer_artifacts
            or tuple(
                sorted(
                    manifest.base_model_artifacts,
                    key=lambda identity: identity.path,
                )
            )
            != manifest.base_model_artifacts
            or len({identity.path for identity in manifest.base_model_artifacts})
            != len(manifest.base_model_artifacts)
            or not any(identity.path == "config.json" for identity in manifest.base_model_artifacts)
            or not any(
                Path(identity.path).suffix == ".safetensors"
                for identity in manifest.base_model_artifacts
            )
            or manifest.seed != config.seed
            or manifest.optimizer_steps != config.max_steps
            or manifest.micro_steps != expected_micro_steps
            or manifest.record_order != expected_order
            or len(manifest.optimizer_losses) != config.max_steps
            or manifest.validation.record_count != len(release.validation)
            or validation_bytes != canonical_validation_metrics_json(manifest.validation)
            or manifest.parameter_counts != _parameter_counts(model)
            or manifest.output_artifacts != actual_output_artifacts
            or manifest.evaluation_command != _EVALUATION_COMMAND
        ):
            raise EncoderDecoderTrainingError("checkpoint run manifest identity does not match")
        evaluated = _evaluate(torch, model, tokenizer, prepared)
        if evaluated != manifest.validation:
            raise EncoderDecoderTrainingError(
                "checkpoint validation does not reproduce the run manifest"
            )
        return evaluated


def evaluate_encoder_decoder_checkpoint(
    config: EncoderDecoderTrainingConfigV1,
    release_path: Path,
    checkpoint: Path,
    run_manifest_sha256: str,
) -> ValidationMetricsV1:
    """Reload and score one safe local checkpoint on the pinned validation split."""

    checkpoint = checkpoint.absolute()
    release = read_training_release(release_path, config.corpus)
    if len(run_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in run_manifest_sha256
    ):
        raise EncoderDecoderTrainingError("run manifest SHA-256 must be 64 lowercase hex digits")
    runtime_modules = _load_neural_runtime()
    with _isolated_deterministic_runtime(runtime_modules[0], config.seed):
        return _evaluate_encoder_decoder_checkpoint(
            config,
            checkpoint,
            run_manifest_sha256,
            release=release,
            runtime_modules=runtime_modules,
        )
