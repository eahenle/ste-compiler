"""Deterministic offline smoke training for the decoder-only LoRA architecture."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib
import importlib.metadata
import json
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
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from operator import index
from pathlib import Path
from typing import Any, Final, Literal, Protocol, SupportsIndex, cast

from pydantic import Field, FiniteFloat

from ste_compiler.realizer.decoder_protocol import (
    DECODER_PROMPT_PROFILE,
    DecoderProtocolError,
    canonical_decoder_prompt,
    segmented_symbol_plan_tokens,
    validate_lora_adapter_identity,
)
from ste_compiler.realizer.neural import NeuralRealizerUnavailable

from .config import (
    ArtifactIdentityV1,
    DecoderOnlyLoRATrainingConfigV1,
    StrictTrainingModel,
    canonical_training_config_json,
    training_config_sha256,
)
from .release_reader import ReleasedTrainingRecordV1, TrainingReleaseSnapshot

IGNORE_INDEX = -100
MODEL_SNAPSHOT_MANIFEST = "snapshot-manifest.json"
MODEL_SNAPSHOT_SCHEMA: Final = "ste-local-causal-lm-snapshot-v1"
FIXTURE_PROFILE: Final = "tiny-byte-bpe-gpt2-v1"
RUN_MANIFEST_SCHEMA: Final = "ste-decoder-lora-run-v1"
MAX_SNAPSHOT_MANIFEST_BYTES = 1024 * 1024
MAX_SNAPSHOT_FILE_BYTES = 64 * 1024 * 1024
MAX_SNAPSHOT_BYTES = 128 * 1024 * 1024
MAX_ADAPTER_METADATA_BYTES = 1024 * 1024
MAX_ADAPTER_WEIGHTS_BYTES = 64 * 1024 * 1024
MAX_ADAPTER_BYTES = 66 * 1024 * 1024
SAFE_MODEL_SUFFIXES = frozenset(
    {
        ".codes",
        ".json",
        ".model",
        ".safetensors",
        ".tokenizer",
        ".txt",
    }
)
SAFE_ADAPTER_FILES = frozenset(
    {
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
    }
)
UNSAFE_ARTIFACT_SUFFIXES = frozenset(
    {
        ".bin",
        ".ckpt",
        ".joblib",
        ".pickle",
        ".pkl",
        ".pt",
        ".pth",
    }
)
_COMMIT = re.compile(r"[0-9a-f]{40}", re.ASCII)
_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)


class DecoderLoRATrainingError(RuntimeError):
    """Decoder-only LoRA training failed closed."""


class TrainingTokenizer(Protocol):
    eos_token_id: int | None
    pad_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]: ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str: ...


class ArtifactDigestV1(StrictTrainingModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)


class LocalModelSnapshotManifestV1(StrictTrainingModel):
    schema_version: Literal["ste-local-causal-lm-snapshot-v1"]
    fixture_profile: Literal["tiny-byte-bpe-gpt2-v1"]
    base_model: ArtifactIdentityV1
    tokenizer: ArtifactIdentityV1
    artifacts: tuple[ArtifactDigestV1, ...]


class DependencyVersionV1(StrictTrainingModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class HardwareProvenanceV1(StrictTrainingModel):
    device: Literal["cpu"]
    platform: str = Field(min_length=1)
    machine: str = Field(min_length=1)
    processor: str
    python: str = Field(min_length=1)
    torch_threads: int = Field(gt=0)


class SourceProvenanceV1(StrictTrainingModel):
    package_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty: Literal[False]
    package_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependency_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DecoderLoRARunManifestV1(StrictTrainingModel):
    schema_version: Literal["ste-decoder-lora-run-v1"]
    architecture: Literal["decoder-only-lora"]
    status: Literal["completed"]
    prompt_profile: Literal["decoder-only-symbol-plan-v1"]
    training_config: DecoderOnlyLoRATrainingConfigV1
    training_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_artifacts: tuple[tuple[str, str], ...]
    model_snapshot_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_snapshot_artifacts: tuple[ArtifactDigestV1, ...]
    source: SourceProvenanceV1
    dependencies: tuple[DependencyVersionV1, ...]
    seed: int = Field(ge=0, le=2**63 - 1)
    optimizer_steps: int = Field(gt=0)
    sample_order: tuple[str, ...] = Field(min_length=1)
    training_losses: tuple[FiniteFloat, ...] = Field(min_length=1)
    validation_loss: FiniteFloat = Field(ge=0)
    total_parameters: int = Field(gt=0)
    trainable_parameters: int = Field(gt=0)
    matched_lora_parameters: tuple[str, ...] = Field(min_length=1)
    hardware: HardwareProvenanceV1
    duration_seconds: FiniteFloat = Field(ge=0)
    output_artifacts: tuple[ArtifactDigestV1, ...] = Field(min_length=1)
    evaluation_command: str = Field(min_length=1)


@dataclass(frozen=True)
class DecoderTrainingExample:
    record_id: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_length: int
    target_length: int


@dataclass(frozen=True)
class VerifiedModelSnapshot:
    manifest: LocalModelSnapshotManifestV1
    manifest_sha256: str
    files: tuple[tuple[str, bytes], ...]


@dataclass(frozen=True)
class RuntimeModules:
    torch: Any
    transformers: Any
    peft: Any
    safetensors: Any


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: object, *, indent: int | None = None) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _artifact(path: str, data: bytes) -> ArtifactDigestV1:
    return ArtifactDigestV1(path=path, sha256=_sha256(data), bytes=len(data))


def _runtime_modules() -> RuntimeModules:
    try:
        return RuntimeModules(
            torch=importlib.import_module("torch"),
            transformers=importlib.import_module("transformers"),
            peft=importlib.import_module("peft"),
            safetensors=importlib.import_module("safetensors"),
        )
    except ImportError as error:
        raise NeuralRealizerUnavailable(
            "decoder-only LoRA training requires the 'neural' extra: install ste-compiler[neural]"
        ) from error


def _integer_tokens(value: Sequence[int], *, field: str) -> tuple[int, ...]:
    token_ids: list[int] = []
    try:
        for item in value:
            if isinstance(item, bool):
                raise TypeError
            token_ids.append(index(cast(SupportsIndex, item)))
    except TypeError as error:
        raise DecoderLoRATrainingError(f"{field} must contain only integer token IDs") from error
    return tuple(token_ids)


def build_decoder_training_example(
    record: ReleasedTrainingRecordV1,
    tokenizer: TrainingTokenizer,
    *,
    max_sequence_tokens: int,
) -> DecoderTrainingExample:
    """Build one exact causal-LM example with prompt masking and one labeled EOS."""

    if max_sequence_tokens <= 0:
        raise ValueError("max_sequence_tokens must be positive")
    prompt = canonical_decoder_prompt(record.serialized_ir)
    prompt_ids = _integer_tokens(
        tokenizer.encode(prompt, add_special_tokens=True),
        field="prompt token sequence",
    )
    if not prompt_ids:
        raise DecoderLoRATrainingError(f"record {record.record_id!r} has an empty encoded prompt")
    prompt_round_trip = tokenizer.decode(
        prompt_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if prompt_round_trip != prompt:
        raise DecoderLoRATrainingError(
            f"record {record.record_id!r} prompt does not round-trip through the tokenizer"
        )
    try:
        target_ids = segmented_symbol_plan_tokens(tokenizer, record.symbols)
    except DecoderProtocolError as error:
        raise DecoderLoRATrainingError(
            f"record {record.record_id!r} has an incompatible symbolic target: {error}"
        ) from error
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise DecoderLoRATrainingError("the tokenizer must define eos_token_id")
    input_ids = (*prompt_ids, *target_ids, eos_token_id)
    if len(input_ids) > max_sequence_tokens:
        raise DecoderLoRATrainingError(
            f"record {record.record_id!r} needs {len(input_ids)} tokens, exceeding "
            f"max_sequence_tokens={max_sequence_tokens}"
        )
    labels = (*([IGNORE_INDEX] * len(prompt_ids)), *target_ids, eos_token_id)
    if labels[len(prompt_ids) :].count(eos_token_id) != 1:
        raise DecoderLoRATrainingError(
            f"record {record.record_id!r} does not have exactly one supervised EOS"
        )
    return DecoderTrainingExample(
        record_id=record.record_id,
        input_ids=input_ids,
        attention_mask=(1,) * len(input_ids),
        labels=labels,
        prompt_length=len(prompt_ids),
        target_length=len(target_ids) + 1,
    )


def _safe_snapshot_name(name: str) -> bool:
    path = Path(name)
    return (
        path.name == name
        and name != MODEL_SNAPSHOT_MANIFEST
        and path.suffix.casefold() in SAFE_MODEL_SUFFIXES
        and path.suffix.casefold() not in UNSAFE_ARTIFACT_SUFFIXES
    )


def _read_regular_file(
    directory_fd: int,
    name: str,
    *,
    max_bytes: int,
    expected_bytes: int | None = None,
) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        file_fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise DecoderLoRATrainingError(f"cannot open artifact {name!r}: {error}") from error
    try:
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise DecoderLoRATrainingError(f"artifact must be a single-link regular file: {name}")
        if before.st_size > max_bytes:
            raise DecoderLoRATrainingError(f"artifact exceeds its size limit: {name}")
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise DecoderLoRATrainingError(f"artifact size does not match its manifest: {name}")
        chunks: list[bytes] = []
        byte_count = 0
        while chunk := os.read(file_fd, min(1024 * 1024, max_bytes + 1 - byte_count)):
            chunks.append(chunk)
            byte_count += len(chunk)
            if byte_count > max_bytes:
                raise DecoderLoRATrainingError(f"artifact exceeds its size limit: {name}")
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
            raise DecoderLoRATrainingError(f"artifact changed while it was read: {name}")
        return data
    except OSError as error:
        raise DecoderLoRATrainingError(f"cannot read artifact {name!r}: {error}") from error
    finally:
        os.close(file_fd)


def read_verified_model_snapshot(
    snapshot: Path,
    config: DecoderOnlyLoRATrainingConfigV1,
    expected_manifest_sha256: str,
) -> VerifiedModelSnapshot:
    """Read a complete content-bound local model/tokenizer snapshot into immutable bytes."""

    if _SHA256.fullmatch(expected_manifest_sha256) is None:
        raise DecoderLoRATrainingError(
            "expected model snapshot manifest SHA-256 must be 64 lowercase hexadecimal characters"
        )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        directory_fd = os.open(snapshot, flags)
    except OSError as error:
        raise DecoderLoRATrainingError(
            f"model snapshot must be a real directory: {snapshot}"
        ) from error
    try:
        names = set(os.listdir(directory_fd))
        if MODEL_SNAPSHOT_MANIFEST not in names:
            raise DecoderLoRATrainingError(f"model snapshot is missing {MODEL_SNAPSHOT_MANIFEST}")
        manifest_bytes = _read_regular_file(
            directory_fd,
            MODEL_SNAPSHOT_MANIFEST,
            max_bytes=MAX_SNAPSHOT_MANIFEST_BYTES,
        )
        manifest_sha256 = _sha256(manifest_bytes)
        if manifest_sha256 != expected_manifest_sha256:
            raise DecoderLoRATrainingError(
                "model snapshot manifest SHA-256 does not match the required run input"
            )
        try:
            manifest = LocalModelSnapshotManifestV1.model_validate_json(manifest_bytes)
        except ValueError as error:
            raise DecoderLoRATrainingError(
                f"model snapshot manifest is invalid: {error}"
            ) from error
        if manifest.base_model != config.base_model or manifest.tokenizer != config.tokenizer:
            raise DecoderLoRATrainingError(
                "model snapshot identity does not match the training configuration"
            )
        artifact_by_path = {artifact.path: artifact for artifact in manifest.artifacts}
        if (
            len(artifact_by_path) != len(manifest.artifacts)
            or set(artifact_by_path) != names - {MODEL_SNAPSHOT_MANIFEST}
            or any(not _safe_snapshot_name(name) for name in artifact_by_path)
        ):
            raise DecoderLoRATrainingError("model snapshot has an invalid artifact set")
        if "config.json" not in artifact_by_path or "model.safetensors" not in artifact_by_path:
            raise DecoderLoRATrainingError(
                "model snapshot requires config.json and model.safetensors"
            )
        if "tokenizer.json" not in artifact_by_path:
            raise DecoderLoRATrainingError("model snapshot requires tokenizer.json")
        if any(artifact.bytes > MAX_SNAPSHOT_FILE_BYTES for artifact in manifest.artifacts):
            raise DecoderLoRATrainingError("model snapshot declares an oversized artifact")
        if sum(artifact.bytes for artifact in manifest.artifacts) > MAX_SNAPSHOT_BYTES:
            raise DecoderLoRATrainingError("model snapshot exceeds the total size limit")
        files: list[tuple[str, bytes]] = []
        for name in sorted(artifact_by_path):
            artifact = artifact_by_path[name]
            data = _read_regular_file(
                directory_fd,
                name,
                max_bytes=MAX_SNAPSHOT_FILE_BYTES,
                expected_bytes=artifact.bytes,
            )
            if _sha256(data) != artifact.sha256:
                raise DecoderLoRATrainingError(
                    f"model snapshot artifact SHA-256 does not match: {name}"
                )
            files.append((name, data))
        return VerifiedModelSnapshot(
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            files=tuple(files),
        )
    finally:
        os.close(directory_fd)


def model_snapshot_manifest_sha256(snapshot: Path) -> str:
    """Read the bounded regular snapshot manifest and return its content digest."""

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        directory_fd = os.open(snapshot, flags)
    except OSError as error:
        raise DecoderLoRATrainingError(
            f"model snapshot must be a real directory: {snapshot}"
        ) from error
    try:
        manifest_bytes = _read_regular_file(
            directory_fd,
            MODEL_SNAPSHOT_MANIFEST,
            max_bytes=MAX_SNAPSHOT_MANIFEST_BYTES,
        )
    finally:
        os.close(directory_fd)
    return _sha256(manifest_bytes)


def _require_two_step_smoke(config: DecoderOnlyLoRATrainingConfigV1) -> None:
    if config.max_steps != 2:
        raise DecoderLoRATrainingError(
            "decoder smoke training requires max_steps=2; use a separate experiment "
            "runner for quality training"
        )


def _write_bytes(path: Path, data: bytes) -> None:
    path.write_bytes(data)


@contextmanager
def _isolated_deterministic_runtime(torch: Any, seed: int) -> Iterator[None]:
    python_random_state = random.getstate()
    torch_random_state = torch.get_rng_state()
    deterministic_enabled = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch_threads = int(torch.get_num_threads())
    try:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        yield
    finally:
        random.setstate(python_random_state)
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
        raise DecoderLoRATrainingError(
            f"atomic no-replace publication is unsupported on {sys.platform}"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise DecoderLoRATrainingError(f"output path already exists: {destination}")
    raise DecoderLoRATrainingError(
        f"cannot atomically publish output {destination}: {os.strerror(error_number)}"
    )


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for path in [*sorted(directories, reverse=True), root]:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _atomic_output_directory(
    output: Path,
    builder: Callable[[Path], Any],
) -> Any:
    if output.exists() or output.is_symlink():
        raise DecoderLoRATrainingError(f"output path already exists: {output}")
    parent = output.parent
    if not parent.is_dir() or parent.is_symlink():
        raise DecoderLoRATrainingError(f"output parent must be a real directory: {parent}")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=parent))
    try:
        result = builder(stage)
        _fsync_tree(stage)
        _rename_no_replace(stage, output)
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return result
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _snapshot_artifacts(directory: Path) -> tuple[ArtifactDigestV1, ...]:
    artifacts: list[ArtifactDigestV1] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
            raise DecoderLoRATrainingError(
                f"fixture produced a non-regular or multiply linked artifact: {path.name}"
            )
        if not _safe_snapshot_name(path.name):
            raise DecoderLoRATrainingError(f"fixture produced an unsafe artifact: {path.name}")
        artifacts.append(_artifact(path.name, path.read_bytes()))
    return tuple(artifacts)


def prepare_decoder_smoke_fixture(
    config: DecoderOnlyLoRATrainingConfigV1,
    release: TrainingReleaseSnapshot,
    output: Path,
) -> LocalModelSnapshotManifestV1:
    """Create one deterministic tiny local causal LM and byte-level tokenizer."""

    _require_two_step_smoke(config)
    modules = _runtime_modules()
    torch = modules.torch
    transformers = modules.transformers
    tokenizers = importlib.import_module("tokenizers")
    tokenizers_models = importlib.import_module("tokenizers.models")
    tokenizers_pre = importlib.import_module("tokenizers.pre_tokenizers")
    tokenizers_decoders = importlib.import_module("tokenizers.decoders")
    tokenizers_trainers = importlib.import_module("tokenizers.trainers")
    tokenizers_processors = importlib.import_module("tokenizers.processors")

    def build(stage: Path) -> LocalModelSnapshotManifestV1:
        tokenizer = tokenizers.Tokenizer(tokenizers_models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = tokenizers_pre.ByteLevel(
            add_prefix_space=False,
            use_regex=False,
        )
        tokenizer.decoder = tokenizers_decoders.ByteLevel()
        special_tokens = ["<pad>", "<eos>", "<unk>", "<bos>"]
        texts = [
            text
            for record in release.train
            for text in (
                canonical_decoder_prompt(record.serialized_ir),
                record.symbols,
            )
        ]
        tokenizer.train_from_iterator(
            texts,
            trainer=tokenizers_trainers.BpeTrainer(
                vocab_size=512,
                min_frequency=1,
                show_progress=False,
                special_tokens=special_tokens,
                initial_alphabet=tokenizers_pre.ByteLevel.alphabet(),
            ),
        )
        bos_token_id = tokenizer.token_to_id("<bos>")
        if bos_token_id is None:
            raise DecoderLoRATrainingError("smoke tokenizer did not create a BOS token")
        tokenizer.post_processor = tokenizers_processors.TemplateProcessing(
            single="<bos> $A",
            special_tokens=[("<bos>", bos_token_id)],
        )
        fast_tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            pad_token="<pad>",
            eos_token="<eos>",
            unk_token="<unk>",
            bos_token="<bos>",
        )
        _preflight_release(release, fast_tokenizer, config)
        fast_tokenizer.save_pretrained(stage)
        model_config = transformers.GPT2Config(
            vocab_size=len(fast_tokenizer),
            n_positions=config.max_sequence_tokens,
            n_ctx=config.max_sequence_tokens,
            n_embd=16,
            n_layer=1,
            n_head=1,
            bos_token_id=fast_tokenizer.bos_token_id,
            eos_token_id=fast_tokenizer.eos_token_id,
            pad_token_id=fast_tokenizer.pad_token_id,
        )
        model = transformers.GPT2LMHeadModel(model_config)
        model.save_pretrained(stage, safe_serialization=True)
        artifacts = _snapshot_artifacts(stage)
        manifest = LocalModelSnapshotManifestV1(
            schema_version=MODEL_SNAPSHOT_SCHEMA,
            fixture_profile=FIXTURE_PROFILE,
            base_model=config.base_model,
            tokenizer=config.tokenizer,
            artifacts=artifacts,
        )
        _write_bytes(
            stage / MODEL_SNAPSHOT_MANIFEST,
            _canonical_json(manifest.model_dump(mode="json"), indent=2),
        )
        return manifest

    with _isolated_deterministic_runtime(torch, config.seed):
        manifest = cast(LocalModelSnapshotManifestV1, _atomic_output_directory(output, build))
    manifest_sha256 = model_snapshot_manifest_sha256(output)
    read_verified_model_snapshot(output, config, manifest_sha256)
    return manifest


@contextmanager
def _materialized_snapshot(snapshot: VerifiedModelSnapshot) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="ste-decoder-base-") as temporary:
        root = Path(temporary)
        for name, data in snapshot.files:
            _write_bytes(root / name, data)
        yield root


@contextmanager
def _materialized_adapter(files: Sequence[tuple[str, bytes]]) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="ste-decoder-adapter-") as temporary:
        root = Path(temporary)
        for name, data in files:
            _write_bytes(root / name, data)
        yield root


def _load_tokenizer_and_base(
    modules: RuntimeModules,
    snapshot_path: Path,
) -> tuple[Any, Any]:
    tokenizer = modules.transformers.AutoTokenizer.from_pretrained(
        str(snapshot_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    base_model = modules.transformers.AutoModelForCausalLM.from_pretrained(
        str(snapshot_path),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    return tokenizer, base_model


def _collate(
    torch: Any,
    examples: Sequence[DecoderTrainingExample],
    *,
    pad_token_id: int,
) -> dict[str, Any]:
    max_length = max(len(example.input_ids) for example in examples)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []
    for example in examples:
        padding = max_length - len(example.input_ids)
        input_ids.append([*example.input_ids, *([pad_token_id] * padding)])
        attention_mask.append([*example.attention_mask, *([0] * padding)])
        labels.append([*example.labels, *([IGNORE_INDEX] * padding)])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _example_batches(
    examples: Sequence[DecoderTrainingExample],
    *,
    seed: int,
    batch_size: int,
) -> Iterator[tuple[DecoderTrainingExample, ...]]:
    if not examples:
        raise DecoderLoRATrainingError("training split is empty")
    rng = random.Random(seed)
    indices: list[int] = []
    position = 0
    while True:
        if position >= len(indices):
            indices = list(range(len(examples)))
            rng.shuffle(indices)
            position = 0
        batch: list[DecoderTrainingExample] = []
        while len(batch) < batch_size:
            if position >= len(indices):
                indices = list(range(len(examples)))
                rng.shuffle(indices)
                position = 0
            batch.append(examples[indices[position]])
            position += 1
        yield tuple(batch)


def _finite_loss(value: object, *, context: str) -> float:
    loss = float(cast(Any, value).detach().cpu().item())
    if not math.isfinite(loss) or loss < 0:
        raise DecoderLoRATrainingError(f"{context} produced a non-finite loss")
    return loss


def _build_examples(
    records: Sequence[ReleasedTrainingRecordV1],
    tokenizer: TrainingTokenizer,
    config: DecoderOnlyLoRATrainingConfigV1,
) -> tuple[DecoderTrainingExample, ...]:
    return tuple(
        build_decoder_training_example(
            record,
            tokenizer,
            max_sequence_tokens=config.max_sequence_tokens,
        )
        for record in records
    )


def _preflight_release(
    release: TrainingReleaseSnapshot,
    tokenizer: TrainingTokenizer,
    config: DecoderOnlyLoRATrainingConfigV1,
) -> None:
    _build_examples(
        (
            *release.train,
            *release.validation,
            *release.test,
            *release.adversarial,
        ),
        tokenizer,
        config,
    )


def _evaluate_model(
    model: Any,
    tokenizer: TrainingTokenizer,
    examples: Sequence[DecoderTrainingExample],
    *,
    batch_size: int,
    torch: Any,
) -> float:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise DecoderLoRATrainingError("tokenizer must define a pad or EOS token")
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for offset in range(0, len(examples), batch_size):
            batch = _collate(
                torch,
                examples[offset : offset + batch_size],
                pad_token_id=pad_token_id,
            )
            outputs = model(**batch)
            losses.append(_finite_loss(outputs.loss, context="validation"))
    if not losses:
        raise DecoderLoRATrainingError("validation split is empty")
    return sum(losses) / len(losses)


def _validate_saved_adapter(
    adapter_directory: Path,
    config: DecoderOnlyLoRATrainingConfigV1,
    modules: RuntimeModules,
) -> tuple[tuple[str, bytes], ...]:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        directory_fd = os.open(adapter_directory, flags)
    except OSError as error:
        raise DecoderLoRATrainingError(
            f"saved adapter must be a real directory: {adapter_directory}"
        ) from error
    try:
        names = set(os.listdir(directory_fd))
        if names != SAFE_ADAPTER_FILES:
            raise DecoderLoRATrainingError("saved adapter has an invalid artifact set")
        files = tuple(
            (
                name,
                _read_regular_file(
                    directory_fd,
                    name,
                    max_bytes=(
                        MAX_ADAPTER_WEIGHTS_BYTES
                        if name == "adapter_model.safetensors"
                        else MAX_ADAPTER_METADATA_BYTES
                    ),
                ),
            )
            for name in sorted(names)
        )
    finally:
        os.close(directory_fd)
    if sum(len(data) for _, data in files) > MAX_ADAPTER_BYTES:
        raise DecoderLoRATrainingError("saved adapter exceeds the total size limit")
    file_bytes = dict(files)
    try:
        raw_config = json.loads(file_bytes["adapter_config.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DecoderLoRATrainingError("saved adapter configuration is invalid") from error
    if not isinstance(raw_config, dict):
        raise DecoderLoRATrainingError("saved adapter configuration must be a JSON object")
    adapter_config = cast(Mapping[str, object], raw_config)

    class ConfigView:
        def __getattr__(self, name: str) -> object | None:
            return adapter_config.get(name)

    try:
        validate_lora_adapter_identity(
            config.base_model.repo_id,
            config.base_model.revision,
            ConfigView(),
        )
    except DecoderProtocolError as error:
        raise DecoderLoRATrainingError(str(error)) from error
    expected_lora = {
        "r": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "bias": config.lora.bias,
    }
    mismatched_lora = [
        name for name, expected in expected_lora.items() if adapter_config.get(name) != expected
    ]
    raw_targets = adapter_config.get("target_modules")
    if (
        not isinstance(raw_targets, list)
        or any(not isinstance(target, str) for target in raw_targets)
        or set(raw_targets) != set(config.lora.target_modules)
    ):
        mismatched_lora.append("target_modules")
    if mismatched_lora:
        raise DecoderLoRATrainingError(
            "saved adapter configuration does not match training LoRA fields: "
            + ", ".join(sorted(mismatched_lora))
        )
    try:
        with (
            _materialized_adapter(files) as captured_adapter,
            modules.safetensors.safe_open(
                captured_adapter / "adapter_model.safetensors",
                framework="pt",
                device="cpu",
            ) as weights,
        ):
            if not list(weights.keys()):
                raise DecoderLoRATrainingError("saved adapter safetensors contain no tensors")
    except DecoderLoRATrainingError:
        raise
    except Exception as error:
        raise DecoderLoRATrainingError("saved adapter safetensors are invalid") from error
    return files


def _package_tree_sha256(package_root: Path) -> str:
    if not package_root.is_dir() or package_root.is_symlink():
        raise DecoderLoRATrainingError(
            f"Python package root must be a real directory: {package_root}"
        )
    paths = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file() and (path.suffix == ".py" or path.name == "py.typed")
    )
    if not paths or any(path.is_symlink() or path.stat().st_nlink != 1 for path in paths):
        raise DecoderLoRATrainingError(
            f"Python package tree contains an invalid source entry: {package_root}"
        )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(package_root).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _source_provenance(source_checkout: Path) -> SourceProvenanceV1:
    if (
        not (source_checkout / "pyproject.toml").is_file()
        or not (source_checkout / "src/ste_compiler").is_dir()
    ):
        raise DecoderLoRATrainingError(
            "source checkout must contain pyproject.toml and src/ste_compiler"
        )
    try:
        commit = subprocess.run(
            ["git", "-C", str(source_checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(source_checkout), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise DecoderLoRATrainingError("cannot derive source-checkout Git provenance") from error
    if _COMMIT.fullmatch(commit) is None:
        raise DecoderLoRATrainingError("source checkout did not resolve to a full Git commit")
    if status:
        raise DecoderLoRATrainingError(
            "source checkout must be clean so its commit identifies the executed package"
        )
    checkout_package = source_checkout / "src/ste_compiler"
    checkout_tree_sha256 = _package_tree_sha256(checkout_package)
    runtime_tree_sha256 = _package_tree_sha256(Path(__file__).parents[1])
    if checkout_tree_sha256 != runtime_tree_sha256:
        raise DecoderLoRATrainingError(
            "source checkout package tree does not match the code executing training"
        )
    lock_path = source_checkout / "uv.lock"
    try:
        lock_bytes = lock_path.read_bytes()
    except OSError as error:
        raise DecoderLoRATrainingError(
            "source checkout does not contain a readable uv.lock"
        ) from error
    return SourceProvenanceV1(
        package_commit=commit,
        dirty=False,
        package_tree_sha256=runtime_tree_sha256,
        dependency_lock_sha256=_sha256(lock_bytes),
    )


def _dependency_versions() -> tuple[DependencyVersionV1, ...]:
    names = (
        "huggingface-hub",
        "peft",
        "safetensors",
        "ste-compiler",
        "tokenizers",
        "torch",
        "transformers",
    )
    versions: list[DependencyVersionV1] = []
    for name in names:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise DecoderLoRATrainingError(
                f"cannot derive runtime dependency version for {name}"
            ) from error
        versions.append(DependencyVersionV1(name=name, version=version))
    return tuple(versions)


def _hardware(torch: Any) -> HardwareProvenanceV1:
    return HardwareProvenanceV1(
        device="cpu",
        platform=platform.platform(),
        machine=platform.machine() or "unknown",
        processor=platform.processor(),
        python=platform.python_version(),
        torch_threads=int(torch.get_num_threads()),
    )


def _model_card(
    config: DecoderOnlyLoRATrainingConfigV1,
    release: TrainingReleaseSnapshot,
) -> bytes:
    return (
        "# Decoder-only LoRA smoke adapter\n\n"
        "This adapter is a two-step offline mechanics smoke artifact. It is not a quality result, "
        "not a published checkpoint, and not suitable for production use.\n\n"
        f"- Prompt profile: `{config.prompt_profile}`\n"
        f"- Dataset: `{release.manifest.dataset_version}`\n"
        f"- Dataset manifest SHA-256: `{release.manifest_sha256}`\n"
        f"- Base identity: `{config.base_model.repo_id}@{config.base_model.revision}`\n"
    ).encode()


def _write_checksums(root: Path) -> None:
    files = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "checksums.sha256"
    ]
    lines = [
        f"{_sha256(path.read_bytes())}  {path.relative_to(root).as_posix()}\n" for path in files
    ]
    _write_bytes(root / "checksums.sha256", "".join(lines).encode("utf-8"))


def evaluate_decoder_lora_adapter(
    config: DecoderOnlyLoRATrainingConfigV1,
    release: TrainingReleaseSnapshot,
    model_snapshot: Path,
    model_snapshot_manifest_sha256: str,
    adapter_directory: Path,
) -> float:
    """Reload one safe local adapter and return deterministic validation loss."""

    _require_two_step_smoke(config)
    verified_snapshot = read_verified_model_snapshot(
        model_snapshot,
        config,
        model_snapshot_manifest_sha256,
    )
    modules = _runtime_modules()
    adapter_files = _validate_saved_adapter(adapter_directory, config, modules)
    torch = modules.torch
    with (
        _isolated_deterministic_runtime(torch, config.seed),
        _materialized_snapshot(verified_snapshot) as snapshot_path,
        _materialized_adapter(adapter_files) as adapter_path,
    ):
        tokenizer, base_model = _load_tokenizer_and_base(modules, snapshot_path)
        _preflight_release(release, tokenizer, config)
        validation_examples = _build_examples(release.validation, tokenizer, config)
        reloaded = modules.peft.PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
            local_files_only=True,
            is_trainable=False,
        )
        reloaded.to("cpu")
        return _evaluate_model(
            reloaded,
            tokenizer,
            validation_examples,
            batch_size=config.micro_batch_size,
            torch=torch,
        )


def run_decoder_lora_training(
    config: DecoderOnlyLoRATrainingConfigV1,
    release: TrainingReleaseSnapshot,
    model_snapshot: Path,
    model_snapshot_manifest_sha256: str,
    output: Path,
    *,
    source_checkout: Path,
    evaluation_command: str,
) -> DecoderLoRARunManifestV1:
    """Run deterministic manual LoRA training and atomically publish a safe adapter."""

    if config.prompt_profile != DECODER_PROMPT_PROFILE:
        raise DecoderLoRATrainingError(
            "training prompt profile does not match the decoder protocol"
        )
    _require_two_step_smoke(config)
    verified_snapshot = read_verified_model_snapshot(
        model_snapshot,
        config,
        model_snapshot_manifest_sha256,
    )
    source = _source_provenance(source_checkout)
    modules = _runtime_modules()
    torch = modules.torch
    dependencies = _dependency_versions()
    started = time.perf_counter()

    def build(stage: Path) -> DecoderLoRARunManifestV1:
        with _materialized_snapshot(verified_snapshot) as snapshot_path:
            tokenizer, base_model = _load_tokenizer_and_base(modules, snapshot_path)
            _preflight_release(release, tokenizer, config)
            train_examples = _build_examples(release.train, tokenizer, config)
            validation_examples = _build_examples(release.validation, tokenizer, config)
            peft_config = modules.peft.LoraConfig(
                task_type=modules.peft.TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config.lora.rank,
                lora_alpha=config.lora.alpha,
                lora_dropout=config.lora.dropout,
                bias=config.lora.bias,
                target_modules=list(config.lora.target_modules),
            )
            model = modules.peft.get_peft_model(base_model, peft_config)
            active_config = model.peft_config["default"]
            active_config.base_model_name_or_path = config.base_model.repo_id
            active_config.revision = config.base_model.revision
            model.to("cpu")
            model.train()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            if not trainable:
                raise DecoderLoRATrainingError(
                    "LoRA configuration selected no trainable parameters"
                )
            matched_names = tuple(
                sorted(
                    name
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and "lora_" in name
                )
            )
            if not matched_names:
                raise DecoderLoRATrainingError("LoRA target modules matched no adapter parameters")
            total_parameters = sum(parameter.numel() for parameter in model.parameters())
            trainable_parameters = sum(parameter.numel() for parameter in trainable)
            optimizer = torch.optim.AdamW(
                trainable,
                lr=config.optimizer.learning_rate,
                weight_decay=config.optimizer.weight_decay,
            )
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = tokenizer.eos_token_id
            if pad_token_id is None:
                raise DecoderLoRATrainingError("tokenizer must define a pad or EOS token")
            batches = _example_batches(
                train_examples,
                seed=config.seed,
                batch_size=config.micro_batch_size,
            )
            training_losses: list[float] = []
            sample_order: list[str] = []
            for _step in range(config.max_steps):
                optimizer.zero_grad(set_to_none=True)
                accumulated_loss = 0.0
                for _accumulation in range(config.gradient_accumulation_steps):
                    examples = next(batches)
                    sample_order.extend(example.record_id for example in examples)
                    batch = _collate(
                        torch,
                        examples,
                        pad_token_id=pad_token_id,
                    )
                    outputs = model(**batch)
                    raw_loss = _finite_loss(outputs.loss, context="training")
                    accumulated_loss += raw_loss
                    (outputs.loss / config.gradient_accumulation_steps).backward()
                optimizer.step()
                training_losses.append(accumulated_loss / config.gradient_accumulation_steps)

            adapter_directory = stage / "adapter"
            adapter_directory.mkdir()
            model.save_pretrained(
                adapter_directory,
                safe_serialization=True,
                save_embedding_layers=False,
            )
            _write_bytes(adapter_directory / "README.md", _model_card(config, release))
            adapter_files = _validate_saved_adapter(adapter_directory, config, modules)

            fresh_base = modules.transformers.AutoModelForCausalLM.from_pretrained(
                str(snapshot_path),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
            )
            reloaded = modules.peft.PeftModel.from_pretrained(
                fresh_base,
                str(adapter_directory),
                local_files_only=True,
                is_trainable=False,
            )
            reloaded.to("cpu")
            validation_loss = _evaluate_model(
                reloaded,
                tokenizer,
                validation_examples,
                batch_size=config.micro_batch_size,
                torch=torch,
            )

        _write_bytes(stage / "training-config.json", canonical_training_config_json(config))
        output_artifacts = tuple(
            _artifact(
                f"adapter/{name}",
                data,
            )
            for name, data in adapter_files
        ) + (
            _artifact(
                "training-config.json",
                (stage / "training-config.json").read_bytes(),
            ),
        )
        duration_seconds = round(time.perf_counter() - started, 6)
        manifest = DecoderLoRARunManifestV1(
            schema_version=RUN_MANIFEST_SCHEMA,
            architecture="decoder-only-lora",
            status="completed",
            prompt_profile=DECODER_PROMPT_PROFILE,
            training_config=config,
            training_config_sha256=training_config_sha256(config),
            corpus_manifest_sha256=release.manifest_sha256,
            corpus_artifacts=release.artifact_sha256,
            model_snapshot_manifest_sha256=verified_snapshot.manifest_sha256,
            model_snapshot_artifacts=verified_snapshot.manifest.artifacts,
            source=source,
            dependencies=dependencies,
            seed=config.seed,
            optimizer_steps=config.max_steps,
            sample_order=tuple(sample_order),
            training_losses=tuple(training_losses),
            validation_loss=validation_loss,
            total_parameters=total_parameters,
            trainable_parameters=trainable_parameters,
            matched_lora_parameters=matched_names,
            hardware=_hardware(torch),
            duration_seconds=duration_seconds,
            output_artifacts=output_artifacts,
            evaluation_command=evaluation_command,
        )
        _write_bytes(
            stage / "run-manifest.json",
            _canonical_json(manifest.model_dump(mode="json"), indent=2),
        )
        _write_checksums(stage)
        return manifest

    with _isolated_deterministic_runtime(torch, config.seed):
        return cast(DecoderLoRARunManifestV1, _atomic_output_directory(output, build))
