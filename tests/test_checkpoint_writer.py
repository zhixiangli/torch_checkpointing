# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import errno
import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch_checkpointing.checkpoint_writer as checkpoint_writer_module
from torch.testing._internal.common_utils import run_tests, TestCase
from torch_checkpointing.barriers import (
    Barrier,
    BarrierConfig,
    DefaultStoreBarrier,
    DefaultStoreBarrierConfig,
)
from torch_checkpointing.checkpoint_base import (
    CheckpointItem,
    CheckpointWriteInfo,
)
from torch_checkpointing.checkpoint_layout import (
    JsonSerialization,
    LayoutInfo,
    RawSerialization,
    TorchSerialization,
)
from torch_checkpointing.checkpoint_manager import CheckpointManager
from torch_checkpointing.checkpoint_writer import (
    CallbackArgs,
    CheckpointWriter,
    CheckpointWriterArgs,
    CheckpointWriterConfig,
    DEFAULT_TEMP_DIR_PREFIX,
)
from torch_checkpointing.config import AsyncCheckpointSaverConfig
from torch_checkpointing.logging_utils import EventLogger
from torch_checkpointing.metadata_serialization import METADATA_FILE_NAME
from torch_checkpointing.staging import CheckpointStagerConfig
from torch_checkpointing.storage.base_storage import (
    OccupiedDestinationError,
    StorageConfig,
)
from torch_checkpointing.storage.filesystem import (
    LocalFileSystemStorage,
    LocalFileSystemStorageConfig,
)
from torch_checkpointing.storage.fsspec_storage import FsspecStorageConfig
from torch_checkpointing.types import RankInfo


def simple_layout(rank: int) -> dict[str, LayoutInfo]:
    """Simple test layout that splits model and metadata."""
    return {
        "model": LayoutInfo(
            file_path=f"model_rank_{rank}.pt",
            serialization_format=TorchSerialization(),
        ),
        "optimizer": LayoutInfo(
            file_path=f"optimizer_rank_{rank}.pt",
            serialization_format=TorchSerialization(),
        ),
        "epoch": LayoutInfo(
            file_path=f"epoch_rank_{rank}.json",
            serialization_format=JsonSerialization(str),
        ),
        "step": LayoutInfo(
            file_path=f"step_rank_{rank}.json",
            serialization_format=JsonSerialization(str),
        ),
    }


def global_file_layout(rank: int) -> dict[str, LayoutInfo]:
    """Test layout that demonstrates global file functionality."""
    return {
        "model": LayoutInfo(
            file_path=f"model_rank_{rank}.pt",  # Per-rank file
            serialization_format=TorchSerialization(),
        ),
        "global_config": LayoutInfo(
            file_path="config.json",  # Global file - same for all ranks
            serialization_format=JsonSerialization(str),
        ),
        "optimizer": LayoutInfo(
            file_path=f"optimizer_rank_{rank}.pt",  # Per-rank file
            serialization_format=TorchSerialization(),
        ),
    }


class _NoopBarrier(Barrier):
    """Stands in for a barrier the ranks really do meet at.

    The commit path only depends on a barrier having passed, so a test that is
    about the commit -- not about coordination -- can hold this instead of
    standing up a store and the ranks to meet on it.
    """

    def __init__(self, config: BarrierConfig) -> None:
        pass

    def execute_barrier(self, timeout_secs: int) -> None:
        pass


@dataclass
class _NoopBarrierConfig(BarrierConfig):
    timeout_barrier_init_sec: int = 0

    def create_barrier(self, rank_info) -> _NoopBarrier:
        return _NoopBarrier(self)


class MockCallback:
    """Mock implementation of callback functions for testing."""

    def __init__(self) -> None:
        self.pre_finalize_called: bool = False
        self.finalize_called: bool = False
        self.pre_finalize_path: str | None = None
        self.finalize_path: str | None = None
        self.pre_finalize_event_logger: EventLogger | None = None
        self.finalize_event_logger: EventLogger | None = None

    def pre_finalize_callback(self, args: CallbackArgs):
        self.pre_finalize_called = True
        self.pre_finalize_path = args.path
        self.pre_finalize_event_logger = args.event_logger

    def finalize_callback(self, args: CallbackArgs):
        self.finalize_called = True
        self.finalize_path = args.path
        self.finalize_event_logger = args.event_logger


class _ConcurrencyProbe:
    def __init__(self, expected_concurrent_writes: int) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self._barrier_entries_remaining = expected_concurrent_writes
        self._barrier = (
            threading.Barrier(expected_concurrent_writes)
            if expected_concurrent_writes > 1
            else None
        )
        self.max_active = 0

    def __enter__(self):
        should_wait = False
        barrier = self._barrier
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            if barrier is not None and self._barrier_entries_remaining > 0:
                self._barrier_entries_remaining -= 1
                should_wait = True
        if should_wait:
            assert barrier is not None
            barrier.wait(timeout=5)

    def __exit__(self, exc_type, exc_value, traceback):
        with self._lock:
            self._active -= 1


class _ProbeStorage(LocalFileSystemStorage):
    def __init__(self, config: "_ProbeStorageConfig") -> None:
        super().__init__(config)
        self._config = config
        self._probe = config.probe

    def mkdir(self, path: Path, recursive: bool = True) -> None:
        self._config.mkdir_paths.append(path)
        super().mkdir(path, recursive)

    def write(self, path: Path, data) -> None:
        with self._probe:
            super().write(path, data)

    def rename(
        self,
        src_path: Path,
        dst_path: Path,
        is_directory: bool = False,
        background_cleanup: bool = False,
    ) -> None:
        self._config.rename_calls.append((src_path, dst_path))
        super().rename(src_path, dst_path, is_directory, background_cleanup)


class _ProbeStorageConfig(LocalFileSystemStorageConfig):
    def __init__(self, probe: _ConcurrencyProbe) -> None:
        super().__init__(use_direct_io=False)
        self.probe = probe
        self.mkdir_paths: list[Path] = []
        self.rename_calls: list[tuple[Path, Path]] = []

    def create_storage(self) -> _ProbeStorage:
        return _ProbeStorage(self)


class _RefusingOnceStorage(LocalFileSystemStorage):
    error: BaseException | None = None

    def rename(
        self,
        src_path: Path,
        dst_path: Path,
        is_directory: bool = False,
        background_cleanup: bool = False,
    ) -> None:
        if is_directory and self.error is not None:
            error = self.error
            self.error = None
            raise error
        super().rename(
            src_path,
            dst_path,
            is_directory=is_directory,
            background_cleanup=background_cleanup,
        )


@dataclass
class _RefusingOnceStorageConfig(LocalFileSystemStorageConfig):
    use_direct_io: bool = False

    def create_storage(self) -> _RefusingOnceStorage:
        return _RefusingOnceStorage(config=self)


class _FlatPrefixStorage(LocalFileSystemStorage):
    def delete(self, path: Path) -> None:
        super().delete(path)
        if not any(path.parent.iterdir()):
            path.parent.rmdir()


@dataclass
class _FlatPrefixStorageConfig(LocalFileSystemStorageConfig):
    use_direct_io: bool = False

    def create_storage(self) -> _FlatPrefixStorage:
        return _FlatPrefixStorage(config=self)


class TestCheckpointWriterConfig(TestCase):
    def test_default_values(self):
        """Test that CheckpointWriterConfig has the correct default values."""
        options = CheckpointWriterConfig()
        self.assertEqual(options.checkpoint_write_barrier_timeout_sec, 600)
        self.assertIsInstance(options.barrier_config, DefaultStoreBarrierConfig)
        self.assertEqual(options.file_write_max_threads, 1)
        self.assertEqual(options.temp_dir_prefix, DEFAULT_TEMP_DIR_PREFIX)
        self.assertTrue(options.write_to_temp_dir)

    def test_custom_values(self):
        """Test that CheckpointWriterConfig can be initialized with custom values."""
        options = CheckpointWriterConfig(checkpoint_write_barrier_timeout_sec=450)
        self.assertEqual(options.checkpoint_write_barrier_timeout_sec, 450)

    def test_file_write_max_threads_rejects_non_positive_values(self):
        """Test that file_write_max_threads must be positive."""
        with pytest.raises(ValueError, match="file_write_max_threads"):
            CheckpointWriterConfig(file_write_max_threads=0)

    def test_empty_temp_dir_prefix_rejected_with_barrier(self):
        """Test that an empty prefix is rejected when a barrier commits the write."""
        with pytest.raises(ValueError, match="temp_dir_prefix"):
            CheckpointWriterConfig(
                barrier_config=_NoopBarrierConfig(), temp_dir_prefix=""
            )

    def test_empty_temp_dir_prefix_allowed_when_writing_in_place(self):
        """Test that the prefix is inert when no temp dir is used."""
        options = CheckpointWriterConfig(
            barrier_config=None, write_to_temp_dir=False, temp_dir_prefix=""
        )
        self.assertEqual(options.temp_dir_prefix, "")

    def test_write_in_place_keeps_the_barrier(self):
        """Dropping the temp dir does not drop the coordination."""
        options = CheckpointWriterConfig(write_to_temp_dir=False)
        self.assertIsNotNone(options.barrier_config)
        self.assertFalse(options.write_to_temp_dir)

    def test_dropping_the_barrier_forces_writing_in_place(self):
        """Warned and forced, not rejected, while callers are being migrated.

        Nothing would ever rename the temp dir with no barrier to wait on, so
        writing in place is what this combination has always silently done.
        """
        with self.assertLogs(
            checkpoint_writer_module.__name__, level="WARNING"
        ) as logs:
            options = CheckpointWriterConfig(barrier_config=None)

        self.assertFalse(options.write_to_temp_dir)
        self.assertIn("write_to_temp_dir=False", "".join(logs.output))

    def test_explicit_write_to_temp_dir_without_a_barrier_is_forced_off(self):
        """Asking for it explicitly is warned about too, not honoured."""
        with self.assertLogs(checkpoint_writer_module.__name__, level="WARNING"):
            options = CheckpointWriterConfig(
                barrier_config=None, write_to_temp_dir=True
            )

        self.assertFalse(options.write_to_temp_dir)

    def test_default_barrier_needs_nothing_of_a_single_rank_writer(self):
        """The default barrier has to be free to hold when there is nobody to meet."""
        writer = CheckpointWriter(
            CheckpointWriterArgs(
                config=CheckpointWriterConfig(),
                rank_info=RankInfo(
                    global_rank=0, global_world_size=1, role_rank=0, role_world_size=1
                ),
                storage_config=LocalFileSystemStorageConfig(),
            )
        )

        # No process group here, and none needed: the barrier is real, it simply
        # has no store to meet on and nothing to wait for.
        self.assertIsInstance(writer._barrier, DefaultStoreBarrier)
        writer._barrier.execute_barrier(timeout_secs=0)

    def test_replace_existing_defaults_to_off_and_is_the_last_field(self):
        self.assertFalse(CheckpointWriterConfig().replace_existing)
        self.assertEqual(
            [f.name for f in dataclasses.fields(CheckpointWriterConfig)][-1],
            "replace_existing",
        )

    def test_replace_existing_requires_a_temp_dir(self):
        with pytest.raises(
            ValueError, match="replace_existing requires write_to_temp_dir=True"
        ):
            CheckpointWriterConfig(
                barrier_config=_NoopBarrierConfig(),
                write_to_temp_dir=False,
                replace_existing=True,
            )

    def test_replace_existing_without_a_barrier_is_rejected(self):
        with (
            pytest.warns(UserWarning),
            pytest.raises(ValueError, match="replace_existing"),
        ):
            CheckpointWriterConfig(barrier_config=None, replace_existing=True)


class TestCheckpointWriter(TestCase):
    def setUp(self):
        # Create a temporary directory for test checkpoints
        self.temp_dir = tempfile.mkdtemp()

        # Create test objects
        self.rank_info = RankInfo(
            global_rank=0,
            global_world_size=1,
            role_rank=0,
            role_world_size=1,
        )
        # Most of these tests are about what reaches storage, so they opt out of the
        # barrier that would commit the write through a temporary directory. The
        # default is exercised by test_default_barrier_commits_through_a_temp_dir.
        self.config = CheckpointWriterConfig(
            barrier_config=None, write_to_temp_dir=False
        )
        self.mock_callback = MockCallback()

        # Create a test state dictionary
        self.state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": 5,
            "step": 1000,
        }
        # Create the storage backend for the writer
        self.storage_config = LocalFileSystemStorageConfig()
        self.storage = self.storage_config.create_storage()

    def _writer(
        self,
        probe: _ConcurrencyProbe,
        *,
        file_write_max_threads: int = 1,
        storage_config: _ProbeStorageConfig | None = None,
        barrier_config: BarrierConfig | None = None,
        temp_dir_prefix: str = DEFAULT_TEMP_DIR_PREFIX,
        pre_finalize_callback=None,
        finalize_callback=None,
    ) -> CheckpointWriter:
        if storage_config is None:
            storage_config = _ProbeStorageConfig(probe)
        return CheckpointWriter(
            CheckpointWriterArgs(
                config=CheckpointWriterConfig(
                    file_write_max_threads=file_write_max_threads,
                    barrier_config=barrier_config,
                    temp_dir_prefix=temp_dir_prefix,
                    # These cases are about what reaches storage, so they stage
                    # only when the caller asked for a barrier to commit behind.
                    write_to_temp_dir=barrier_config is not None,
                ),
                rank_info=self.rank_info,
                storage_config=storage_config,
                pre_finalize_callback=pre_finalize_callback,
                finalize_callback=finalize_callback,
            )
        )

    def _raw_checkpoint_info(self) -> CheckpointWriteInfo:
        return CheckpointWriteInfo(
            checkpoint_items={
                "first": CheckpointItem(
                    value=b"first",
                    layout=LayoutInfo("first.bin", RawSerialization()),
                ),
                "second": CheckpointItem(
                    value=b"second",
                    layout=LayoutInfo("second.bin", RawSerialization()),
                ),
                "metadata": CheckpointItem(
                    value=b"metadata",
                    requires_copy=False,
                    layout=LayoutInfo("metadata.bin", RawSerialization()),
                ),
            }
        )

    def _replacing_writer(
        self, *, storage_config: StorageConfig | None = None
    ) -> CheckpointWriter:
        if storage_config is None:
            storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
        return CheckpointWriter(
            CheckpointWriterArgs(
                config=CheckpointWriterConfig(
                    barrier_config=_NoopBarrierConfig(),
                    replace_existing=True,
                ),
                rank_info=self.rank_info,
                storage_config=storage_config,
            )
        )

    def _populated_final(self, name: str, *, marker: bool = False) -> Path:
        final_path = Path(self.temp_dir) / name
        final_path.mkdir()
        (final_path / "old.bin").write_bytes(b"old")
        if marker:
            (final_path / METADATA_FILE_NAME).write_bytes(b"old-marker")
        return final_path

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_default_get_write_stream_delegates_to_stream_write(self):
        path = Path(self.temp_dir) / "default_stream.pt"
        value = {"tensor": torch.arange(4)}

        with self.storage.get_write_stream(path, value) as stream:
            torch.save(value, stream)

        torch.testing.assert_close(
            torch.load(path, weights_only=False)["tensor"],
            value["tensor"],
        )

    def test_file_write_max_threads_controls_parallel_key_writes(self):
        """Test that file_write_max_threads controls concurrent key writes."""
        for max_threads, expected_max_active in ((1, 1), (2, 2), (3, 3)):
            with self.subTest(max_threads=max_threads):
                probe = _ConcurrencyProbe(expected_max_active)

                self._writer(probe, file_write_max_threads=max_threads).write(
                    path=os.path.join(self.temp_dir, f"checkpoint_{max_threads}"),
                    checkpoint_info=self._raw_checkpoint_info(),
                )

                self.assertEqual(probe.max_active, expected_max_active)

    def test_write_prepares_shared_checkpoint_parent_once(self):
        """Test that shared checkpoint parent directories are deduplicated."""
        probe = _ConcurrencyProbe(expected_concurrent_writes=1)
        storage_config = _ProbeStorageConfig(probe)

        self._writer(probe, storage_config=storage_config).write(
            path=os.path.join(self.temp_dir, "checkpoint_mkdir"),
            checkpoint_info=self._raw_checkpoint_info(),
        )

        self.assertEqual(
            storage_config.mkdir_paths,
            [Path(self.temp_dir) / "checkpoint_mkdir"],
        )

    def test_key_write_metrics_emit_save_task_metric(self):
        """Test that per-key write metrics use the save_task event name."""
        probe = _ConcurrencyProbe(expected_concurrent_writes=1)

        with mock.patch.object(checkpoint_writer_module.logger, "info") as info:
            self._writer(probe).write(
                path=os.path.join(self.temp_dir, "checkpoint_metrics"),
                checkpoint_info=self._raw_checkpoint_info(),
            )

        metric_names = {
            call.kwargs["extra"].get("metric_name")
            for call in info.call_args_list
            if "extra" in call.kwargs
        }

        self.assertIn(
            "train.checkpoint_write.execute.storage.first.save_task.e2e.latency_ms",
            metric_names,
        )
        self.assertIn(
            "train.checkpoint_write.execute.storage.mkdir.latency_ms",
            metric_names,
        )

    def test_write_with_barrier_commits_from_prefixed_temp_dir(self):
        """Test that a barriered write stages in <prefix><name> then renames."""
        prefix = "tmp_job123_attempt2_"
        final_path = Path(self.temp_dir) / "checkpoint_0003000"
        expected_tmp_path = Path(self.temp_dir) / f"{prefix}checkpoint_0003000"
        mock_pre_finalize_callback = mock.MagicMock()
        mock_finalize_callback = mock.MagicMock()

        probe = _ConcurrencyProbe(1)
        writer = self._writer(
            probe,
            barrier_config=_NoopBarrierConfig(),
            temp_dir_prefix=prefix,
            pre_finalize_callback=mock_pre_finalize_callback,
            finalize_callback=mock_finalize_callback,
        )
        writer.write(str(final_path), self._raw_checkpoint_info())
        self.assertEqual(mock_pre_finalize_callback.call_count, 1)
        self.assertEqual(
            mock_pre_finalize_callback.call_args.args[0].path, str(expected_tmp_path)
        )
        self.assertEqual(mock_finalize_callback.call_count, 1)
        self.assertEqual(mock_finalize_callback.call_args.args[0].path, str(final_path))
        self.assertTrue(final_path.exists())
        self.assertFalse(expected_tmp_path.exists())

    def test_write_with_barrier_only_role_rank_0_commits(self):
        """Test that only role rank 0 renames the shared temp dir to final."""
        self.rank_info = RankInfo(
            global_rank=1,
            global_world_size=2,
            role_rank=1,
            role_world_size=2,
        )
        prefix = "tmp_job123_attempt2_"
        final_path = Path(self.temp_dir) / "checkpoint_0003000"
        expected_tmp_path = Path(self.temp_dir) / f"{prefix}checkpoint_0003000"
        mock_pre_finalize_callback = mock.MagicMock()
        mock_finalize_callback = mock.MagicMock()
        probe = probe = _ConcurrencyProbe(1)
        writer = self._writer(
            probe,
            barrier_config=_NoopBarrierConfig(),
            temp_dir_prefix=prefix,
            pre_finalize_callback=mock_pre_finalize_callback,
            finalize_callback=mock_finalize_callback,
        )
        writer.write(str(final_path), self._raw_checkpoint_info())
        self.assertEqual(mock_pre_finalize_callback.call_count, 1)
        self.assertEqual(
            mock_pre_finalize_callback.call_args.args[0].path, str(expected_tmp_path)
        )
        self.assertEqual(mock_finalize_callback.call_count, 1)
        self.assertEqual(mock_finalize_callback.call_args.args[0].path, str(final_path))

        # Not renamed -- temp dir still exists
        self.assertFalse(final_path.exists())
        self.assertTrue(expected_tmp_path.exists())

    def test_write_with_barrier_rejects_preexisting_final_dir(self):
        """Test that committing onto a non-empty final path fails loudly."""
        final_path = Path(self.temp_dir) / "checkpoint_0004000"
        final_path.mkdir()
        (final_path / "leftover.bin").write_bytes(b"leftover")

        probe = _ConcurrencyProbe(1)
        writer = self._writer(probe, barrier_config=_NoopBarrierConfig())

        with pytest.raises(OSError):
            writer.write(str(final_path), self._raw_checkpoint_info())

        # The pre-existing checkpoint must be left untouched, not merged into or
        # replaced, and the staged data must not be nested underneath it.
        self.assertEqual(sorted(p.name for p in final_path.iterdir()), ["leftover.bin"])

    def test_write_with_barrier_commits_into_empty_final_dir(self):
        """Test that an empty final path is replaced rather than nested into."""
        final_path = Path(self.temp_dir) / "checkpoint_0005000"
        final_path.mkdir()

        probe = _ConcurrencyProbe(1)
        writer = self._writer(probe, barrier_config=_NoopBarrierConfig())
        writer.write(str(final_path), self._raw_checkpoint_info())

        self.assertEqual(
            sorted(p.name for p in final_path.iterdir()),
            ["first.bin", "metadata.bin", "second.bin"],
        )
        self.assertEqual((final_path / "first.bin").read_bytes(), b"first")

    def test_default_barrier_commits_through_a_temp_dir(self):
        """Test that a writer left at its defaults commits atomically."""
        final_path = Path(self.temp_dir) / "checkpoint_0004000"
        tmp_path = Path(self.temp_dir) / f"{DEFAULT_TEMP_DIR_PREFIX}checkpoint_0004000"
        writer = CheckpointWriter(
            CheckpointWriterArgs(
                config=CheckpointWriterConfig(),
                rank_info=self.rank_info,
                storage_config=_ProbeStorageConfig(_ConcurrencyProbe(1)),
                pre_finalize_callback=self.mock_callback.pre_finalize_callback,
            )
        )
        # This rank is alone, so its barrier has nothing to coordinate.
        self.assertIsInstance(writer._barrier, DefaultStoreBarrier)

        writer.write(str(final_path), self._raw_checkpoint_info())

        self.assertEqual(self.mock_callback.pre_finalize_path, str(tmp_path))
        self.assertTrue(final_path.exists())
        self.assertFalse(tmp_path.exists())

    def test_write_in_place_keeps_the_barrier_and_never_renames(self):
        """``write_to_temp_dir=False`` writes at the final path under a barrier.

        The rename must not fire. Renaming a directory onto itself looks
        harmless on a POSIX filesystem, where it is a no-op, and destroys the
        checkpoint on a backend that copies every object and then deletes the
        source.
        """
        final_path = Path(self.temp_dir) / "checkpoint_0005000"
        tmp_path = Path(self.temp_dir) / f"{DEFAULT_TEMP_DIR_PREFIX}checkpoint_0005000"
        storage_config = _ProbeStorageConfig(_ConcurrencyProbe(1))
        writer = CheckpointWriter(
            CheckpointWriterArgs(
                config=CheckpointWriterConfig(
                    barrier_config=_NoopBarrierConfig(), write_to_temp_dir=False
                ),
                rank_info=self.rank_info,
                storage_config=storage_config,
                pre_finalize_callback=self.mock_callback.pre_finalize_callback,
                finalize_callback=self.mock_callback.finalize_callback,
            )
        )
        self.assertIsInstance(writer._barrier, _NoopBarrier)

        writer.write(str(final_path), self._raw_checkpoint_info())

        self.assertEqual(storage_config.rename_calls, [])
        self.assertFalse(tmp_path.exists())
        self.assertEqual((final_path / "first.bin").read_bytes(), b"first")
        # Both hooks see the final path, because it is the only path there is.
        self.assertEqual(self.mock_callback.pre_finalize_path, str(final_path))
        self.assertEqual(self.mock_callback.finalize_path, str(final_path))

    def test_write_calls_callbacks(self):
        """Test that write calls the callbacks with correct parameters."""
        # Create writer with callbacks
        args = CheckpointWriterArgs(
            config=self.config,
            rank_info=self.rank_info,
            storage_config=self.storage_config,
            pre_finalize_callback=self.mock_callback.pre_finalize_callback,
            finalize_callback=self.mock_callback.finalize_callback,
        )
        writer = CheckpointWriter(args=args)

        checkpoint_path = os.path.join(self.temp_dir, "checkpoint")

        # Build CheckpointInfo from state_dict
        items = {
            key: CheckpointItem(value=value, layout=None)
            for key, value in self.state_dict.items()
        }
        checkpoint_info = CheckpointWriteInfo(checkpoint_items=items)

        # Call write
        writer.write(checkpoint_path, checkpoint_info)

        # Verify callbacks were called
        self.assertTrue(self.mock_callback.pre_finalize_called)
        self.assertEqual(self.mock_callback.pre_finalize_path, checkpoint_path)
        self.assertIsNotNone(self.mock_callback.pre_finalize_event_logger)
        self.assertIsInstance(self.mock_callback.pre_finalize_event_logger, EventLogger)

        self.assertTrue(self.mock_callback.finalize_called)
        self.assertEqual(self.mock_callback.finalize_path, checkpoint_path)
        self.assertIsNotNone(self.mock_callback.finalize_event_logger)
        self.assertIsInstance(self.mock_callback.finalize_event_logger, EventLogger)

    def test_callback_context_and_order_around_atomic_rename(self):
        checkpoint_path = Path(self.temp_dir) / "checkpoint_with_callback"
        temporary_path = checkpoint_path.parent / f"tmp_{checkpoint_path.name}"
        events: list[str] = []

        def pre_write_callback(args: CallbackArgs) -> None:
            events.append("pre_write")
            self.assertEqual(Path(args.path), checkpoint_path)
            self.assertIs(args.storage, writer._storage)
            self.assertIsInstance(args.event_logger, EventLogger)
            self.assertFalse(temporary_path.exists())
            self.assertFalse(checkpoint_path.exists())

        def pre_finalize_callback(args: CallbackArgs) -> None:
            self.assertIs(args.storage, writer._storage)
            self.assertIsInstance(args.event_logger, EventLogger)
            events.append("pre_finalize")
            self.assertEqual(Path(args.path), temporary_path)
            self.assertTrue(temporary_path.is_dir())
            self.assertFalse(checkpoint_path.exists())

        def finalize_callback(args: CallbackArgs) -> None:
            self.assertIs(args.storage, writer._storage)
            self.assertIsInstance(args.event_logger, EventLogger)
            events.append("finalize")
            self.assertEqual(Path(args.path), checkpoint_path)
            self.assertTrue(checkpoint_path.is_dir())

        writer = CheckpointWriter(
            CheckpointWriterArgs(
                config=replace(
                    self.config,
                    barrier_config=_NoopBarrierConfig(),
                    write_to_temp_dir=True,
                ),
                rank_info=self.rank_info,
                storage_config=self.storage_config,
                pre_write_callback=pre_write_callback,
                pre_finalize_callback=pre_finalize_callback,
                finalize_callback=finalize_callback,
            )
        )
        # The config is what puts the writer on the commit path; this barrier
        # replaces the one it built only to record when the wait happened.
        barrier = mock.Mock()
        barrier.execute_barrier.side_effect = lambda _timeout: events.append("barrier")
        writer._barrier = barrier

        items = {
            key: CheckpointItem(value=value, layout=None)
            for key, value in self.state_dict.items()
        }
        writer.write(
            str(checkpoint_path),
            CheckpointWriteInfo(checkpoint_items=items),
        )

        self.assertEqual(
            events,
            ["pre_write", "pre_finalize", "barrier", "finalize"],
        )
        self.assertFalse(temporary_path.exists())

    def test_write_without_callbacks(self):
        """Test that write works correctly without callbacks."""
        args = CheckpointWriterArgs(
            config=self.config,
            rank_info=self.rank_info,
            storage_config=self.storage_config,
        )
        writer = CheckpointWriter(args=args)

        checkpoint_path = os.path.join(self.temp_dir, "checkpoint_no_callbacks")

        # Build CheckpointInfo from state_dict
        items = {
            key: CheckpointItem(value=value, layout=None)
            for key, value in self.state_dict.items()
        }
        checkpoint_info = CheckpointWriteInfo(checkpoint_items=items)

        # Should not raise any errors
        writer.write(checkpoint_path, checkpoint_info)

    def test_default_layout_writes_item_rank_file(self):
        checkpoint_path = Path(self.temp_dir) / "checkpoint_default_layout"
        writer = CheckpointWriter(
            CheckpointWriterArgs(
                config=self.config,
                rank_info=self.rank_info,
                storage_config=self.storage_config,
            )
        )

        writer.write(
            str(checkpoint_path),
            CheckpointWriteInfo(
                checkpoint_items={
                    "model": CheckpointItem(value={"weight": torch.ones(1)})
                }
            ),
        )

        self.assertTrue((checkpoint_path / "model_0.pt").is_file())

    def test_close(self):
        """Test that close doesn't raise any exceptions."""
        args = CheckpointWriterArgs(
            config=self.config,
            rank_info=self.rank_info,
            storage_config=self.storage_config,
        )
        writer = CheckpointWriter(args=args)
        # This is a no-op in the base class, so just verify it doesn't raise
        writer.close()

    def test_write_with_simple_layout(self):
        """Test writing checkpoint with simple layout."""
        # Create a writer without checkpoint_layout
        args = CheckpointWriterArgs(
            config=self.config,
            rank_info=self.rank_info,
            storage_config=self.storage_config,
        )
        writer = CheckpointWriter(args=args)

        # Create test state dict that matches the layout keys
        state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": "5",  # JSON needs string representation
            "step": "1000",
        }

        checkpoint_path = os.path.join(self.temp_dir, "checkpoint_layout")

        # Compute layout_info_mappings by calling simple_layout with rank
        layout_info_mappings = simple_layout(self.rank_info.global_rank)

        # Build CheckpointInfo with layout_info_mappings
        items = {
            key: CheckpointItem(
                value=value,
                layout=layout_info_mappings.get(key),
            )
            for key, value in state_dict.items()
        }
        checkpoint_info = CheckpointWriteInfo(checkpoint_items=items)

        # Write checkpoint with CheckpointInfo
        writer.write(checkpoint_path, checkpoint_info)

        # Verify files exist based on layout
        model_file = os.path.join(
            checkpoint_path, f"model_rank_{self.rank_info.global_rank}.pt"
        )
        optimizer_file = os.path.join(
            checkpoint_path, f"optimizer_rank_{self.rank_info.global_rank}.pt"
        )
        epoch_file = os.path.join(
            checkpoint_path, f"epoch_rank_{self.rank_info.global_rank}.json"
        )
        step_file = os.path.join(
            checkpoint_path, f"step_rank_{self.rank_info.global_rank}.json"
        )

        self.assertTrue(os.path.exists(model_file))
        self.assertTrue(os.path.exists(optimizer_file))
        self.assertTrue(os.path.exists(epoch_file))
        self.assertTrue(os.path.exists(step_file))

        # Verify content of torch files
        loaded_model = torch.load(model_file)
        loaded_optimizer = torch.load(optimizer_file)
        self.assertIn("weight", loaded_model)
        self.assertEqual(loaded_optimizer["param_groups"][0]["lr"], 0.01)

        # Verify content of JSON files
        with open(epoch_file, "r") as f:
            epoch_content = json.load(f)
            self.assertEqual(epoch_content, "5")  # epoch

        with open(step_file, "r") as f:
            step_content = json.load(f)
            self.assertEqual(step_content, "1000")  # step

    def test_write_with_layout_extra_keys(self):
        """Test that writer ignores extra keys when no layout is provided for them."""
        args = CheckpointWriterArgs(
            config=self.config,
            rank_info=self.rank_info,
            storage_config=self.storage_config,
        )
        writer = CheckpointWriter(args=args)

        # Create state dict with extra keys not covered by layout
        extra_keys_state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": "5",
            "step": "1000",
            "extra_key": "not_covered_by_layout",  # This key is not in the layout
        }

        checkpoint_path = os.path.join(self.temp_dir, "checkpoint_extra_keys")

        # Compute layout_info_mappings by calling simple_layout with rank
        # This layout doesn't include "extra_key"
        layout_info_mappings = simple_layout(self.rank_info.global_rank)

        # Build CheckpointInfo - only include keys that are in the layout
        # extra_key is not in layout, so it's excluded from checkpoint_info
        items = {
            key: CheckpointItem(
                value=value,
                layout=layout_info_mappings.get(key),
            )
            for key, value in extra_keys_state_dict.items()
            if key in layout_info_mappings
        }
        checkpoint_info = CheckpointWriteInfo(checkpoint_items=items)

        # Write checkpoint with CheckpointInfo - extra keys should be ignored
        writer.write(checkpoint_path, checkpoint_info)

        # Verify that files for layout-covered keys exist
        model_file = os.path.join(
            checkpoint_path, f"model_rank_{self.rank_info.global_rank}.pt"
        )
        self.assertTrue(os.path.exists(model_file))

        # Verify that extra_key was NOT written to any file (it was ignored)
        # We just check that the covered files exist and don't check for extra_key

    def test_write_with_global_file_layout(self):
        """Test writing checkpoint with global file layout."""
        args = CheckpointWriterArgs(
            config=self.config,
            rank_info=self.rank_info,
            storage_config=self.storage_config,
        )
        writer = CheckpointWriter(args=args)

        # Create test state dict that matches the global layout keys
        state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "global_config": "some_global_config_value",
        }

        checkpoint_path = os.path.join(self.temp_dir, "checkpoint_global_layout")

        # Compute layout_info_mappings by calling global_file_layout with rank
        layout_info_mappings = global_file_layout(self.rank_info.global_rank)

        # Build CheckpointInfo with layout_info_mappings
        items = {
            key: CheckpointItem(
                value=value,
                layout=layout_info_mappings.get(key),
            )
            for key, value in state_dict.items()
        }
        checkpoint_info = CheckpointWriteInfo(checkpoint_items=items)

        # Write checkpoint with CheckpointInfo
        writer.write(checkpoint_path, checkpoint_info)

        # Verify per-rank files exist with rank suffix
        model_file = os.path.join(
            checkpoint_path, f"model_rank_{self.rank_info.global_rank}.pt"
        )
        optimizer_file = os.path.join(
            checkpoint_path, f"optimizer_rank_{self.rank_info.global_rank}.pt"
        )

        # Verify global file exists WITHOUT rank suffix
        global_config_file = os.path.join(checkpoint_path, "config.json")

        self.assertTrue(os.path.exists(model_file))
        self.assertTrue(os.path.exists(optimizer_file))
        self.assertTrue(
            os.path.exists(global_config_file)
        )  # No rank suffix for global files

        # Verify content of torch files
        loaded_model = torch.load(model_file)
        loaded_optimizer = torch.load(optimizer_file)
        self.assertIn("weight", loaded_model)
        self.assertEqual(loaded_optimizer["param_groups"][0]["lr"], 0.01)

        # Verify content of global JSON file
        with open(global_config_file, "r") as f:
            import json

            global_data = json.load(f)
            # The JSON file contains just the raw value, not wrapped in an object
            self.assertEqual(global_data, "some_global_config_value")

    def test_replace_existing_replaces_a_populated_final_dir(self):
        for label, storage_config in (
            ("local", LocalFileSystemStorageConfig(use_direct_io=False)),
            ("fsspec", FsspecStorageConfig(protocol="memory")),
        ):
            with self.subTest(backend=label):
                final_path = Path(self.temp_dir) / f"checkpoint_{label}"
                tmp_path = (
                    Path(self.temp_dir) / f"{DEFAULT_TEMP_DIR_PREFIX}checkpoint_{label}"
                )
                writer = self._replacing_writer(storage_config=storage_config)
                storage = writer._storage
                try:
                    storage.mkdir(final_path)
                    storage.write(final_path / "old.bin", b"old")
                    storage.write(final_path / METADATA_FILE_NAME, b"old-marker")
                    with self.assertLogs(
                        checkpoint_writer_module.__name__, level="WARNING"
                    ) as logs:
                        writer.write(str(final_path), self._raw_checkpoint_info())
                    self.assertTrue(
                        any(str(final_path) in line for line in logs.output)
                    )
                    self.assertEqual(
                        sorted(storage.ls(final_path)),
                        ["first.bin", "metadata.bin", "second.bin"],
                    )
                    self.assertEqual(storage.read(final_path / "first.bin"), b"first")
                    self.assertFalse(storage.exists(tmp_path))
                finally:
                    if label == "fsspec" and storage.exists(Path(self.temp_dir)):
                        storage.rmdir(Path(self.temp_dir))

    def test_replace_existing_removes_the_old_marker_before_the_rest(self):
        final_path = self._populated_final("checkpoint_0013000", marker=True)
        real_rmdir = LocalFileSystemStorage.rmdir

        def fail_on_final(storage_self, path: Path) -> None:
            if path == final_path:
                raise RuntimeError("killed mid-delete")
            real_rmdir(storage_self, path)

        writer = self._replacing_writer()
        with (
            mock.patch.object(LocalFileSystemStorage, "rmdir", fail_on_final),
            pytest.raises(RuntimeError, match="killed mid-delete"),
        ):
            writer.write(str(final_path), self._raw_checkpoint_info())

        self.assertFalse((final_path / METADATA_FILE_NAME).exists())
        self.assertTrue((final_path / "old.bin").exists())

    def test_replace_existing_touches_nothing_before_the_barrier(self):
        final_path = self._populated_final("checkpoint_0014000")
        writer = self._replacing_writer()
        seen: list[bool] = []
        barrier = mock.Mock()
        barrier.execute_barrier.side_effect = lambda _timeout: seen.append(
            (final_path / "old.bin").exists()
        )
        writer._barrier = barrier

        writer.write(str(final_path), self._raw_checkpoint_info())

        self.assertEqual(seen, [True])
        self.assertFalse((final_path / "old.bin").exists())

    def test_replace_existing_keeps_the_old_dir_when_the_write_fails(self):
        final_path = self._populated_final("checkpoint_0015000", marker=True)
        writer = self._replacing_writer()

        with (
            mock.patch.object(
                LocalFileSystemStorage, "write", side_effect=OSError("disk full")
            ),
            pytest.raises(OSError, match="disk full"),
        ):
            writer.write(str(final_path), self._raw_checkpoint_info())

        self.assertEqual(
            sorted(p.name for p in final_path.iterdir()),
            [METADATA_FILE_NAME, "old.bin"],
        )

    def test_only_role_rank_0_replaces_the_destination(self):
        self.rank_info = RankInfo(
            global_rank=1, global_world_size=2, role_rank=1, role_world_size=2
        )
        final_path = self._populated_final("checkpoint_0016000")
        writer = self._replacing_writer()

        writer.write(str(final_path), self._raw_checkpoint_info())

        self.assertEqual([p.name for p in final_path.iterdir()], ["old.bin"])

    def test_replace_existing_commits_onto_an_empty_dir_without_deleting(self):
        final_path = Path(self.temp_dir) / "checkpoint_0017000"
        final_path.mkdir()
        rmdir_calls: list[Path] = []
        real_rmdir = LocalFileSystemStorage.rmdir

        def record_rmdir(storage_self, path: Path) -> None:
            if path == final_path:
                rmdir_calls.append(path)
            real_rmdir(storage_self, path)

        writer = self._replacing_writer()
        with mock.patch.object(LocalFileSystemStorage, "rmdir", record_rmdir):
            writer.write(str(final_path), self._raw_checkpoint_info())

        self.assertEqual(rmdir_calls, [])
        self.assertEqual(
            sorted(p.name for p in final_path.iterdir()),
            ["first.bin", "metadata.bin", "second.bin"],
        )

    def test_replace_existing_survives_a_failed_second_rename(self):
        final_path = self._populated_final("checkpoint_0018000", marker=True)
        tmp_path = Path(self.temp_dir) / f"{DEFAULT_TEMP_DIR_PREFIX}checkpoint_0018000"
        real_rename = LocalFileSystemStorage.rename
        directory_rename_calls = 0

        def fail_second_directory_rename(
            storage_self,
            src_path: Path,
            dst_path: Path,
            is_directory: bool = False,
            background_cleanup: bool = False,
        ) -> None:
            nonlocal directory_rename_calls
            if is_directory:
                directory_rename_calls += 1
                if directory_rename_calls == 2:
                    raise TimeoutError("second rename")
            real_rename(
                storage_self,
                src_path,
                dst_path,
                is_directory=is_directory,
                background_cleanup=background_cleanup,
            )

        writer = self._replacing_writer()
        with (
            mock.patch.object(
                LocalFileSystemStorage, "rename", fail_second_directory_rename
            ),
            pytest.raises(TimeoutError, match="second rename"),
        ):
            writer.write(str(final_path), self._raw_checkpoint_info())
        self.assertFalse(final_path.exists())
        self.assertEqual(
            sorted(p.name for p in tmp_path.iterdir()),
            ["first.bin", "metadata.bin", "second.bin"],
        )
        self._replacing_writer().write(str(final_path), self._raw_checkpoint_info())
        self.assertEqual(
            sorted(p.name for p in final_path.iterdir()),
            ["first.bin", "metadata.bin", "second.bin"],
        )

    def test_replace_existing_replaces_only_after_an_occupied_destination_refusal(
        self,
    ):
        replaced = [
            OccupiedDestinationError(errno.EEXIST, "x"),
            OccupiedDestinationError(errno.ENOTEMPTY, "x"),
        ]
        propagated = [
            OccupiedDestinationError(errno.ENOTDIR, "x"),
            OccupiedDestinationError(errno.EISDIR, "x"),
            OccupiedDestinationError("no errno"),  # a backend that passed no errno
            OSError(errno.ENOTEMPTY, "x"),  # a backend that does not use the class
            FileExistsError(errno.EEXIST, "x"),  # a driver error raised mid-move
            PermissionError(errno.EACCES, "x"),
            TimeoutError("cut short"),
            FileExistsError("no errno"),
        ]
        writer = self._replacing_writer(storage_config=_RefusingOnceStorageConfig())
        storage = writer._storage
        assert isinstance(storage, _RefusingOnceStorage)

        for index, error in enumerate(replaced):
            with self.subTest(replaced=index):
                final_path = self._populated_final(f"replaced_{index}", marker=True)
                storage.error = error
                writer.write(str(final_path), self._raw_checkpoint_info())
                self.assertEqual(
                    sorted(p.name for p in final_path.iterdir()),
                    ["first.bin", "metadata.bin", "second.bin"],
                )

        for index, error in enumerate(propagated):
            with self.subTest(propagated=index):
                final_path = self._populated_final(f"propagated_{index}", marker=True)
                storage.error = error
                with self.assertRaises(BaseException) as raised:
                    writer.write(str(final_path), self._raw_checkpoint_info())
                self.assertIs(raised.exception, error)
                self.assertEqual(
                    sorted(p.name for p in final_path.iterdir()),
                    [METADATA_FILE_NAME, "old.bin"],
                )

    def test_replace_existing_refuses_a_file_at_the_destination(self):
        final_path = Path(self.temp_dir) / "checkpoint_file"
        final_path.write_bytes(b"a file")
        writer = self._replacing_writer()
        with pytest.raises(OSError):
            writer.write(str(final_path), self._raw_checkpoint_info())
        self.assertEqual(final_path.read_bytes(), b"a file")

    def test_replace_existing_refuses_a_symlink_to_a_checkpoint(self):
        target = self._populated_final("step-10", marker=True)
        link = Path(self.temp_dir) / "latest"
        link.symlink_to(target)
        writer = self._replacing_writer()
        with pytest.raises(OSError):
            writer.write(str(link), self._raw_checkpoint_info())
        self.assertTrue(link.is_symlink())
        self.assertEqual(
            sorted(p.name for p in target.iterdir()),
            [METADATA_FILE_NAME, "old.bin"],
        )

    def test_replace_existing_handles_a_marker_only_dir_on_a_flat_store(self):
        final_path = Path(self.temp_dir) / "checkpoint_marker_only"
        final_path.mkdir()
        (final_path / METADATA_FILE_NAME).write_bytes(b"old-marker")
        writer = self._replacing_writer(storage_config=_FlatPrefixStorageConfig())
        writer.write(str(final_path), self._raw_checkpoint_info())
        self.assertEqual(
            sorted(p.name for p in final_path.iterdir()),
            ["first.bin", "metadata.bin", "second.bin"],
        )

    def test_replace_existing_spares_the_destination_when_nothing_was_written(self):
        final_path = self._populated_final("checkpoint_0014000", marker=True)
        leftover = final_path.parent / f"{DEFAULT_TEMP_DIR_PREFIX}{final_path.name}"
        leftover.mkdir()
        (leftover / "stale.bin").write_bytes(b"stale")
        with pytest.raises(OSError):
            self._replacing_writer().write(
                str(final_path), CheckpointWriteInfo(checkpoint_items={})
            )
        self.assertEqual(
            sorted(p.name for p in final_path.iterdir()),
            [METADATA_FILE_NAME, "old.bin"],
        )

    def test_replace_existing_spares_the_destination_when_only_metadata_was_written(
        self,
    ):
        # Rank 0 writes the distributed metadata even with no items, so the
        # staging directory exists; that alone must not replace a checkpoint.
        for label, storage_config in (
            ("local", LocalFileSystemStorageConfig(use_direct_io=False)),
            ("fsspec", FsspecStorageConfig(protocol="memory")),
        ):
            with self.subTest(backend=label):
                final_path = Path(self.temp_dir) / f"metadata_only_{label}"
                writer = self._replacing_writer(storage_config=storage_config)
                storage = writer._storage
                try:
                    storage.mkdir(final_path)
                    storage.write(final_path / "old.bin", b"old")
                    storage.write(final_path / METADATA_FILE_NAME, b"old-marker")
                    # Local storage creates no staging directory for a save with
                    # no items; an interrupted save may have left one.
                    storage.mkdir(
                        final_path.parent
                        / f"{DEFAULT_TEMP_DIR_PREFIX}{final_path.name}"
                    )
                    with pytest.raises(OccupiedDestinationError):
                        writer.write(
                            str(final_path),
                            CheckpointWriteInfo(
                                checkpoint_items={},
                                serialized_distributed_metadata=b"meta",
                            ),
                        )
                    self.assertEqual(
                        sorted(storage.ls(final_path)), [METADATA_FILE_NAME, "old.bin"]
                    )
                    self.assertEqual(
                        storage.read(final_path / METADATA_FILE_NAME), b"old-marker"
                    )
                finally:
                    if label == "fsspec" and storage.exists(Path(self.temp_dir)):
                        storage.rmdir(Path(self.temp_dir))

    def test_replace_existing_reports_the_replacement_apart_from_the_rename(self):
        final_path = self._populated_final("checkpoint_metric", marker=True)
        real_rmdir = LocalFileSystemStorage.rmdir

        def slow_rmdir(storage_self, path: Path) -> None:
            if path == final_path:
                time.sleep(0.5)
            real_rmdir(storage_self, path)

        def metrics(info: mock.Mock) -> dict[str, int]:
            return {
                call.kwargs["extra"]["metric_name"].rsplit(".execute.", 1)[-1]: (
                    call.kwargs["extra"]["value"]
                )
                for call in info.call_args_list
                if call.kwargs.get("extra", {}).get("metric_name")
            }

        with (
            mock.patch.object(LocalFileSystemStorage, "rmdir", slow_rmdir),
            mock.patch.object(checkpoint_writer_module.logger, "info") as info,
        ):
            self._replacing_writer().write(str(final_path), self._raw_checkpoint_info())
        replaced = metrics(info)
        self.assertGreaterEqual(replaced["replace_existing.latency_ms"], 500)
        self.assertLess(replaced["rename.latency_ms"], 500)

        with mock.patch.object(checkpoint_writer_module.logger, "info") as info:
            self._replacing_writer().write(
                str(Path(self.temp_dir) / "checkpoint_metric_fresh"),
                self._raw_checkpoint_info(),
            )
        fresh = metrics(info)
        self.assertIn("rename.latency_ms", fresh)
        self.assertNotIn("replace_existing.latency_ms", fresh)

    def test_without_replace_existing_the_refusal_propagates_after_one_rename(self):
        final_path = self._populated_final("checkpoint_0019000", marker=True)
        writer = self._writer(_ConcurrencyProbe(1), barrier_config=_NoopBarrierConfig())

        with pytest.raises(FileExistsError) as raised:
            writer.write(str(final_path), self._raw_checkpoint_info())

        self.assertEqual(raised.value.errno, errno.ENOTEMPTY)
        self.assertIn("the destination is occupied", str(raised.value))
        self.assertEqual(len(writer._storage._config.rename_calls), 1)
        self.assertEqual(
            sorted(p.name for p in final_path.iterdir()),
            [METADATA_FILE_NAME, "old.bin"],
        )

    def test_replace_existing_refuses_a_symlink_on_the_fsspec_file_driver(self):
        target = self._populated_final("step-10", marker=True)
        link = Path(self.temp_dir) / "latest"
        link.symlink_to(target)
        writer = self._replacing_writer(
            storage_config=FsspecStorageConfig(protocol="file")
        )

        with pytest.raises(FileExistsError):
            writer.write(str(link), self._raw_checkpoint_info())

        self.assertTrue(link.is_symlink())
        self.assertEqual(
            sorted(p.name for p in target.iterdir()), [METADATA_FILE_NAME, "old.bin"]
        )

    def test_replace_existing_removes_nested_directories(self):
        final_path = self._populated_final("checkpoint_0020000", marker=True)
        (final_path / "sharded").mkdir()
        (final_path / "sharded" / "model_0.safetensors").write_bytes(b"old")

        self._replacing_writer().write(str(final_path), self._raw_checkpoint_info())

        self.assertEqual(
            sorted(p.name for p in final_path.iterdir()),
            ["first.bin", "metadata.bin", "second.bin"],
        )

    def test_replace_existing_recovers_from_an_interrupted_delete(self):
        final_path = self._populated_final("checkpoint_0021000", marker=True)
        real_rmdir = LocalFileSystemStorage.rmdir

        def fail_on_final(storage_self, path: Path) -> None:
            if path == final_path:
                raise RuntimeError("killed mid-delete")
            real_rmdir(storage_self, path)

        with (
            mock.patch.object(LocalFileSystemStorage, "rmdir", fail_on_final),
            pytest.raises(RuntimeError, match="killed mid-delete"),
        ):
            self._replacing_writer().write(str(final_path), self._raw_checkpoint_info())
        self.assertEqual([p.name for p in final_path.iterdir()], ["old.bin"])

        self._replacing_writer().write(str(final_path), self._raw_checkpoint_info())
        self.assertEqual(
            sorted(p.name for p in final_path.iterdir()),
            ["first.bin", "metadata.bin", "second.bin"],
        )


def test_generic_pre_write_receives_active_storage_and_final_path_before_creation(
    tmp_path,
):
    path = tmp_path / "checkpoint"
    args = CheckpointWriterArgs(
        config=CheckpointWriterConfig(),
        rank_info=RankInfo(0, 1, 0, 1),
        storage_config=LocalFileSystemStorageConfig(use_direct_io=False),
    )
    args.config.write_to_temp_dir = True
    # A single-rank real default-store barrier requires no process group.
    args.config.barrier_config = DefaultStoreBarrierConfig()
    received = []
    args.pre_write_callback = lambda callback_args: received.append(
        (
            callback_args.path,
            callback_args.storage,
            path.exists(),
            (tmp_path / "tmp_checkpoint").exists(),
        )
    )
    writer = CheckpointWriter(args)
    writer.write(str(path), CheckpointWriteInfo(checkpoint_items={}))
    assert received == [(str(path), writer._storage, False, False)]


def test_async_save_replaces_the_same_path_twice(tmp_path: Path) -> None:
    config = CheckpointManager.Config(
        save=AsyncCheckpointSaverConfig(
            writer_config=CheckpointWriterConfig(replace_existing=True),
            staging_config=CheckpointStagerConfig(
                use_pinned_memory=False,
                use_non_blocking_copy=False,
            ),
            wait_timeout_secs=30,
        ),
        storage_config=LocalFileSystemStorageConfig(use_direct_io=False),
    )
    manager = config.build()
    path = os.path.join(tmp_path, "checkpoint")
    try:
        first_future = manager.save(path, {"model": torch.zeros(4), "step": 1})
        assert first_future is not None
        first_future.result(timeout=30)
        second_future = manager.save(path, {"model": torch.ones(4), "step": 2})
        assert second_future is not None
        second_future.result(timeout=30)
        loaded = manager.load(path, into={"model": torch.zeros(4), "step": 0})
        assert loaded["step"] == 2 and torch.equal(loaded["model"], torch.ones(4))
    finally:
        manager.close()


if __name__ == "__main__":
    run_tests()
