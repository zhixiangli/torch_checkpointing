# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import errno
import os
import pickle
import posixpath
import uuid
from collections.abc import Iterator
from pathlib import Path

import fsspec
import pytest
import torch
from fsspec.core import split_protocol
from fsspec.implementations.memory import MemoryFileSystem
from torch_checkpointing import ItemSpec
from torch_checkpointing.checkpoint_manager import CheckpointManager
from torch_checkpointing.config import AsyncCheckpointSaverConfig
from torch_checkpointing.default_resharder import DefaultResharder
from torch_checkpointing.staging import CheckpointStagerConfig
from torch_checkpointing.storage.base_storage import (
    METADATA_FILE_NAME,
    OccupiedDestinationError,
    ReadArgs,
)
from torch_checkpointing.storage.fsspec_storage import (
    FsspecStorage,
    FsspecStorageConfig,
)

# Point this at a real store (for example gs://bucket/prefix) to run the tests
# that take the ``storage`` fixture against its fsspec driver. Defaults to the
# in-memory filesystem.
_TEST_URL = os.environ.get(
    "TORCH_CHECKPOINTING_TEST_FSSPEC_URL", "memory://torch-checkpointing-tests"
)


@pytest.fixture
def storage() -> FsspecStorage:
    protocol, _ = split_protocol(_TEST_URL)
    return FsspecStorageConfig(protocol=protocol).create_storage()


@pytest.fixture
def root(storage: FsspecStorage) -> Iterator[Path]:
    _, base = split_protocol(_TEST_URL)
    path = Path(base) / uuid.uuid4().hex
    yield path
    if storage.exists(path):
        storage.rmdir(path)


def _put(storage: FsspecStorage, path: Path, data: bytes = b"payload") -> Path:
    storage.mkdir(path.parent)
    storage.write(path, data)
    return path


@pytest.fixture
def memory_root() -> Iterator[Path]:
    """A root on memory://, for tests that patch MemoryFileSystem."""
    path = Path(f"/marker-last-{uuid.uuid4().hex}")
    yield path
    fs = fsspec.filesystem("memory")
    if fs.exists(str(path)):
        fs.rm(str(path), recursive=True)


def _memory_storage() -> FsspecStorage:
    return FsspecStorageConfig(protocol="memory").create_storage()


def _staged_checkpoint(storage: FsspecStorage, src: Path) -> None:
    _put(storage, src / METADATA_FILE_NAME, b"marker")
    _put(storage, src / "model_0.pt", b"model")
    _put(storage, src / "sub" / "optim_0.pt", b"optim")


@pytest.mark.parametrize(
    "data",
    [
        b"Hello bytes! ",
        bytearray(b"Hello bytearray! "),
        memoryview(b"Hello memoryview! "),
    ],
    ids=["bytes", "bytearray", "memoryview"],
)
def test_write_then_read_round_trips_every_buffer_type(storage, root, data):
    storage.mkdir(root)
    storage.write(root / "a.bin", data)

    assert storage.read(root / "a.bin") == bytes(memoryview(data))


def test_stream_write_commits_on_close(storage, root):
    storage.mkdir(root)
    with storage.stream_write(root / "a.bin") as stream:
        stream.write(b"ab")
        stream.write(bytearray(b"cd"))

    assert storage.read(root / "a.bin") == b"abcd"


def test_ranged_read_supports_seek_and_readinto(storage, root):
    _put(storage, root / "a.bin", b"0123456789")

    read_args = ReadArgs(pre_read_full_file=False, direct_io=True)
    with storage.stream_read(root / "a.bin", read_args) as stream:
        stream.seek(3)
        buffer = bytearray(4)
        assert stream.readinto(buffer) == 4
        assert bytes(buffer) == b"3456"
        assert stream.tell() == 7


def test_read_of_a_missing_file_raises(storage, root):
    with pytest.raises(FileNotFoundError):
        storage.read(root / "missing.bin")


def test_ls_returns_entry_basenames(storage, root):
    _put(storage, root / "dir" / "a.bin")
    _put(storage, root / "dir" / "sub" / "b.bin")

    assert sorted(storage.ls(root / "dir")) == ["a.bin", "sub"]


def test_probes_distinguish_files_directories_and_absent_paths(storage, root):
    _put(storage, root / "dir" / "a.bin", b"0123456789")

    assert storage.exists(root / "dir" / "a.bin")
    assert storage.exists(root / "dir")
    assert storage.isdir(root / "dir")
    assert not storage.isdir(root / "dir" / "a.bin")
    assert storage.getsize(root / "dir" / "a.bin") == 10
    # A checkpoint folder that was never written is absent, not an error.
    assert not storage.exists(root / "never-written")
    assert not storage.isdir(root / "never-written")
    with pytest.raises(FileNotFoundError):
        storage.getsize(root / "dir" / "missing.bin")


def test_delete_removes_one_file(storage, root):
    _put(storage, root / "dir" / "a.bin")
    _put(storage, root / "dir" / "b.bin")

    storage.delete(root / "dir" / "a.bin")

    assert storage.ls(root / "dir") == ["b.bin"]


def test_rmdir_removes_a_populated_directory_and_nothing_else(storage, root):
    _put(storage, root / "step-1" / "a.bin")
    _put(storage, root / "step-1" / "sub" / "b.bin")
    # "step-1" is a string prefix of "step-10". A prefix-based delete must not
    # reach the sibling.
    _put(storage, root / "step-10" / "a.bin", b"keep")

    storage.rmdir(root / "step-1")

    assert not storage.exists(root / "step-1")
    assert storage.read(root / "step-10" / "a.bin") == b"keep"


def test_rmdir_of_a_missing_directory_raises(storage, root):
    with pytest.raises(FileNotFoundError):
        storage.rmdir(root / "missing")


def test_remap_path_returns_the_full_uri(storage, root):
    path = _put(storage, root / "a.bin")

    protocol, stripped = split_protocol(storage.remap_path(path))

    assert protocol is not None
    assert stripped.lstrip("/") == str(path).lstrip("/")


def test_config_survives_pickling():
    config = FsspecStorageConfig(protocol="memory")

    restored = pickle.loads(pickle.dumps(config))

    assert restored == config
    assert isinstance(restored.create_storage(), FsspecStorage)


def test_rename_directory_onto_absent_destination(storage, root):
    _put(storage, root / "src" / "payload.bin")
    _put(storage, root / "src" / "sub" / "nested.bin")

    storage.rename(root / "src", root / "dst", is_directory=True)

    assert not storage.exists(root / "src")
    assert sorted(storage.ls(root / "dst")) == ["payload.bin", "sub"]
    assert storage.ls(root / "dst" / "sub") == ["nested.bin"]


def test_rename_directory_replaces_empty_destination(storage, root):
    _put(storage, root / "src" / "payload.bin")
    storage.mkdir(root / "dst")

    storage.rename(root / "src", root / "dst", is_directory=True)

    assert not storage.exists(root / "src")
    # The contents land directly at dst, not nested under dst/src.
    assert storage.ls(root / "dst") == ["payload.bin"]


def test_rename_directory_rejects_populated_destination(storage, root):
    _put(storage, root / "src" / "payload.bin")
    _put(storage, root / "dst" / "existing.bin")

    with pytest.raises(OSError):
        storage.rename(root / "src", root / "dst", is_directory=True)

    # Neither side is touched: no nesting, no merge, no deletion.
    assert storage.ls(root / "src") == ["payload.bin"]
    assert storage.ls(root / "dst") == ["existing.bin"]


def test_rename_directory_rejects_file_destination(storage, root):
    _put(storage, root / "src" / "payload.bin")
    _put(storage, root / "dst", b"not a directory")

    with pytest.raises(OSError):
        storage.rename(root / "src", root / "dst", is_directory=True)

    assert storage.read(root / "dst") == b"not a directory"
    assert storage.ls(root / "src") == ["payload.bin"]


def test_rename_directory_leaves_prefix_siblings_alone(storage, root):
    _put(storage, root / "tmp_step-2" / "payload.bin")
    # "step-2" is a string prefix of "step-20".
    _put(storage, root / "step-20" / "keep.bin", b"keep")

    storage.rename(root / "tmp_step-2", root / "step-2", is_directory=True)

    assert storage.ls(root / "step-2") == ["payload.bin"]
    assert storage.read(root / "step-20" / "keep.bin") == b"keep"
    assert sorted(storage.ls(root)) == ["step-2", "step-20"]


def test_rename_file_onto_absent_destination(storage, root):
    _put(storage, root / "a.bin", b"one")

    storage.rename(root / "a.bin", root / "b.bin")

    assert not storage.exists(root / "a.bin")
    assert storage.read(root / "b.bin") == b"one"


def test_rename_file_replaces_existing_file(storage, root):
    _put(storage, root / "a.bin", b"one")
    _put(storage, root / "b.bin", b"two")

    storage.rename(root / "a.bin", root / "b.bin")

    assert not storage.exists(root / "a.bin")
    assert storage.read(root / "b.bin") == b"one"


def test_rename_file_rejects_directory_destination(storage, root):
    _put(storage, root / "a.bin", b"one")
    _put(storage, root / "dst" / "existing.bin")

    with pytest.raises(OSError):
        storage.rename(root / "a.bin", root / "dst")

    assert storage.read(root / "a.bin") == b"one"
    assert storage.ls(root / "dst") == ["existing.bin"]


def _empty_directory(storage: FsspecStorage, path: Path) -> Path:
    storage.mkdir(path)
    if not storage.isdir(path):
        pytest.skip("the store keeps no empty directories")
    return path


def test_rename_onto_itself_leaves_the_directory_alone(storage, root):
    _put(storage, root / "full" / "a.bin")
    empty = _empty_directory(storage, root / "empty")

    storage.rename(root / "full", root / "full", is_directory=True)
    storage.rename(empty, empty, is_directory=True)

    assert storage.ls(root / "full") == ["a.bin"]
    assert storage.isdir(empty)


def test_rename_of_a_missing_source_raises_and_spares_the_destination(storage, root):
    empty = _empty_directory(storage, root / "empty")

    with pytest.raises(FileNotFoundError):
        storage.rename(root / "missing", empty, is_directory=True)
    with pytest.raises(FileNotFoundError):
        storage.rename(root / "missing", root / "missing")

    assert storage.isdir(empty)


def test_rename_file_flagged_as_directory_rejects_empty_directory(storage, root):
    # The source's type decides, not the is_directory hint.
    _put(storage, root / "a.bin", b"one")
    empty = _empty_directory(storage, root / "empty")

    with pytest.raises(OSError):
        storage.rename(root / "a.bin", empty, is_directory=True)

    assert storage.read(root / "a.bin") == b"one"
    assert storage.isdir(empty)


def test_rename_directory_moves_the_marker_last(memory_root, monkeypatch):
    storage = _memory_storage()
    src, dst = memory_root / "tmp_step-20", memory_root / "step-20"
    _staged_checkpoint(storage, src)
    fs = fsspec.filesystem("memory")
    copy_file = MemoryFileSystem.cp_file
    landed: list[str] = []

    def cp_file(self, path1, path2, **kwargs):
        if fs.isfile(path1) and path2.startswith(str(dst) + "/"):
            if not path2.endswith("/" + METADATA_FILE_NAME):
                assert not fs.exists(str(src / METADATA_FILE_NAME))
                assert not fs.exists(str(dst / METADATA_FILE_NAME))
            landed.append(path2)
        copy_file(self, path1, path2, **kwargs)

    monkeypatch.setattr(MemoryFileSystem, "cp_file", cp_file)
    storage.rename(src, dst, is_directory=True)

    assert landed[-1] == str(dst / METADATA_FILE_NAME)
    assert len(landed) == 4  # hidden marker, two shards, then the marker
    assert sorted(storage.ls(dst)) == [METADATA_FILE_NAME, "model_0.pt", "sub"]
    assert storage.read(dst / METADATA_FILE_NAME) == b"marker"


def _file_copies_in_a_rename() -> int:
    storage = _memory_storage()
    fs = fsspec.filesystem("memory")
    scratch = Path(f"/marker-last-{uuid.uuid4().hex}")
    copy_file = MemoryFileSystem.cp_file
    copied = 0

    def cp_file(self, path1, path2, **kwargs):
        nonlocal copied
        if fs.isfile(path1):
            copied += 1
        copy_file(self, path1, path2, **kwargs)

    try:
        _staged_checkpoint(storage, scratch / "src")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(MemoryFileSystem, "cp_file", cp_file)
            storage.rename(scratch / "src", scratch / "dst", is_directory=True)
        return copied
    finally:
        if fs.exists(str(scratch)):
            fs.rm(str(scratch), recursive=True)


def test_rename_directory_cut_short_never_leaves_the_marker(memory_root, monkeypatch):
    storage = _memory_storage()
    fs = fsspec.filesystem("memory")
    copy_file = MemoryFileSystem.cp_file

    for k in range(_file_copies_in_a_rename()):
        src = memory_root / f"tmp_step-{k}"
        dst = memory_root / f"step-{k}"
        _staged_checkpoint(storage, src)
        copied = 0

        def cp_file(self, path1, path2, **kwargs):
            nonlocal copied
            if fs.isfile(path1):
                copied += 1
                if copied == k + 1:
                    raise TimeoutError("cut short")
            copy_file(self, path1, path2, **kwargs)

        monkeypatch.setattr(MemoryFileSystem, "cp_file", cp_file)
        with pytest.raises(TimeoutError, match="cut short"):
            storage.rename(src, dst, is_directory=True)
        assert not storage.exists(dst / METADATA_FILE_NAME)


def test_rename_directory_replaces_a_stale_hidden_copy(storage, root):
    _staged_checkpoint(storage, root / "src")
    _put(storage, root / "src" / ".metadata.pkl.tmp", b"stale")
    storage.rename(root / "src", root / "dst", is_directory=True)
    assert storage.read(root / "dst" / METADATA_FILE_NAME) == b"marker"
    assert ".metadata.pkl.tmp" not in storage.ls(root / "dst")


def test_rename_directory_without_a_marker_publishes_no_stale_hidden_copy(
    storage, root
):
    _put(storage, root / "src" / "model_0.pt", b"model")
    _put(storage, root / "src" / ".metadata.pkl.tmp", b"stale")
    storage.rename(root / "src", root / "dst", is_directory=True)
    assert sorted(storage.ls(root / "dst")) == [".metadata.pkl.tmp", "model_0.pt"]


def test_rename_directory_refused_leaves_the_source_marker(storage, root):
    _staged_checkpoint(storage, root / "src")
    _put(storage, root / "dst" / "existing.bin")
    with pytest.raises(OSError):
        storage.rename(root / "src", root / "dst", is_directory=True)
    assert storage.read(root / "src" / METADATA_FILE_NAME) == b"marker"
    assert ".metadata.pkl.tmp" not in storage.ls(root / "src")


def test_rename_directory_holding_only_the_marker(storage, root):
    _put(storage, root / "src" / METADATA_FILE_NAME, b"marker")
    storage.rename(root / "src", root / "dst", is_directory=True)
    assert storage.ls(root / "dst") == [METADATA_FILE_NAME]
    assert not storage.exists(root / "src")


def test_rename_directory_treats_a_nested_metadata_file_as_ordinary(storage, root):
    _put(storage, root / "src" / "sub" / METADATA_FILE_NAME, b"nested")
    storage.rename(root / "src", root / "dst", is_directory=True)
    assert storage.ls(root / "dst") == ["sub"]
    assert storage.ls(root / "dst" / "sub") == [METADATA_FILE_NAME]


def test_rename_directory_with_a_marker_replaces_empty_destination(storage, root):
    _empty_directory(storage, root / "dst")
    _staged_checkpoint(storage, root / "src")
    storage.rename(root / "src", root / "dst", is_directory=True)
    assert sorted(storage.ls(root / "dst")) == [METADATA_FILE_NAME, "model_0.pt", "sub"]


def test_rename_directory_that_fails_before_moving_puts_the_marker_back(
    memory_root, monkeypatch
):
    storage = _memory_storage()
    src, dst = memory_root / "tmp_step-20", memory_root / "step-20"
    _staged_checkpoint(storage, src)
    move = MemoryFileSystem.mv

    def mv(self, path1, path2, recursive=False, **kwargs):
        if recursive:
            raise TimeoutError("before anything moved")
        move(self, path1, path2, recursive=recursive, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(MemoryFileSystem, "mv", mv)
        with pytest.raises(TimeoutError, match="before anything moved"):
            storage.rename(src, dst, is_directory=True)

    assert storage.read(src / METADATA_FILE_NAME) == b"marker"
    assert ".metadata.pkl.tmp" not in storage.ls(src)
    storage.rename(src, dst, is_directory=True)
    assert storage.read(dst / METADATA_FILE_NAME) == b"marker"


def test_rename_directory_raises_the_move_error_when_the_put_back_fails(
    memory_root, monkeypatch, caplog
):
    storage = _memory_storage()
    src, dst = memory_root / "tmp_step-20", memory_root / "step-20"
    _staged_checkpoint(storage, src)
    move = MemoryFileSystem.mv

    def mv(self, path1, path2, recursive=False, **kwargs):
        if recursive:
            raise TimeoutError("move")
        if path2 == str(src / METADATA_FILE_NAME):
            raise OSError("put back")
        move(self, path1, path2, recursive=recursive, **kwargs)

    with (
        monkeypatch.context() as patch,
        caplog.at_level("WARNING", logger="torch_checkpointing.storage.fsspec_storage"),
    ):
        patch.setattr(MemoryFileSystem, "mv", mv)
        with pytest.raises(TimeoutError, match="move"):
            storage.rename(src, dst, is_directory=True)

    assert ".metadata.pkl.tmp" in storage.ls(src)
    assert METADATA_FILE_NAME not in storage.ls(src)
    assert "put back" in caplog.text


def test_metadata_file_name_has_one_definition():
    from torch_checkpointing import metadata_serialization
    from torch_checkpointing.storage import base_storage

    assert base_storage.METADATA_FILE_NAME == "metadata.pkl"
    assert (
        metadata_serialization.METADATA_FILE_NAME is base_storage.METADATA_FILE_NAME
    )


def test_rename_directory_interrupted_makes_no_put_back(memory_root, monkeypatch):
    storage = _memory_storage()
    src, dst = memory_root / "tmp_step-20", memory_root / "step-20"
    _staged_checkpoint(storage, src)
    move = MemoryFileSystem.mv
    destinations: list[str] = []

    def mv(self, path1, path2, recursive=False, **kwargs):
        destinations.append(path2)
        if recursive:
            raise KeyboardInterrupt("killed mid-commit")
        move(self, path1, path2, recursive=recursive, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(MemoryFileSystem, "mv", mv)
        with pytest.raises(KeyboardInterrupt):
            storage.rename(src, dst, is_directory=True)

    assert destinations == [str(src / ".metadata.pkl.tmp"), str(dst)]
    assert not storage.exists(dst / METADATA_FILE_NAME)


def test_rename_directory_reveal_failure_leaves_no_marker(memory_root, monkeypatch):
    storage = _memory_storage()
    src, dst = memory_root / "tmp_step-20", memory_root / "step-20"
    _staged_checkpoint(storage, src)
    move = MemoryFileSystem.mv

    def mv(self, path1, path2, recursive=False, **kwargs):
        if path2 == str(dst / METADATA_FILE_NAME):
            raise TimeoutError("reveal")
        move(self, path1, path2, recursive=recursive, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(MemoryFileSystem, "mv", mv)
        with pytest.raises(TimeoutError, match="reveal"):
            storage.rename(src, dst, is_directory=True)

    assert sorted(storage.ls(dst)) == [".metadata.pkl.tmp", "model_0.pt", "sub"]
    assert not storage.exists(src)


def test_directory_operations_reject_a_file(storage, root):
    path = _put(storage, root / "a.bin", b"one")

    with pytest.raises(NotADirectoryError):
        storage.ls(path)
    with pytest.raises(NotADirectoryError):
        storage.rmdir(path)

    assert storage.read(path) == b"one"


# fsspec expands a glob in the path of a recursive delete, a copy or a move, so
# a name with one of these characters would take every path it matches with it.
_GLOB_NAMES = ["step-*", "step-?", "step-[12]"]


@pytest.mark.parametrize("name", _GLOB_NAMES)
def test_rmdir_rejects_glob_characters_and_spares_what_they_match(storage, root, name):
    _put(storage, root / name / "a.bin", b"literal")
    _put(storage, root / "step-1" / "a.bin", b"keep")

    with pytest.raises(ValueError, match="glob"):
        storage.rmdir(root / name)

    assert storage.read(root / name / "a.bin") == b"literal"
    assert storage.read(root / "step-1" / "a.bin") == b"keep"


@pytest.mark.parametrize("name", _GLOB_NAMES)
def test_rename_rejects_glob_characters_and_spares_what_they_match(storage, root, name):
    _put(storage, root / name / "a.bin", b"literal")
    _put(storage, root / "src" / "a.bin", b"src")
    _put(storage, root / "step-1" / "a.bin", b"keep")

    with pytest.raises(ValueError, match="glob"):
        storage.rename(root / name, root / "dst", is_directory=True)
    with pytest.raises(ValueError, match="glob"):
        storage.rename(root / "src", root / name, is_directory=True)
    with pytest.raises(ValueError, match="glob"):
        storage.rename(root / name / "a.bin", root / "b.bin")

    assert sorted(storage.ls(root)) == sorted([name, "src", "step-1"])
    assert storage.read(root / name / "a.bin") == b"literal"
    assert storage.read(root / "src" / "a.bin") == b"src"
    assert storage.read(root / "step-1" / "a.bin") == b"keep"


def test_ranged_reads_have_independent_positions(storage, root):
    # Parallel loads give each worker its own stream of the same file.
    path = _put(storage, root / "a.bin", b"0123456789")
    read_args = ReadArgs(pre_read_full_file=False)

    with storage.stream_read(path, read_args) as first:
        first.seek(6)
        with storage.stream_read(path, read_args) as second:
            assert second.read(2) == b"01"
        assert first.read(2) == b"67"


class _ListingCachingMemoryFileSystem(MemoryFileSystem):
    """In-memory filesystem that caches directory listings the way gcsfs does.

    Listings are cached per instance, so changes made through another
    instance are not seen. A cached listing of the parent or grandparent is
    trusted when looking a path up, so an entry missing from it reads as
    absent, and opening a file for reading looks it up first. The plain memory
    filesystem never caches, so it cannot go stale.
    """

    protocol = "listingcachingmemory"

    def ls(self, path, detail=True, **kwargs):
        path = self._strip_protocol(path)
        try:
            entries = self.dircache[path]
        except KeyError:
            entries = super().ls(path, detail=True, **kwargs)
            self.dircache[path] = entries
        # pyrefly: ignore[bad-index]
        return entries if detail else sorted(entry["name"] for entry in entries)

    def info(self, path, **kwargs):
        path = self._strip_protocol(path)
        # Each raises FileNotFoundError when a cached listing lacks the entry.
        self._ls_from_cache(self._parent(path))
        self._ls_from_cache(path)
        return super().info(path, **kwargs)

    def _open(self, path, mode="rb", **kwargs):  # pyrefly: ignore[bad-override]
        if "r" in mode:
            self.info(path)
        return super()._open(path, mode, **kwargs)

    def invalidate_cache(self, path=None):
        # Drops the listing of path and of every ancestor.
        if path is None:
            self.dircache.clear()
            return
        path = self._strip_protocol(path).rstrip("/")
        while path:
            self.dircache.pop(path, None)
            parent = self._parent(path).rstrip("/")
            if parent == path:
                break
            path = parent


def test_sees_a_checkpoint_written_through_another_client():
    # A long-lived reader lists the folder; another rank or the async-save
    # subprocess then writes a checkpoint through its own client, and the
    # reader loads it with ranged reads.
    fsspec.register_implementation(
        "listingcachingmemory", _ListingCachingMemoryFileSystem, clobber=True
    )
    storage = FsspecStorageConfig(protocol="listingcachingmemory").create_storage()
    root = Path(f"test-{uuid.uuid4().hex}")
    other_client = fsspec.filesystem("memory")
    try:
        storage.write(root / "step-5" / "metadata.pkl", b"m")
        assert storage.ls(root) == ["step-5"]

        other_client.pipe_file(f"/{root}/step-10/model_0.pt", b"0123456789")

        read_args = ReadArgs(pre_read_full_file=False)
        with storage.stream_read(root / "step-10" / "model_0.pt", read_args) as stream:
            stream.seek(3)
            assert stream.read(4) == b"3456"
        assert storage.getsize(root / "step-10" / "model_0.pt") == 10
        assert sorted(storage.ls(root)) == ["step-10", "step-5"]
    finally:
        _ListingCachingMemoryFileSystem.clear_instance_cache()
        if other_client.exists(f"/{root}"):
            other_client.rm(f"/{root}", recursive=True)


def test_enabling_the_listing_cache_is_rejected():
    config = FsspecStorageConfig(
        protocol="memory", storage_options={"use_listings_cache": True}
    )

    with pytest.raises(ValueError, match="use_listings_cache"):
        config.create_storage()


def test_disabling_the_listing_cache_explicitly_is_accepted():
    config = FsspecStorageConfig(
        protocol="memory", storage_options={"use_listings_cache": False}
    )

    assert isinstance(config.create_storage(), FsspecStorage)


class _MarkerListingMemoryFileSystem(MemoryFileSystem):
    """In-memory filesystem that lists a directory's own marker object.

    An object store has no directories. A console or tool that creates a
    folder stores an empty object named ``<dir>/``, and drivers list it along
    with the folder's contents: s3fs as a file named ``<dir>/``, gcsfs as a
    directory named ``<dir>``.
    """

    protocol = "markerlistingmemory"

    def __init__(self, *args, marker_suffix, marker_type, **kwargs):
        super().__init__(*args, **kwargs)
        self._marker_suffix = marker_suffix
        self._marker_type = marker_type

    def ls(self, path, detail=True, **kwargs):
        path = self._strip_protocol(path)
        entries = super().ls(path, detail=True, **kwargs)
        if self.isdir(path):
            marker = {
                "name": path + self._marker_suffix,
                "type": self._marker_type,
                "size": 0,
            }
            entries = [marker, *entries]
        # pyrefly: ignore[bad-index]
        return entries if detail else sorted(entry["name"] for entry in entries)


@pytest.mark.parametrize(
    "marker_suffix, marker_type",
    [("/", "file"), ("", "directory")],
    ids=["as_s3fs_lists_it", "as_gcsfs_lists_it"],
)
def test_ls_leaves_out_the_marker_of_the_directory(marker_suffix, marker_type):
    fsspec.register_implementation(
        "markerlistingmemory", _MarkerListingMemoryFileSystem, clobber=True
    )
    storage = FsspecStorageConfig(
        protocol="markerlistingmemory",
        storage_options={"marker_suffix": marker_suffix, "marker_type": marker_type},
    ).create_storage()
    root = Path(f"test-{uuid.uuid4().hex}")
    try:
        storage.mkdir(root / "empty")
        storage.write(root / "full" / "a.bin", b"payload")

        # An empty folder holds only its marker; it is still a directory.
        assert storage.ls(root / "empty") == []
        assert storage.ls(root / "full") == ["a.bin"]
        with pytest.raises(NotADirectoryError):
            storage.ls(root / "full" / "a.bin")
    finally:
        _MarkerListingMemoryFileSystem.clear_instance_cache()
        fsspec.filesystem("memory").rm(f"/{root}", recursive=True)


def _payload() -> dict:
    return {"model": torch.arange(4), "step": 10}


def test_sync_save_then_load_round_trip_over_fsspec(tmp_path: Path) -> None:
    config = CheckpointManager.Config.with_sync_save()
    config.storage_config = FsspecStorageConfig(protocol="file")
    manager = config.build()

    checkpoint_path = os.path.join(tmp_path, "sync_round_trip")
    try:
        assert manager.save(checkpoint_path, _payload()) is None

        loaded = manager.load(
            checkpoint_path,
            into={"model": torch.zeros(4, dtype=torch.long), "step": 0},
        )

        assert torch.equal(loaded["model"], torch.arange(4))
        assert loaded["step"] == 10
    finally:
        manager.close()


def test_async_save_then_load_round_trip_over_fsspec(tmp_path: Path) -> None:
    # The memory filesystem is per process, so the save subprocess needs a
    # store both processes can see.
    config = CheckpointManager.Config(
        save=AsyncCheckpointSaverConfig(
            staging_config=CheckpointStagerConfig(
                use_pinned_memory=False,
                use_non_blocking_copy=False,
            )
        )
    )
    config.save.wait_timeout_secs = 30
    config.storage_config = FsspecStorageConfig(protocol="file")
    manager = config.build()

    checkpoint_path = os.path.join(tmp_path, "async_round_trip")
    try:
        write_future = manager.save(checkpoint_path, _payload())
        assert write_future is not None
        write_future.result(timeout=30)

        loaded = manager.load(
            checkpoint_path,
            into={"model": torch.zeros(4, dtype=torch.long), "step": 0},
        )

        assert torch.equal(loaded["model"], torch.arange(4))
        assert loaded["step"] == 10
    finally:
        manager.close()


def test_commit_cut_short_leaves_no_marker_at_the_final_path(memory_root, monkeypatch):
    config = CheckpointManager.Config.with_sync_save()
    config.default = ItemSpec(resharder=DefaultResharder())  # writes metadata.pkl
    config.storage_config = FsspecStorageConfig(protocol="memory")
    state = {"model": torch.arange(4), "optimizer": torch.ones(4)}
    dst = memory_root / "step-20"
    fs = fsspec.filesystem("memory")
    copy_file = MemoryFileSystem.cp_file
    copied = 0

    def cp_file(self, path1, path2, **kwargs):
        nonlocal copied
        # cp_file raises RuntimeError("preempted") on the second file copy whose
        # destination is under memory_root / "step-20".
        if fs.isfile(path1) and path2.startswith(f"{dst}/"):
            copied += 1
            if copied == 2:
                raise RuntimeError("preempted")
        copy_file(self, path1, path2, **kwargs)

    monkeypatch.setattr(MemoryFileSystem, "cp_file", cp_file)
    manager = config.build()
    try:
        with pytest.raises(RuntimeError, match="preempted"):
            manager.save(str(memory_root / "step-20"), state)
    finally:
        manager.close()
    final = fsspec.filesystem("memory").find(str(memory_root / "step-20"))
    assert len(final) == 1  # the move really was cut short
    assert str(memory_root / "step-20" / METADATA_FILE_NAME) not in final


def test_load_ignores_a_stray_hidden_marker_copy(memory_root):
    config = CheckpointManager.Config.with_sync_save()
    config.default = ItemSpec(resharder=DefaultResharder())  # writes metadata.pkl
    config.storage_config = FsspecStorageConfig(protocol="memory")
    state = {"model": torch.arange(4), "optimizer": torch.ones(4)}
    manager = config.build()
    try:
        manager.save(str(memory_root / "step-20"), state)
        fsspec.filesystem("memory").pipe_file(
            str(memory_root / "step-20" / ".metadata.pkl.tmp"), b"stray"
        )
        loaded = manager.load(
            str(memory_root / "step-20"),
            into={
                "model": torch.zeros(4, dtype=torch.long),
                "optimizer": torch.zeros(4),
            },
        )
        assert torch.equal(loaded["model"], torch.arange(4))
    finally:
        manager.close()


def _fail_recursive_mv(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    move = MemoryFileSystem.mv

    def mv(self, path1, path2, recursive=False, **kwargs):
        if recursive:
            raise error
        return move(self, path1, path2, recursive=recursive, **kwargs)

    monkeypatch.setattr(MemoryFileSystem, "mv", mv)


def test_refused_directory_rename_raises_occupied_destination_error(storage, root):
    _put(storage, root / "src" / "payload.bin")
    _put(storage, root / "dst" / "existing.bin")

    with pytest.raises(OccupiedDestinationError) as raised:
        storage.rename(root / "src", root / "dst", is_directory=True)

    assert isinstance(raised.value, FileExistsError)
    assert raised.value.errno == errno.ENOTEMPTY
    assert "the destination is occupied" in str(raised.value)


@pytest.mark.parametrize("driver_errno", [errno.ENOTEMPTY, errno.EEXIST])
def test_a_driver_error_during_the_move_passes_through_unchanged(
    driver_errno, memory_root, monkeypatch
):
    storage = _memory_storage()
    _put(storage, memory_root / "src" / "payload.bin")
    error = OSError(driver_errno, "driver")
    _fail_recursive_mv(monkeypatch, error)

    with pytest.raises(OSError) as raised:
        storage.rename(memory_root / "src", memory_root / "dst", is_directory=True)

    assert raised.value is error
    assert not isinstance(raised.value, OccupiedDestinationError)


def test_symlink_destination_keeps_the_plain_refusal(tmp_path):
    storage = FsspecStorageConfig(protocol="file").create_storage()
    target = tmp_path / "real"
    _put(storage, target / "metadata.pkl", b"marker")
    _put(storage, tmp_path / "src" / "model_0.pt")
    (tmp_path / "step-1").symlink_to(target)

    with pytest.raises(FileExistsError) as raised:
        storage.rename(tmp_path / "src", tmp_path / "step-1", is_directory=True)

    assert type(raised.value) is FileExistsError
    assert raised.value.errno == errno.ENOTEMPTY
    assert (target / "metadata.pkl").read_bytes() == b"marker"
    assert (tmp_path / "src" / "model_0.pt").exists()


@pytest.mark.parametrize(
    "error",
    [RuntimeError("preempted"), KeyboardInterrupt("killed mid-commit")],
    ids=["exception", "kill"],
)
def test_saving_again_replaces_what_a_cut_short_commit_left(error, monkeypatch):
    root = f"test-{uuid.uuid4().hex}"
    fs = fsspec.filesystem("memory")

    def make_manager() -> CheckpointManager:
        config = CheckpointManager.Config.with_sync_save()
        config.default = ItemSpec(resharder=DefaultResharder())
        config.storage_config = FsspecStorageConfig(protocol="memory")
        config.save.writer_config.replace_existing = True
        return config.build()

    original_cp_file = MemoryFileSystem.cp_file
    copied = 0
    dst_prefix = f"/{root}/step-20/"

    def cut_short_cp_file(self, path1, path2, **kwargs):
        nonlocal copied
        if self.isfile(self._strip_protocol(path1)) and self._strip_protocol(
            path2
        ).startswith(dst_prefix):
            copied += 1
            if copied == 2:
                raise error
        return original_cp_file(self, path1, path2, **kwargs)

    try:
        first_manager = make_manager()
        try:
            with monkeypatch.context() as patch:
                patch.setattr(MemoryFileSystem, "cp_file", cut_short_cp_file)
                with pytest.raises(type(error)):
                    first_manager.save(
                        f"{root}/step-20",
                        {
                            "model": torch.zeros(4, dtype=torch.long),
                            "optimizer": torch.ones(4, dtype=torch.long),
                        },
                    )
        finally:
            first_manager.close()

        manager = make_manager()
        try:
            manager.save(
                f"{root}/step-20",
                {"model": torch.arange(4), "optimizer": torch.arange(4, 8)},
            )
            names = sorted(posixpath.basename(p) for p in fs.find(f"{root}/step-20"))
            assert names == ["metadata.pkl", "model_0.pt", "optimizer_0.pt"]

            loaded = manager.load(
                f"{root}/step-20",
                into={
                    "model": torch.zeros(4, dtype=torch.long),
                    "optimizer": torch.zeros(4, dtype=torch.long),
                },
            )
            assert torch.equal(loaded["model"], torch.arange(4))
            assert torch.equal(loaded["optimizer"], torch.arange(4, 8))
        finally:
            manager.close()
    finally:
        if fs.exists(f"/{root}"):
            fs.rm(f"/{root}", recursive=True)


def test_replace_existing_saves_over_what_the_store_holds(storage, root):
    # Runs on the store _TEST_URL names, so the marker delete, a flat prefix
    # vanishing with its last object, and the second rename hit its driver.
    protocol, _ = split_protocol(_TEST_URL)
    config = CheckpointManager.Config.with_sync_save()
    config.default = ItemSpec(resharder=DefaultResharder())
    config.storage_config = FsspecStorageConfig(protocol=protocol)
    config.save.writer_config.replace_existing = True
    zeros = {
        "model": torch.zeros(4, dtype=torch.long),
        "optimizer": torch.zeros(4, dtype=torch.long),
    }
    _put(storage, root / "step-1" / "model_0.pt", b"partial")  # commit cut short
    _put(storage, root / "step-2" / METADATA_FILE_NAME, b"marker")  # only the marker
    manager = config.build()
    try:
        manager.save(str(root / "step-3"), zeros)  # a complete checkpoint
        for step in (1, 2, 3):
            path = root / f"step-{step}"
            manager.save(
                str(path),
                {"model": torch.arange(4) + step, "optimizer": torch.arange(4)},
            )
            assert sorted(storage.ls(path)) == [
                METADATA_FILE_NAME,
                "model_0.pt",
                "optimizer_0.pt",
            ]
            loaded = manager.load(str(path), into=dict(zeros))
            assert torch.equal(loaded["model"], torch.arange(4) + step)
        assert sorted(storage.ls(root)) == ["step-1", "step-2", "step-3"]
    finally:
        manager.close()
