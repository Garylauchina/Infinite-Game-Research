# Market Camera v2 — Zstd segment writer for frames / metrics.
# Append-only, segmented JSONL.zst. Roll by wall-clock.

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Literal

import zstandard as zstd


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


class ZstdSegmentWriter:
    """Append-only segment writer. Roll by wall-clock every segment_seconds."""

    def __init__(
        self,
        kind: Literal["frames", "metrics"],
        store_dir: Path,
        segment_seconds: float = 300.0,
        level: int = 3,
    ):
        self.kind = kind
        self.store_dir = Path(store_dir)
        self.subdir = self.store_dir / kind
        self.subdir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.subdir / "manifest.json"
        self.segment_seconds = segment_seconds
        self.level = level

        self._pending_path: Path | None = None
        self._writer = None
        self._file = None
        self._start_ts: float | None = None
        self._start_t: int | None = None
        self._last_t: int | None = None
        self._lines: int = 0
        self._manifest: list = []

        self._ensure_manifest()

    def _ensure_manifest(self) -> None:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r") as f:
                    self._manifest = json.load(f)
            except Exception:
                self._manifest = []
        else:
            self._manifest = []
        if not isinstance(self._manifest, list):
            self._manifest = []

    def _roll(self, current_t: int) -> None:
        now = time.time()
        do_roll = False
        if self._start_ts is None:
            do_roll = True
        elif (now - self._start_ts) >= self.segment_seconds:
            do_roll = True

        if not do_roll:
            return

        self.close()

        stamp = _utc_stamp()
        start_t = current_t
        name = f"segment_{stamp}_{start_t}_pending.jsonl.zst"
        self._pending_path = self.subdir / name
        self._start_ts = now
        self._start_t = start_t
        self._last_t = start_t
        self._lines = 0

        self._file = open(self._pending_path, "wb")
        cctx = zstd.ZstdCompressor(level=self.level)
        self._writer = cctx.stream_writer(self._file)

    def write_jsonline(self, obj: dict, current_t: int) -> None:
        """Append one JSON line. Rolls segment if needed."""
        self._roll(current_t)
        if not self._writer:
            return
        t = obj.get("t", current_t)
        self._last_t = int(t)
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        self._writer.write(line.encode("utf-8"))
        self._lines += 1

    def roll_if_needed(self, current_t: int) -> None:
        """Force roll check (e.g. after each tick)."""
        self._roll(current_t)

    def close(self) -> None:
        """Close current segment, rename pending -> final, update manifest."""
        if not self._writer or not self._pending_path:
            return

        try:
            self._writer.close()
        except Exception:
            pass
        self._writer = None

        try:
            if self._file and not self._file.closed:
                self._file.flush()
                os.fsync(self._file.fileno())
                self._file.close()
        except Exception:
            pass
        self._file = None

        if not self._pending_path.exists():
            self._pending_path = None
            self._start_ts = None
            self._start_t = None
            return

        size = self._pending_path.stat().st_size
        final_name = self._pending_path.name.replace("_pending", "")
        final_path = self._pending_path.parent / final_name
        self._pending_path.rename(final_path)
        rel = final_path.relative_to(self.store_dir)
        path_str = str(rel).replace("\\", "/")

        end_t = self._last_t if self._last_t is not None else self._start_t
        rec = {
            "path": path_str,
            "start_t": self._start_t,
            "end_t": end_t,
            "lines": self._lines,
            "bytes": size,
            "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pending": False,
        }
        self._manifest.append(rec)

        try:
            with open(self.manifest_path, "w") as f:
                json.dump(self._manifest, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        except Exception:
            pass

        self._pending_path = None
        self._start_ts = None
        self._start_t = None
        self._last_t = None
        self._lines = 0
