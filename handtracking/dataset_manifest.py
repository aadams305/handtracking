"""JSONL manifest for distilled samples."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List, Optional


@dataclass
class LetterboxMeta:
    scale: float
    pad_x: float
    pad_y: float
    src_w: int
    src_h: int
    dst: int = 256


@dataclass
class DistilledSample:
    """One training row: keypoints in ``dst``×``dst`` letterboxed pixel space."""

    image_path: str
    letterbox: LetterboxMeta
    keypoints_xy: List[List[float]]  # NUM_HAND_JOINTS x 2 in [0, dst)

    def to_json_line(self) -> str:
        d = {
            "image_path": self.image_path,
            "letterbox": asdict(self.letterbox),
            "keypoints_xy": self.keypoints_xy,
        }
        return json.dumps(d, separators=(",", ":"))

    @staticmethod
    def from_json_line(line: str) -> "DistilledSample":
        d = json.loads(line)
        lb = LetterboxMeta(**d["letterbox"])
        return DistilledSample(
            image_path=d["image_path"],
            letterbox=lb,
            keypoints_xy=d["keypoints_xy"],
        )


def iter_manifest(path: Path) -> Iterator[DistilledSample]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield DistilledSample.from_json_line(line)


def write_manifest(samples: List[DistilledSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.to_json_line() + "\n")


def count_manifest(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n
