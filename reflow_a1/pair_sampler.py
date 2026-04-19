"""Hierarchical pair graph construction for ReFlow A.1.

The graph intentionally avoids an all-pairs video graph. A full graph is O(N^2)
and distant video frames often have poor co-visibility. The ReFlow A.1 scaffold
instead builds a keyframe coarse graph for the global skeleton and local clip
graphs for fine temporal completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

Pair = Tuple[str, str]


@dataclass(frozen=True)
class PairGraph:
    clips: List[List[str]]
    keyframes: List[str]
    coarse_pairs: List[Pair]
    fine_pairs_per_clip: List[List[Pair]]

    def as_dict(self) -> Dict[str, object]:
        return {
            "clips": self.clips,
            "keyframes": self.keyframes,
            "coarse_pairs": self.coarse_pairs,
            "fine_pairs_per_clip": self.fine_pairs_per_clip,
        }


def _unique_pairs(pairs: Iterable[Pair]) -> List[Pair]:
    seen = set()
    out: List[Pair] = []
    for a, b in pairs:
        if a == b:
            continue
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def split_into_clips(frame_ids: Sequence[str], clip_len: int = 10) -> List[List[str]]:
    if clip_len <= 0:
        raise ValueError(f"clip_len must be positive, got {clip_len}")
    return [list(frame_ids[i : i + clip_len]) for i in range(0, len(frame_ids), clip_len)]


def select_keyframes(clips: Sequence[Sequence[str]]) -> List[str]:
    return [clip[0] for clip in clips if len(clip) > 0]


def build_coarse_pairs(
    keyframes: Sequence[str],
    add_skip_pairs: bool = True,
    coarse_max_offset: int | None = None,
) -> List[Pair]:
    """Build keyframe coarse edges with configurable temporal offsets.

    By default this preserves the old behavior:
    - add_skip_pairs=False -> only (i, i+1)
    - add_skip_pairs=True  -> (i, i+1) and (i, i+2)
    """

    if coarse_max_offset is None:
        coarse_max_offset = 2 if add_skip_pairs else 1
    coarse_max_offset = int(coarse_max_offset)
    if coarse_max_offset < 1:
        raise ValueError(f"coarse_max_offset must be >= 1, got {coarse_max_offset}")

    pairs: List[Pair] = []
    max_offset = min(coarse_max_offset, max(len(keyframes) - 1, 0))
    for offset in range(1, max_offset + 1):
        pairs.extend((keyframes[i], keyframes[i + offset]) for i in range(len(keyframes) - offset))
    return _unique_pairs(pairs)


def build_fine_pairs_for_clip(clip: Sequence[str], fine_max_offset: int = 4) -> List[Pair]:
    if len(clip) <= 1:
        return []
    fine_max_offset = int(fine_max_offset)
    if fine_max_offset < 1:
        raise ValueError(f"fine_max_offset must be >= 1, got {fine_max_offset}")
    keyframe = clip[0]
    pairs: List[Pair] = []
    pairs.extend((keyframe, frame_id) for frame_id in clip[1:])
    max_offset = min(fine_max_offset, max(len(clip) - 1, 0))
    for offset in range(1, max_offset + 1):
        pairs.extend((clip[i], clip[i + offset]) for i in range(len(clip) - offset))
    return _unique_pairs(pairs)


def build_reflow_a1_pair_graph(
    frame_ids: Sequence[str],
    clip_len: int = 10,
    add_skip_pairs: bool = True,
    coarse_max_offset: int | None = None,
    fine_max_offset: int = 4,
) -> PairGraph:
    clips = split_into_clips(frame_ids, clip_len=clip_len)
    keyframes = select_keyframes(clips)
    coarse_pairs = build_coarse_pairs(
        keyframes,
        add_skip_pairs=add_skip_pairs,
        coarse_max_offset=coarse_max_offset,
    )
    fine_pairs_per_clip = [build_fine_pairs_for_clip(clip, fine_max_offset=fine_max_offset) for clip in clips]
    return PairGraph(clips, keyframes, coarse_pairs, fine_pairs_per_clip)
