"""Constant-shape recurrent caches. detach() is EXPLICIT, never automatic."""
from __future__ import annotations
from dataclasses import dataclass, fields, is_dataclass
from typing import Any
import torch
from torch import Tensor


def _map(value: Any, fn):
    if isinstance(value, Tensor):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_map(x, fn) for x in value)
    if is_dataclass(value):
        return type(value)(**{f.name: _map(getattr(value, f.name), fn) for f in fields(value)})
    return value


def tensors(value):
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, tuple):
        for x in value:
            yield from tensors(x)
    elif is_dataclass(value):
        for f in fields(value):
            yield from tensors(getattr(value, f.name))


@dataclass(frozen=True)
class LevelState:
    inner: Tensor       # A: [B,H,Dk,M], interpreted S^K = A^T
    outer: Tensor       # B: [B,H,M,Dv]
    count: Tensor       # [B,H,M], soft occurrence counts
    position: Tensor | None = None  # [B,H,Dk,M], additive positional moments


@dataclass(frozen=True)
class RelationState:
    source: int
    target: int
    memory: Tensor      # [B,H,M_source,M_target], FULL, not a low-rank factor


@dataclass(frozen=True)
class ResidualSnapshot:
    inner: Tensor
    outer: Tensor


@dataclass(frozen=True)
class GPAState:
    levels: tuple[LevelState, ...]
    relations: tuple[RelationState, ...]
    position: int = 0
    chunk_size: int = 1
    chunk_offset: int = 0
    frozen: tuple[ResidualSnapshot, ...] | None = None

    def detach(self) -> "GPAState":
        return _map(self, lambda x: x.detach())

    def to(self, *args, **kwargs) -> "GPAState":
        return _map(self, lambda x: x.to(*args, **kwargs))

    def numel(self) -> int:
        """Logical tensor elements, including optional lagged snapshots.

        The integer position counter requires O(log T) bits; 'fixed state'
        refers to tensor shapes, not arbitrary-precision count bit length.
        """
        return sum(x.numel() for x in tensors(self))

