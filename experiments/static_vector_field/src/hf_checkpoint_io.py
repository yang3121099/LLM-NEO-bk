"""Streaming HuggingFace checkpoint reader for the static vector-field audit.

Design goals
------------
* Never load a whole 8B model into RAM.  We build a ``name -> shard file`` index
  and pull one tensor at a time, on demand.
* Support sharded safetensors (``model.safetensors.index.json``), a single
  ``model.safetensors``, and — as a best-effort fallback — PyTorch ``.bin``
  shards.  safetensors is always preferred.
* Tensors are returned as flat ``float64`` numpy arrays (via torch, so that
  bfloat16 checkpoints upcast correctly).

The loader is deliberately torch-backed because Llama-3.1 checkpoints are
bfloat16, which numpy cannot represent natively.
"""

from __future__ import annotations

import json
import os
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import torch
from safetensors import safe_open

# Names that should be treated as tied (folded into a group but flagged).
_TIED_HINTS = ("embed_tokens", "lm_head")


def _is_floating_dtype(dtype: torch.dtype) -> bool:
    return dtype.is_floating_point


class CheckpointReader:
    """Lazily reads individual tensors from one HF checkpoint directory."""

    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self._st_index: Dict[str, str] = {}      # tensor name -> safetensors file
        self._bin_state: Dict[str, torch.Tensor] | None = None  # fallback
        self._st_handles: Dict[str, object] = {}  # file -> safe_open handle
        self._meta: Dict[str, Tuple[Tuple[int, ...], str]] = {}  # name -> (shape, dtype)
        self._build_index()

    # -- index construction -------------------------------------------------- #
    def _build_index(self) -> None:
        if not os.path.isdir(self.path):
            raise FileNotFoundError(f"[{self.name}] not a directory: {self.path}")

        index_json = os.path.join(self.path, "model.safetensors.index.json")
        single = os.path.join(self.path, "model.safetensors")
        st_files = sorted(glob(os.path.join(self.path, "*.safetensors")))

        if os.path.isfile(index_json):
            with open(index_json) as f:
                weight_map = json.load(f)["weight_map"]
            for tname, fname in weight_map.items():
                self._st_index[tname] = os.path.join(self.path, fname)
            self._scan_safetensors_meta(sorted(set(self._st_index.values())))
        elif st_files:
            for fpath in st_files:
                with safe_open(fpath, framework="pt", device="cpu") as f:
                    for tname in f.keys():
                        self._st_index[tname] = fpath
            self._scan_safetensors_meta(st_files)
        else:
            self._build_bin_index()

    def _scan_safetensors_meta(self, files: List[str]) -> None:
        for fpath in files:
            with safe_open(fpath, framework="pt", device="cpu") as f:
                for tname in f.keys():
                    sl = f.get_slice(tname)
                    shape = tuple(sl.get_shape())
                    dtype = sl.get_dtype()  # str like "BF16", "F32"
                    self._meta[tname] = (shape, dtype)

    def _build_bin_index(self) -> None:
        bin_files = sorted(glob(os.path.join(self.path, "*.bin")))
        if not bin_files:
            raise FileNotFoundError(
                f"[{self.name}] no .safetensors or .bin files in {self.path}"
            )
        # PyTorch bin cannot be sliced lazily; load all shards once into memory.
        self._bin_state = {}
        for bf in bin_files:
            sd = torch.load(bf, map_location="cpu", weights_only=True)
            self._bin_state.update(sd)
        for tname, t in self._bin_state.items():
            self._meta[tname] = (tuple(t.shape), str(t.dtype))

    # -- public API ---------------------------------------------------------- #
    def keys(self) -> List[str]:
        if self._bin_state is not None:
            return sorted(self._bin_state.keys())
        return sorted(self._st_index.keys())

    def meta(self) -> Dict[str, Tuple[Tuple[int, ...], str]]:
        return self._meta

    def is_floating(self, name: str) -> bool:
        if self._bin_state is not None:
            return _is_floating_dtype(self._bin_state[name].dtype)
        _, dtype = self._meta[name]
        # safetensors dtype strings: F16/F32/F64/BF16 are floating.
        return dtype.upper() in {"F16", "F32", "F64", "BF16"}

    def shape(self, name: str) -> Tuple[int, ...]:
        return self._meta[name][0]

    def _handle(self, fpath: str):
        h = self._st_handles.get(fpath)
        if h is None:
            h = safe_open(fpath, framework="pt", device="cpu")
            self._st_handles[fpath] = h
        return h

    def load_flat_f64(self, name: str) -> np.ndarray:
        """Return tensor ``name`` as a flat float64 numpy array."""
        if self._bin_state is not None:
            t = self._bin_state[name]
        else:
            t = self._handle(self._st_index[name]).get_tensor(name)
        return t.to(torch.float32).reshape(-1).numpy().astype(np.float64, copy=False)

    def close(self) -> None:
        self._st_handles.clear()


def is_tied_name(name: str) -> bool:
    return any(h in name for h in _TIED_HINTS)


def common_floating_names(
    readers: List[CheckpointReader],
    ignore_patterns: List[str] | None = None,
    include_non_floating: bool = False,
) -> Tuple[List[str], Dict[str, str]]:
    """Return tensor names shared by all readers with matching shape/dtype.

    Returns ``(names, notes)`` where ``notes`` maps a skipped name to the reason.
    """
    import re

    ignore_res = [re.compile(p) for p in (ignore_patterns or [])]
    key_sets = [set(r.keys()) for r in readers]
    shared = set.intersection(*key_sets) if key_sets else set()

    names: List[str] = []
    notes: Dict[str, str] = {}

    # Record names that are not universally present.
    union = set.union(*key_sets) if key_sets else set()
    for n in sorted(union - shared):
        notes[n] = "not present in all checkpoints"

    for name in sorted(shared):
        if any(rx.search(name) for rx in ignore_res):
            notes[name] = "ignored by regex"
            continue
        if not include_non_floating and not all(r.is_floating(name) for r in readers):
            notes[name] = "non-floating dtype"
            continue
        shapes = {r.shape(name) for r in readers}
        if len(shapes) != 1:
            notes[name] = f"shape mismatch: {shapes}"
            continue
        names.append(name)
    return names, notes
