# Running this pipeline on B300

Everything here used to assume one machine shape: 8×H100 (Hopper, `sm_90`),
torch 2.6.0, FlashAttention-2, a per-GPU micro-batch of 1. A B300 node
(Blackwell Ultra, `sm_103`, ~279 GB HBM3e per GPU) breaks the first three and
wastes the fourth. This is what changed, what it decides for you, and what it
cannot decide for you.

If you only read one line: run `bash scripts/gpu_profile.sh --print` on the
node. It reports what stack that machine needs, and every setup and generator
script reads the same values.

---

## 1. Quick start on a fresh B300 node

```bash
# 1. what does this machine need?
bash scripts/gpu_profile.sh --print

# 2. build the env (installs the torch that matches, not the H100 pin)
bash setup_env.sh

# 3. prove the GPU can actually run kernels before starting a long job
python3 shadow_rl/check_env.py --full

# 4. generate + run as usual; the generator picks the attention kernel and the
#    micro-batch from the hardware it is run on
bash run.sh
bash scripts/train_<MODEL>_<TIMESTAMP>.sh
```

Step 3 is not optional. The failure it catches — a torch with no kernels for
this compute capability — imports cleanly, prints the right GPU name, and only
dies at the first matmul, which in this pipeline is after the data pass and the
merge.

## 1b. B200 and B300 are not the same target

They are both Blackwell, and that is exactly what makes the difference easy to
miss. B200 is `sm_100`; B300 (Blackwell Ultra) is `sm_103`.

| | B200 | B300 |
|---|---|---|
| compute capability | 10.0 (`sm_100`) | 10.3 (`sm_103`) |
| CUDA toolkit for codegen | ≥ 12.8 | **≥ 12.9** |
| wheel index | cu128 | cu130 (cu129 also works) |
| torch floor | ≥ 2.7 | **≥ 2.9** |
| `TORCH_CUDA_ARCH_LIST` | `10.0a` | `10.3a` |
| vllm floor | ≥ 0.9 | ≥ 0.11 |
| memory / GPU | ~180 GB | ~279 GB |

The trap is **arch-conditional code**. CUDA kernels that need arch-specific
features — the FP8/NVFP4 CUTLASS paths in flash-attn, vLLM and friends — are
compiled as `sm_100a` or `sm_103a`, and that suffix means *this arch only*:

```
wheel built for B200 (sm_100a) running on B300 : nothing
wheel built for B200 (sm_100)  running on B300 : runs (minor-version compatible)
wheel built for B300 (sm_103a) running on B200 : nothing
```

So a container that works on your B200 fleet can fail on a B300 with `no kernel
image is available for execution on the device`, even though both are Blackwell
and `nvidia-smi` looks fine. Plain (non-`a`) `sm_100` code does run on a B300.

Nothing here needs a per-machine branch: `scripts/gpu_profile.sh` detects the
capability and emits the row above, and `check_env.py` distinguishes native
kernels from merely-compatible ones. Confirm on the node with:

```bash
bash scripts/gpu_profile.sh --print
```

For the RL pipeline the practical differences are the wheel set, and that a
B300's 279 GB lets you raise `MICRO_BATCH_PER_GPU` — which is accumulation
chunking, not part of the update — where a B200 may not.

## 2. What differs between the two machines

| | H100 (was) | B300 (now) |
|---|---|---|
| compute capability | `sm_90` | `sm_103` |
| torch | `2.6.0` (pinned) | `>=2.9.0` from the cu130 index |
| wheel index | cu126 | cu130 (cu129 also works; override `TORCH_INDEX`) |
| `TORCH_CUDA_ARCH_LIST` | `9.0` | `10.3a` |
| attention | `--flash_attn fa2` | `--flash_attn sdpa` unless flash-attn ≥ 2.8 is built for `sm_103` |
| memory / GPU | 80 GB | ~279 GB |
| micro-batch × grad accum | 1 × 256 | 4 × 64 (same effective batch) |
| vllm | any release | `>=0.11.0`, built against CUDA ≥ 12.9 |
| NGC base image | `pytorch:24.12-py3` | a release with CUDA ≥ 12.9, e.g. `pytorch:25.09-py3` |

Nothing above is hardcoded per machine. `scripts/gpu_profile.sh` detects the
compute capability (`nvidia-smi --query-gpu=compute_cap`, falling back to the
device name on older drivers) and exports the row. Hopper and older keep exactly
the pins the existing results were produced with, so old runs stay reproducible.

## 3. The three things that actually bite

### "no kernel image is available for execution on the device"

The wheel has no code for this GPU. Two distinct causes:

* **Too old.** Any torch below 2.9 predates `sm_103`. The fix is a reinstall
  from the matching index; `shadow_rl/check_env.py` prints the exact command.
* **Built for the wrong Blackwell.** CUDA kernels can be compiled
  *arch-conditionally* (`sm_100a`), which is what the FP8/NVFP4 CUTLASS paths
  need. Arch-conditional code runs on that arch and nothing else — so a wheel
  built for a B200 has no kernels a B300 can run, even though both are
  Blackwell. Plain (non-`a`) `sm_100` code does run on `sm_103`, by CUDA's
  minor-version binary compatibility.

`check_env.py` distinguishes these: it compares `torch.cuda.get_arch_list()`
against the device, tells you whether you have native kernels, compatible
kernels, PTX-only (works, JITs for minutes on first use), or nothing — and then
runs a bf16 matmul and an SDPA call to confirm, because the arch list is a claim
and a launched kernel is evidence.

### FlashAttention-2 is not a given

`--flash_attn fa2` was unconditional in the generators. flash-attn added
`sm_100` kernels in 2.8, and whether a given wheel covers `sm_103` depends on
how it was built. The profile therefore *calls* `flash_attn_func` on the GPU and
falls back to `sdpa` when it raises. On Blackwell, `sdpa` resolves to the cuDNN
attention backend, which is a fine default — this is not a slow fallback path.

To force one or the other:

```bash
FLASH_ATTN=fa2 bash run.sh     # I built flash-attn for this arch myself
FLASH_ATTN=sdpa bash run.sh    # don't probe, just use sdpa
```

If you do want to build flash-attn from source here, `nvcc` must know `sm_103`
(CUDA ≥ 12.9) — `gpu_profile.sh --print` says whether it does — and
`TORCH_CUDA_ARCH_LIST=10.3a` must be exported, or you will spend the build
producing a wheel for the wrong arch.

### The card is 3.5× bigger, so the batch shape was wrong

A micro-batch of 1 on 279 GB leaves the GPU mostly idle. The generators now
multiply `per_device_train_batch_size` and divide
`gradient_accumulation_steps` by the same factor, so the *effective* batch size
— and therefore the optimisation trajectory and any comparison against an H100
run — is unchanged. Only the number of forward passes per step changes.

The factor comes from per-GPU memory (≥240 GB → 4, ≥160 GB → 2, else 1) and
applies on Blackwell only; Hopper and older are left alone. Opt out with
`AUTO_MICRO_BS=false`. If a model OOMs at the scaled shape, that is the knob —
the effective batch size is preserved either way.

## 4. What is not handled automatically

These need a human decision on the node. They are flagged by
`check_env.py`/`setup_env.sh` rather than silently worked around.

* **Evaluation backend.** The generated OpenCompass configs use lmdeploy
  TurboMind (`TurboMindModelwithChatTemplate`), which ships prebuilt kernels.
  Blackwell Ultra support is recent; if the engine dies at start-up rather than
  at import, upgrade lmdeploy or switch those model entries to the vllm backend.
  `check_env.py` warns whenever lmdeploy or vllm is present on `sm_100+`.
* **vllm wheel availability.** vllm pins its own torch, and the torch it names
  on PyPI is not necessarily built for CUDA ≥ 12.9. The setup scripts pass the
  CUDA-matched index as an extra index so the resolver can satisfy both; if it
  cannot, use the NGC container (`docker/docker-cuda`, with `BASE_IMAGE` and
  `CUDA_ARCH_LIST` overridden) instead of fighting pip.
* **The transformers ceiling.** `requirements.txt` caps transformers at
  `<=4.52.1`, which is a LLaMA-Factory constraint, not a hardware one. It is
  kept as-is here. Newer models or FP8 checkpoints may need that cap raised,
  which is a separate change with its own testing.
* **FP8 / NVFP4.** B300's headline feature is not used by this pipeline at all —
  training is bf16, as it was on H100. Enabling it is new work, not a port.

RL runs have one more Blackwell-specific catch: verl's sequence-packing path
calls flash-attn's varlen kernels directly, so it is off unless flash-attn is
built for `sm_103`. See [`verl_rl/README.md`](../verl_rl/README.md) — the
pipeline detects it and reports which path it took.

## 5. Overrides

Every value is overridable, which is also how you test a B300 config from a CPU
login node:

```bash
FORCE_GPU_CC=10.3 FORCE_GPU_MEM_GB=279 bash scripts/gpu_profile.sh --print
FORCE_GPU_CC=10.3 bash setup_env.sh          # build a B300 env anywhere
TORCH_INDEX=https://download.pytorch.org/whl/cu129 bash setup_env.sh
FLASH_ATTN=sdpa AUTO_MICRO_BS=false bash run.sh
```

| variable | effect |
|---|---|
| `FORCE_GPU_CC` | pretend the device has this compute capability |
| `FORCE_GPU_NAME` / `FORCE_GPU_COUNT` / `FORCE_GPU_MEM_GB` | skip the corresponding `nvidia-smi` query |
| `TORCH_INDEX`, `TORCH_SPEC`, `TORCH_CUDA_ARCH_LIST`, `VLLM_SPEC` | used as-is if already set |
| `ATTN_IMPL` / `FLASH_ATTN` | skip the flash-attn probe |
| `AUTO_MICRO_BS=false` | keep the original batch shape |

## 6. Going back to H100

Nothing to revert: run the same commands on the H100 node. The profile detects
`sm_90` and reinstates exactly what each script installed before — `setup_env.sh`
the `torch==2.6.0` / cu126 pin, `shadow_rl/setup.sh` the newest torch from cu128
— along with `fa2` and the 1×256 batch shape. The generated training scripts are
byte-identical to the ones from before this change apart from three comment lines
in the header recording the hardware they were generated for.
