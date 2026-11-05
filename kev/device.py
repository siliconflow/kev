"""The accelerator this process uses: cuda, then mps, then cpu."""
import re

import torch


def default_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def kind(device):
    """The accelerator kind of a device given as a string ("cuda", "mps", "cpu") or a torch.device ("cuda:0"):
    a device object never compares equal to its string, so normalise before branching."""
    return getattr(device, "type", None) or str(device).split(":", 1)[0]


def sync(device):
    """Wait for queued kernels, so wall-clock timings around a forward pass are real."""
    if kind(device) == "mps": torch.mps.synchronize()
    elif kind(device) == "cuda": torch.cuda.synchronize()


def empty_cache(device):
    if kind(device) == "mps": torch.mps.empty_cache()
    elif kind(device) == "cuda": torch.cuda.empty_cache()


def out_of_memory(e):
    """Whether a torch allocator ran out: CUDA raises torch.OutOfMemoryError, MPS a plain RuntimeError with this message.
    The MLX backend's Metal errors are neither."""
    return isinstance(e, torch.OutOfMemoryError) or isinstance(e, RuntimeError) and str(e).startswith("MPS backend out of memory")


def allocated_bytes(device):
    """Bytes currently allocated on the device (MPS) or the peak since the process started (CUDA); 0 on CPU."""
    if kind(device) == "mps": return torch.mps.current_allocated_memory()
    if kind(device) == "cuda": return torch.cuda.max_memory_allocated()
    return 0


def free_bytes(device):
    """Free device memory as the driver sees it, or None on a backend with no such query.

    Reserved-but-unallocated caching-allocator slack is NOT free for a new segment mapping (a fragmented allocator
    fails a 20 MiB map with 1.9 MiB free while its own slack is larger), so a caller deciding whether a mapping can
    succeed must read the driver's number, not the allocator's."""
    return torch.cuda.mem_get_info(device)[0] if kind(device) == "cuda" else None


_ACCELERATOR_ERROR = getattr(torch, "AcceleratorError", None)   # torch >= 2.6 raises it for CUDA runtime failures

_UNRECOVERABLE = re.compile(
    r"Sticky error|CUDA_ERROR_|unspecified launch failure|illegal memory access|device-side assert|"
    r"CUDA error:|illegal instruction|context is destroyed|context has been destroyed", re.IGNORECASE)

# Runtime errors that keep the context usable: a workspace allocation that failed can be shed and retried, and a
# launch configuration only the host got wrong is that request's 500, not the instance's death.
_RECOVERABLE_CUDA = re.compile(r"ALLOC_FAILED|invalid (configuration )?argument", re.IGNORECASE)


def unrecoverable(e):
    """Whether a failure poisons the accelerator context: shedding and retrying cannot serve again, only a restart of
    the process can (CUDA's sticky errors: after a launch failure every later CUDA call fails in-process).

    Out-of-memory is NOT unrecoverable even when the message says "CUDA error: out of memory": kev.serve sheds and
    retries those, and the eager path serves without the graphs. A plain RuntimeError is not either ("CUDA graphs are
    off: ..." is one), and neither is every torch.AcceleratorError: torch raises that one type for every checked
    cudaError_t, including CUBLAS_STATUS_ALLOC_FAILED and "invalid configuration argument", which leave the context
    valid - restarting the container over them would loop on a request that only deserves its own 500."""
    msg = str(e) or repr(e)
    if out_of_memory(e) or "out of memory" in msg.lower(): return False
    if _RECOVERABLE_CUDA.search(msg): return False
    if _ACCELERATOR_ERROR is not None and isinstance(e, _ACCELERATOR_ERROR): return True
    return bool(_UNRECOVERABLE.search(msg))
