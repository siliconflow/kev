"""The accelerator this process uses: cuda, then mps, then cpu."""
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
