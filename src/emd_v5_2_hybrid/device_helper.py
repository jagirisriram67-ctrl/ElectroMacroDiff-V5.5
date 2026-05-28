"""Unified device helper for CPU, CUDA, and explicit TPU/XLA training.

The V5.5 Kaggle staircase defaults to CUDA/CPU because the current RDKit,
graph, and docking workflows are GPU/CPU-native. XLA/TPU is available only
when explicitly requested, so ``--device auto`` cannot accidentally send a
non-XLA-safe script to TPU.
"""

from __future__ import annotations


def resolve_device(preference: str = "auto") -> str:
    """Return the best PyTorch device string for the given preference.

    ``auto`` tries CUDA, then CPU.
    ``auto-xla`` tries XLA, then CUDA, then CPU.
    ``xla``/``tpu`` forces TPU and raises if unavailable.
    ``cuda`` forces CUDA and raises if unavailable.
    ``cpu`` always returns CPU.
    """
    preference = str(preference).lower().strip()

    if preference == "cpu":
        return "cpu"

    if preference in ("xla", "tpu"):
        return _require_xla()

    if preference == "cuda":
        return _require_cuda()

    if preference in ("auto-xla", "auto_tpu", "auto-tpu"):
        xla = _try_xla()
        if xla is not None:
            return xla
        cuda = _try_cuda()
        if cuda is not None:
            return cuda
        return "cpu"

    cuda = _try_cuda()
    if cuda is not None:
        return cuda
    return "cpu"


def resolve_xla_device_for_experimental_training() -> str:
    """Return an XLA device only for scripts that implement real XLA stepping."""
    result = _try_xla()
    if result is None:
        raise RuntimeError("XLA/TPU is not available in this runtime.")
    return result


def _try_xla() -> str | None:
    try:
        import torch_xla.core.xla_model as xm  # type: ignore[import-untyped]

        device = xm.xla_device()
        return str(device)
    except Exception:
        return None


def _require_xla() -> str:
    result = _try_xla()
    if result is None:
        raise RuntimeError(
            "XLA/TPU requested but torch_xla is not available. "
            "Use --device auto for CUDA/CPU, or use an XLA-adapted script."
        )
    return result


def _try_cuda() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return None


def _require_cuda() -> str:
    result = _try_cuda()
    if result is None:
        raise RuntimeError("CUDA requested but no GPU is available. Use --device auto or --device cpu.")
    return result


def move_to(tensor_or_dict, device: str):
    """Move a tensor or a dict of tensors to the given device."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install torch") from exc

    if isinstance(tensor_or_dict, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in tensor_or_dict.items()
        }
    if hasattr(tensor_or_dict, "to"):
        return tensor_or_dict.to(device)
    return tensor_or_dict


def xla_step(optimizer=None):
    """Call XLA mark_step if running on TPU; no-op otherwise."""
    try:
        import torch_xla.core.xla_model as xm  # type: ignore[import-untyped]

        xm.mark_step()
    except Exception:
        pass


def device_summary(device: str) -> dict:
    """Return a JSON-serializable summary of the resolved device."""
    summary = {"device": device, "device_type": "cpu"}
    try:
        import torch

        summary["torch_version"] = torch.__version__
        if "xla" in device.lower() or "tpu" in device.lower():
            summary["device_type"] = "tpu"
            try:
                import torch_xla  # type: ignore[import-untyped]

                summary["torch_xla_version"] = torch_xla.__version__
            except Exception:
                pass
        elif "cuda" in device.lower():
            summary["device_type"] = "cuda"
            summary["cuda_device_name"] = torch.cuda.get_device_name(0)
            summary["cuda_memory_gb"] = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2)
    except Exception:
        pass
    return summary
