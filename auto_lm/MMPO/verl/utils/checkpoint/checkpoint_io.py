"""Publish a PyTorch shard only after Python I/O and filesystem sync succeed."""
import os
from pathlib import Path
import tempfile
import zipfile

import torch


def atomic_torch_save(value, destination):
    destination = Path(destination)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        with zipfile.ZipFile(temporary) as archive:
            if not archive.infolist():
                raise ValueError("empty checkpoint archive")
        os.replace(temporary, destination)
    except Exception as exc:
        # torch's ZIP finalizer can mask the original Python write exception.
        cause = exc
        while cause is not None:
            if isinstance(cause, OSError) and cause.errno is not None:
                raise OSError(cause.errno, f"Checkpoint write/sync failed: {cause}", str(destination)) from exc
            cause = cause.__context__
        raise
    finally:
        Path(temporary).unlink(missing_ok=True)
