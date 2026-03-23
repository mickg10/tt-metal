# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TraceRetainer — prevent DRAM address reuse during trace capture.

When decode traces are captured, intermediate tensors are allocated in regular DRAM
(not the trace region). After capture, these intermediates are freed, but the trace
command buffer still references their DRAM addresses. If prefill reuses those addresses,
trace replay reads/writes corrupted data.

The TraceRetainer prevents this by holding references to intermediates after trace
capture, keeping their DRAM addresses occupied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import ttnn


@dataclass
class TraceRetainer:
    """Holds intermediate tensor references to prevent DRAM address reuse."""
    enabled: bool = False
    held: list = field(default_factory=list)

    def release_all(self) -> int:
        """Free all retained tensors. Returns count freed."""
        count = len(self.held)
        while self.held:
            t = self.held.pop()
            if t is not None:
                try:
                    ttnn.deallocate(t, force=True)
                except Exception:
                    pass
        return count


# ---------------------------------------------------------------------------
# Module-level active retainer (set during trace capture, cleared after)
# ---------------------------------------------------------------------------

_active_trace_retainer: TraceRetainer | None = None


def set_trace_retainer(retainer: TraceRetainer | None) -> None:
    """Activate a TraceRetainer for the decode forward path."""
    global _active_trace_retainer
    _active_trace_retainer = retainer


def clear_trace_retainer() -> None:
    """Deactivate the TraceRetainer."""
    global _active_trace_retainer
    _active_trace_retainer = None


def _dealloc(tensor: Any, *, force: bool = False) -> None:
    """Deallocate or retain a tensor depending on the active TraceRetainer."""
    if tensor is None:
        return
    if _active_trace_retainer is not None and _active_trace_retainer.enabled:
        _active_trace_retainer.held.append(tensor)
        return
    ttnn.deallocate(tensor, force=force)
