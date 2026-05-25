"""Asynchronous tier migration engine.

Handles token movement between tiers without blocking the main
inference path. Uses CUDA streams for GPU↔CPU transfers and
background quantization/dequantization.

Architecture:
    Main Stream (inference)    Migration Stream (background)
    ─────────────────────     ──────────────────────────────
    forward() ──┐              idle
    attention() │              idle
    decode()    │──event──►    quantize evicted tokens
    forward()   │              transfer to CPU (cold tier)
    attention() │              ◄──complete event──
    decode()    │              idle
                               ...

The key insight: tier migrations are not on the critical path.
We can overlap migration work with the next forward pass, hiding
the latency of quantization and PCIe transfers entirely.

Usage:
    migrator = AsyncMigrator(config)
    migrator.schedule_demotion(layer_idx, evict_indices, keys, values)
    # ... continue inference on main stream ...
    migrator.wait_pending()  # Only block when absolutely needed
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

import torch

from akv.quantizer import KVQuantizer, QuantConfig, QuantizedTensor

logger = logging.getLogger(__name__)


class MigrationType(str, Enum):
    DEMOTE_HOT_TO_WARM = "hot_to_warm"
    DEMOTE_WARM_TO_COLD = "warm_to_cold"
    PROMOTE_COLD_TO_WARM = "cold_to_warm"
    PROMOTE_WARM_TO_HOT = "warm_to_hot"


@dataclass
class MigrationTask:
    """A single tier migration task."""
    migration_type: MigrationType
    layer_idx: int
    token_indices: torch.Tensor
    keys: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None
    quantized_keys: Optional[QuantizedTensor] = None
    quantized_values: Optional[QuantizedTensor] = None
    callback: Optional[Callable] = None
    priority: int = 0  # Higher = more urgent


@dataclass
class MigrationStats:
    """Statistics for migration operations."""
    total_demotions: int = 0
    total_promotions: int = 0
    total_tokens_migrated: int = 0
    hot_to_warm_count: int = 0
    warm_to_cold_count: int = 0
    cold_to_warm_count: int = 0
    warm_to_hot_count: int = 0
    avg_demotion_ms: float = 0.0
    avg_promotion_ms: float = 0.0
    peak_queue_depth: int = 0
    migrations_overlapped: int = 0  # Migrations hidden behind compute


@dataclass
class AsyncMigratorConfig:
    """Configuration for async migration."""
    # Enable async migrations (disable for debugging)
    enabled: bool = True

    # Use a dedicated CUDA stream for migration work
    use_migration_stream: bool = True

    # Maximum pending migrations before forcing a sync
    max_pending: int = 8

    # Batch size for migrations (process multiple layers together)
    batch_size: int = 4

    # Quantization configs
    warm_bits: int = 4
    cold_bits: int = 2
    group_size: int = 128

    # CPU offload settings
    pin_memory: bool = True  # Use pinned memory for faster CPU↔GPU transfers
    non_blocking: bool = True  # Use non-blocking transfers


class AsyncMigrator:
    """Manages asynchronous token migration between cache tiers.

    Core idea: tier migrations (quantization, dequantization, PCIe transfers)
    are not on the critical inference path. By overlapping them with the
    next forward pass using CUDA streams, we can hide their latency entirely.

    For a typical 7B model:
    - Hot→Warm demotion (64 tokens, 32 layers): ~1.2ms
    - Warm→Cold transfer (64 tokens to CPU): ~0.3ms
    - Cold→Warm promotion (16 tokens from CPU): ~0.2ms

    These fit comfortably within a single decode step (~8ms), allowing
    complete overlap.
    """

    def __init__(self, config: Optional[AsyncMigratorConfig] = None, device: str = "cuda"):
        self.config = config or AsyncMigratorConfig()
        self.device = device
        self.stats = MigrationStats()

        # CUDA streams
        self._main_stream = None
        self._migration_stream = None
        self._transfer_stream = None  # Dedicated for PCIe transfers

        if self.config.use_migration_stream and device == "cuda" and torch.cuda.is_available():
            self._migration_stream = torch.cuda.Stream(device=device, priority=-1)  # Low priority
            self._transfer_stream = torch.cuda.Stream(device=device, priority=-1)

        # Task queue
        self._pending_tasks: deque[MigrationTask] = deque()
        self._completed_events: list[torch.cuda.Event] = []

        # Quantizers
        self._warm_quantizer = KVQuantizer(QuantConfig(
            bits=self.config.warm_bits,
            group_size=self.config.group_size,
        ))
        self._cold_quantizer = KVQuantizer(QuantConfig(
            bits=self.config.cold_bits,
            group_size=self.config.group_size,
        ))

        # Lock for thread-safe queue access
        self._lock = threading.Lock()

    @property
    def pending_count(self) -> int:
        return len(self._pending_tasks)

    @property
    def is_idle(self) -> bool:
        return len(self._pending_tasks) == 0

    def schedule_demotion(
        self,
        layer_idx: int,
        token_indices: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        target_tier: str = "warm",
        callback: Optional[Callable] = None,
    ) -> None:
        """Schedule async demotion of tokens from hot to warm/cold tier.

        This returns immediately. The actual quantization and data movement
        happens on the migration stream, overlapped with inference.

        Args:
            layer_idx: Which layer's cache to modify
            token_indices: Indices of tokens to demote
            keys: (n_tokens, D) key tensors to quantize
            values: (n_tokens, D) value tensors to quantize
            target_tier: "warm" or "cold"
            callback: Called when migration completes (on migration stream)
        """
        migration_type = (MigrationType.DEMOTE_HOT_TO_WARM if target_tier == "warm"
                         else MigrationType.DEMOTE_WARM_TO_COLD)

        task = MigrationTask(
            migration_type=migration_type,
            layer_idx=layer_idx,
            token_indices=token_indices,
            keys=keys,
            values=values,
            callback=callback,
        )

        with self._lock:
            self._pending_tasks.append(task)
            self.stats.peak_queue_depth = max(
                self.stats.peak_queue_depth, len(self._pending_tasks)
            )

        # If too many pending, force a flush
        if len(self._pending_tasks) >= self.config.max_pending:
            self._flush_pending()
        else:
            self._try_process_next()

    def schedule_promotion(
        self,
        layer_idx: int,
        token_indices: torch.Tensor,
        quantized_keys: QuantizedTensor,
        quantized_values: QuantizedTensor,
        source_tier: str = "cold",
        callback: Optional[Callable] = None,
    ) -> None:
        """Schedule async promotion of tokens from cold/warm to higher tier.

        Used when the model's attention pattern suddenly focuses on tokens
        that were previously demoted. The promotion brings them back to
        a faster tier for subsequent attention computation.
        """
        migration_type = (MigrationType.PROMOTE_COLD_TO_WARM if source_tier == "cold"
                         else MigrationType.PROMOTE_WARM_TO_HOT)

        task = MigrationTask(
            migration_type=migration_type,
            layer_idx=layer_idx,
            token_indices=token_indices,
            quantized_keys=quantized_keys,
            quantized_values=quantized_values,
            callback=callback,
            priority=1,  # Promotions are higher priority than demotions
        )

        with self._lock:
            # Insert promotions at front (higher priority)
            self._pending_tasks.appendleft(task)

        self._try_process_next()

    def _try_process_next(self) -> None:
        """Process the next migration task on the migration stream."""
        if not self.config.enabled:
            self._process_sync()
            return

        if self._migration_stream is None:
            self._process_sync()
            return

        with self._lock:
            if not self._pending_tasks:
                return
            task = self._pending_tasks.popleft()

        # Process on migration stream (non-blocking)
        with torch.cuda.stream(self._migration_stream):
            self._execute_task(task)

        # Record event for synchronization
        event = torch.cuda.Event()
        event.record(self._migration_stream)
        self._completed_events.append(event)
        self.stats.migrations_overlapped += 1

    def _process_sync(self) -> None:
        """Process all pending tasks synchronously (fallback)."""
        with self._lock:
            tasks = list(self._pending_tasks)
            self._pending_tasks.clear()

        for task in tasks:
            self._execute_task(task)

    def _execute_task(self, task: MigrationTask) -> None:
        """Execute a single migration task."""
        if task.migration_type == MigrationType.DEMOTE_HOT_TO_WARM:
            self._do_hot_to_warm(task)
        elif task.migration_type == MigrationType.DEMOTE_WARM_TO_COLD:
            self._do_warm_to_cold(task)
        elif task.migration_type == MigrationType.PROMOTE_COLD_TO_WARM:
            self._do_cold_to_warm(task)
        elif task.migration_type == MigrationType.PROMOTE_WARM_TO_HOT:
            self._do_warm_to_hot(task)

        if task.callback:
            task.callback(task)

    def _do_hot_to_warm(self, task: MigrationTask) -> None:
        """Quantize tokens from FP16 to INT4 (hot → warm)."""
        q_keys = self._warm_quantizer.quantize(task.keys)
        q_values = self._warm_quantizer.quantize(task.values)

        task.quantized_keys = q_keys
        task.quantized_values = q_values

        self.stats.hot_to_warm_count += 1
        self.stats.total_demotions += 1
        self.stats.total_tokens_migrated += task.token_indices.shape[0]

    def _do_warm_to_cold(self, task: MigrationTask) -> None:
        """Transfer quantized tokens from GPU to CPU (warm → cold)."""
        if task.quantized_keys is not None:
            # Re-quantize to lower precision if needed
            if self.config.cold_bits < self.config.warm_bits:
                # Dequantize warm, then requantize to cold precision
                keys_fp = self._warm_quantizer.dequantize(task.quantized_keys)
                values_fp = self._warm_quantizer.dequantize(task.quantized_values)
                task.quantized_keys = self._cold_quantizer.quantize(keys_fp)
                task.quantized_values = self._cold_quantizer.quantize(values_fp)

            # Transfer to CPU
            if self._transfer_stream:
                with torch.cuda.stream(self._transfer_stream):
                    task.quantized_keys = task.quantized_keys.to_device(
                        "cpu", non_blocking=self.config.non_blocking
                    )
                    task.quantized_values = task.quantized_values.to_device(
                        "cpu", non_blocking=self.config.non_blocking
                    )
            else:
                task.quantized_keys = task.quantized_keys.to_device("cpu")
                task.quantized_values = task.quantized_values.to_device("cpu")

        self.stats.warm_to_cold_count += 1
        self.stats.total_demotions += 1
        self.stats.total_tokens_migrated += task.token_indices.shape[0]

    def _do_cold_to_warm(self, task: MigrationTask) -> None:
        """Transfer tokens from CPU back to GPU (cold → warm)."""
        if task.quantized_keys is not None:
            if self._transfer_stream:
                with torch.cuda.stream(self._transfer_stream):
                    task.quantized_keys = task.quantized_keys.to_device(
                        self.device, non_blocking=self.config.non_blocking
                    )
                    task.quantized_values = task.quantized_values.to_device(
                        self.device, non_blocking=self.config.non_blocking
                    )
                self._transfer_stream.synchronize()
            else:
                task.quantized_keys = task.quantized_keys.to_device(self.device)
                task.quantized_values = task.quantized_values.to_device(self.device)

        self.stats.cold_to_warm_count += 1
        self.stats.total_promotions += 1
        self.stats.total_tokens_migrated += task.token_indices.shape[0]

    def _do_warm_to_hot(self, task: MigrationTask) -> None:
        """Dequantize tokens from INT4 back to FP16 (warm → hot)."""
        if task.quantized_keys is not None:
            task.keys = self._warm_quantizer.dequantize(task.quantized_keys)
            task.values = self._warm_quantizer.dequantize(task.quantized_values)

        self.stats.warm_to_hot_count += 1
        self.stats.total_promotions += 1
        self.stats.total_tokens_migrated += task.token_indices.shape[0]

    def _flush_pending(self) -> None:
        """Process all pending tasks (blocks until complete)."""
        while self._pending_tasks:
            self._try_process_next()
        self.wait_pending()

    def wait_pending(self) -> None:
        """Wait for all pending migrations to complete.

        Only call this when you absolutely need the migration results
        (e.g., before accessing promoted tokens). Otherwise, let
        migrations overlap with inference.
        """
        if self._migration_stream:
            self._migration_stream.synchronize()
        if self._transfer_stream:
            self._transfer_stream.synchronize()

        # Clear completed events
        self._completed_events.clear()

    def synchronize_before_attention(self) -> None:
        """Lightweight sync: only wait if there are pending promotions.

        Call this before attention computation to ensure promoted tokens
        are available. Demotions don't need to be waited on since they
        only affect evicted tokens.
        """
        # Check if any pending promotions need to complete
        has_promotions = any(
            t.migration_type in (MigrationType.PROMOTE_COLD_TO_WARM, MigrationType.PROMOTE_WARM_TO_HOT)
            for t in self._pending_tasks
        )

        if has_promotions:
            self._flush_pending()

    def get_stats(self) -> dict:
        """Get migration statistics."""
        return {
            "total_demotions": self.stats.total_demotions,
            "total_promotions": self.stats.total_promotions,
            "total_tokens_migrated": self.stats.total_tokens_migrated,
            "hot_to_warm": self.stats.hot_to_warm_count,
            "warm_to_cold": self.stats.warm_to_cold_count,
            "cold_to_warm": self.stats.cold_to_warm_count,
            "warm_to_hot": self.stats.warm_to_hot_count,
            "peak_queue_depth": self.stats.peak_queue_depth,
            "migrations_overlapped": self.stats.migrations_overlapped,
            "pending": self.pending_count,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = MigrationStats()
