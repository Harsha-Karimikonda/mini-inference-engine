from collections import OrderedDict
from dataclasses import dataclass

from .errors import CachePressure


@dataclass
class Allocation:
    request_id: str
    blocks: list[int]


class KVCache:
    """Logical paged cache model; it deliberately does not own model tensors."""

    def __init__(self, blocks: int = 256, block_tokens: int = 16):
        if blocks <= 0 or block_tokens <= 0:
            raise ValueError("blocks and block_tokens must be positive")
        self.capacity = blocks
        self.block_tokens = block_tokens
        self.free = set(range(blocks))
        self.allocations: dict[str, Allocation] = {}
        self.idle = OrderedDict[str, None]()

    def allocate(self, request_id: str, tokens: int) -> Allocation:
        if request_id in self.allocations:
            return self.allocations[request_id]
        needed = max(1, (tokens + self.block_tokens - 1) // self.block_tokens)
        while len(self.free) < needed and self.idle:
            victim = next(iter(self.idle))
            self.idle.pop(victim, None)
            allocation = self.allocations.pop(victim, None)
            if allocation:
                self.free.update(allocation.blocks)
        if len(self.free) < needed:
            raise CachePressure(f"need {needed} blocks, only {len(self.free)} free")
        chosen = sorted(self.free)[:needed]
        self.free.difference_update(chosen)
        allocation = Allocation(request_id, chosen)
        self.allocations[request_id] = allocation
        return allocation

    def release(self, request_id: str, reusable: bool = False) -> None:
        allocation = self.allocations.pop(request_id, None)
        if not allocation:
            self.idle.pop(request_id, None)
            return
        self.free.update(allocation.blocks)
        self.idle.pop(request_id, None)

    def mark_idle(self, request_id: str) -> None:
        if request_id in self.allocations:
            self.idle[request_id] = None

    @property
    def utilization(self) -> float:
        return (self.capacity - len(self.free)) / self.capacity

    @property
    def fragmentation(self) -> float:
        if not self.free:
            return 0.0
        largest = run = previous = 0
        for block in sorted(self.free):
            run = run + 1 if block == previous + 1 else 1
            largest = max(largest, run)
            previous = block
        return 1.0 - (largest / max(1, len(self.free)))
