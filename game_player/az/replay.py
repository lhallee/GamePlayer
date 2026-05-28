from __future__ import annotations

import random
from collections import deque
from collections.abc import Iterable
from typing import Deque

from game_player.az.self_play import SelfPlaySample


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        assert capacity > 0
        self.capacity = capacity
        self.samples: Deque[SelfPlaySample] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.samples)

    def add(self, sample: SelfPlaySample) -> None:
        self.samples.append(sample)

    def add_many(self, samples: Iterable[SelfPlaySample]) -> None:
        for sample in samples:
            self.add(sample)

    def sample(
        self,
        batch_size: int,
        rng: random.Random | None = None,
    ) -> list[SelfPlaySample]:
        assert batch_size > 0
        assert len(self.samples) >= batch_size
        if rng is None:
            rng = random.Random()
        return rng.sample(list(self.samples), batch_size)
