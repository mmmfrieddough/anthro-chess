"""Data-parallel training across the processes `torchrun` launches.

The launch decides how many ranks a run uses; no configuration field asks for it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from hashlib import sha256

import torch
import torch.distributed as dist

#: How long a rank waits at a collective for the others. Torch's ten-minute
#: default is shorter than a cadence reading or checkpoint the primary rank takes
#: alone while the rest wait at the next reduction. A rank that dies is not
#: waited on this long: `torchrun` stops the rest as soon as one exits.
COLLECTIVE_TIMEOUT = timedelta(hours=6)


class DistributedError(ValueError):
    """Raised when a launch cannot be run as data-parallel training."""


@dataclass(frozen=True)
class DataParallel:
    """Which of a run's processes this one is."""

    rank: int
    local_rank: int
    world_size: int

    @property
    def primary(self) -> bool:
        """Return whether this rank writes what the run writes once."""

        return self.rank == 0

    @property
    def distributed(self) -> bool:
        """Return whether there is anyone to reduce with."""

        return self.world_size > 1


SINGLE_PROCESS = DataParallel(rank=0, local_rank=0, world_size=1)


def launched_parallelism() -> DataParallel:
    """Read this process's place in the launch from the `torchrun` environment."""

    if not dist.is_torchelastic_launched():
        return SINGLE_PROCESS
    try:
        return DataParallel(
            rank=int(os.environ["RANK"]),
            local_rank=int(os.environ["LOCAL_RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
        )
    except (KeyError, ValueError) as error:
        raise DistributedError(
            f"the torchrun environment does not name this process's rank: {error}"
        ) from error


def rank_device(parallel: DataParallel, device: torch.device) -> torch.device:
    """Return the device one rank trains on, given the backend the run resolved.

    CUDA ranks take one card each by local rank. CPU ranks share the host. MPS
    has one device and no collective backend.
    """

    if not parallel.distributed:
        return device
    if device.type == "cuda":
        count = torch.cuda.device_count()
        if parallel.local_rank >= count:
            raise DistributedError(
                f"local rank {parallel.local_rank} has no CUDA device: this host "
                f"has {count}, so launch at most {count} process(es) per node"
            )
        selected = torch.device("cuda", parallel.local_rank)
        torch.cuda.set_device(selected)
        return selected
    if device.type == "cpu":
        return device
    raise DistributedError(
        f"data-parallel training is not supported on {device.type.upper()}; "
        f"launch one process"
    )


def start_process_group(device: torch.device) -> None:
    """Join the launch's process group over the backend ``device`` reduces on."""

    dist.init_process_group(
        backend="nccl" if device.type == "cuda" else "gloo",
        timeout=COLLECTIVE_TIMEOUT,
        device_id=device if device.type == "cuda" else None,
    )


def rank_seed(seed: int, rank: int, step: int) -> int:
    """Return the seed a non-primary rank's random streams continue from.

    Ranks initialize from the run seed so the model starts identical everywhere,
    then reseed so their dropout draws differ. A checkpoint holds only the
    primary rank's streams, so a resumed rank reseeds from the step it resumed
    at rather than replaying its opening draws.
    """

    digest = sha256(f"{seed}\0rank\0{rank}\0step\0{step}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1
