"""Data-parallel training across the processes `torchrun` launches.

A run is one process unless `torchrun` started several, so nothing in a
configuration asks for this: the launch decides how many cards a run uses, and
the configuration decides what the run computes. Every rank runs the same loop
on its own share of each optimizer step's micro-batches, and gradients are
averaged across ranks before the update.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from hashlib import sha256

import torch
import torch.distributed as dist

#: How long a rank waits at a collective for the others. Torch's own default is
#: ten minutes, and the primary rank alone takes every in-training cadence
#: reading and writes every checkpoint while the others wait at the next
#: gradient reduction, so a long benchmark preview would otherwise abort a run
#: that was working. A rank that genuinely dies is not waited on for this long:
#: `torchrun` stops the rest as soon as one exits.
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

    CUDA ranks take one card each by local rank. CPU ranks share the host, which
    is what lets the suite exercise this path without a card. MPS has one device
    and no collective backend.
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

    Every rank initializes from the run seed, so the model starts identical
    everywhere. Afterwards the streams have to differ, or each rank draws the
    same history-dropout mask for different data and the step sees fewer
    independent draws than one process taking every micro-batch would. A
    checkpoint holds the primary rank's streams alone, so a resumed rank reseeds
    from the step it resumed at rather than replaying its opening draws.
    """

    digest = sha256(f"{seed}\0rank\0{rank}\0step\0{step}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1
