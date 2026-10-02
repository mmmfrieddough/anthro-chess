import pytest
import torch

from anthro_chess.training import distributed
from anthro_chess.training.distributed import (
    SINGLE_PROCESS,
    DataParallel,
    DistributedError,
    launched_parallelism,
    rank_device,
    rank_seed,
)

PAIR = DataParallel(rank=1, local_rank=1, world_size=2)


def test_one_process_keeps_the_device_it_resolved() -> None:
    assert rank_device(SINGLE_PROCESS, torch.device("mps")) == torch.device("mps")


def test_a_backend_without_collectives_refuses_several_ranks() -> None:
    with pytest.raises(DistributedError, match="not supported on MPS"):
        rank_device(PAIR, torch.device("mps"))


def test_a_rank_with_no_card_of_its_own_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(DistributedError, match="local rank 1 has no CUDA device"):
        rank_device(PAIR, torch.device("cuda"))


def test_cpu_ranks_share_the_host() -> None:
    assert rank_device(PAIR, torch.device("cpu")) == torch.device("cpu")


def test_a_launch_that_does_not_name_the_rank_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(distributed.dist, "is_torchelastic_launched", lambda: True)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.delenv("RANK", raising=False)

    with pytest.raises(DistributedError, match="RANK"):
        launched_parallelism()


def test_ranks_and_resumes_draw_from_their_own_streams() -> None:
    seeds = {rank_seed(17, rank, step) for rank in (1, 2) for step in (0, 25000)}

    assert len(seeds) == 4
    assert rank_seed(17, 1, 0) == rank_seed(17, 1, 0)
