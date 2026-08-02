"""What the suite says about the host it is running on.

Every sentence here is one a CPU-only continuous-integration run never
produces, which is precisely why it is worth pinning: the case these lines
exist for is a provisioned accelerator host, and that host is not the one the
merge check runs on.
"""

from __future__ import annotations

from accelerators import HOST, AcceleratorSurface, detect_surface

CUDA_HOST = AcceleratorSurface(
    present=(("cuda", 2),),
    selectable_training=("mps",),
    selectable_inference=("mps",),
)
APPLE_HOST = AcceleratorSurface(
    present=(("mps", 1),),
    selectable_training=("mps",),
    selectable_inference=("mps",),
)
CPU_HOST = AcceleratorSurface(
    present=(),
    selectable_training=("mps",),
    selectable_inference=("mps",),
)
#: A CUDA host whose inference selection accepts the card and whose training
#: selection does not. Not a contrived case: it is what this repository looked
#: like between the two halves of CUDA support landing, and it is what every
#: future backend will look like while one path has it and the other does not.
HALF_SELECTABLE_HOST = AcceleratorSurface(
    present=(("cuda", 2),),
    selectable_training=("mps",),
    selectable_inference=("cuda", "mps"),
)


def test_an_unselectable_accelerator_is_reported_rather_than_left_implied() -> None:
    lines = CUDA_HOST.report_lines()

    assert lines[0] == "accelerators present: cuda (2 device(s))"
    assert lines[1] == "accelerators the device selection accepts: mps"
    assert "no present accelerator is selectable" in lines[2]


def test_a_host_whose_accelerator_is_selectable_needs_no_warning() -> None:
    lines = APPLE_HOST.report_lines()

    assert lines == [
        "accelerators present: mps (1 device(s))",
        "accelerators the device selection accepts: mps",
    ]
    assert APPLE_HOST.usable_training == ("mps",)


def test_a_host_with_no_accelerator_reports_that_rather_than_warning() -> None:
    lines = CPU_HOST.report_lines()

    assert lines == [
        "accelerators present: none",
        "accelerators the device selection accepts: mps",
    ]
    assert CPU_HOST.usable_training == ()


def test_an_accelerator_only_one_selection_accepts_is_reported_per_selection() -> None:
    """The two selections are reported apart because they can disagree.

    Saying only "no present accelerator is selectable" here would be false,
    and saying nothing would let a run that exercised no training path on a
    working GPU read exactly like one that did.
    """

    lines = HALF_SELECTABLE_HOST.report_lines()

    assert lines[0] == "accelerators present: cuda (2 device(s))"
    assert lines[1] == "accelerators the device selection accepts: cuda, mps"
    assert lines[2] == (
        "the training device selection does not accept cuda, so nothing here "
        "exercised a training path on it"
    )
    assert len(lines) == 3
    assert HALF_SELECTABLE_HOST.usable_inference == ("cuda",)
    assert HALF_SELECTABLE_HOST.usable_training == ()


def test_a_host_both_selections_accept_needs_no_warning() -> None:
    both = AcceleratorSurface(
        present=(("cuda", 1),),
        selectable_training=("cuda", "mps"),
        selectable_inference=("cuda", "mps"),
    )

    assert both.report_lines() == [
        "accelerators present: cuda (1 device(s))",
        "accelerators the device selection accepts: cuda, mps",
    ]


def test_skip_reasons_name_the_selection_that_rejected_the_accelerator() -> None:
    """Each selection explains itself, because they can reject different hosts.

    The same GPU is rejected by training here and accepted by inference, so a
    shared sentence would be wrong for one of them.
    """

    training = HALF_SELECTABLE_HOST.training_skip_reason("cuda")
    inference = CUDA_HOST.inference_skip_reason("cuda")

    assert training.endswith("which the training device selection does not accept")
    assert inference.endswith("which the inference device selection does not accept")
    assert HALF_SELECTABLE_HOST.inference_skip_reason("cuda") != inference


def test_skip_reasons_separate_an_absent_accelerator_from_a_rejected_one() -> None:
    absent = CPU_HOST.training_skip_reason("mps")
    rejected = CUDA_HOST.training_skip_reason("mps")

    assert absent.endswith("this host has no accelerator the project targets")
    assert "cuda (2 device(s))" in rejected
    assert rejected.endswith("which the training device selection does not accept")
    assert absent != rejected


def test_a_supported_accelerator_that_is_simply_missing_reads_plainly() -> None:
    mixed = AcceleratorSurface(
        present=(("mps", 1),),
        selectable_training=("cuda", "mps"),
        selectable_inference=("cuda", "mps"),
    )

    reason = mixed.training_skip_reason("cuda")

    assert reason == (
        "requires an available PyTorch CUDA backend; this host has mps (1 device(s))"
    )


def test_the_detected_surface_describes_this_process() -> None:
    detected = detect_surface()

    assert detected == HOST
    assert set(detected.present_backends) <= {"mps", "cuda"}
    assert all(count > 0 for _, count in detected.present)
    assert "mps" in detected.selectable_training
    assert "cpu" not in detected.selectable_training
    assert "auto" not in detected.selectable_inference
    assert "cuda" in detected.selectable_inference
    assert "cuda" in detected.selectable_training
