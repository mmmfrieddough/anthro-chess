"""The frozen training configuration every candidate change is read against.

``docs/decisions/0065-a-frozen-ablation-vehicle-is-the-base-a-seed-floor-can-live-on.md``
owns why this exists. What lives here is the one thing that has to be checked by
machine: the digest of the configuration, so that editing it fails loudly
instead of silently invalidating every comparison ever read against it.

Two of the digest's inputs describe a prepared corpus rather than anything in
this repository, so they are pinned here as values. Everything else is recomputed
from the checked-in configuration and from code, which is what lets the identity
be verified where no corpus exists. A corpus prepared differently moves
``dataset_sha256``, and the arm trained against it then carries an identity the
stored seed dispersion has no entry for, which is reported as having no floor
rather than quoted against the wrong one.
"""

from __future__ import annotations

from collections.abc import Mapping

from anthro_chess.data.streaming import shard_loader_configuration_sha256
from anthro_chess.models.move_model import model_identity
from anthro_chess.training.checkpoints import training_identity_sha256
from anthro_chess.training.config import TrainingConfig
from anthro_chess.training.runner import compatibility_record

#: The widened corpus the vehicle trains on, as the manifest that describes it
#: and the shard set the shard-backed loader resolved from it. Both are content
#: derived, so a corpus prepared from the same pinned selection reproduces them.
VEHICLE_CORPUS_MANIFEST_SHA256 = (
    "371655fd0e809f11b34b617be41fffc5e5dd77b1613bb15d3e34d8783f5a71da"
)
VEHICLE_CORPUS_DATASET_SHA256 = (
    "2f6d948982fcd6d43a34c9fdc49a01eeb28c2d77d6c21a7765d23d9aae3eac93"
)

#: What every reading taken against the vehicle carries in its result envelope.
#: A comparison finds the vehicle's seed dispersion by this exact value or
#: reports that it has none.
VEHICLE_TRAINING_SHA256 = (
    "13ebf875b3116fa04f2d0321637be7b843deccbbc9e4ff2ee72ed60db319d088"
)


def vehicle_compatibility(config: TrainingConfig) -> Mapping[str, object]:
    """Return the compatibility record a vehicle arm of ``config`` would write.

    The corpus half is the pinned pair above rather than a reading of the
    machine, so this answers what the checked-in configuration means rather than
    what one host happens to hold.
    """

    return compatibility_record(
        config,
        data={
            "train": {
                "manifest_sha256": VEHICLE_CORPUS_MANIFEST_SHA256,
                "dataset_sha256": VEHICLE_CORPUS_DATASET_SHA256,
                "loader_configuration_sha256": _loader_identity(config),
            },
            "validation": None,
        },
        model=model_identity(config.model),
    )


def _loader_identity(config: TrainingConfig) -> str:
    """Return the loader digest a vehicle arm records, not the selection's own.

    A shard-backed run records the selection digest wrapped with its planning
    window, and the vehicle declares ``[train.streaming]``, so it takes that
    path. Digesting the selection alone here would name an identity no arm can
    carry, and the constant below would then be a key nothing looks anything up
    by.
    """

    if config.train.streaming is None:
        raise ValueError(
            "the vehicle trains through the shard-backed loader; a "
            "configuration without a streaming section is not it"
        )
    return shard_loader_configuration_sha256(
        config.train.loader,
        config.train.streaming,
    )


def vehicle_training_sha256(config: TrainingConfig) -> str:
    """Return the training identity ``config`` produces as the vehicle."""

    return training_identity_sha256(vehicle_compatibility(config))
