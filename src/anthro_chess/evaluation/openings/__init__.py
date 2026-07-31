"""Opening classification from the project's own versioned book.

Game-level classification only: one label per game at each granularity level,
for grouping and distribution comparison. Per-ply multi-label opening metadata
for preference conditioning is separate later work that should extend this
rather than replace it.
"""

from anthro_chess.evaluation.openings.book import (
    BOOK_FILE_NAME,
    BOOK_METADATA_FILE_NAME,
    OpeningBook,
    OpeningBookError,
    OpeningContinuation,
    OpeningEntry,
    load_book,
    opening_book_identity,
)
from anthro_chess.evaluation.openings.classification import (
    OPENING_CLASSIFICATION_VERSION,
    UNCLASSIFIED_LABEL,
    OpeningClassificationError,
    OpeningLabel,
    classify_action_ids,
    classify_moves,
    classify_progression,
    opening_distribution,
    progression_for_action_ids,
    repertoire_distribution,
)
from anthro_chess.evaluation.openings.names import (
    UNCLASSIFIED,
    OpeningLevel,
    opening_level,
    opening_levels,
)
from anthro_chess.evaluation.openings.tree import (
    OPENING_TREE_VERSION,
    ActionPolicy,
    OpeningTreeError,
    RepertoireWalk,
    walk_repertoire,
)

__all__ = [
    "BOOK_FILE_NAME",
    "BOOK_METADATA_FILE_NAME",
    "OPENING_CLASSIFICATION_VERSION",
    "OPENING_TREE_VERSION",
    "UNCLASSIFIED",
    "UNCLASSIFIED_LABEL",
    "ActionPolicy",
    "OpeningBook",
    "OpeningBookError",
    "OpeningClassificationError",
    "OpeningContinuation",
    "OpeningEntry",
    "OpeningLabel",
    "OpeningLevel",
    "OpeningTreeError",
    "RepertoireWalk",
    "classify_action_ids",
    "classify_moves",
    "classify_progression",
    "load_book",
    "opening_book_identity",
    "opening_distribution",
    "opening_level",
    "opening_levels",
    "progression_for_action_ids",
    "repertoire_distribution",
    "walk_repertoire",
]
