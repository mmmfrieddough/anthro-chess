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
    OpeningEntry,
    load_book,
    opening_book_identity,
)
from anthro_chess.evaluation.openings.classification import (
    OPENING_CLASSIFICATION_VERSION,
    UNCLASSIFIED,
    UNCLASSIFIED_LABEL,
    OpeningClassificationError,
    OpeningLabel,
    OpeningLevel,
    classify_action_ids,
    classify_moves,
    opening_distribution,
    opening_levels,
)

__all__ = [
    "BOOK_FILE_NAME",
    "BOOK_METADATA_FILE_NAME",
    "OPENING_CLASSIFICATION_VERSION",
    "UNCLASSIFIED",
    "UNCLASSIFIED_LABEL",
    "OpeningBook",
    "OpeningBookError",
    "OpeningClassificationError",
    "OpeningEntry",
    "OpeningLabel",
    "OpeningLevel",
    "classify_action_ids",
    "classify_moves",
    "load_book",
    "opening_book_identity",
    "opening_distribution",
    "opening_levels",
]
