"""Run the command-line interface as ``python -m anthro_chess``.

The console script is the interface people use. This exists for the case that
cannot rely on one: a subprocess launched from inside the package, which knows
its interpreter but not whether the script directory is on ``PATH``.
"""

from __future__ import annotations

import sys

from anthro_chess.interfaces.cli import main

if __name__ == "__main__":
    sys.exit(main())
