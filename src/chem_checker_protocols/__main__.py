"""Allow ``python -m chem_checker_protocols``."""

import sys

from .cli import main

sys.exit(main())
