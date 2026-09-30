"""Allow ``python -m chemcheck_protocols``."""

import sys

from .cli import main

sys.exit(main())
