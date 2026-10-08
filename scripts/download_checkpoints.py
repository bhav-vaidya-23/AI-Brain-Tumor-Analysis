#!/usr/bin/env python
"""Wrapper for backend.download_checkpoints."""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.download_checkpoints import main

if __name__ == "__main__":
    main()
