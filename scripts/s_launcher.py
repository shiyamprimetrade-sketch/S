"""Source checkout launcher independent of the user's current directory."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s_developer.cli import main
raise SystemExit(main())
