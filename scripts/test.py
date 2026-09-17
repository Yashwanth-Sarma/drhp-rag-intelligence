"""Run this project's tests without collecting legacy scripts or calling APIs."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.runtime'),str(ROOT)]
if __name__=='__main__':
    import pytest
    raise SystemExit(pytest.main([str(ROOT/'tests'),'-q',*sys.argv[1:]]))
