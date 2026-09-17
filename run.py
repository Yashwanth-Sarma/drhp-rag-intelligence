"""Local launcher. Workspace dependencies are optional; standard venv also works."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / '.runtime'))
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('finsight.api:create_app', factory=True, host='127.0.0.1', port=8765)
