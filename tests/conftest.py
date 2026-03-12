# Корень репозитория в PYTHONPATH, чтобы импортировать app и ragas_eval
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
