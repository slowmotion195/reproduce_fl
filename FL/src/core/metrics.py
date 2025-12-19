import os
import json
from typing import Dict, Any

class MetricsWriter:
    def __init__(self, run_dir: str, filename: str = "metrics.jsonl"):
        self.path = os.path.join(run_dir, filename)
        self.f = open(self.path, "a", encoding="utf-8")

    def write(self, rec: Dict[str, Any]):
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass
