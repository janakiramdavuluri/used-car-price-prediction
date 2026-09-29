import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
from pathlib import Path
from src.automate_lifecycle import inspect_candidate
ROOT=Path(__file__).resolve().parents[1]
def generate_report():
    report=inspect_candidate()
    p=ROOT/'reports/registry_report.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(report,indent=2))
    return report
if __name__=='__main__':print(generate_report())
