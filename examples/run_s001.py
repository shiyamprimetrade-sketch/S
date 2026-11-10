"""Run S-001 and write a reproducible local JSON evidence file."""
import json
from pathlib import Path
from continuum.s001 import run_experiment

def main():
    result=run_experiment()
    target=Path("reports")/"S-001-evidence.json"
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"correct":result["correct"],"total":result["total"],"baseline":result["baseline"],"sha256":result["sha256"],"report":str(target)},indent=2))
if __name__=="__main__":main()
