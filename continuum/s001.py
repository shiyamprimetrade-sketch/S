"""S-001: infer compact suffix rewrite rules from demonstrations.

An experimental symbolic compiler, not an autonomous general-purpose inventor.
The learned rules are executable data, not Python code or arbitrary shell commands.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import hashlib
import json

@dataclass(frozen=True)
class Rule:
    suffix: str
    replacement: str
    support: int

    def apply(self, word: str) -> str:
        return word[:-len(self.suffix)] + self.replacement if self.suffix else word + self.replacement

class MeaningCompiler:
    """Induce suffix rewrite instructions with support and conflict checks."""

    def __init__(self, min_support: int = 2, max_suffix: int = 3):
        if min_support < 2 or not 1 <= max_suffix <= 8:
            raise ValueError("Invalid compiler configuration")
        self.min_support = min_support
        self.max_suffix = max_suffix
        self.rules: list[Rule] = []
        self.examples: list[tuple[str, str]] = []

    def fit(self, examples: list[tuple[str, str]]) -> "MeaningCompiler":
        if len(examples) < 2:
            raise ValueError("At least two examples required")
        if len({a for a, _ in examples}) != len(examples):
            raise ValueError("Duplicate training inputs")
        if any(not isinstance(a, str) or not isinstance(b, str) or not a or len(a)>500 or len(b)>1000 for a,b in examples):
            raise ValueError("Invalid training example")
        self.examples = list(examples)
        candidates: dict[tuple[str,str], set[str]] = {}
        for word, expected in examples:
            # A rewrite is possible if the output preserves the input prefix
            # preceding a chosen suffix.
            for size in range(0, min(self.max_suffix, len(word)-1)+1):
                suffix = word[-size:] if size else ""
                prefix = word[:-size] if size else word
                if expected.startswith(prefix):
                    replacement = expected[len(prefix):]
                    candidates.setdefault((suffix, replacement), set()).add(word)
        rules = []
        for (suffix, replacement), words in candidates.items():
            if len(words) < self.min_support:
                continue
            # A rule must fit every training word with its suffix.
            matching = [(w,e) for w,e in examples if w.endswith(suffix)]
            if not matching or any((w[:-len(suffix)] if suffix else w) + replacement != e for w,e in matching):
                continue
            rules.append(Rule(suffix, replacement, len(words)))
        # Specific rules take precedence. Ties use deterministic ordering.
        self.rules = sorted(rules, key=lambda r: (-len(r.suffix), -r.support, r.suffix, r.replacement))
        return self

    def predict(self, word: str) -> dict:
        if not isinstance(word, str) or not word or len(word)>500:
            raise ValueError("Invalid input")
        if not self.examples:
            raise ValueError("Compiler has not been trained")
        for rule in self.rules:
            if word.endswith(rule.suffix):
                return {"input":word, "output":rule.apply(word), "rule":{"suffix":rule.suffix,"replacement":rule.replacement,"support":rule.support}, "status":"hypothesis"}
        return {"input":word, "output":None, "rule":None, "status":"abstain"}

    def program(self) -> dict:
        return {"language":"S-001 suffix rewrite", "version":1, "rules":[r.__dict__ for r in self.rules], "min_support":self.min_support, "max_suffix":self.max_suffix}

    def evaluate(self, holdout: list[tuple[str,str]]) -> dict:
        if not holdout:
            raise ValueError("Independent holdout is required")
        train_inputs = {w for w,_ in self.examples}
        if any(w in train_inputs for w,_ in holdout):
            raise ValueError("Training and holdout overlap")
        cases=[]
        for word, expected in holdout:
            result=self.predict(word)
            cases.append({"input":word,"expected":expected,"actual":result["output"],"correct":result["output"]==expected,"status":result["status"],"rule":result["rule"]})
        evidence={"program":self.program(),"training_examples":len(self.examples),"holdout_cases":cases,"correct":sum(c["correct"] for c in cases),"total":len(cases),"abstentions":sum(c["status"]=="abstain" for c in cases)}
        evidence["sha256"]=hashlib.sha256(json.dumps(evidence,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        return evidence

def baseline_append_s(word: str) -> str:
    return word + "s"

TRAIN=[("cat","cats"),("dog","dogs"),("bus","buses"),("gas","gases"),("box","boxes"),("fox","foxes"),("city","cities"),("baby","babies")]
HOLDOUT=[("hat","hats"),("mug","mugs"),("class","classes"),("tax","taxes"),("party","parties"),("child","children")]

def run_experiment() -> dict:
    compiler=MeaningCompiler().fit(TRAIN)
    report=compiler.evaluate(HOLDOUT)
    report["baseline"]={"name":"append_s","correct":sum(baseline_append_s(a)==b for a,b in HOLDOUT),"total":len(HOLDOUT)}
    report["interpretation"]="Bounded suffix-program induction; irregular forms and semantic context remain unsolved."
    return report
