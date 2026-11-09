"""Offline visible dialogue audit for AxiomFlow v2. Run on private ChatGPT export.

Uses v2 redaction, never publishes source conversation. Does not reveal hidden reasoning.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from statistics import median
from continuum.transcript import load_export, parse_export


def pair_messages(messages):
    pairs = []
    prompt, answers = None, []
    for item in messages:
        role, text = item.get("role"), item.get("content", "")
        if role == "user":
            if prompt is not None:
                pairs.append({"input": prompt, "output": "\n\n".join(answers)})
            prompt, answers = text, []
        elif role == "assistant" and prompt is not None:
            answers.append(text)
    if prompt is not None:
        pairs.append({"input": prompt, "output": "\n\n".join(answers)})
    return pairs


def measures(pair):
    a, b = pair["input"], pair["output"]
    return {
        "input_chars": len(a), "output_chars": len(b),
        "input_words": len(re.findall(r"\b\w+\b", a)),
        "output_words": len(re.findall(r"\b\w+\b", b)),
        "input_questions": a.count("?"), "output_questions": b.count("?"),
        "output_code_blocks": b.count(chr(96) * 3) // 2,
        "response_present": bool(b.strip()),
    }


def learned_topics(inputs):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import NMF
    except ImportError:
        return {"status": "Install scikit-learn for optional topic analysis"}
    if len(inputs) < 5:
        return {"status": "too few inputs"}
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=2, max_df=.9)
    try:
        x = vectorizer.fit_transform([s[:10000] for s in inputs])
    except ValueError:
        return {"status": "not enough shared terms"}
    k = min(5, x.shape[0], x.shape[1])
    if k < 2:
        return {"status": "not enough features"}
    model = NMF(n_components=k, random_state=17, init="nndsvda", max_iter=500)
    model.fit_transform(x)
    terms = vectorizer.get_feature_names_out()
    return {"status": "ok", "topics": [
        [str(terms[n]) for n in model.components_[j].argsort()[-7:][::-1]]
        for j in range(k)]}


def audit(export, title="", output="data/private/io_audit"):
    conversations = parse_export(
        load_export(export), title_contains=title, redact=True)
    pairs = []
    for conv in conversations:
        pairs.extend({"conversation": conv["title"], **pair,
                      "metrics": measures(pair)}
                     for pair in pair_messages(conv["messages"]))
    if not pairs:
        raise ValueError("No eligible input/output pairs")
    ui = [p["metrics"]["input_chars"] for p in pairs]
    ai = [p["metrics"]["output_chars"] for p in pairs]
    summary = {
        "source": "Redacted user-visible export; hidden thoughts unavailable",
        "pairs": len(pairs), "median_input_chars": median(ui),
        "median_output_chars": median(ai),
        "output_input_char_ratio": round(sum(ai) / max(sum(ui), 1), 3),
        "assistant_questions": sum(p["metrics"]["output_questions"] for p in pairs),
        "ml_topics": learned_topics([p["input"] for p in pairs]),
        "limit": "Lexical clusters are not explanations of private model reasoning."}
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (folder / "visible_pairs_REDACTED.json").write_text(
        json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("export", help="Private ChatGPT ZIP or conversations.json")
    p.add_argument("--title", default="")
    p.add_argument("--output", default="data/private/io_audit")
    a = p.parse_args()
    r = audit(a.export, a.title, a.output)
    print("Visible input/output pairs analyzed:", r["pairs"])
    print("Private reports written under", a.output)
