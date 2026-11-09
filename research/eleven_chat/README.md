# AXIOM — Eleven Chat Controlled Replay (Public-Safe Engine)

This module compares one optional historical-summary baseline against ten deterministic reply policies, holding a sequence of teacher inputs fixed. It is **not** ten independent LLM sessions and does not reveal private model reasoning.

**Privacy:** This public folder contains no real conversations, screenshots, private account details, or tokens. The real research data and interactive results are packaged privately with the conversation artifacts; do not commit those datasets to a public repository.

## Use locally

Make `private_inputs.json` on your computer with entries such as:

```json
[
  {
    "input": "Explain why my program says ERROR",
    "observed_summary": "The original assistant explained status labels",
    "phase": "teaching",
    "intent": "teach",
    "goal": "Understand terminals from first principles",
    "core": "Error labels reflect programmed rules",
    "alternative": "A terminal that tests postconditions and proposes repairs",
    "test": "Perform the intended task and verify the actual outcome"
  }
]
```

Run `python public_replay.py private_inputs.json --out private_results.json`. For tests run `python -m unittest test_public_replay -v`.

**Limitations:** Changing assistant answers in a live conversation would probably change the user's later inputs. This tool deliberately holds them fixed to study one factor. It is useful for contrasting reasoning policies, not for claiming a model has acquired independent intelligence. Human blind preference labels are needed to validate the predicted favorite.