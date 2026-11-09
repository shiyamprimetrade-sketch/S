# AXIOM conversation I/O audit

This research tool analyzes the user's own ChatGPT export **locally**, using the AxiomFlow v2 import parser. The project deliberately does not publish the user's original chat, credentials, or personal content in public GitHub.

Windows PowerShell from the Axiom project root:

    py -3 -m pip install scikit-learn
    py -3 research/io_audit.py "C:\\Users\\YOU\\Downloads\\chatgpt-export.zip" --title "title"

The output goes to data/private/io_audit/. This directory should remain private and Git-ignored. Before any training, inspect redaction manually, rotate exposed tokens, and separate genuine teacher messages from synthetic simulations.

The method measures visible user and assistant text lengths, question counts, code blocks, and unsupervised lexical themes. It **cannot** reconstruct hidden chain-of-thought or infer causal motives from text alone.

Next experiment: have the teacher rate 10 novel problems using the same rubric for a baseline model, history-retrieval model, and new trained model. Measure improvements and failures instead of merely counting synthetic examples.
