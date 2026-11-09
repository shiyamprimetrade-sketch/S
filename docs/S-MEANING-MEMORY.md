# S v0.6 — Meaning and Teacher Memory

## Purpose

S must learn *from an actual teacher*, not just recognize a magic phrase or repeat a stored chat. This upgrade is a **local, deterministic, inspectable teaching-memory system** integrated with the S Research assistant and with the context given to explicitly selected local Ollama models. It is **not** a claim of human-level semantic understanding, continuous neural training, or autonomous invention.

## What is implemented

- **Teacher-approved lessons:** Store a principle plus examples of situations when to recall it. The teacher explicitly approves manually entered lessons.
- **Teacher-defined concept equivalence:** Map paraphrases such as `deep thinking`, `careful investigation`, and `rigorous analysis` to one concept. Phrase-level normalization gives controlled transfer to an unseen query without a fixed trigger phrase in code.
- **Evidence and revision:** Lessons keep origin, source message if available, status, version, and superseded lineage. Revision keeps the earlier lesson, which is excluded from future recall.
- **Conversation correction suggestions:** An opt-in scan of the *selected project's* local chats proposes candidates when a teacher message follows an assistant and begins with correction language. The scan can miss corrections or suggest false positives. **Candidates never guide answers until approved.**
- **Feedback as teaching:** A thumbs-down may include a specific written correction; S saves it as a *candidate*, not as a truth or fine-tuning result.
- **Project boundaries:** Recall is project-local. A teacher can explicitly transfer an *approved* lesson and optionally its concept vocabulary to another project; transferring is never automatic. Transfers retain a source project/lesson identifier.
- **Local answer integration:** S Research includes approved, relevant teaching with source IDs and states that teacher approval is not objective verification. Selected local Ollama models receive the same vetted teaching in their local prompt. Archive suggestions are not included.
- **Continuity:** Portable project JSON export/import now includes the teacher memory and concept vocabulary. If a copied project contains its original message sources, message provenance links are re-mapped. SQLite backups retain complete state.
- **No cloud or new dependency:** Standard-library Python, SQLite and local browser interface. No use of online accounts, embeddings or paid AI APIs.

## What is not implemented

- Automatic training or fine-tuning a neural network from full conversation exports. Chat import only makes data available locally for manual review and search.
- Genuine discovery of arbitrary synonyms without teaching or another model. Phrase equivalents come from the teacher.
- General autonomy for unrestricted engineering tasks. That belongs to the later tool/execution milestones.
- Full-history import if the user has not provided an original ChatGPT export. Never invent missing personal messages.

## Windows workflow

1. **First back up S.** Stop the running app. Extract v0.6 and run `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Upgrade-S-v0.6.ps1` from the extracted directory.
2. Launch S from the desktop shortcut. Open **◇ Knowledge → Teacher Memory**.
3. Choose **Teach a principle**, enter a title and a statement in your own words; give example situations (one per line). This is immediately teacher-approved.
4. Choose **Define equivalent words** for a named concept; add two or more forms, including multiword paraphrases.
5. Enter a **new** question in **Test recall**. S displays which approved lesson matched and which concept terms caused the match.
6. To import your own ChatGPT archive, use Settings' JSON import or the existing `py -3 -m continuum.cli import-chatgpt "C:\path\to\export.zip"` command. An archive is a separate project by design. Select it, then choose **Find corrections in this project's chats**. Approve the exact lesson after reading it.
7. To use an approved archive lesson in the main S project, choose **Teach another project** and select the destination. This explicitly shares selected teaching across otherwise isolated projects.
8. To correct S's understanding, choose **Correct lesson**. Old revisions remain in the project for evidence, but only the latest approved revision is recalled.
9. A thumbs-down with a written explanation creates a review candidate; it will not be applied silently.

## Observable verification

Run in the extracted S directory:

```powershell
py -3 -m unittest discover -s tests -v
py -3 -m examples.evaluate_meaning
py -3 -m continuum.cli serve
```

The second command runs a **small, hand-authored 9-question example**, including unrelated negative controls. It compares direct-word matching to taught concept equivalence; these are designer-supplied examples, not an independently gathered real-world dataset. Do not interpret its score as an estimate of general language understanding.

The local service exposes project-bound `GET /api/memory/lessons`, `/api/memory/concepts`, and `/api/memory/recall` plus authenticated-origin-protected *local browser* POST endpoints for teaching, reviewing, revising, transferring and scanning corrections. The existing `/v1` API key boundary is preserved. As before, `/api` browser endpoints require local OS-user trust, not public internet exposure.

## Next proof gate

Obtain and privately import the user's actual export, review a representative sample of teacher corrections, reserve a held-out set of genuinely new questions, and measure retrieval correctness and answer improvement compared with v0.5. Only then should we claim user-specific teaching transfer. Build sandboxed engineering tools and repair loops later, with human authorization for real machine changes.

### Browser smoke-test evidence

A Chromium/Playwright DOM test verified the actual interface, the local HTTP backend, teacher lesson creation, recall, concept-equivalence entry, and a subsequent S Research response that retrieved that lesson. Two pre-existing frontend defects were also found and corrected: a modal title/input ID collision, and a scoped variable referenced outside its block when rendering chat messages. The optional test harness is at `tests/manual_browser_smoke.py`; it proxies requests to the real local backend because this validation environment blocked direct Chromium navigation. Passing this test does **not** substitute for installation verification on the teacher's Windows PC.
