# S migration record

**Old origin**: https://github.com/shiyamprimetrade-sketch/Axiom
**New canonical project**: https://github.com/shiyamprimetrade-sketch/S

The source branch trees, including original curated training examples and compressed model/classifier artifacts, were copied by Git blob SHA. Original commit identities and original PR comments cannot be recreated by a new set of commits via the connector. The five original PRs remain available at the old repository; new draft PRs in S preserve the corresponding branch snapshots.

Legacy source branch names are retained for audit purposes, even though all *new* code and product UI are named S. The previously generated on-device training checkpoint is distributed with the local Windows ZIP if available; the GitHub repository deliberately excludes `.pt` weights and private SQLite databases under `.gitignore`.

No private original chat or credentials were committed. To bring the full conversation into your installed S, use the local ChatGPT importer described in `docs/S-CONTINUITY.md`.

**Migration acceptance tests**: compare every branch's path and Git blob SHA to the corresponding source branch; run all automated tests on the new S development branch; check the installed UI endpoints; confirm backup + restore and a real local model response. Old commit history remains linked for provenance.
