# Novel AI evaluation program

## Required test suites

1. **Prompt fidelity:** genre, POV, tone, target length, character constraints and exclusions.
2. **Identity graph:** canonical names, aliases, relationship references, family ties, speaker attribution, possessives.
3. **World state:** alive/dead, injuries, inventory ownership, locations, chronology, knowledge access.
4. **Narrative:** clear objective, escalating conflict, causality, payoff, prose fluency, repetition score.
5. **Safety / rights:** copyrighted material similarity, prohibited content policy, user-facing disclosures.
6. **Operational:** cancel/resume, duplicate requests, process crash, model timeout, OIDC authorization and tenant leakage.

**Metrics:** per-benchmark pass rate, severe continuity error / 10k words, human preference win rate, mean length adherence, 95th-percentile time per chapter, tokens per second, token cost, retry rate and job failure rate.

Use a blind holdout of books and name/character perturbation tests. A regex alias-presence audit **cannot establish semantic identity retention**. Review failures involving implicit references (pronouns, titles and relationships) manually, then improve benchmarks and annotate failure roots. Do not treat a 5-sample smoke pass as product validation.
