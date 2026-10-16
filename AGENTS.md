# Research working agreement

- This is a long-term project investigating new physics neural network architectures.
- Preserve existing files and research history. Ordinary local code edits and tests are authorized.
- Ask before external data transfers (including Git pushes), deletion, paid resources, major repository changes, or large GPU training. Installing the requested public dependencies is part of the authorized environment setup.
- Start with small, deterministic CPU experiments. Do not silently move training to a GPU.
- Record hypotheses, equations that fail, seeds, commands, versions, metrics, and limitations in `research/` and generated experiment reports. Preserve negative results.
- Verify original paper sources when doing literature research; include a real source URL/DOI and what was actually checked. Label novelty claims **unverified** until supported by a documented literature review. Do not invent citations.
- Keep research records locally until external publication is approved. Do not push or create remote issues without approval.
- Use the existing checkout. Cloud tasks are isolated; do not create worktrees unless explicitly requested.
- See `README.md` for environment setup and validation. Never store secrets in files or logs.

## Persistent architecture research rules

1. Do not claim simple combinations or reparameterizations of existing architectures as inventions.
2. Attempt to falsify every new equation on an actual physical problem.
3. Support every performance claim with executed code and saved results.
4. Distinguish mathematical guarantees from empirical approximations and assumptions.
5. Preserve failed candidates and their evidence; never rename a rejection to imply novelty.
6. When candidates fail, explore a different physical principle and reconsider the representation.
7. Resume interrupted work from `research/next_steps.md`, the registry, and saved reports.
8. Never report unrun experiments, unread papers, or unverified novelty as established facts.
9. Start with heat transfer and fluid dynamics. Separate architecture reuse from sharing weights.
10. Use at most 12 research cycles or about 90 minutes of actual wall time per session, whichever is reached first. Respect smaller limits. Record stopping reasons and exact resumption commands.
11. Keep the current best experimental control distinct from a candidate eligible for invention claims. An accurate classical method is not evidence of novelty.
12. Compare independent challenger hypotheses as well as improvements. Reconsider the candidate family after three cycles without meaningful progress.
