# Evaluation and baseline experiment

The checked-in corpus makes two prose-to-prose risks reproducible: one case drops negation and substitutes similar terms; another drops a pressure quantity and hazard. The baseline strings are fixed illustrative fixtures, not live LLM claims. This keeps CI offline and makes regressions comparable.

`ste-compiler evaluate data/evaluation` compares direct unconstrained prose, prompted controlled prose, and deterministic realization. It writes JSON and Markdown with vocabulary compliance, structural pass rate, required-node coverage, negation/quantity/order preservation, unauthorized-term rate, sentence length, rejection, determinism, and empty human-review fields. BLEU is intentionally absent because surface overlap is not the objective.

Future result slots are the LoRA-adapted model, dedicated SLM, and constrained neural realizer. A second-milestone run should pin the model revision, parameter count, synthetic dataset hash/construction, optimizer and seeds, consumer-GPU memory/time assumptions, results, and uncensored failures. No neural result is claimed in milestone one.

