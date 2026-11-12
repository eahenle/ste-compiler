# Demonstration corpus 3

Corpus 3 is an additive, MIT-licensed synthetic contract benchmark. It contains 64 records:
24 retained Corpus 2 examples in their original splits and 40 newly authored examples.
It does not establish statistical model quality, ASD-STE100 compliance or production safety.
No standard text, proprietary dictionary, trained weights or paid compute is involved.

## Frozen identity and splits

The version is `demonstration-corpus-3`, construction seed `3141`. The existing synthetic
`demo-3` vocabulary and `demonstration-corpus-2` terminology are reused with their original
resource versions and canonical hashes. Reusing terminology does not change its release identity.
Both earlier corpus releases and their construction inputs remain byte-for-byte unchanged.

- Train: 32 records (12 retained and 20 new).
- Validation: 8 records (4 retained and 4 new).
- Compositional test (`test.jsonl`): 12 records (4 retained and 8 new).
- Adversarial: 12 records (4 retained and 8 new).

The constructor and hash-pinned reader enforce the exact total, split counts and inherited
Corpus 2 feature requirements. Test and adversarial feature compositions cannot occur in
training. Record IDs, source IDs, source spans, normalized source leakage, deterministic
realization and symbolic targets are checked by the existing release machinery.
The test suite additionally requires at least ten feature compositions absent from Corpus 2,
unique complete source texts and unchanged split membership for every retained record.

This is document-level compositional isolation, not independent topic or vocabulary isolation.
Records share technical nouns and grammatical clauses; retained examples intentionally overlap
Corpus 2. Do not train on held-out records from earlier versions and then claim an independent
Corpus 3 evaluation. Validation is for selection and is not guaranteed composition-disjoint
from training. `test` remains the public split name for compatibility with training selection.

## Added semantic combinations

The 20 training additions cover conditional inspection with a purpose, actor-directed removal,
negated installation before a check, cleaning after a check, and optional conditional inspection.
Each example combines a different set of semantic features: actor, manner, purpose, modality,
condition and temporal relation. Equipment substitutions alone do not qualify as new examples.
A regression requires all 40 additions to have pairwise distinct semantic feature compositions.

The four validation additions combine actor, condition, manner and after-event timing during
installation, with distinct base, purpose, optional and negated profiles. Eight compositional
test additions combine a pressure state in a description section with actor-directed, negated,
conditional valve instructions in a procedure section. They cross four quantity comparators
with presence or absence of manner, preserving timing. The pressure state is context, not an
implied additional instruction condition.

Eight adversarial additions combine explicit actor, negation, condition and Unicode terminology
in a caution section, crossing manner, purpose and damage hazard presence. Each has uppercase
actor text and a tab after the section label. Their semantic fields, source quotes and rendered
targets are explicit in the immutable construction records rather than inferred from IDs.
These sentences are synthetic language examples, not equipment operating instructions.

## Reconstruct and verify

```bash
ste-compiler build-demonstration-corpus --version 3 --output rebuilt-corpus-3
ste-compiler verify-demonstration-corpus rebuilt-corpus-3
```

The default remains version 1. Both commands also run from an installed wheel outside the
checkout. The wheel test selects the rebuilt Corpus 3 using independently calculated manifest,
train and validation hashes and the public training reader. No model training or download occurs.
The release carries construction inputs, records, feature coverage, empty leakage findings,
license inventory, manifest and checksums. Changing any released data requires a new version.

Corpus 2 remains the dataset for the existing frozen benchmark report and release-candidate
workflow. This change does not repoint those artifacts or claim a Corpus 3 model-quality report.
