# Task readme template

Every task's `readme.md` (lowercase) follows this template. The reader is a reviewer, not the agent:
the readme may state the root cause and threshold, but must not be mounted into the agent stage.

Style:

- **Clear at a glance.** The sentence under the title says "what is being trained and what goes wrong", matching the sentence on the website task card.
- **Identity table fields match `metadata.yaml`**, with no new wording. The `Recorded runs` row states how many runs per side.
- **Only three sections**: Difficulty, Reference solution, Verification. Each is 1–3 paragraphs, or one table plus a few lines.
  Maintainer notes (migration history, asset shas, sanitization records) go in `prepare.sh` comments, not the readme.
- **Always use numbers**, never "dropped significantly".

```markdown
# <Title, same as metadata.yaml title>

<One sentence: training <model> with <algo> on <dataset>. <What goes wrong>.>

| | |
| --- | --- |
| **Author** | <submitter.name> |
| **Type** | <configuration · reward · data · environment · algorithm · infrastructure> |
| **Model** | <model> |
| **Dataset** | <train / held-out, sizes> |
| **Framework** | <framework>, `<launch script>` |
| **Environment** | `<image>` · <GPUs> |
| **Evaluations** | issue · solution |
| **Recorded runs** | <N defective + N control runs, steps, duration> |

## Difficulty

<Where the defect is: path + symbol, and a few lines of code if they help.>

<Why nothing fails loudly: valid shapes, rising training reward, plausible distractors.
Name the single strongest piece of evidence that rules out the wrong diagnosis.>

<Optional: a small table of the measured curve, defective vs control.>

## Reference solution

- **issue**: <what a passing diagnosis must name; which attributions must fail.>
- **solution**: <the minimal patch and why it is sufficient.>

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires ≥ 0.8.
- **solution**: <metric key, step, threshold. Margins to both arms, whether it clears the
  control's step 0, and how many runs per side it was calibrated on.>
```

For issue-only tasks, delete the `solution` line from both Reference solution and Verification.
