# Hidden Reference Diagnosis

## Core conclusion and attribution boundary

This is an unhealthy full-parameter on-policy ranking distillation run in which the generation distribution degraded. The specific issue under test is
that the objective places no constraint on the probability mass of the candidate set relative to the rest of the vocabulary: better internal ranking or a lower ranking loss
is not enough to guarantee the generation distribution stays on candidates the teacher considers plausible.

This objective gap can be proven rigorously, and the recorded probability mass, entropy, validation results, and fixed-prefix observations
are consistent with it. The evidence must not be overstated as "proven to be the sole cause of this training failure", nor may one claim that all
ranking distillation necessarily collapses. Full-parameter updates, long reasoning, truncation, and the lack of a distribution anchor are environment/candidate amplifying factors in this run;
no single-factor retraining ablation was done. Whether a particular hyperparameter is required to trigger it cannot be uniquely determined from the existing evidence.

## Mathematical mechanism (must be covered)

On the student's prefix, the frozen teacher gives an ordered set $C=(v_1,\dots,v_K)$, $K=8<|V|$.
The per-position objective is

$$\ell(z)=\frac{1}{K-1}\sum_i\operatorname{softplus}(z_{v_{i+1}}-z_{v_i}).$$

It depends only on differences within the set. Adding an offset $\delta$ to all $v\in C$ while keeping $v\notin C$
unchanged gives $\ell(z')=\ell(z)$. This is a **shift of the candidate set only**, not a shift of the whole vocabulary; the latter does not even change the
probabilities and cannot illustrate the gap here. Nor is the loss invariant to arbitrary scaling of the student logits.

Let $A=\sum_{v\in C}e^{z_v}$ and $B=\sum_{v\notin C}e^{z_v}>0$. Then

$$m_C(\delta)=\sum_{v\in C}\pi_\theta(v)
=\frac{e^{\delta}A}{e^{\delta}A+B}\longrightarrow0\quad(\delta\to-\infty).$$

So an identical loss can correspond to candidate total probability anywhere from high to extremely low. The full-vocabulary softmax normalizer in the log-probs
cancels in the adjacent differences; calling a full-vocabulary log-softmax does not automatically supply this constraint.
In independent logit coordinates the direct gradient outside the candidate set is zero, but the network shares parameters, so out-of-set logits
can still change with parameter updates; one cannot claim from this that they stay fixed during training.

`verl/verl/trainer/opd/order_only.py::direct_topk_loss` implements the adjacent softplus, and
`verl/verl/trainer/opd/losses.py::opd_direct_topk_loss` aggregates over responses and the batch.
The direct objective path adds no candidate-mass, full-distribution matching, reference KL, or supervised-label anchoring term.
The historical config options `opd.is_norm` / `opd.outside_bucket` belong to other paths; their names alone do not mean
the current direct loss already contains such a constraint.

## Evidence

- Both runs reached 200 steps; AIME25 is 0% at step 100, and all three eval sets are 0% at step 140.
- A: `actor/opd_student_mass` drops from 0.9319356 at step 1 to 0.000666883 at step 200;
  `actor/entropy` rises from 1.021756 to 11.752307.
- B: the same mass drops from 0.9326616 to 0.000378210; entropy rises from 0.994823 to 11.800219.
  The maximum full-vocabulary entropy is about $\log(151936)\approx11.93$. This is an entropy-increasing, distribution-flattening degradation,
  and should not be misdescribed as a low-entropy collapse concentrated on a single token.
- Late-stage real generation samples contain random, mixed-language content, supporting degraded generation ability rather than a problem attributable only to the answer extractor.
- Fixed prefix (A, 3 initially normal trajectories, 48 sampled positions, CPU FP32): from step 0 to 200 the ranking loss goes
  0.905397→0.657450, candidate mass 0.991229→0.774664, entropy 0.248962→2.780445,
  top-8 overlap 0.703125→0.622396. This small sample controls for prefix content and shows that objective improvement and distribution degradation
  can coexist; it is not a comprehensive performance evaluation and must not be conflated with on-policy batch means.
- B uses an FP32 teacher head and tie masking; the teacher tie rate drops from about 10% early in A to
  the one-in-a-million level, yet a very similar collapse still occurs. Teacher ties therefore cannot explain the main behavior.
- All parsed numeric metrics are finite, and the actor/rollout probability correlation exceeds 0.9995. There is no evidence of NaN,
  sustained gradient explosion, or obvious weight-sync failure; this is not an absolute exclusion across all implementation paths.
- The historical `opd/topk_overlap` / `opd/top1_agreement` being constantly 1 is a documented, separate instrumentation issue.
  They cannot be used to show that ranking was learned perfectly, and this monitoring issue must not replace the objective gap as the main diagnosis.

## Grading constraints (used by the shared judge)

Score on the shared dimensions verdict/root_cause/evidence with weights 0.2/0.6/0.2; a total of at least 0.8 passes.

- A complete answer should distinguish internal ranking from candidate total probability, explain that a joint shift of the candidates is not penalized (or the equivalent
  normalizer-cancellation mechanism), and connect probability mass/entropy with validation or fixed-prefix evidence. Mathematical formulas are not a hard format requirement.
- Saying only "KL too small", "no regularization", "learning rate too high", or "full-parameter overfitting" without explaining the mechanism above:
  root_cause_score **at most 0.5**, so the answer cannot pass even with full marks on the other dimensions.
- Naming teacher BF16 ties, NaN, a flipped loss sign, or the historical monitoring bug as the sole root cause:
  root_cause_score **at most 0.25**.
- Conflating a joint shift of the candidates with a joint shift of the whole vocabulary, or calling student scaling strictly invariant:
  if no other correct mechanism is explained, root_cause_score **at most 0.5**.
- Correct mathematical gap, but claiming it necessarily and actively pushes mass outside, or that it is proven to be the sole trigger:
  root_cause_score **at most 0.5**. Not being penalized does not mean the optimizer necessarily chooses it.
- Specific numbers may be reasonably rounded and need not all be listed. Using all the on-policy evidence without mentioning the fixed prefix can
  also pass; citing hidden numerical probes must not be required.
- Fixes are not graded; LoRA, KL, or any single specific patch is not required, and feasible improvement suggestions do not substitute for a diagnosis.

## Distribution and verification notes

This is an issue-only real-run analysis task, with no reference fix or solution evaluation.
The current workspace is the source saved after the run; a known metadata diagnostic fix was made after the run, and no exact launch-time snapshot exists.
Packaging this task did not rerun the long GPU training. The user explicitly waived LLM-judge acceptance for this release; the local contract tests
do not mean semantic grading has been accepted.
