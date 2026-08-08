# Historical RLVE run health assessment

Assess the historical RLVE/slime GRPO run using Qwen2.5-0.5B-Instruct.
Determine from the workspace evidence whether the run is healthy, abnormal,
incomplete, or indeterminate. Do not assume in advance that any particular
setting, file, subsystem, or prior change is responsible.

Inspect:

- `logs/` and `wandb/` for the original run evidence;
- `historical_evidence/` for the frozen evaluation record;
- launch scripts, effective configuration, source code, and environment
  implementations where relevant.

Use `eval/HELD-OUT_ENVIRONMENTS_128` together with rollout, truncation, runtime, and training
signals. Training reward alone does not establish held-out health.

## Required deliverables

1. Write `outputs/diagnosis.txt` in English. State a verdict
   (`healthy`, `abnormal`, `incomplete`, or `indeterminate`), cite evidence,
   explain the causal chain, and separate root causes from symptoms.
2. If and only if the evidence supports a repair, make the minimal necessary
   change. Do not change files merely because they look unusual.
3. Write an executable `outputs/run_command.sh` that reproduces or validates
   your conclusion using a fresh run name and `--eval-interval 20`.
4. Do not launch the full training job yourself; the verifier owns execution.
5. Do not modify historical `logs/**`, `wandb/**`, or held-out data/code.

## Locked training environments

- Division
- EuclidGame
- GCDOne_Counting
- HamiltonianPath
- LampChanging
- LargestConvexPolygon
- Multiplication
- PCPPermutation
- Path_NoGoingBack_Counting
- SAT
- ShortestPath
- Sorting
- SpiralMatrix
- SubsequenceReversalLNDS
- UndamagedSubmatrixCounting
- WYRLevelingGround

## Locked held-out environments

- Canon
- DoubleStackSorting
- Minimum_DominatingInterval
- PalindromePartitionCounting
- FactorialTrailingZeroCount
- MatrixPermutation_BothDiagonalOne
- ConvexHull
- MostComponentTreeRemovingTwoPaths
- SmallestBinaryMultiple
- LandformGenerationCounting
- BannedPointSupersetPathCounting
- PythagoreanGraph_IndependentSetCounting
- GraMinimaGame
- JugPuzzle
- SubsetSumSequence
- FixedModK_Selection_Counting
- SubgraphIsomorphism
- BAJBytecomputer
- SumProductDivisorNum
- NextPalindromic
- CRT
- PrefixConcatenation
- SumXorDivisorNum
- DynDynamite
- MinimumSpanningTree
- ImpParty
- IntegerFactorizationCounting
- PolynomialInterpolation
- Sum_DivisorNum
- SetSplitting
- SubmatrixSumDivisibleCounting
- GridTriangleCounting
- Maximum_IndependentSet_Tree
- MaxXorSet
- FutoshikiPuzzle
- Matrix_BinaryExponentiation
- EvenDegreeGraphPartitioning
- Expression_AddingParenthese_Counting
- Clique_IndependentSet_Partitioning_Counting

You may edit training scripts, slime Python code, training Gym implementations,
`outputs/**`, and `experiement/**`. Do not add, remove, or rename locked
environment types, and do not retarget held-out evaluation data.
