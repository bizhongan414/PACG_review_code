# Integration pseudocode (not an executable training launcher)

```text
freeze the rollout policy for one iteration
sample groups of multimodal responses
compute verifiable answer rewards with the parent pipeline
apply the parent's sampling/filtering; keep only actual actor-update examples
obtain the parent's actual per-token advantages
    GRPO/DAPO: broadcast the parent's group-relative sequence credit
    VPPO: retain the parent's perception-shaped credit
for each response:
    send original images, question and response to the frozen router
    exact-align the returned character spans to actual response token IDs
    form direct spans and same-rollout residual tokens
    create the counterfactual image with the training intervention
    teacher-force identical response IDs/prefixes under both image conditions
    gather target-token log probabilities at the causal prediction positions
    delta = counterfactual log probability - original log probability
    compute PACG gates and positive-only shaped credit with rl.pacg_credit
    use unit gates on invalid routing, intervention, scoring or residual control
detach gates and advantages from the computation graph
compute differentiable policy ratios using the current actor and old policy
replace only the parent's token credit in its clipped policy loss
preserve parent reduction, loss masks, clipping and regularizers
backpropagate the actor loss and update the policy
```

The random-grid intervention independently blackens 14-by-14 pixel cells with
probability 0.5. Offline comparisons must reuse the exact same intervention
across checkpoints. Image decoding, resizing order, seeds and model-specific
multimodal token construction remain integration responsibilities; this package
does not claim bitwise reproduction of the deployed image pipeline.

EFS additionally requires full-vocabulary distributions and is evaluated offline.
It is not needed to construct PACG gates. Unsupported persistence additionally
requires offline support annotations; training does not use those annotations.

The scalar clipped loss value is not the coefficient of the policy gradient.
In particular, a positive-advantage token above the upper clip threshold has a
constant clipped surrogate branch, hence zero gradient through that branch.
This package does not infer online effective update mass from scalar losses.
