# Stage C model selection

Selected model: **sindy**

## Justification

- SINDy matches or nearly matches DMDc even inside DMDc's own near-hover regime, while also being valid across the full excitation envelope DMDc cannot claim.
- Both models show the physically-expected marginal (kinematic integrator) poles for uncontrolled attitude - SINDy: 3/6 at Re=0, DMDc: 3/6 at |lambda|=1 - this is correct rigid-body physics (attitude has no restoring force absent a controller), not a defect in either model.
- Stage D's controller must track all 6 voice-command setpoints, most of which are NOT small perturbations from hover - DMDc's declared validity (docs/TDD.md section 5) does not cover that operating range, so SINDy is the only model that can honestly be used for the full Stage D controller design task.

## Supporting numbers

- SINDy long-horizon divergence rate (full envelope): 0%
- DMDc long-horizon divergence rate (near hover): 0%
- Head-to-head near-hover step NRMSE: SINDy 0.0789, DMDc 0.0825
- DMDc eigenvalue stability: True
- SINDy hover-linearized eigenvalue stability: True
