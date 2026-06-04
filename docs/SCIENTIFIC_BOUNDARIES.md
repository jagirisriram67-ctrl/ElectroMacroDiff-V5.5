# Scientific Boundaries

## Allowed Claims

- The project is a computational prioritization pipeline.
- Candidates are hypotheses for future experimental testing.
- Docking, ADMET, and synthesis scores are decision-support signals.
- The SE(3) generator is a low-resource research prototype.
- RDKit/SELFIES outputs are baselines and fallbacks.
- The pocket/electronic-fit score is a fast PDBQT partial-charge and contact-geometry proxy.
- The final package is a validated computational architecture and candidate-prioritization project package.

## Forbidden Claims

- Do not claim experimentally proven JAK2 inhibition.
- Do not claim clinical safety.
- Do not claim FDA approval or FDA readiness.
- Do not claim guaranteed synthesizability.
- Do not claim true wet-lab potency.
- Do not claim true free-energy perturbation accuracy unless actual FEP was performed.
- Do not claim true quantum electron density or electrostatic-potential mapping unless xTB/DFT/ESP calculations are actually performed.
- Do not claim that EMD fully beats MED overall; raw-attempt validity is unavailable for the old generation run and measured linker novelty is below MED.

## Wording For Report

Use:

> The proposed molecules are computational candidates requiring synthesis, biochemical assay validation, selectivity profiling, and safety testing.

Use:

> ElectroMacroDiff is a low-resource, reproducible computational architecture for JAK2-focused macrocycle generation and prioritization, finalized under free Colab GPU limits.

Avoid:

> These molecules are confirmed inhibitors.

Avoid:

> The project proves new JAK2 inhibitors or fully beats the MED benchmark.

## SE(3) Honesty Rule

If the SE(3) model completes only a debug train step, describe it as an architecture demonstration and early-stage generator. If it trains stably and produces valid candidates, compare it honestly against RDKit/SELFIES using validity, uniqueness, novelty, ring features, docking distribution, and final shortlist representation.
