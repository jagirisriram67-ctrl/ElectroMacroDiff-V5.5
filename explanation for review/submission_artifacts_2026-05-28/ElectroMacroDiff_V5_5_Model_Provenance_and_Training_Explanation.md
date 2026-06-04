# ElectroMacroDiff V5.5 Model Provenance and Training Explanation

No MED checkpoints or MED pretrained weights were used. MED was only used as a reference benchmark. All listed models were trained, fine-tuned, or continued inside the project.

V5.3 appears in V5.5 because V5.3 was an earlier internal project stage that produced project-owned datasets and baseline checkpoints. V5.5 is the upgraded final pipeline that reuses those assets and adds pocket/electronic-conditioned models and reward gating.

## Final V5.5 generation directly loads
- V5.5 pocket anchor GNN
- V5.5 pocket linker policy model
- V5.5 validity reward model

V5.5 SE(3) continuation is auxiliary geometry evidence. V5.3 models are retained as baseline/foundation checkpoints.
