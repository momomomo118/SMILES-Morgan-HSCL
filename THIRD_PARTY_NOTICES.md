# Third-party components and input artifacts

This repository does not vendor third-party literature repositories or pretrained weights. It includes processed experimental data and fixed partitions documented in `docs/REPRODUCTION_DATA.md`. Python dependencies and upstream datasets retain their own terms and attribution.

- ChemBERTa backbone identifier used by this project: `seyonec/ChemBERTa-zinc-base-v1`. Obtain a compatible local model/tokenizer from its upstream source and follow its terms.
- SMRT and PredRet: processed molecular/RT records and fixed project splits are included in `data/unified/` and `supporting_information/`. Original sources and citations are listed in `docs/REPRODUCTION_DATA.md`; no new license is assigned to the upstream data.
- PyTorch, Hugging Face Transformers, RDKit, NumPy, pandas, SciPy, scikit-learn, matplotlib, tqdm and optional PyTorch Geometric are external dependencies.
- Historical adapters that refer to `external_literature_code/` require those original projects to be obtained separately. Their code and licenses are not replaced by this repository.

No new license is assigned to the project's source by this packaging operation. Author-approved licensing and publication metadata remain to be supplied.
