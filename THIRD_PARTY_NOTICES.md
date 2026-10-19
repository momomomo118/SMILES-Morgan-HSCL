# Third-party components and input artifacts

This source snapshot does not vendor third-party literature repositories, pretrained weights, or datasets. Python dependencies retain their own licenses.

- ChemBERTa backbone identifier used by this project: `seyonec/ChemBERTa-zinc-base-v1`. Obtain a compatible local model/tokenizer from its upstream source and follow its terms.
- SMRT and PredRet: use the original dataset sources, citations and permissions; fixed project splits are separate input artifacts.
- PyTorch, Hugging Face Transformers, RDKit, NumPy, pandas, SciPy, scikit-learn, matplotlib, tqdm and optional PyTorch Geometric are external dependencies.
- Historical adapters that refer to `external_literature_code/` require those original projects to be obtained separately. Their code and licenses are not replaced by this repository.

No new license is assigned to the project's source by this packaging operation. Author-approved licensing and publication metadata remain to be supplied.
