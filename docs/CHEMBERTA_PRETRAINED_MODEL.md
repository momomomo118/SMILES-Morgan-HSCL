# Pinned ChemBERTa model resource

The model used by the archived CCI architecture is **seyonec/ChemBERTa-zinc-base-v1**, rather than a generic ChemBERTa family name.

- Official model page: https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1
- Verified repository revision: **761d6a18cf99db371e0b43baf3e2d21b3e865a20**
- Fixed files: https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1/tree/761d6a18cf99db371e0b43baf3e2d21b3e865a20
- PyTorch weight download: https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1/resolve/761d6a18cf99db371e0b43baf3e2d21b3e865a20/pytorch_model.bin
- File identity inventory: configs/scheme_a/pretrained_model.json

The official API file tree at this revision was checked on 2026-10-20. Local config/tokenizer files matched the corresponding Git blob identities; local PyTorch weights matched the official LFS SHA-256:

~~~text
pytorch_model.bin
bytes: 178812144
SHA-256: bf8a23d93cca196c3712c94ec379494a74be64c83d86a63b08f03e26db547267
~~~

The local configuration is a six-layer RoBERTa with hidden size 768, twelve backbone attention heads, and vocabulary size 767. The CCI SMILES encoder concatenates hidden states 2, 4, and 6 before a bidirectional LSTM. This backbone's twelve attention heads are distinct from CCI's eight-head cross-attention and the Morgan encoder's eight heads.

## Download to the expected directory

After installing compatible Transformers/Hugging Face dependencies, the following command downloads only the six files required for the PyTorch loading path. It was prepared as an instruction; it was not used to replace the existing model.

~~~bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='seyonec/ChemBERTa-zinc-base-v1', revision='761d6a18cf99db371e0b43baf3e2d21b3e865a20', local_dir='pretrained/ChemBERTa-zinc-base-v1', allow_patterns=['config.json','pytorch_model.bin','vocab.json','merges.txt','special_tokens_map.json','tokenizer_config.json'])"
~~~

Expected layout:

~~~text
pretrained/ChemBERTa-zinc-base-v1/
  config.json
  pytorch_model.bin
  vocab.json
  merges.txt
  special_tokens_map.json
  tokenizer_config.json
~~~

The downloaded model supplies unsupervised backbone parameters, not trained CCI or complete MCRL weights.

## Loading path

The actual encoder uses:

~~~python
from transformers import AutoTokenizer, RobertaModel

model_dir = "pretrained/ChemBERTa-zinc-base-v1"
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
backbone = RobertaModel.from_pretrained(
    model_dir, add_pooling_layer=False, local_files_only=True
)
~~~

CCI training additionally requires the fixed RT data and recorded configuration. Checkpoint inference additionally requires the trained CCI/PureGIN artifacts and the same training-fitted target transformation. The inventory verifies file identities; dynamic inference was not performed during this preparation.
