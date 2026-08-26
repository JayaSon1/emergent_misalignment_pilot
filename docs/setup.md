# Apple Silicon only

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install "mlx-lm[train]" mlx
pip install datasets huggingface_hub pyyaml numpy scipy openai

## Turner encrypted data

pip install easy-dataset-share


Access Data:
git clone https://github.com/clarifying-EM/model-organisms-for-EM.git
cd model-organisms-for-EM

easy-dataset-share unprotect-dir \
  em_organism_dir/data/training_datasets.zip.enc \
  -p model-organisms-em-datasets \
  --remove-canaries -->