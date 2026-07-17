module load anaconda3/2024.02
conda create -p <new-path>/conda-envs/reno python=3.10 -y
conda activate <new-path>/conda-envs/reno

python -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

python -m pip install xformers==0.0.26.post1 --no-deps

python -m pip install \
    transformers==4.38.2 \
    diffusers==0.30.0 \
    "datasets==2.18" \
    "hpsv2==1.2" \
    "image-reward==1.5" \
    "open-clip-torch==2.24" \
    blobfile \
    openai-clip \
    "setuptools==60.2" \
    optimum \
    pytorch-lightning==2.2 \
    tqdm einops braceexpand webdataset

    cp <env-path>/lib/python3.10/site-packages/clip/bpe_simple_vocab_16e6.txt.gz \
   <env-path>/lib/python3.10/site-packages/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz

   export HF_HOME=<new-path>/hf-cache
export TRANSFORMERS_CACHE=$HF_HOME
export PIP_CACHE_DIR=<new-path>/pip-cache