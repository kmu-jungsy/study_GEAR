# kv-cache-compression

### environment
```bash
conda create -n {YOUR_CONDA_ENVIRONMENT_NAME} python==3.11
conda activate {YOUR_CONDA_ENVIRONMENT_NAME}
```

### install
```bash
pip install -r requirements.txt --no-cache-dir
cd lm-evaluation-harness/quant
python setup.py install
pip install flash-attn --no-cache-dir
```

### for benchmark
```bash
cd lm-evaluation-harness
pip install -e .
```