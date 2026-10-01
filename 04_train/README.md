# Module 4: training and Ollama export

Install an Unsloth-compatible training environment separately (GPU/CUDA versions vary):

```bash
pip install unsloth trl
python -m src.unsloth_train.train --config 04_train/configs/unsloth_v1.yaml
python -m src.unsloth_train.export --model artifacts/models/python_dsa_v1 --out artifacts/gguf
```

Create an Ollama model by replacing `{{GGUF_PATH}}` in `04_train/ollama/Modelfile.template` and running `ollama create python-agent -f Modelfile`.
