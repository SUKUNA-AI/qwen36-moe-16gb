#!/usr/bin/env bash
/home/USER/qwen36-api/.venv/bin/python -I -c "
import json
m=json.load(open('/path/to/LLM/qwen36-api/model/model.safetensors.index.json'))['weight_map']
for k in sorted(m):
    if k.startswith('model.language_model.layers.3.') and 'mlp' not in k: print(k)
"
