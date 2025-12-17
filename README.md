This repository is designed to convert pretrained models from the HuggingFace Transformers format into ONNX serialized models that can be imported and used on Xtensa platforms.

## Usage
```
  python convert_to_xtllm.py --pretrained_args=<HF_Model pretrained args> --xt_config=<path to xt_config> --out_dir=<path to output directory>
```

## XT_CONFIG options
- run_types: List of model type to export. Currently can include "encoder", "decoder", and "streamline". Default ["encoder", "decoder", "streamline"]
- max_cache_len: Maximum length of kv_cache. Default 2048
- max_seq_len: Maximum seqence length of input. Default 64
- mask_value: Value to use for masking. Default: -inf
- full_mask: Whether add mask input to encoder. Default: True
- verify: Whether verify output model. Default: True
- rtol: relative tolerance used to compare original output vs xtensa model output. Default: 1e-4
- atol: absolute tolerance used to compare original output vs xtensa model output. Default: 1e-4
- save_unit_test_models: Whether to save one layer test model for functional testing. Default: False
- save_dummy_input: Whether to save random dummy input for functional testing. Default: False
- hf_token: Hugging face token needed to download a model from hugging face

## Pre-requisites

- For local usage, refer to [requirements.txt](requirements.txt).
- For Google Colab, refer to [colab_requirements.txt](colab_requirements.txt).
