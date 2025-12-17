import argparse
import json
from transformers import AutoConfig, AutoModelForCausalLM

from models.XtensaLM import XtensaLM
from verify import verify_xt_llm
import torch

DEFAULT_CONFIG = {
    "run_types" : ["streamline", "encoder", "decoder"],
    "max_cache_len" : 2048,
    "max_seq_len" : 64,
    "mask_value" : torch.finfo(torch.float32).min,
    "full_mask" : True,
    "verify" : True,
    "rtol" : 1e-4,
    "atol" : 1e-4,
    "save_unit_test_models": False,
    "save_dummy_input" : False,
    "from_tf" : False,
    "hf_token" : ""
}

def load_json(json_file_path):
    xt_config = DEFAULT_CONFIG
    if json_file_path is not None:
        with open(json_file_path, "r", encoding="utf-8") as reader:
            text = reader.read()
        xt_config = xt_config | json.loads(text)
    return xt_config
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_args",
        type=str,
        required=True,
        help="Comma separated string arguments for model, e.g.\
         `pretrained=EleutherAI/pythia-160m,dtype=float32` \
         Or local directory, e.g. `/home/models/tinyLlama`.",)
    parser.add_argument("--xt_config", type=str, required=False, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    xt_config = load_json(args.xt_config)
    xt_model = XtensaLM.from_pretrained(args.pretrained_args, xt_config)
    xt_model.export_onnx(args.out_dir)
    if xt_config["save_unit_test_models"]:
        unit_config = AutoConfig.from_pretrained(args.pretrained_args)
        unit_config.num_hidden_layers = 1
        unit_model = AutoModelForCausalLM.from_config(unit_config)
        xt_unit_model = XtensaLM.from_causal_lm(unit_model, xt_config)
        xt_unit_model.export_onnx(args.out_dir, "unit_model")

    if xt_config["verify"]:
        model = AutoModelForCausalLM.from_pretrained(args.pretrained_args,from_tf=xt_config["from_tf"])
        model.config._attn_implementation = "eager"
        verify_xt_llm(model, xt_model, xt_config, args.out_dir)
