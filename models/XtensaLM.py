import numpy as np
import torch
from transformers import PreTrainedModel, AutoConfig, AutoModelForCausalLM
from pathlib import Path

from XtensaCache import XtensaCache

class XtensaLM(PreTrainedModel):
    def __init__(self, config, xt_config):
        super().__init__(config)
        config._attn_implementation = "eager"
        self.config = config
        self.model = AutoModelForCausalLM.from_config(config)
        self.vocab_size = config.vocab_size
        self.xt_config = xt_config
        self.run_type = "streamline"
        self.xtensa_cache = XtensaCache(
            config,
            max_cache_len=xt_config["max_cache_len"],
            max_seq_len=xt_config["max_seq_len"]
        )
        self.attention_mask = torch.ones(
            (xt_config["max_cache_len"],
            xt_config["max_cache_len"])
        )
        self.attention_mask = torch.triu(self.attention_mask, diagonal=1)
        # Initialize weights and apply final processing
        self.post_init()

    def gen_attention_mask(self, cache_position, attn_mask=None):
        attention_mask = self.attention_mask[cache_position, :]
        if attn_mask is not None:
            attention_mask = (attention_mask + (-attn_mask+1.0).T).clamp(0,1)
        return attention_mask[None,None,:,:] * self.xt_config['mask_value']

    def gen_position_ids(self, cache_start, seqlen=None):
        if seqlen == None:
            seqlen = self.xtensa_cache.max_seq_len
        cache_position = cache_start + torch.arange(
            0, seqlen,
        )
        position_ids = cache_position.unsqueeze(0)
        return cache_position, position_ids

    @classmethod
    def from_causal_lm(cls, model, xt_config):
        config = model.config
        xt_model = cls(config, xt_config)
        xt_model.model = model
        xt_model.model.config._attn_implementation = "eager"
        return xt_model

    @classmethod
    def from_pretrained(cls, pretrained, xt_config):
        if xt_config["hf_token"]:
            config = AutoConfig.from_pretrained(pretrained, use_auth_token=xt_config["hf_token"])
            xt_model = cls(config, xt_config)
            xt_model.model = AutoModelForCausalLM.from_pretrained(pretrained, from_tf=xt_config["from_tf"], use_auth_token=xt_config["hf_token"])
        else:
            config = AutoConfig.from_pretrained(pretrained)
            xt_model = cls(config, xt_config)
            xt_model.model = AutoModelForCausalLM.from_pretrained(pretrained, from_tf=xt_config["from_tf"])
        xt_model.model.config._attn_implementation = "eager"
        return xt_model
    
    def export_onnx(self, output_dir, model_name="full_model"):
        for run_tpye in self.xt_config["run_types"]:
            self.run_type = run_tpye
            match run_tpye:
                case "streamline":
                    self.export_streamline(output_dir, model_name)
                case "encoder":
                    self.export_encoder(output_dir, model_name)
                case "decoder":
                    self.export_decoder(output_dir, model_name)
                case _:
                    raise ValueError("Run_type not defined.")

    def export_streamline(self, output_dir, model_name):
        self.model.eval()
        with torch.no_grad():
            dummy_input_ids = torch.randint(
                self.vocab_size,
                (1, self.xtensa_cache.max_seq_len),
                dtype=torch.int32
            )
            attention_mask = torch.zeros(
                (1, self.xtensa_cache.max_seq_len),
                dtype=torch.float32
            )
            attention_mask[:, :10] = 1.0
            cache_position = torch.tensor([0]).float()

            past_key = [torch.randn(
                self.xtensa_cache.cache_shape,
                dtype=self.dtype
            ) for _ in range(self.config.num_hidden_layers)]
            past_value = [torch.randn(
                self.xtensa_cache.cache_shape,
                dtype=self.dtype
            ) for _ in range(self.config.num_hidden_layers)]

            dir_path = Path(output_dir) / "streamline" / model_name
            dir_path.mkdir(parents=True, exist_ok=True)
            if self.xt_config["save_dummy_input"]:
                dummy_inputs_dir_path = dir_path / "dummy_inputs"
                dummy_inputs_dir_path.mkdir(parents=True, exist_ok=True)
                np.save(dummy_inputs_dir_path/"input_ids", dummy_input_ids)
                np.save(dummy_inputs_dir_path/"attn_mask", attention_mask)
                np.save(dummy_inputs_dir_path/"cache_position", cache_position)
                for i in range(self.config.num_hidden_layers):
                    np.save(dummy_inputs_dir_path/f"cache_k_{i}", past_key[i])
                    np.save(dummy_inputs_dir_path/f"cache_v_{i}", past_value[i])
            onnx_path = dir_path / "model.onnx"
            torch.onnx.export(
                self,
                (
                    dummy_input_ids,
                    attention_mask,
                    cache_position,
                    past_key,
                    past_value,
                ),
                str(onnx_path),
                input_names=[
                    "input_ids",
                    "attention_mask",
                    "cache_position",
                    ] + [f"cache_k_{i}" for i in range(self.config.num_hidden_layers)]
                    + [f"cache_v_{i}" for i in range(self.config.num_hidden_layers)],
                output_names=[
                    "logits",
                    "cache_position_out",
                ] + [f"cache_k_{i}_out" for i in range(self.config.num_hidden_layers)]
                + [f"cache_v_{i}_out" for i in range(self.config.num_hidden_layers)],
                dynamic_axes=None,
                opset_version=16,
                do_constant_folding=True,
            )

    def export_encoder(self, output_dir, model_name):
        self.model.eval()
        with torch.no_grad():
            dummy_input_ids = torch.randint(
                self.vocab_size,
                (1, self.xtensa_cache.max_seq_len),
                dtype=torch.int32
            )
            if self.xt_config['full_mask']:
                attention_mask = torch.zeros(
                    (1, self.xtensa_cache.max_seq_len),
                    dtype=torch.float32
                )
                attention_mask[:, :10] = 1.0

            dir_path = Path(output_dir) / "encoder" / model_name
            dir_path.mkdir(parents=True, exist_ok=True)
            if self.xt_config["save_dummy_input"]:
                dummy_inputs_dir_path = dir_path / "dummy_inputs"
                dummy_inputs_dir_path.mkdir(parents=True, exist_ok=True)
                np.save(dummy_inputs_dir_path/"input_ids", dummy_input_ids)
                np.save(dummy_inputs_dir_path/"attn_mask", attention_mask)
            onnx_path = dir_path / "model.onnx"
            torch.onnx.export(
                self,
                (
                    dummy_input_ids,
                    attention_mask
                ) if self.xt_config['full_mask'] else
                (
                    dummy_input_ids
                ),
                str(onnx_path),
                input_names=[
                    "input_ids",
                    "attention_mask"
                ] if self.xt_config['full_mask'] else
                [
                    "input_ids"
                ],
                output_names=[
                    "logits",
                ] + [f"cache_k_{i}_out" for i in range(self.config.num_hidden_layers)]
                + [f"cache_v_{i}_out" for i in range(self.config.num_hidden_layers)],
                dynamic_axes=None,
                opset_version=16,
                do_constant_folding=True,
            )

    def export_decoder(self, output_dir, model_name):
        self.model.eval()
        with torch.no_grad():
            dummy_input_ids = torch.randint(
                self.vocab_size,
                (1, 1),
                dtype=torch.float32
            )
            cache_position = torch.tensor([0], dtype=torch.int32)
            past_key = [torch.randn(
                self.xtensa_cache.cache_shape,
                dtype=self.dtype
            ) for _ in range(self.config.num_hidden_layers)]
            past_value = [torch.randn(
                self.xtensa_cache.cache_shape,
                dtype=self.dtype
            ) for _ in range(self.config.num_hidden_layers)]


            dir_path = Path(output_dir) / "decoder" / model_name
            dir_path.mkdir(parents=True, exist_ok=True)
            if self.xt_config["save_dummy_input"]:
                dummy_inputs_dir_path = dir_path / "dummy_inputs"
                dummy_inputs_dir_path.mkdir(parents=True, exist_ok=True)
                np.save(dummy_inputs_dir_path/"input_ids", dummy_input_ids)
                np.save(dummy_inputs_dir_path/"cache_position", cache_position)
                for i in range(self.config.num_hidden_layers):
                    np.save(dummy_inputs_dir_path/f"cache_k_{i}", past_key[i])
                    np.save(dummy_inputs_dir_path/f"cache_v_{i}", past_value[i])
            onnx_path = dir_path / "model.onnx"
            torch.onnx.export(
                self,
                (
                    dummy_input_ids,
                    None, cache_position, past_key, past_value,
                ),
                str(onnx_path),
                input_names=[
                    "input_ids", "cache_position"
                ] + [f"cache_k_{i}" for i in range(self.config.num_hidden_layers)]
                + [f"cache_v_{i}" for i in range(self.config.num_hidden_layers)],
                output_names=[
                    "output_ids",
                ] + [f"cache_k_{i}_out" for i in range(self.config.num_hidden_layers)]
                + [f"cache_v_{i}_out" for i in range(self.config.num_hidden_layers)],
                dynamic_axes=None,
                opset_version=16,
                do_constant_folding=True,
            )

    def forward(
            self, 
            input_ids,
            attention_mask=None, 
            cache_start=None, 
            past_key=None, 
            past_value=None):
        match self.run_type:
            case 'streamline':
                cache_start = cache_start.int()
                seqlen = attention_mask.int().sum()
                self.xtensa_cache.init_cache(past_key, past_value)
                cache_position, position_ids = self.gen_position_ids(cache_start)
                attention_mask_4d = self.gen_attention_mask(
                    cache_position,
                    attention_mask
                )
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask_4d,
                    position_ids=position_ids,
                    past_key_values=self.xtensa_cache,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=False,
                    cache_position=cache_position
                )
                logits = output[0]
                cache_start += seqlen
                return logits, cache_start.float(), self.xtensa_cache.cache_k, self.xtensa_cache.cache_v
            case 'encoder':
                past_key = [torch.zeros(
                    self.xtensa_cache.cache_shape,
                    dtype=self.dtype
                ) for _ in range(self.config.num_hidden_layers)]
                past_value = [torch.zeros(
                    self.xtensa_cache.cache_shape,
                    dtype=self.dtype
                ) for _ in range(self.config.num_hidden_layers)]
                self.xtensa_cache.init_cache(past_key, past_value)
                cache_start = torch.tensor(0)
                cache_position, position_ids = self.gen_position_ids(cache_start)
                attention_mask_4d = self.gen_attention_mask(cache_position, attention_mask)
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask_4d,
                    position_ids=position_ids,
                    past_key_values=self.xtensa_cache,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=False,
                    cache_position=cache_position
                )
                logits = output[0]
                return logits, self.xtensa_cache.cache_k, self.xtensa_cache.cache_v
            case 'decoder':
                self.xtensa_cache.init_cache(past_key, past_value)
                cache_position, position_ids = self.gen_position_ids(cache_start, seqlen=1)
                attention_mask = self.gen_attention_mask(cache_position)
                output = self.model(
                    input_ids=input_ids.int(),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=self.xtensa_cache,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=False,
                    cache_position=cache_position
                )
                logits = output[0]
                output_ids = torch.argmax(logits, dim=-1)
                return output_ids.float(), self.xtensa_cache.cache_k, self.xtensa_cache.cache_v
            case _:
                raise ValueError('run_type not defined.')
