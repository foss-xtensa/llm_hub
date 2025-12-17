import torch
import onnxruntime
import numpy as np

from pathlib import Path
from transformers import DynamicCache

def verify_xt_streamline(model, xt_model, xt_config, out_dir):
    # compare model with xt_model
    model.eval()
    xt_model.eval()
    with torch.no_grad():
        dummy_input_ids = torch.randint(
            xt_model.vocab_size,
            (1, xt_model.xtensa_cache.max_seq_len),
            dtype=torch.int32
        )
        seqlen = torch.tensor([10])
        attention_mask = torch.zeros(
            (1, xt_model.xtensa_cache.max_seq_len),
            dtype=torch.float32
        )
        attention_mask[:, :seqlen] = 1.0
        cache_position = torch.tensor([0])

        past_key = [torch.randn(
            xt_model.xtensa_cache.cache_shape
        ) for _ in range(xt_model.model.config.num_hidden_layers)]
        past_value = [torch.randn(
            xt_model.xtensa_cache.cache_shape
        ) for _ in range(xt_model.model.config.num_hidden_layers)]
        dummy_input_ids_org = dummy_input_ids[:, :seqlen]

        dummy_output = xt_model(
            dummy_input_ids,
            attention_mask,
            cache_position.float(),
            past_key,
            past_value
        )
        dummy_output_org = model(dummy_input_ids_org)
        for i in range(xt_model.config.num_hidden_layers):
            if not torch.allclose(
                dummy_output[2][i][:,:seqlen,:],
                dummy_output_org.past_key_values[i][0].squeeze(0),
                rtol=xt_config["rtol"], atol=xt_config["atol"]
            ):
                print(f"Warning: streamline hf xt_model layer {i} cache k max absolute difference: {torch.max(torch.abs(dummy_output[2][i][:,:seqlen,:] - dummy_output_org.past_key_values[i][0].squeeze(0)))}")
            
            if not torch.allclose(
                dummy_output[3][i][:,:seqlen,:],
                dummy_output_org.past_key_values[i][1].squeeze(0),
                rtol=xt_config["rtol"], atol=xt_config["atol"]
            ):
                print(f"Warning: streamline hf xt_model layer {i} cache v max absolute difference: {torch.max(torch.abs(dummy_output[3][i][:,:seqlen,:] - dummy_output_org.past_key_values[i][1].squeeze(0)))}")
            
        if not torch.allclose(
            dummy_output[0][:,:seqlen,:],
            dummy_output_org.logits,
            rtol=xt_config["rtol"], atol=xt_config["atol"]
        ):
            print(f"Warning: streamline hf xt_model output max absolute difference: {torch.max(torch.abs(dummy_output[0][:,:seqlen,:] - dummy_output_org.logits))}")

    # Compare onnxruntime results with huggingface results
    session = onnxruntime.InferenceSession(Path(out_dir) / "streamline" / "full_model" / "model.onnx", None)
    
    outputs = session.get_outputs()
    output_names = []
    for out in outputs:
        output_names.append(out.name)
    
    output = session.run(
        output_names,
        {'input_ids' : dummy_input_ids.numpy(),
         'attention_mask' : attention_mask.numpy(), 'cache_position' : cache_position.float().numpy()} | 
         {f'cache_k_{i}' : past_key[i].numpy() for i in range(xt_model.config.num_hidden_layers)} |
         {f'cache_v_{i}' : past_value[i].numpy() for i in range(xt_model.config.num_hidden_layers)}
    )
    for i in range(xt_model.config.num_hidden_layers):
        if not np.allclose(
            dummy_output[2][i],
            output[i+2],
            rtol=xt_config["rtol"], atol=xt_config["atol"]
        ):
            print(f"Warning: streamline model layer {i} cache k max absolute difference: {np.max(np.abs(np.array(dummy_output[2][i]) - output[i+2]))}")
        if not np.allclose(
            dummy_output[3][i],
            output[xt_model.config.num_hidden_layers+2+i],
            rtol=xt_config["rtol"], atol=xt_config["atol"]
        ):
            print(f"Warning: streamline model layer {i} cache v max absolute difference: {np.max(np.abs(np.array(dummy_output[3][i]) - output[xt_model.config.num_hidden_layers+2+i]))}")
    if not np.allclose(
        dummy_output[0],
        output[0],
        rtol=xt_config["rtol"], atol=xt_config["atol"]
    ):
        print(f"Warning: streamline model output max absolute difference: {np.max(np.abs(np.array(dummy_output[0]) - output[0]))}")

def verify_xt_enc(model, xt_model, xt_config, out_dir):
    # compare model with xt_model
    model.eval()
    xt_model.eval()
    with torch.no_grad():
        dummy_input_ids = torch.randint(
            xt_model.vocab_size,
            (1, xt_model.xtensa_cache.max_seq_len),
            dtype=torch.int32
        )
        seqlen = torch.tensor(10)
        if xt_config['full_mask']:
            attention_mask = torch.zeros(
                (1, xt_model.xtensa_cache.max_seq_len),
                dtype=torch.float32
            )
            attention_mask[:, :seqlen] = 1.0
        else:
            attention_mask = None

        dummy_input_ids_org = dummy_input_ids[:, :seqlen]

        dummy_output = xt_model(dummy_input_ids, attention_mask)
        dummy_output_org = model(dummy_input_ids_org)
        for i in range(xt_model.config.num_hidden_layers):
            if not torch.allclose(
                dummy_output[1][i][:,:seqlen,:],
                dummy_output_org.past_key_values[i][0].squeeze(0),
                rtol=xt_config["rtol"], atol=xt_config["atol"]
            ):
                print(f"Warning: encoder hf xt_model layer {i} cache k max absolute difference: {torch.max(torch.abs(dummy_output[1][i][:,:seqlen,:] - dummy_output_org.past_key_values[i][0].squeeze(0)))}")
            if not torch.allclose(
                dummy_output[2][i][:,:seqlen,:],
                dummy_output_org.past_key_values[i][1].squeeze(0),
                rtol=xt_config["rtol"], atol=xt_config["atol"]
            ):
                print(f"Warning: encoder hf xt_model layer {i} cache v max absolute difference: {torch.max(torch.abs(dummy_output[2][i][:,:seqlen,:] - dummy_output_org.past_key_values[i][1].squeeze(0)))}")
        if not torch.allclose(
            dummy_output[0][:,:seqlen,:],
            dummy_output_org.logits,
            rtol=xt_config["rtol"], atol=xt_config["atol"]
        ):
            print(f"Warning: encoder hf xt_model output max absolute difference: {torch.max(torch.abs(dummy_output[0][:,:seqlen,:] - dummy_output_org.logits))}")

    # Compare onnxruntime results with huggingface results
    session = onnxruntime.InferenceSession(Path(out_dir) / "encoder" / "full_model" / "model.onnx", None)
    
    outputs = session.get_outputs()
    output_names = []
    for out in outputs:
        output_names.append(out.name)
    
    output = session.run(
        output_names,
        {'input_ids' : dummy_input_ids.numpy(),
        'attention_mask' : attention_mask.numpy()}
        if xt_config['full_mask'] else
        {'input_ids' : dummy_input_ids.numpy()}
    )
    for i in range(xt_model.config.num_hidden_layers):
        if not np.allclose(
            dummy_output[1][i],
            output[i+1],
            rtol=xt_config["rtol"], atol=xt_config["atol"]
        ):
            print(f"Warning: encoder model layer {i} cache k max absolute difference: {np.max(np.abs(np.array(dummy_output[1][i]) - output[i+1]))}")
        if not np.allclose(
            dummy_output[2][i],
            output[xt_model.config.num_hidden_layers+1+i],
            rtol=xt_config["rtol"], atol=xt_config["atol"]
        ):
            print(f"Warning: encoder model layer {i} cache v max absolute difference: {np.max(np.abs(np.array(dummy_output[2][i]) - output[xt_model.config.num_hidden_layers+1+i]))}")
    if not np.allclose(
        dummy_output[0],
        output[0],
        rtol=xt_config["rtol"], atol=xt_config["atol"]
    ):
        print(f"Warning: encoder model output max absolute difference: {np.max(np.abs(np.array(dummy_output[0]) - output[0]))}")

def verify_xt_dec(model, xt_model, xt_config, out_dir):
    # compare model with xt_model
    model.eval()
    xt_model.eval()
    with torch.no_grad():
        dummy_input_ids = torch.randint(
            xt_model.vocab_size,
            (1, 1),
            dtype=torch.float32
        )
        cache_position = torch.tensor([10], dtype=torch.int32)

        past_key = [torch.randn(
            xt_model.xtensa_cache.cache_shape
        ) for _ in range(xt_model.model.config.num_hidden_layers)]
        past_value = [torch.randn(
            xt_model.xtensa_cache.cache_shape
        ) for _ in range(xt_model.model.config.num_hidden_layers)]

        dummy_output = xt_model(
            dummy_input_ids,
            cache_start=cache_position, 
            past_key=past_key, 
            past_value=past_value)
        cache = DynamicCache()
        cache.key_cache = [past_key[i][:, :cache_position, :].unsqueeze(0)
                            for i in range(xt_model.model.config.num_hidden_layers)]
        cache.value_cache = [past_value[i][:, :cache_position, :].unsqueeze(0)
                            for i in range(xt_model.model.config.num_hidden_layers)]
        dummy_output_org = model(dummy_input_ids.int(), past_key_values=cache)
        for i in range(xt_model.config.num_hidden_layers):
            if not torch.allclose(
                dummy_output[1][i][:,:cache_position+1,:],
                dummy_output_org.past_key_values[i][0].squeeze(0),
                rtol=xt_config["rtol"], atol=xt_config["atol"]
            ):
                print(f"Warning: decoder hf xt_model layer {i} cache k max absolute difference: {torch.max(torch.abs(dummy_output[1][i][:,:cache_position+1,:] - dummy_output_org.past_key_values[i][0].squeeze(0)))}")
            if not torch.allclose(
                dummy_output[2][i][:,:cache_position+1,:],
                dummy_output_org.past_key_values[i][1].squeeze(0),
                rtol=xt_config["rtol"], atol=xt_config["atol"]
            ):
                print(f"Warning: decoder hf xt_model layer {i} cache v max absolute difference: {torch.max(torch.abs(dummy_output[2][i][:,:cache_position+1,:] - dummy_output_org.past_key_values[i][1].squeeze(0)))}")
        if not torch.allclose(
            dummy_output[0].long(),
            torch.argmax(dummy_output_org.logits, dim=-1),
            rtol=xt_config["rtol"], atol=xt_config["atol"]
        ):
            print(f"Warning: decoder hf xt_model output max absolute difference: {torch.max(torch.abs(dummy_output[0].long() - torch.argmax(dummy_output_org.logits, dim=-1)))}")

    # Compare onnxruntime results with huggingface results
    session = onnxruntime.InferenceSession(Path(out_dir) / "decoder" / "full_model" / "model.onnx", None)
    
    outputs = session.get_outputs()
    output_names = []
    for out in outputs:
        output_names.append(out.name)
    
    output = session.run(
        output_names,
        {'input_ids' : dummy_input_ids.numpy(),
         'cache_position' : cache_position.numpy()} | 
         {f'cache_k_{i}' : past_key[i].numpy() for i in range(xt_model.config.num_hidden_layers)} |
         {f'cache_v_{i}' : past_value[i].numpy() for i in range(xt_model.config.num_hidden_layers)}
    )
    for i in range(xt_model.config.num_hidden_layers):
        if not np.allclose(
            dummy_output[1][i],
            output[i+1],
            rtol=xt_config["rtol"], atol=xt_config["atol"]
        ):
            print(f"Warning: decoder model layer {i} cache k max absolute difference: {np.max(np.abs(np.array(dummy_output[1][i]) - output[i+1]))}")
        if not np.allclose(
            dummy_output[2][i],
            output[xt_model.config.num_hidden_layers+1+i],
            rtol=xt_config["rtol"], atol=xt_config["atol"]
        ):
            print(f"Warning: decoder model layer {i} cache v max absolute difference: {np.max(np.abs(np.array(dummy_output[2][i]) - output[xt_model.config.num_hidden_layers+i+1]))}")
    if not np.allclose(
        dummy_output[0],
        output[0],
        rtol=xt_config["rtol"], atol=xt_config["atol"]
    ):
        print(f"Warning: decoder model output max absolute difference: {np.max(np.abs(np.array(dummy_output[0]) - output[0]))}")

def verify_xt_llm(model, xt_model, xt_config, out_dir):
    for run_type in xt_config["run_types"]:
        xt_model.run_type = run_type
        match run_type:
            case "streamline":
                verify_xt_streamline(model, xt_model, xt_config, out_dir)
            case "encoder":
                verify_xt_enc(model, xt_model, xt_config, out_dir)
            case "decoder":
                verify_xt_dec(model, xt_model, xt_config, out_dir)
            case _:
                raise ValueError('run_type not defined.')