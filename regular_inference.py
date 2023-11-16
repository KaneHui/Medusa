import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # define GPU id, remove if you want to use all GPUs available
import torch
torch.set_printoptions(profile="full") # reset

from contextlib import contextmanager
import numpy as np
from medusa.model.medusa_model import MedusaModel
from medusa.model.kv_cache import *
from medusa.model.utils import *
from medusa.model.medusa_choices import *
from copy import deepcopy
import matplotlib.pyplot as plt


mudusa_model_name = '../medusa_2heads/'
# base_model = '../Vicuna-7B-Base/'
base_model = '../Vicuna-7B-Quant/'

model = MedusaModel.from_pretrained(
    mudusa_model_name,
    base_model = base_model,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
tokenizer = model.get_tokenizer()

medusa_choices = mc_sim_7b_63

# initialize the kv cache
past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model.base_model)
model.past_key_values = past_key_values
model.past_key_values_data = past_key_values_data
model.current_length_data = current_length_data

# set the prompt
model.current_length_data.zero_() # this is for rerun
prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hi, could you share a tale about a charming llama that grows Medusa-like hair and starts its own coffee shop? ASSISTANT:"
prompt = "What do you think of China?"
print(prompt)
input_ids = tokenizer([prompt]).input_ids
input_len = len(input_ids[0])
print('Input token length:', len(input_ids[0]))
print('input_ids:', input_ids)
print('Init KV cache shape for attention modules:', model.past_key_values[0][0].shape, model.past_key_values[0][1].shape)

inference_count = 0

# first inference
with torch.inference_mode():
    model.current_length_data.zero_() # this is for rerun
    output = model.base_model(torch.as_tensor(input_ids).cuda(), past_key_values=model.past_key_values,)
    print('output shape:', output.logits.shape)
    pred = output.logits.argmax(-1)
    input_ids[0] = input_ids[0] + pred[0, -1:].tolist()
    inference_count += 1
    print('KV cache shape for attention modules after first inference:', model.past_key_values[0][0].shape, model.past_key_values[0][1].shape)
    print(f'Prediction @ {inference_count} : ', tokenizer.batch_decode(pred[..., -1:]))

# second inference
with torch.inference_mode():
    print('pred[..., -1:]=', pred[..., -1:].size())
    output = model.base_model(pred[..., -1:], past_key_values=model.past_key_values, use_cache=True) # note we only need to put last token in the input_ids
    print('output shape:', output.logits.shape)
    pred = output.logits.argmax(-1)
    input_ids[0] = input_ids[0] + pred[0, -1:].tolist()
    print('KV cache shape for attention modules after second inference:', model.past_key_values[0][0].shape, model.past_key_values[0][1].shape)
    print('pred shape:', pred.size())
    print(pred.data)
    print('input ids :', input_ids)
    print('pred[0, -1:]=', pred[0, -1:])
    inference_count += 1
    print(f'Prediction @ {inference_count} : ', tokenizer.batch_decode(pred[..., -1:]))

with torch.inference_mode():
    for index in range(1024):
        output = model.base_model(pred[..., -1:], past_key_values=model.past_key_values, use_cache=True)
        pred = output.logits.argmax(-1)
        # pred_topk = output.logits.topk(10, dim = -1).indices[0]
        input_ids[0] = input_ids[0] + pred[0, -1:].tolist()
        inference_count += 1
        print(f'Prediction @ {inference_count} : ', tokenizer.batch_decode(pred[..., -1:]))
        if tokenizer.eos_token_id in pred[0, -1:]:
            # print(f' Last Prediction @ {inference_count} : ', tokenizer.batch_decode(pred[..., -1:]))
            break
        # print(tokenizer.batch_decode(pred_topk), 'topk:', pred_topk)
print("Inference Count:", inference_count)
print(tokenizer.decode(input_ids[0]))

