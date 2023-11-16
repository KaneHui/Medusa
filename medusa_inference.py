import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # define GPU id, remove if you want to use all GPUs available
import torch
from contextlib import contextmanager
import numpy as np
from medusa.model.medusa_model import MedusaModel
from medusa.model.kv_cache import *
from medusa.model.utils import *
from medusa.model.medusa_choices import *
from copy import deepcopy
import matplotlib.pyplot as plt


# mudusa_model_name = '../medusa_2heads/'
# base_model = '../Vicuna-7B-Base/'
# model = MedusaModel.from_pretrained(
#     mudusa_model_name,
#     base_model = base_model,
#     torch_dtype=torch.float16,
#     low_cpu_mem_usage=True,
#     device_map="auto"
# )
# tokenizer = model.get_tokenizer()

# medusa_choices = mc_sim_7b_63

# # initialize the kv cache
# past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model.base_model)
# model.past_key_values = past_key_values
# model.past_key_values_data = past_key_values_data
# model.current_length_data = current_length_data

# # set the prompt
# model.current_length_data.zero_() # this is for rerun
# prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hi, could you share a tale about a charming llama that grows Medusa-like hair and starts its own coffee shop? ASSISTANT:"
# print(prompt)
# input_ids = tokenizer([prompt]).input_ids
# input_len = len(input_ids[0])
# print('Input token length:', len(input_ids[0]))
# print('Init KV cache shape for attention modules:', model.past_key_values[0][0].shape, model.past_key_values[0][1].shape)


# print(model.medusa_head)

# # first inference with medusa heads
# with torch.inference_mode():
#     print("### First inference with medusa heads ###")
#     input_ids = tokenizer([prompt]).input_ids
#     input_len = len(input_ids[0])
#     input_ids = torch.as_tensor(input_ids).cuda()
#     model.current_length_data.zero_() # this is for rerun
#     medusa_logits, outputs, logits = model(input_ids, output_orig = True, past_key_values=model.past_key_values)
#     medusa_pred = torch.argmax(medusa_logits[..., -1, :], dim = -1)
#     pred = torch.argmax(logits[..., -1, :], dim = -1)
#     print('Base model prediction:', tokenizer.batch_decode(pred))
#     print('Medusa prediction:', tokenizer.batch_decode(medusa_pred))
#     preds = torch.cat([pred, medusa_pred[:, 0 ]], dim = -1)
#     print('Combined prediction:', tokenizer.batch_decode(preds))
#     print('Medusa logits shape:', medusa_logits.shape, 'logits shape:', logits.shape)

# # second inference with medusa heads
# with torch.inference_mode():
#   print("### Second inference with medusa heads ###")
#   medusa_logits, outputs, logits = model(preds.cuda().unsqueeze(0), output_orig = True, past_key_values = model.past_key_values)
#   medusa_pred = torch.argmax(medusa_logits[..., (0 - model.medusa - 1):, :], dim = -1)
#   pred = torch.argmax(logits[..., :, :], dim = -1)
#   print('Base model prediction:', tokenizer.batch_decode(pred[0]))
#   print('truncated input tokens:', preds[1:].tolist())
#   print('Output tokens:', pred[0, :].tolist())

#   posterior_mask = (
#               preds[1:] == pred[0, :-1]
#           ).int()
#   accept_length = torch.cumprod(posterior_mask, dim = -1).sum().item()
#   cur_length = accept_length + input_len + 1
#   print('Posterior mask:', posterior_mask.tolist())
#   print('Accept length:', accept_length)
#   print('Current KV cache length for attention modules:', model.current_length_data[0].item())
#   print('Start length:', input_len, ',current length:', cur_length)
#   # update kv cache
#   model.current_length_data.fill_(cur_length)
#   # create new input
#   preds = torch.cat([pred[:, accept_length], medusa_pred[:,0,accept_length]], dim = -1)
#   print('Combined prediction:', tokenizer.batch_decode(preds))


# # Third inference with medusa heads
# with torch.inference_mode():
#   print("### Third inference with medusa heads ###")
#   medusa_logits, outputs, logits = model(preds.cuda().unsqueeze(0), output_orig = True, past_key_values = model.past_key_values)
#   medusa_pred = torch.argmax(medusa_logits[..., (0 - model.medusa - 1):, :], dim = -1)
#   pred = torch.argmax(logits[..., :, :], dim = -1)
#   print('Base model prediction:', tokenizer.batch_decode(pred[0]))
#   print('truncated input tokens:', preds[1:].tolist())
#   print('Output tokens:', pred[0, :].tolist())

#   posterior_mask = (
#               preds[1:] == pred[0, :-1]
#           ).int()
#   accept_length = torch.cumprod(posterior_mask, dim = -1).sum().item()
#   cur_length = accept_length + input_len + 1
#   print('Posterior mask:', posterior_mask.tolist())
#   print('Accept length:', accept_length)
#   print('Current KV cache length for attention modules:', model.current_length_data[0].item())
#   print('Start length:', input_len, ',current length:', cur_length)
#   # update kv cache
#   model.current_length_data.fill_(cur_length)
#   # create new input
#   preds = torch.cat([pred[:, accept_length], medusa_pred[:,0,accept_length]], dim = -1)
#   print('Combined prediction:', tokenizer.batch_decode(preds))



# mudusa_model_name = '../medusa_2heads/'
mudusa_model_list = ['../medusa_3heads/']
with torch.inference_mode():
    base_model = '../Vicuna-7B-Base/'
    for mudusa_model_name in mudusa_model_list:
        inference_count = 0
        accept_lengths = []
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
        print(prompt)
        input_ids = tokenizer([prompt]).input_ids
        input_len = len(input_ids[0])
        print('Input token length:', len(input_ids[0]))
        print('Init KV cache shape for attention modules:', model.past_key_values[0][0].shape, model.past_key_values[0][1].shape)
        input_ids = tokenizer([prompt]).input_ids
        input_len = len(input_ids[0])
        input_ids = torch.as_tensor(input_ids).cuda()
        model.current_length_data.zero_() # this is for rerun
        medusa_logits, outputs, logits = model(input_ids, output_orig = True, past_key_values=model.past_key_values)
        inference_count += 1

        medusa_pred = torch.argmax(medusa_logits[..., -1, :], dim = -1)
        pred = torch.argmax(logits[..., -1, :], dim = -1)
        preds = torch.cat([pred, medusa_pred[:, 0 ]], dim = -1)
        print('base_lm_pred: ', pred)
        print('medusa_pred: ', medusa_pred)
        print('concat preds: ', preds)

        print(f'Prediction @ {inference_count}: {tokenizer.batch_decode(pred)}')
        cur_length = input_len
        accept_lengths.append(1)
        for index in range(1024):
            medusa_logits, outputs, logits = model(preds.cuda().unsqueeze(0), output_orig = True, past_key_values = model.past_key_values)
            inference_count += 1

            medusa_pred = torch.argmax(medusa_logits[..., (0 - model.medusa - 1):, :], dim = -1)
            pred = torch.argmax(logits[..., :, :], dim = -1)
            posterior_mask = (
                        preds[1:] == pred[0, :-1]
                    ).int()

            accept_length = torch.cumprod(posterior_mask, dim = -1).sum().item()
            cur_length = cur_length + accept_length + 1
            # update kv cache
            model.current_length_data.fill_(cur_length)

            if index < 3:
                print('old concat preds: ', preds)
                print('cur_length: ', cur_length)
                print('accept_length: ', accept_length)
                print('base_lm_pred: ', pred)
                print('medusa_pred: ', medusa_pred)
                print('posterior_mask: ', posterior_mask)
            # create new input
            preds = torch.cat([pred[:, accept_length], medusa_pred[:,0,accept_length]], dim = -1)
            if index < 3:
                print('new concat preds: ', preds)

            if inference_count < 5 or inference_count > 250:
                print(f'Prediction @ {inference_count}: {tokenizer.batch_decode(pred[0, :accept_length + 1])}')
            accept_lengths.append(accept_length + 1)
            if tokenizer.eos_token_id in pred[0, :accept_length + 1]:
                break
        title = 'Medusa-{}-Heads'.format(model.medusa)
        plt.plot(accept_lengths)
        plt.xlabel('Inference step')
        plt.ylabel('Tokens Accepted')
        plt.title(title)
        plt.show()
        plt.savefig('figs/'+ title + '.png', bbox_inches="tight", dpi=600)
        print('Total Inference Count: {} \nTotal Output Tokens: {} \nAvg Tokens Per Infer: {:.1f}'.format(inference_count, np.sum(accept_lengths), np.mean(accept_lengths)))
