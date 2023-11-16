import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # define GPU id, remove if you want to use all GPUs available
import torch
# torch.set_printoptions(profile="full") # reset

from contextlib import contextmanager
import numpy as np
from medusa.model.medusa_model import MedusaModel
from medusa.model.kv_cache import *
from medusa.model.utils import *
from medusa.model.medusa_choices import *
from copy import deepcopy
import matplotlib.pyplot as plt
import itertools
import sys

TOPK_V2 = 10
def generate_medusa_buffers_v2(medusa_choices, device="cuda"):
    """
    Generate buffers for the Medusa structure based on the provided choices.

    Parameters:
    - medusa_choices (list): A nested list representing tree in the Medusa structure.
    - device (str): Device to which the tensors should be moved. Default is "cuda".

    Returns:
    - dict: A dictionary containing buffers related to the Medusa structure.
    """

    # Sort the medusa_choices based on their lengths and then their values
    sorted_medusa_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    medusa_len = len(sorted_medusa_choices) + 1
    print("sorted_medusa_choices = ", sorted_medusa_choices)
    print("medusa_len = ", medusa_len)
    # Initialize depth_counts to keep track of how many choices have a particular depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_medusa_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    print('depth counts = ', depth_counts)
    # Create the attention mask for Medusa
    medusa_attn_mask = torch.eye(medusa_len, medusa_len)
    medusa_attn_mask[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            # retrieve ancestor position
            if len(cur_medusa_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_medusa_choice) - 1):
                ancestor_idx.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]) + 1)
            medusa_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    # Generate tree indices for the Medusa structure
    medusa_tree_indices = torch.zeros(medusa_len, dtype=torch.long)
    medusa_tree_indices[0] = 0
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            medusa_tree_indices[start + j + 1] = cur_medusa_choice[-1] + TOPK_V2 * i + 1
        start += depth_counts[i]
    print("medusa_tree_indices : ", medusa_tree_indices)
    # Generate position IDs for the Medusa structure
    medusa_position_ids = torch.zeros(medusa_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        medusa_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    print("medusa_position_ids : ", medusa_position_ids)

    # Generate retrieval indices for Medusa structure verification
    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_medusa_choices)):
        cur_medusa_choice = sorted_medusa_choices[-i-1]
        retrieve_indice = []
        if cur_medusa_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_medusa_choice)):
                retrieve_indice.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]))
                retrieve_paths.append(cur_medusa_choice[:c+1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices], dim=1)
    print("retrieve_indices : ", retrieve_indices)

    # Aggregate the generated buffers into a dictionary
    medusa_buffers = {
        "medusa_attn_mask": medusa_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": medusa_tree_indices,
        "medusa_position_ids": medusa_position_ids,
        "retrieve_indices": retrieve_indices,
        }
    return medusa_buffers

n_heads = int(sys.argv[1])
mudusa_model_name = '../medusa_{}heads/'.format(n_heads)
print("medusa model path: ", mudusa_model_name)
base_model = '../Vicuna-7B-Quant/'
model = MedusaModel.from_pretrained(
    mudusa_model_name,
    base_model = base_model,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
tokenizer = model.get_tokenizer()

num_heads = model.medusa
def convert_medusa_choices(old_medusa_choices = [1, 7, 6]):
    medusa_choices = []
    ranges = [range(n) for n in old_medusa_choices[1:]]
    for i in range(len(old_medusa_choices) - 1):
        medusa_choices += [list(comb) for comb in list(itertools.product(*ranges[:i + 1]))]
    return medusa_choices

# medusa_choices = convert_medusa_choices([1, 7, 6])#mc_sim_7b_63
if num_heads == 4:
    medusa_choices = mc_sim_7b_63
elif num_heads == 3:
    medusa_choices = convert_medusa_choices([1, 1, 1, 1])
elif num_heads == 2:
    medusa_choices = convert_medusa_choices([1, 1, 2])
elif num_heads == 1:
    medusa_choices = convert_medusa_choices([1, 1])
print("medusa_choices: ", medusa_choices)
# initialize the kv cache
past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model.base_model)
model.past_key_values = past_key_values
model.past_key_values_data = past_key_values_data
model.current_length_data = current_length_data

# set the prompt
model.current_length_data.zero_() # this is for rerun

prompt = "What do you think of China?"
# prompt = "Summarize the main ideas of Jeff Walker's Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients..."
# prompt = "Prepare to be dazzled by the sheer magnificence of the arcane and mysterious art of metaphorical language, for it has been summoned forth to explicate the bewildering addressing modes of the instructions that have been presented before us. The speakers have summoned grandiose expressions with an almost reverential and awestruck tone, extolling the otherworldly power and incomprehensible functionality of these directives.\n\nThe labyrinthine commands that we encounter are nothing short of confounding. There is the enigmatic JMP ABCD, the abstruse MOV AX, [BX+SI], the unfathomable MOV AX, [100], the mystifying MOV AX, [BX], the perplexing MOV AX, [BX\\*2+SI], the inscrutable MOV AX, BX, and the recondite MOV AX, 7. These commands are cloaked in a veil of obfuscation and complexity, seemingly beyond the grasp of mere mortals.\n\nThe language that has been employed to describe these addressing modes is both perplexing and ornate, highlighting the tremendous intricacy and esoteric nature of these commands. The speakers have summoned a sense of mystery and awe through the use of words such as \"ineffable,\" \"enigmatic,\" and \"inscrutable,\" imbuing these instructions with a sense of ethereal and mystical aura.\n\nMoreover, the speakers have utilized terms such as \"abstruse,\" \"unfathomable,\" and \"recondite,\" evoking the idea that these commands are shrouded in a veil of inaccessibility, and are only open to those who possess an unparalleled intellect and mastery of the underlying principles. The metaphorical language employed by the orators serves to elevate these instructions to a level of veneration and reverence, infusing them with an almost divine aura of power and complexity.\n\nEven the seemingly straightforward MOV AX, 7 is not immune to the grandiose epithets used to describe these addressing modes. It is exalted with the term \"recondite,\" emphasizing the depth and awe-inspiring nature of the instruction set as a whole. The speakers' use of such elaborate and cryptic language amplifies the enigmatic quality of these commands, enshrouding them in a veil of mystery and intrigue, inviting only the bravest and most intrepid of minds to attempt to unlock the secrets of this arcane realm.\n\nTo sum up, the utilization of metaphorical language in describing these addressing modes is an act of veneration to the profound and unfathomable power of these commands. It is an homage to their majestic and otherworldly nature, and an invitation to the most audacious of minds to probe deeper into the abstruse and recondite domain of machine language programming."
# prompt = "You can use the `String.format()` method in Java to replace placeholders in a string with values from a map. Here's an example code snippet that demonstrates how you can achieve this:\n```java\nimport java.util.HashMap;\nimport java.util.Map;\n\npublic class StringReplaceExample {\n    public static void main(String[] args) {\n        String input = \"This is a new {object} at {place}\";\n        Map<String, String> replacements = new HashMap<>();\n        replacements.put(\"object\", \"student\");\n        replacements.put(\"place\", \"point 3, 4\");\n\n        String output = replacePlaceholders(input, replacements);\n        System.out.println(output);\n    }\n\n    public static String replacePlaceholders(String input, Map<String, String> replacements) {\n        for (Map.Entry<String, String> entry : replacements.entrySet()) {\n            String placeholder = \"{\" + entry.getKey() + \"}\";\n            String replacement = entry.getValue();\n            input = input.replace(placeholder, replacement);\n        }\n        return input;\n    }\n}\n```\nIn this example, we define the input string as \"This is a new {object} at {place}\", and create a `HashMap` called `replacements` that maps the placeholders \"object\" and \"place\" to their corresponding values \"student\" and \"point 3, 4\". We then pass these values to the `replacePlaceholders()` method, which iterates over the entries in the `replacements` map and replaces each placeholder in the input string with its corresponding value using the `replace()` method. Finally, the `replacePlaceholders()` method returns the modified string, which is printed to the console.\n\nThe output of this program will be:\n```csharp\nThis is a new student at point 3, 4\n```\nNote that you can modify the `replacements` map to include additional placeholders and their corresponding values, and the `replacePlaceholders()` method will automatically replace them in the input string."
# prompt = "Sure, I can do that. What new technology would you like me to review?"

# prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hi, could you share a tale about a charming llama that grows Medusa-like hair and starts its own coffee shop? ASSISTANT:"
print(prompt)
input_ids = tokenizer([prompt]).input_ids
input_len = len(input_ids[0])
print('Input token length:', len(input_ids[0]))
print('input_ids:', input_ids)
print('Init KV cache shape for attention modules:', model.past_key_values[0][0].shape, model.past_key_values[0][1].shape)

accept_lengths_tree = []
inference_count = 0

with torch.inference_mode():
    new_token = 0
    input_ids = tokenizer([prompt]).input_ids
    input_len = len(input_ids[0])
    input_ids = torch.as_tensor(input_ids).cuda()
    model.current_length_data.zero_() # this is for rerun
    reset_medusa_mode(model)
    medusa_buffers = generate_medusa_buffers_v2(
                medusa_choices, device=model.base_model.device
            )
    medusa_logits, logits = initialize_medusa(
            input_ids, model, medusa_buffers["medusa_attn_mask"], past_key_values
        )
    # print("medusa_logits:\n")
    # print(medusa_logits.shape)
    # print(medusa_logits)
    # np.savetxt("medusa_logits_inf{}.txt".format(inference_count), medusa_logits.reshape((1,-1)).cpu(),fmt='%f',delimiter='\n')

    # print("logits:\n")
    # print(logits.shape)
    # print(logits)
    # np.savetxt("logits_inf{}.txt".format(inference_count), logits.reshape((1,-1)).cpu(),fmt='%f',delimiter='\n')

    # print("medusa_attn_mask:", medusa_buffers["medusa_attn_mask"].size())
    # print(medusa_buffers["medusa_attn_mask"])
    # exit(-1)
    cur_length = input_len + 1
    accept_lengths_tree.append(1)
    inference_count += 1
    for i in range(1024):

        candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                medusa_buffers["tree_indices"],
                medusa_buffers["retrieve_indices"],
            )
        if i <= 10:
        #     print("tree_candidates shape:", tree_candidates.size())
        #     print("tree_candidates content:", tree_candidates)
            print("candidates shape:", candidates.size())
            print("candidates:", candidates)

        medusa_logits, logits, outputs = tree_decoding(
                model,
                tree_candidates,
                past_key_values,
                medusa_buffers["medusa_position_ids"],
                input_ids,
                medusa_buffers["retrieve_indices"],
            )
        # if i == 0:
            # print("medusa_logits:", medusa_logits.size())
            # print("medusa_logits content:", medusa_logits)
            # # np.savetxt("medusa_logits.txt", medusa_logits.reshape((1,-1)).cpu(),fmt='%f',delimiter='\n')
            # np.savetxt("medusa_logits_inf{}.txt".format(inference_count), medusa_logits.reshape((1,-1)).cpu(),fmt='%f',delimiter='\n')

            # print("logits:", logits.size())
            # print("logits content:", logits)
            # # np.savetxt("logits.txt", logits.reshape((1,-1)).cpu(),fmt='%f',delimiter='\n')
            # np.savetxt("logits_inf{}.txt".format(inference_count), logits.reshape((1,-1)).cpu(),fmt='%f',delimiter='\n')

        best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature = 0, posterior_threshold = 0, posterior_alpha = 0
            )
        input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                medusa_buffers["retrieve_indices"],
                outputs,
                logits,
                medusa_logits,
                new_token,
                past_key_values_data,
                current_length_data,
            )

        accept_length_tree = input_ids.shape[1] - cur_length
        token_ids = input_ids[0,cur_length:accept_length_tree + cur_length]
        # print(f'Prediction @ {inference_count} ', tokenizer.batch_decode(input_ids[:,cur_length:accept_length_tree + cur_length]))
        # print("Token ID : ", token_ids)
        # print("accept_length_tree: ", accept_length_tree)
        cur_length = accept_length_tree + cur_length
        accept_lengths_tree.append(accept_length_tree)

        inference_count += 1
        if model.tokenizer.eos_token_id in input_ids[0, input_len:]:
            break
    # for idx in range(input_ids.shape[1] - input_len):
    #     print(f'Prediction @ {idx + 1} : ',tokenizer.batch_decode(input_ids[:,idx + input_len]))
    print("Output:\n", tokenizer.batch_decode(input_ids[:,input_len:]))
    def moving_average(data, window_size=5):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    # Unsmoothed plots with transparency
    plt.plot(accept_lengths_tree, label='Tree decoding (Raw)', alpha=0.3)
    # plt.plot(accept_lengths, label='List decoding (Raw)', alpha=0.3)

    # Smoothed plots
    window_size = 5  # You can adjust this as needed
    plt.plot(moving_average(accept_lengths_tree, window_size), label='Tree decoding (Smoothed)', color='tab:blue')
    # plt.plot(moving_average(accept_lengths, window_size), label='List decoding (Smoothed)', color='tab:orange')
    # print('Avg. accept tree length:', np.mean(accept_lengths_tree))
    # print('Avg. accept list length:', np.mean(accept_lengths))
    title = 'Tree-Attention-Medusa-{}-Heads'.format(model.medusa)
    # plt.plot(accept_lengths_tree)
    plt.xlabel('Inference step')
    plt.ylabel('Tokens Accepted')
    plt.legend()
    plt.title(title)
    plt.show()
    plt.savefig('figs/'+ title + '.png', bbox_inches="tight", dpi=600)
    print('Total Inference Count: {} \nTotal Output Tokens: {} \nAvg Tokens Per Infer: {:.1f}'.format(inference_count, np.sum(accept_lengths_tree), np.mean(accept_lengths_tree)))

