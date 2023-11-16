import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # define GPU id, remove if you want to use all GPUs available
import torch
torch.set_printoptions(profile="full") # reset
import networkx as nx
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
def test_generate_medusa_buffers(medusa_choices, device="cuda"):
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

    # Move the tensors in the dictionary to the specified device
    medusa_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v,  device=device)
        for k, v in medusa_buffers.items()
    }
    return medusa_buffers

n_heads = int(sys.argv[1])
num_heads = n_heads
def convert_medusa_choices(old_medusa_choices = [1, 7, 6]):
    print("old_medusa_choices[1:]=", old_medusa_choices[1:])
    medusa_choices = []
    ranges = [range(n) for n in old_medusa_choices[1:]]
    for i in range(len(old_medusa_choices) - 1):
        medusa_choices += [list(comb) for comb in list(itertools.product(*ranges[:i + 1]))]
        print("medusa_choices=", medusa_choices)
    return medusa_choices

# medusa_choices = convert_medusa_choices([1, 7, 6])#mc_sim_7b_63
if num_heads == 4:
    medusa_choices = mc_sim_7b_63
elif num_heads == 3:
    medusa_choices = convert_medusa_choices([1, 1, 3, 4])
elif num_heads == 2:
    medusa_choices = convert_medusa_choices([1, 1, 2])
elif num_heads == 1:
    medusa_choices = convert_medusa_choices([1, 2])
print("medusa_choices: ", medusa_choices)
# initialize the kv cache
paths = medusa_choices[:]

# print("paths: ", paths)
G = nx.DiGraph()

for path in paths:
    for i in range(len(path)):
        if i == 0:
            parent = 'root'
        else:
            parent = tuple(path[:i])
        child = tuple(path[:i+1])
        G.add_edge(parent, child)

# print(G)
# Use the Graphviz layout for drawing.
# pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
# nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, width=2, edge_color="gray")
# plt.savefig('../figs/medusa_configuration_explained.png', dpi=300)

medusa_buffers = test_generate_medusa_buffers(medusa_choices, device='cpu')
for item in medusa_buffers:
    print(item)

# medusa_attn_mask = medusa_buffers['medusa_attn_mask']
# print('Medusa attention mask shape: ', medusa_attn_mask.shape)
# print('Medusa attention mask:')
# print(medusa_attn_mask)
# plt.imsave("figs/medusa_attn_mask.png", medusa_attn_mask[0,0].cpu().numpy())
