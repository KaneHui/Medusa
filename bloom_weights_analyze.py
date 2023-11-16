
import torch
import torch.nn as nn
import numpy as np
import math
import os
import time
import fakequant_cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

skip_layers_tag = ['word_embeddings', 'lm_head']


def cosine_diff_kernel(a, b):
  div_sum = torch.norm(a) * torch.norm(b)
  ab_sum  = torch.sum(a * b)
  if div_sum.item() == 0 and ab_sum.item() == 0:
    return 1.0
  cos_tensor = ab_sum / div_sum
  return cos_tensor.item()

def mse_diff_kernel(a, b):
  mse_tensor = ((a - b) ** 2).mean()
  return mse_tensor.item()

def compute_cos_distance(array_float, dequant_float):
  array_sum = np.sum(array_float * dequant_float)
  x_pow_sum = np.sum(array_float * array_float)
  y_pow_sum = np.sum(dequant_float * dequant_float)
  if array_sum == 0 and x_pow_sum == 0:
    return 1.0
  arary_sqrt = math.sqrt(x_pow_sum) * math.sqrt(y_pow_sum)
  cos = array_sum/ arary_sqrt
  #print("cos:", cos) #>0.9 0.99
  return cos

def compute_mse(float_array, dequant_float):
  sub_array = float_array - dequant_float
  delta_array = sub_array * sub_array
  mse_dis = delta_array.mean()
  return mse_dis


def get_mat_scale(src_data, data_len, group_size, intN=4):
  q_max = 7
  input_data = src_data.reshape(data_len// group_size, group_size)
  scales = np.zeros((input_data.shape[0]), dtype=np.float32)
  for block_id in range(input_data.shape[0]):
      block_array = input_data[block_id]
#       print("block_id:", block_id)
#       print(block_array)
      block_min,block_max   = block_array.min(),block_array.max()
      if block_max == 0 and block_max == block_min:
        block_max = block_max + 0.01
      abs_max = max(abs(block_max), abs(block_min))
      block_s = abs_max / float(q_max)
      scales[block_id] = block_s
      input_data[block_id] = block_array
  return scales




def build_bloom_fakequant_weights(input_file_path, output_file_path, intN = 4, group_size=32, skip_embdding = False, show_diff=False):
  print("*************begin load dict *************:", input_file_path)
  model_dict = torch.load(input_file_path)
  print("*************end load dict *************:", input_file_path)
  show_diff_dict = {'tensor':[],'shape':[],'minmax':[],'cos_diff': [],'mse_diff': []}
  absmax = int(pow(2, intN-1) - 1)
  start0 = time.time()
  for name in model_dict.keys():
#     print("----------------:", name, "  shape:", model_dict[name].shape, " data type:",model_dict[name].dtype, "  size:",model_dict[name].numel()/1024/1024, " MB")
#     continue
    if len(model_dict[name].shape) == 2 :
      if "word_embeddings.weight" in name or "lm_head.weight" in name:
        if skip_embdding:
          print("skip weight:", name)
          continue
    if len(model_dict[name].shape) == 2:
      src_tensor = model_dict[name]
      data_type = src_tensor.dtype
      data_size = src_tensor.numel()
      print("----------------:", name, "  shape:", src_tensor.shape, " data type:",data_type, "  size:",data_size/1024/1024, " MB")
      
      if data_type == torch.float16:
        print("begin fakequant")
        scalestensor = torch.ones((src_tensor.shape[0]*src_tensor.shape[1]//group_size), dtype = torch.float32).to(device)
        src_tensor = src_tensor.to(device)
        if intN == 4:
          fakequant_cuda.compute_scale_q4_0(src_tensor , scalestensor, int(32))
        else:
          fakequant_cuda.compute_scale_intN_q0(src_tensor , scalestensor, int(32), float(absmax))    
        output_tensor = torch.ones(src_tensor.shape, dtype = data_type).to(device)
        if intN == 4:
          fakequant_cuda.fakequant_q4_0_half(src_tensor ,output_tensor, scalestensor, int(group_size))
        else:
          fakequant_cuda.fakequant_intN_q0_half(src_tensor ,output_tensor, scalestensor, int(group_size), float(absmax))
        #replace
        model_dict[name] = output_tensor.to('cpu')

        if show_diff:
#           src_tensor = src_tensor.to('cpu').numpy()
#           output_tensor = output_tensor.to('cpu').numpy()
#           cos_dif = compute_cos_distance(src_tensor, output_tensor)
#           mse_dif = compute_mse(src_tensor, output_tensor)
          cos_dif = cosine_diff_kernel(src_tensor, output_tensor)
          mse_dif = mse_diff_kernel(src_tensor, output_tensor)
          show_diff_dict['tensor'].append(name)
          show_diff_dict['shape'].append(str(src_tensor.shape))
          show_diff_dict['minmax'].append("("+str(src_tensor.min().item())+','+str(src_tensor.max().item())+")")
          show_diff_dict['cos_diff'].append(cos_dif)
          show_diff_dict['mse_diff'].append(mse_dif)
          print("********: cos diff:",cos_dif, "  mse diff:", mse_dif)


  end0 = time.time()
  print("deal all weight spend : ", (end0 -start0)*1000.0, " ms")        
  print("************* save fakequant dict *************:", output_file_path)
  torch.save(model_dict, output_file_path)
  if show_diff:
    import pandas as pd
    show_diff_dict = pd.DataFrame(show_diff_dict)
    show_diff_dict.to_excel("bloom_weight_diff.xlsx", index=True)


def build_bloom_fakequant_weights_perchannel(input_file_path, output_file_path, intN = 8, skip_embdding = False, show_diff=False):
  print("*************begin load dict *************:", input_file_path)
  model_dict = torch.load(input_file_path)
  print("*************end load dict *************:", input_file_path)
  show_diff_dict = {'tensor':[],'shape':[],'minmax':[],'cos_diff': [],'mse_diff': []}
  absmax = int(pow(2, intN-1) - 1)
  start0 = time.time()
  for name in model_dict.keys():
#     print("----------------:", name, "  shape:", model_dict[name].shape, " data type:",model_dict[name].dtype, "  size:",model_dict[name].numel()/1024/1024, " MB")
#     continue
    if len(model_dict[name].shape) == 2 :
      if "word_embeddings.weight" in name or "lm_head.weight" in name:
        if skip_embdding:
          print("skip weight:", name)
          continue
    if len(model_dict[name].shape) == 2:
      src_tensor = model_dict[name]
      data_type = src_tensor.dtype
      data_size = src_tensor.numel()
      print("----------------:", name, "  shape:", src_tensor.shape, " data type:",data_type, "  size:",data_size/1024/1024, " MB")
      
      if data_type == torch.float16:
        group_size = src_tensor.shape[1]
        print("begin fakequant group:", group_size)
        scalestensor = torch.ones((src_tensor.shape[0]*src_tensor.shape[1]//group_size), dtype = torch.float32).to(device)
        src_tensor = src_tensor.to(device)
        fakequant_cuda.compute_scale_percahnnel_intN_q0(src_tensor , scalestensor, int(group_size), float(absmax))    
        output_tensor = torch.ones(src_tensor.shape, dtype = data_type).to(device)
        fakequant_cuda.fakequant_intN_perchannel_q0_half(src_tensor ,output_tensor, scalestensor, int(group_size), float(absmax))
        #replace

        model_dict[name] = output_tensor.to('cpu')

        if show_diff:
#           src_tensor = src_tensor.to('cpu').numpy()
#           output_tensor = output_tensor.to('cpu').numpy()
#           cos_dif = compute_cos_distance(src_tensor, output_tensor)
#           mse_dif = compute_mse(src_tensor, output_tensor)
          cos_dif = cosine_diff_kernel(src_tensor, output_tensor)
          mse_dif = mse_diff_kernel(src_tensor, output_tensor)
          show_diff_dict['tensor'].append(name)
          show_diff_dict['shape'].append(str(src_tensor.shape))
          show_diff_dict['minmax'].append("("+str(src_tensor.min().item())+','+str(src_tensor.max().item())+")")
          show_diff_dict['cos_diff'].append(cos_dif)
          show_diff_dict['mse_diff'].append(mse_dif)
          print("********: cos diff:",cos_dif, "  mse diff:", mse_dif)


  end0 = time.time()
  print("deal all weight spend : ", (end0 -start0)*1000.0, " ms")        
  print("************* save fakequant dict *************:", output_file_path)
  torch.save(model_dict, output_file_path)
  if show_diff:
    import pandas as pd
    show_diff_dict = pd.DataFrame(show_diff_dict)
    show_diff_dict.to_excel("bloom_weight_diff.xlsx", index=True)


    
def build_bloom_fakequant_weights_pertensor(input_file_path, output_file_path, intN = 8, skip_embdding = False, show_diff=False):
  print("*************begin load dict *************:", input_file_path)
  model_dict = torch.load(input_file_path)
  print("*************end load dict *************:", input_file_path)
  show_diff_dict = {'tensor':[],'shape':[],'minmax':[],'cos_diff': [],'mse_diff': []}
  absmax = int(pow(2, intN-1) - 1)
  start0 = time.time()
  for name in model_dict.keys():
#     print("----------------:", name, "  shape:", model_dict[name].shape, " data type:",model_dict[name].dtype, "  size:",model_dict[name].numel()/1024/1024, " MB")
#     continue
    if len(model_dict[name].shape) == 2 :
      if "word_embeddings.weight" in name or "lm_head.weight" in name:
        if skip_embdding:
          print("skip weight:", name)
          continue
    if len(model_dict[name].shape) == 2:
      src_tensor = model_dict[name]
      data_type = src_tensor.dtype
      data_size = src_tensor.numel()
      print("----------------:", name, "  shape:", src_tensor.shape, " data type:",data_type, "  size:",data_size/1024/1024, " MB")
      
      if data_type == torch.float16:
        group_size = src_tensor.shape[1]
        print("begin fakequant :")
        src_tensor = src_tensor.to(device)
        tmp_min, tmp_max = abs(src_tensor.min().item()), abs(src_tensor.max().item())
        if tmp_min == tmp_max and tmp_max == 0:
            tmp_max = 0.001
        scale = max(tmp_min, tmp_max) / float(absmax) 
        print("scale:", scale)
        output_tensor = torch.ones(src_tensor.shape, dtype = data_type).to(device)
        fakequant_cuda.fakequant_intN_pertensor_q0_half(src_tensor ,output_tensor, scale, float(absmax))
        #replace

        model_dict[name] = output_tensor.to('cpu')

        if show_diff:
#           src_tensor = src_tensor.to('cpu').numpy()
#           output_tensor = output_tensor.to('cpu').numpy()
#           cos_dif = compute_cos_distance(src_tensor, output_tensor)
#           mse_dif = compute_mse(src_tensor, output_tensor)
          cos_dif = cosine_diff_kernel(src_tensor, output_tensor)
          mse_dif = mse_diff_kernel(src_tensor, output_tensor)
          show_diff_dict['tensor'].append(name)
          show_diff_dict['shape'].append(str(src_tensor.shape))
          show_diff_dict['minmax'].append("("+str(src_tensor.min().item())+','+str(src_tensor.max().item())+")")
          show_diff_dict['cos_diff'].append(cos_dif)
          show_diff_dict['mse_diff'].append(mse_dif)
          print("********: cos diff:",cos_dif, "  mse diff:", mse_dif)


  end0 = time.time()
  print("deal all weight spend : ", (end0 -start0)*1000.0, " ms")        
  print("************* save fakequant dict *************:", output_file_path)
  torch.save(model_dict, output_file_path)
  if show_diff:
    import pandas as pd
    show_diff_dict = pd.DataFrame(show_diff_dict)
    show_diff_dict.to_excel("bloom_weight_diff.xlsx", index=True)    
    
    
#input dir  , output dir
def build_bloom_model_bin(input_dir, output_dir, intN =4, group_size=32, skip_embdding = False, show_diff=False):
  dir_files = os.listdir(input_dir)
  for path_file in dir_files:
    abs_path =  input_dir + "/" + path_file
    if os.path.isfile(abs_path) and os.path.exists(abs_path):
      print(path_file)
      if os.path.splitext(path_file)[-1] == ".bin" and "pytorch_model" in path_file:
        input_path = input_dir + "/" + path_file
        ouputput_path = output_dir + "/" + path_file
        build_bloom_fakequant_weights(input_path, ouputput_path, intN=intN, group_size=int(group_size), skip_embdding=skip_embdding, show_diff=show_diff)
    
#input dir  , output dir
def build_bloom_model_bin_perchannel(input_dir, output_dir, intN =8, skip_embdding = False, show_diff=False):
  dir_files = os.listdir(input_dir)
  for path_file in dir_files:
    abs_path =  input_dir + "/" + path_file
    if os.path.isfile(abs_path) and os.path.exists(abs_path):
      print(path_file)
      if os.path.splitext(path_file)[-1] == ".bin" and "pytorch_model" in path_file:
        input_path = input_dir + "/" + path_file
        ouputput_path = output_dir + "/" + path_file
        build_bloom_fakequant_weights_perchannel(input_path, ouputput_path, intN=intN, skip_embdding=skip_embdding, show_diff=show_diff)
    
#input dir  , output dir
def build_bloom_model_bin_pertensor(input_dir, output_dir, intN =8, skip_embdding = False, show_diff=False):
  dir_files = os.listdir(input_dir)
  for path_file in dir_files:
    abs_path =  input_dir + "/" + path_file
    if os.path.isfile(abs_path) and os.path.exists(abs_path):
      print(path_file)
      if os.path.splitext(path_file)[-1] == ".bin" and "pytorch_model" in path_file:
        input_path = input_dir + "/" + path_file
        ouputput_path = output_dir + "/" + path_file
        build_bloom_fakequant_weights_pertensor(input_path, ouputput_path, intN=intN, skip_embdding=skip_embdding, show_diff=show_diff)    
    
def run_bloom_weight_quant(input_dir, output_dir, granularity ='per-group', intN =4, group_size=32, skip_embdding = False,               show_diff=False):

    if granularity == 'per-group':
        build_bloom_model_bin(input_dir, output_dir, intN =intN, group_size=group_size, skip_embdding = skip_embdding, show_diff=show_diff)
    if granularity == 'per-channel':
        build_bloom_model_bin_perchannel(input_dir, output_dir, intN =intN, skip_embdding = skip_embdding, show_diff=show_diff)
    if granularity == 'per-tensor':
        build_bloom_model_bin_pertensor(input_dir, output_dir, intN =intN, skip_embdding = skip_embdding, show_diff=show_diff)
    
if __name__ == '__main__':
#   input_path = "/data/juicefs_sharing_data/11070475/models/bloomz/self7b/pytorch_model_00001-of-00032.bin"
#   output_path = "/data/juicefs_sharing_data/11070475/models/bloomz/test7b/pytorch_model_00001-of-00032.bin"
#   build_bloom_fakequant_weights(input_path, output_path, 32)
  #tmp_test(input_path, output_path)
  #input_dir = "/data/vjuicefs_ai_system_platform/11070475/models/bloom1bself/model"
#   input_dir = "/data/vjuicefs_ai_system_platform/11070475/project/summary_project/model"
#   output_dir = "/data/vjuicefs_ai_system_platform/public_data/llmmodel/tmp"

  #build_bloom_model_bin(input_dir, output_dir, int(32), skip_embdding = True, show_diff=True)

#   input_dir = "/data/vjuicefs_ai_system_platform/11070475/project/summary_project/model_20wSFT_5w_prune_vocab"
#   output_dir =   '/data/vjuicefs_ai_system_platform/11070475/project/summary_project/modelint8_prune'
#   build_bloom_model_bin(input_dir, output_dir, int(8), int(32), skip_embdding = False, show_diff=True)
#   tmp_test("/data/vjuicefs_ai_system_platform/11070475/models/bloom1bself/model/pytorch_model.bin",\
#            "/data/vjuicefs_ai_system_platform/11070475/project/summary_project/model2/pytorch_model.bin")

#   input_dir = "/data/vjuicefs_ai_system_platform/11070475/project/summary_project/model_20wSFT_5w_prune_vocab"
#   output_dir =   '/data/vjuicefs_ai_system_platform/11070475/project/summary_project/modelint8perchannel_prune'
#   build_bloom_model_bin_perchannel(input_dir, output_dir, int(8), skip_embdding = False, show_diff=True)
    pass
#   input_dir = "/data/vjuicefs_ai_system_platform/11070475/project/summary_project/model_20wSFT_5w_prune_vocab"
#   output_dir =   '/data/vjuicefs_ai_system_platform/11070475/project/summary_project/modelint8pertensor_prune'
#   build_bloom_model_bin_pertensor(input_dir, output_dir, int(8), skip_embdding = False, show_diff=True)    
