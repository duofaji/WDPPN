# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 11:00:07 2025

@author: fangwenji
"""

import torch
import numpy as np
import h5py
from tqdm import tqdm
import sys
import os

try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIR = os.getcwd()
    
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SAFARI_ROOT = os.path.join(PROJECT_ROOT, "safari")

# 加入 sys.path
sys.path.append(PROJECT_ROOT)
sys.path.append(SAFARI_ROOT)

from models.model_s import model_s
from detecta import detect_peaks

def generate_arrays_from_file(file_list, step):
    
    n_loops = int(np.ceil(len(file_list) / step))
    b = 0
    while True:
        for i in range(n_loops):
            e = i*step + step 
            if e > len(file_list):
                e = len(file_list)
            chunck = file_list[b:e]
            b=e
            yield chunck

def test_model(model, trace_test_names, data_paths, batch_size, device):
    ts_pre = torch.zeros((1, 6000), dtype=torch.float32)
    ts_true = []
    
    list_generator = generate_arrays_from_file(trace_test_names, batch_size)                
    
    for it in tqdm(range(int(np.ceil(len(trace_test_names) / batch_size)))):
        data = h5py.File(data_paths,'r')
        new_list = next(list_generator)
    
        data_test_batch = np.zeros((len(new_list), 3, 6000))
        
        for i in range(0,int(len(new_list))):
            data_test_batch[i] = np.array(data.get('data/'+new_list[i])).transpose(1,0)
            
            dataset = data.get('data/'+new_list[i])
            ts = dataset.attrs['s_arrival_sample']
            ts_true.append(ts)
            
        data.close()
        
        data_test_batch_t = torch.tensor(data_test_batch, dtype = torch.float32).to(device)
        
        ts_t, y_denoised = model(data_test_batch_t)
        
        ts_t = torch.nn.Sigmoid()(ts_t)
        
        ts_predict = torch.squeeze(ts_t).detach().cpu()
        
        ts_pre = torch.cat((ts_pre, ts_predict),dim=0)
        
    ts_predict = ts_pre[1:].numpy()
    ts_true = np.array(ts_true)
    
    return ts_predict, ts_true

def evaluate_picks(predict, truth):
    TP, FP, FN, Miss = 0, 0, 0, 0
    TP_pre, TP_true = [], []

    for i in tqdm(range(len(predict))):
        picks = detect_peaks(predict[i], mph=0.325, mpd=1, show=False)
        picks_in_range = []
        probs_in_range = []
        
        if len(picks) == 0:
            Miss += 1
        
        for pick in picks:
            if abs(pick - truth[i]) <= 50:
                picks_in_range.append(pick)
                probs_in_range.append(predict[i][pick])
            else:
                FP += 1

        if len(picks_in_range) == 0:
            FN += 1
        else:
            best_idx = np.argmax(probs_in_range)
            TP_pick = picks_in_range[best_idx]
            TP += 1
            TP_pre.append(TP_pick)
            TP_true.append(truth[i])
            
    return TP, FP, FN, TP_pre, TP_true, Miss

def evaluate_result(ts_predict_all, ts_true_all):
    TP_s, FP_s, FN_s, TP_pre_s, TP_true_s, Miss_s = evaluate_picks(ts_predict_all, ts_true_all)

    precision_s = TP_s / (TP_s + FP_s)
    recall_s = TP_s / (TP_s + FN_s)
    F1scores_s = 2 *(precision_s*recall_s)/(precision_s + recall_s)

    s_error = (np.array(TP_true_s) - np.array(TP_pre_s))*0.01
    s_mae = np.mean(np.abs(s_error))
    s_std = np.std(s_error)
    
    return precision_s, recall_s, F1scores_s, s_mae, s_std, Miss_s

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_s().to(device)
    checkpoint = torch.load('../pretrained_model_parameters/parameters_s.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    trace_test_names_8 = np.load('../sample_data/noisy_data/s/trace_use_8.npy')
    trace_test_names_12 = np.load('../sample_data/noisy_data/s/trace_use_12.npy')
    trace_test_names_16 = np.load('../sample_data/noisy_data/s/trace_use_16.npy')
    trace_test_names_20 = np.load('../sample_data/noisy_data/s/trace_use_20.npy')
    
    data_paths_8 = '../sample_data/noisy_data/s/data_8.hdf5'
    data_paths_12 = '../sample_data/noisy_data/s/data_12.hdf5'
    data_paths_16 = '../sample_data/noisy_data/s/data_16.hdf5'
    data_paths_20 = '../sample_data/noisy_data/s/data_20.hdf5'

    batch_size = 16

    ts_predict_all_8, ts_label_all_8 = test_model(model, trace_test_names_8, data_paths_8, batch_size, device)
    ts_predict_all_12, ts_label_all_12 = test_model(model, trace_test_names_12, data_paths_12, batch_size, device)
    ts_predict_all_16, ts_label_all_16 = test_model(model, trace_test_names_16, data_paths_16, batch_size, device)
    ts_predict_all_20, ts_label_all_20 = test_model(model, trace_test_names_20, data_paths_20, batch_size, device)
    
    precision_s_8, recall_s_8, F1scores_s_8, s_mae_8, s_std_8, Miss_s_8 = evaluate_result(ts_predict_all_8, ts_label_all_8)
    precision_s_12, recall_s_12, F1scores_s_12, s_mae_12, s_std_12, Miss_s_12 = evaluate_result(ts_predict_all_12, ts_label_all_12)
    precision_s_16, recall_s_16, F1scores_s_16, s_mae_16, s_std_16, Miss_s_16 = evaluate_result(ts_predict_all_16, ts_label_all_16)
    precision_s_20, recall_s_20, F1scores_s_20, s_mae_20, s_std_20, Miss_s_20 = evaluate_result(ts_predict_all_20, ts_label_all_20)
    
    print('Test Results\n')
    print('8 dB')
    print(f"Mean Absolute Error (s): {s_mae_8:.4f}")
    print(f"F1 Score: {F1scores_s_8:.4f}")
    print(f"Miss number: {Miss_s_8}\n")
    print('12 dB')
    print(f"Mean Absolute Error (s): {s_mae_12:.4f}")
    print(f"F1 Score: {F1scores_s_12:.4f}")
    print(f"Miss number: {Miss_s_12}\n")
    print('16 dB')
    print(f"Mean Absolute Error (s): {s_mae_16:.4f}")
    print(f"F1 Score: {F1scores_s_16:.4f}")
    print(f"Miss number: {Miss_s_16}\n")
    print('20 dB')
    print(f"Mean Absolute Error (s): {s_mae_20:.4f}")
    print(f"F1 Score: {F1scores_s_20:.4f}")
    print(f"Miss number: {Miss_s_20}\n")
