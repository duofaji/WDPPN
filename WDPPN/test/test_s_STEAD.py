# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 15:48:55 2025

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

def test_model(model, trace_test_names, data_paths, label_paths, batch_size, device):
    ts_pre = torch.zeros((1, 6000), dtype=torch.float32)
    ts_true = torch.zeros((1, 6000), dtype=torch.float32)

    for trace_list, data_path, label_path in zip(trace_test_names, data_paths, label_paths):
        list_generator = generate_arrays_from_file(trace_list, batch_size)
        for _ in tqdm(range(int(np.ceil(len(trace_list) / batch_size)))):
            new_list = next(list_generator)
            
            with h5py.File(data_path, 'r') as data_file, h5py.File(label_path, 'r') as label_file:
                data_test_batch = np.zeros((len(new_list), 3, 6000))
                ts_test_batch = np.zeros((len(new_list), 1, 6000))
                
                for i, key in enumerate(new_list):
                    data_test_batch[i] = np.array(data_file['data/' + key]).transpose(1, 0)
                    ts_test_batch[i] = label_file['ts/' + key][0, :, :]
            
            data_test_batch = torch.tensor(data_test_batch, dtype=torch.float32).to(device)
            ts_test_batch = torch.tensor(ts_test_batch, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                ts_predict, y_denoised = model(data_test_batch)
            
            ts_predict = torch.sigmoid(ts_predict)
            ts_predict = ts_predict.squeeze().cpu()
            ts_test_batch = ts_test_batch.squeeze().cpu()
            
            ts_pre = torch.cat((ts_pre, ts_predict), dim=0)
            ts_true = torch.cat((ts_true, ts_test_batch), dim=0)
    
    return ts_pre[1:].numpy(), ts_true[1:].numpy()

def evaluate_picks(predict, truth):
    TP, FP, FN = 0, 0, 0
    TP_pre, TP_true = [], []

    for i in range(len(predict)):
        picks = detect_peaks(predict[i], mph=0.325, mpd=1, show=False)
        picks_in_range = []
        probs_in_range = []

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

    return TP, FP, FN, TP_pre, TP_true

def evaluate_result(ts_predict_all, ts_label_all):
    ts_true = []
    for i in range(ts_label_all.shape[0]):
        ts_true.append(np.argmax(ts_label_all[i]))
    ts_true = np.array(ts_true)
    
    TP_s, FP_s, FN_s, TP_pre_s, TP_true_s = evaluate_picks(ts_predict_all, ts_true)
    
    precision_s = TP_s / (TP_s + FP_s)
    recall_s = TP_s / (TP_s + FN_s)
    F1scores_s = 2 *(precision_s*recall_s)/(precision_s + recall_s)

    s_error = (np.array(TP_true_s) - np.array(TP_pre_s))*0.01
    s_mae = np.mean(np.abs(s_error))
    s_std = np.std(s_error)
    
    return precision_s, recall_s, F1scores_s, s_mae, s_std
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_s().to(device)
    checkpoint = torch.load('../pretrained_model_parameters/parameters_s.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    trace_test_name1 = np.load('../sample_data/STEAD/trace_test1.npy')
    
    trace_test_names = [trace_test_name1]
    data_paths = [
        '../sample_data/STEAD/data1.hdf5'
    ]
    label_paths = [
        '../sample_data/STEAD/label1.hdf5'
    ]
    
    batchsize = 16

    ts_predict_all, ts_label_all = test_model(model, trace_test_names, data_paths, label_paths, batchsize, device)
    
    precision_s, recall_s, F1scores_s, s_mae, s_std = evaluate_result(ts_predict_all, ts_label_all)

    print(f"Precision: {precision_s:.4f}")
    print(f"Recall: {recall_s:.4f}")
    print(f"F1 Score: {F1scores_s:.4f}")
    print(f"Mean Absolute Error (s): {s_mae:.4f}")
    print(f"Standard Deviation (s): {s_std:.4f}")
