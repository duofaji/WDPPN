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

from models.model_p import model_p
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
    tp_pre = torch.zeros((1, 6000), dtype=torch.float32)
    tp_true = torch.zeros((1, 6000), dtype=torch.float32)

    for trace_list, data_path, label_path in zip(trace_test_names, data_paths, label_paths):
        list_generator = generate_arrays_from_file(trace_list, batch_size)
        for _ in tqdm(range(int(np.ceil(len(trace_list) / batch_size)))):
            new_list = next(list_generator)
            
            with h5py.File(data_path, 'r') as data_file, h5py.File(label_path, 'r') as label_file:
                data_test_batch = np.zeros((len(new_list), 3, 6000))
                tp_test_batch = np.zeros((len(new_list), 1, 6000))
                
                for i, key in enumerate(new_list):
                    data_test_batch[i] = np.array(data_file['data/' + key]).transpose(1, 0)
                    tp_test_batch[i] = label_file['tp/' + key][0, :, :]
            
            data_test_batch = torch.tensor(data_test_batch, dtype=torch.float32).to(device)
            tp_test_batch = torch.tensor(tp_test_batch, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                tp_predict, y_denoised = model(data_test_batch)
            
            tp_predict = torch.sigmoid(tp_predict)
            tp_predict = tp_predict.squeeze().cpu()
            tp_test_batch = tp_test_batch.squeeze().cpu()
            
            tp_pre = torch.cat((tp_pre, tp_predict), dim=0)
            tp_true = torch.cat((tp_true, tp_test_batch), dim=0)
    
    return tp_pre[1:].numpy(), tp_true[1:].numpy()

def evaluate_picks(predict, truth):
    TP, FP, FN = 0, 0, 0
    TP_pre, TP_true = [], []

    for i in range(len(predict)):
        picks = detect_peaks(predict[i], mph=0.3, mpd=1, show=False)
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

def evaluate_result(tp_predict_all, tp_label_all):
    tp_true = []
    for i in range(tp_label_all.shape[0]):
        tp_true.append(np.argmax(tp_label_all[i]))
    tp_true = np.array(tp_true)
    
    TP_p, FP_p, FN_p, TP_pre_p, TP_true_p = evaluate_picks(tp_predict_all, tp_true)
    
    precision_p = TP_p / (TP_p + FP_p)
    recall_p = TP_p / (TP_p + FN_p)
    F1scores_p = 2 *(precision_p*recall_p)/(precision_p + recall_p)

    p_error = (np.array(TP_true_p) - np.array(TP_pre_p))*0.01
    p_mae = np.mean(np.abs(p_error))
    p_std = np.std(p_error)
    
    return precision_p, recall_p, F1scores_p, p_mae, p_std
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_p().to(device)
    checkpoint = torch.load('../pretrained_model_parameters/parameters_p.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    trace_test_name1 = np.load('../dataset/STEAD/trace_test1.npy')
    trace_test_name2 = np.load('../dataset/STEAD/trace_test2.npy')
    trace_test_name3 = np.load('../dataset/STEAD/trace_test3.npy')
    trace_test_name4 = np.load('../dataset/STEAD/trace_test4.npy')
    trace_test_name5 = np.load('../dataset/STEAD/trace_test5.npy')
    
    trace_test_names = [trace_test_name1, trace_test_name2, trace_test_name3, trace_test_name4, trace_test_name5]
    data_paths = [
        '../dataset/STEAD/data1.hdf5',
        '../dataset/STEAD/data2.hdf5',
        '../dataset/STEAD/data3.hdf5',
        '../dataset/STEAD/data4.hdf5',
        '../dataset/STEAD/data5.hdf5'
    ]
    label_paths = [
        '../dataset/STEAD/label1.hdf5',
        '../dataset/STEAD/label2.hdf5',
        '../dataset/STEAD/label3.hdf5',
        '../dataset/STEAD/label4.hdf5',
        '../dataset/STEAD/label5.hdf5'
    ]
    
    batch_size = 128

    tp_predict_all, tp_label_all = test_model(model, trace_test_names, data_paths, label_paths, batch_size, device)
    
    precision_p, recall_p, F1scores_p, p_mae, p_std = evaluate_result(tp_predict_all, tp_label_all)

    print(f"Precision: {precision_p:.4f}")
    print(f"Recall: {recall_p:.4f}")
    print(f"F1 Score: {F1scores_p:.4f}")
    print(f"Mean Absolute Error (s): {p_mae:.4f}")
    print(f"Standard Deviation (s): {p_std:.4f}")
