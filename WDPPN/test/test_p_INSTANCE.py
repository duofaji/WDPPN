# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 16:21:49 2025

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

def test_model(model, trace_test_names, data_paths, batch_size, device):
    tp_pre = torch.zeros((1, 6000), dtype=torch.float32)
    tp_true = []
    
    list_generator = generate_arrays_from_file(trace_test_names, batch_size)                
    
    for it in tqdm(range(int(np.ceil(len(trace_test_names) / batch_size)))):
        data = h5py.File(data_paths,'r')
        new_list = next(list_generator)
    
        data_test_batch = np.zeros((len(new_list), 3, 6000))
        
        for i in range(0,int(len(new_list))):
            data_test_batch[i] = np.array(data.get('data/'+new_list[i])).transpose(1,0)
            
            dataset = data.get('data/'+new_list[i])
            tp = dataset.attrs['p_arrival_sample']
            tp_true.append(tp)
            
        data.close()
        
        data_test_batch_t = torch.tensor(data_test_batch, dtype = torch.float32).to(device)
        
        tp_t, y_denoised = model(data_test_batch_t)
        
        tp_t = torch.nn.Sigmoid()(tp_t)
        
        tp_predict = torch.squeeze(tp_t).detach().cpu()
        
        tp_pre = torch.cat((tp_pre, tp_predict),dim=0)
        
    tp_predict = tp_pre[1:].numpy()
    tp_true = np.array(tp_true)
    tp_true = tp_true - 1
    
    return tp_predict, tp_true

def evaluate_picks(predict, truth, threshold=0.3, tol=50):
    TP, FP, FN = 0, 0, 0
    TP_pre, TP_true = [], []

    for i in range(len(predict)):
        picks = detect_peaks(predict[i], mph=threshold, mpd=1, show=False)

        picks_in_range = []
        probs_in_range = []

        for pick in picks:
            if abs(pick - truth[i]) <= tol:
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

def evaluate_result(tp_predict_all, tp_true_all):
    TP_p, FP_p, FN_p, TP_pre_p, TP_true_p = evaluate_picks(tp_predict_all, tp_true_all)

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

    trace_test_names = np.load('../dataset/INSTANCE/trace.npy')
    
    data_paths = '../dataset/INSTANCE/data_instance.hdf5'

    batch_size = 128

    tp_predict_all, tp_label_all = test_model(model, trace_test_names, data_paths, batch_size, device)
    
    precision_p, recall_p, F1scores_p, p_mae, p_std = evaluate_result(tp_predict_all, tp_label_all)

    print(f"Precision: {precision_p:.4f}")
    print(f"Recall: {recall_p:.4f}")
    print(f"F1 Score: {F1scores_p:.4f}")
    print(f"Mean Absolute Error (s): {p_mae:.4f}")
    print(f"Standard Deviation (s): {p_std:.4f}")