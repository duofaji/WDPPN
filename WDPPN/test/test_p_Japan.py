# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 10:36:26 2025

@author: fangwenji
"""

import torch
import numpy as np
import h5py
from tqdm import tqdm
from obspy import UTCDateTime
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
    
    list_generator = generate_arrays_from_file(trace_test_names, batch_size)                
    
    for it in tqdm(range(int(np.ceil(len(trace_test_names) / batch_size)))):
        data = h5py.File(data_paths,'r')
        new_list = next(list_generator)
    
        data_test_batch = np.zeros((len(new_list), 3, 6000))
        
        for i in range(0,int(len(new_list))):
            data_test_batch[i] = np.array(data.get('data/'+new_list[i])).transpose(1,0)
            
        data.close()
        
        data_test_batch_t = torch.tensor(data_test_batch, dtype = torch.float32).to(device)
        
        tp_t, y_denoised = model(data_test_batch_t)
        
        tp_t = torch.nn.Sigmoid()(tp_t)
        
        tp_predict = torch.squeeze(tp_t,dim=1).detach().cpu()
        
        tp_pre = torch.cat((tp_pre, tp_predict),dim=0)
        
    tp_predict = tp_pre[1:].numpy()
    
    return tp_predict

def label_cal(origin_path, trace_origin_name):
    p_label = {}
    data_japan = h5py.File(origin_path, 'r')
    
    for i in range(100):
        name = trace_origin_name[i]
        tp = np.array(data_japan.get('tp/'+name)).tolist()
        tp_UTC = [UTCDateTime(t) for t in tp]
        p_label[name] = tp_UTC
    
    return p_label

def filter_picks(times):
    if len(times) <= 1:
        return times
    
    else:
        result = [times[0]]

        for i in range(1, len(times)):
            if times[i] - result[-1] > 200:
                result.append(times[i])

    return result

def evaluate_picks(tp_pick_f_UTC, tp_label_UTC, tp_pro_f, tol=0.5):
    TP = []
    FP = []
    FN = []

    matched_best = set()

    for label in tp_label_UTC:
        in_tol = [i for i, pick in enumerate(tp_pick_f_UTC)
                  if abs(pick - label) <= tol]
        if in_tol:
            best_idx = max(in_tol, key=lambda i: tp_pro_f[i])
            TP.append((label, tp_pick_f_UTC[best_idx], tp_pro_f[best_idx]))
            matched_best.add(best_idx)
        else:
            FN.append(label)

    for idx, pick in enumerate(tp_pick_f_UTC):
        if not any(abs(pick - label) <= tol for label in tp_label_UTC):
            FP.append((pick, tp_pro_f[idx]))

    return TP, FP, FN
    
def evaluate_result(window_number, tp_predict, origin_path, trace_origin_name):
    p_result = {}

    start_index = 0
    
    data_japan = h5py.File(origin_path, 'r')
    p_label = label_cal(origin_path, trace_origin_name)

    for i in range(100):
        size = window_number[i]
        end_index = int(start_index + size)
        tp_group = tp_predict[start_index:end_index]
        
        if tp_group.shape[0] > 1:
            candidate_dict = {}
            for j in range(tp_group.shape[0]-1):
                picks = detect_peaks(tp_group[j], mph=0.3, mpd=1, show=False)
                for local_idx in picks:
                    abs_index = local_idx + 1200 * j
                    prob = tp_group[j][local_idx].item()
                    if abs_index in candidate_dict:
                        candidate_dict[abs_index] = max(candidate_dict[abs_index], prob)
                    else:
                        candidate_dict[abs_index] = prob
            
            wave = np.array(data_japan.get('data/'+trace_origin_name[i]))
            wave_starttime = UTCDateTime(data_japan.get('data/'+trace_origin_name[i]).attrs['start_time'])
            
            picks = detect_peaks(tp_group[-1], mph=0.3, mpd=1, show=False)
            for local_idx in picks:
                abs_index = local_idx + (wave.shape[0]-6000)
                prob = tp_group[j][local_idx].item()
                if abs_index in candidate_dict:
                    candidate_dict[abs_index] = max(candidate_dict[abs_index], prob)
                else:
                    candidate_dict[abs_index] = prob
        else:
            candidate_dict = {}
            for j in range(tp_group.shape[0]):
                picks = detect_peaks(tp_group[j], mph=0.3, mpd=1, show=False)
                for local_idx in picks:
                    abs_index = local_idx + 1200 * j
                    prob = tp_group[j][local_idx].item()
                    if abs_index in candidate_dict:
                        candidate_dict[abs_index] = max(candidate_dict[abs_index], prob)
                    else:
                        candidate_dict[abs_index] = prob

        all_candidates = sorted(candidate_dict.keys())
        tp_pick_f = filter_picks(all_candidates)
        
        tp_pro_f = []
        for k in range(len(tp_pick_f)):
            tp_pro_f.append(candidate_dict[tp_pick_f[k]])
        
        tp_pick_f_UTC = [wave_starttime + tp*0.01 for tp in tp_pick_f]
        
        tp_label_UTC = p_label[trace_origin_name[i]]
        
        TP, FP, FN = evaluate_picks(tp_pick_f_UTC, tp_label_UTC, tp_pro_f)
        
        p_result[trace_origin_name[i]] = {'TP':TP, 'FP': FP, 'FN': FN}
        
        start_index = end_index
        
    TP_p_number = 0
    FP_p_number = 0
    FN_p_number = 0
    TP_error_p = []

    for i in range(100):
        name = trace_origin_name[i]
        TP = p_result[name]['TP']
        FP = p_result[name]['FP']
        FN = p_result[name]['FN']
        TP_p_number += len(TP)
        FP_p_number += len(FP)
        FN_p_number += len(FN)
        
        for j in range(len(TP)):
            tp_true = TP[j][0]
            tp_pre = TP[j][1]
            tp_error = tp_true - tp_pre
            TP_error_p.append(tp_error)
            
    precision_p = TP_p_number / (TP_p_number + FP_p_number)
    recall_p = TP_p_number / (TP_p_number + FN_p_number)
    F1scores_p = 2 *(precision_p*recall_p)/(precision_p + recall_p)

    p_mae = np.mean(np.abs(TP_error_p))
    p_std = np.std(TP_error_p)
    
    return precision_p, recall_p, F1scores_p, p_mae, p_std

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_p().to(device)
    checkpoint = torch.load('../pretrained_model_parameters/parameters_p.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    trace_test_names = np.load('../sample_data/Japan/window_names.npy')
    window_number = np.load('../sample_data/Japan/window_counts.npy')
    trace_origin_name = np.load('../sample_data/Japan/trace.npy')
    
    origin_path = '../sample_data/Japan/origin_records.hdf5'
    data_paths = '../sample_data/Japan/records_segment.hdf5'

    batch_size = 16

    tp_predict_all = test_model(model, trace_test_names, data_paths, batch_size, device)
    
    precision_p, recall_p, F1scores_p, p_mae, p_std = evaluate_result(window_number, tp_predict_all, origin_path, trace_origin_name)

    print(f"Precision: {precision_p:.4f}")
    print(f"Recall: {recall_p:.4f}")
    print(f"F1 Score: {F1scores_p:.4f}")
    print(f"Mean Absolute Error (s): {p_mae:.4f}")
    print(f"Standard Deviation (s): {p_std:.4f}")