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
    
    list_generator = generate_arrays_from_file(trace_test_names, batch_size)                
    
    for it in tqdm(range(int(np.ceil(len(trace_test_names) / batch_size)))):
        data = h5py.File(data_paths,'r')
        new_list = next(list_generator)
    
        data_test_batch = np.zeros((len(new_list), 3, 6000))
        
        for i in range(0,int(len(new_list))):
            data_test_batch[i] = np.array(data.get('data/'+new_list[i])).transpose(1,0)
            
        data.close()
        
        data_test_batch_t = torch.tensor(data_test_batch, dtype = torch.float32).to(device)
        
        ts_t, y_denoised = model(data_test_batch_t)
        
        ts_t = torch.nn.Sigmoid()(ts_t)
        
        ts_predict = torch.squeeze(ts_t,dim=1).detach().cpu()
        
        ts_pre = torch.cat((ts_pre, ts_predict),dim=0)
        
    ts_predict = ts_pre[1:].numpy()
    
    return ts_predict

def label_cal(origin_path, trace_origin_name):
    s_label = {}
    data_japan = h5py.File(origin_path, 'r')
    
    for i in range(100):
        name = trace_origin_name[i]
        ts = np.array(data_japan.get('ts/'+name)).tolist()
        ts_UTC = [UTCDateTime(t) for t in ts]
        s_label[name] = ts_UTC
    
    return s_label

def filter_picks(times):
    if len(times) <= 1:
        return times
    
    else:
        result = [times[0]]

        for i in range(1, len(times)):
            if times[i] - result[-1] > 200:
                result.append(times[i])

    return result

def evaluate_picks(ts_pick_f_UTC, ts_label_UTC, ts_pro_f, tol=0.5):
    TP = []
    FP = []
    FN = []

    matched_best = set()

    for label in ts_label_UTC:
        in_tol = [i for i, pick in enumerate(ts_pick_f_UTC)
                  if abs(pick - label) <= tol]
        if in_tol:
            best_idx = max(in_tol, key=lambda i: ts_pro_f[i])
            TP.append((label, ts_pick_f_UTC[best_idx], ts_pro_f[best_idx]))
            matched_best.add(best_idx)
        else:
            FN.append(label)

    for idx, pick in enumerate(ts_pick_f_UTC):
        if not any(abs(pick - label) <= tol for label in ts_label_UTC):
            FP.append((pick, ts_pro_f[idx]))

    return TP, FP, FN
    
def evaluate_result(window_number, ts_predict, origin_path, trace_origin_name):
    s_result = {}

    start_index = 0
    
    data_japan = h5py.File(origin_path, 'r')
    s_label = label_cal(origin_path, trace_origin_name)

    for i in range(100):
        size = window_number[i]
        end_index = int(start_index + size)
        ts_group = ts_predict[start_index:end_index]
        
        if ts_group.shape[0] > 1:
            candidate_dict = {}
            for j in range(ts_group.shape[0]-1):
                picks = detect_peaks(ts_group[j], mph=0.325, mpd=1, show=False)
                for local_idx in picks:
                    abs_index = local_idx + 1200 * j
                    prob = ts_group[j][local_idx].item()
                    if abs_index in candidate_dict:
                        candidate_dict[abs_index] = max(candidate_dict[abs_index], prob)
                    else:
                        candidate_dict[abs_index] = prob
            
            wave = np.array(data_japan.get('data/'+trace_origin_name[i]))
            wave_starttime = UTCDateTime(data_japan.get('data/'+trace_origin_name[i]).attrs['start_time'])
            
            picks = detect_peaks(ts_group[-1], mph=0.325, mpd=1, show=False)
            for local_idx in picks:
                abs_index = local_idx + (wave.shape[0]-6000)
                prob = ts_group[j][local_idx].item()
                if abs_index in candidate_dict:
                    candidate_dict[abs_index] = max(candidate_dict[abs_index], prob)
                else:
                    candidate_dict[abs_index] = prob
        else:
            candidate_dict = {}
            for j in range(ts_group.shape[0]):
                picks = detect_peaks(ts_group[j], mph=0.325, mpd=1, show=False)
                for local_idx in picks:
                    abs_index = local_idx + 1200 * j
                    prob = ts_group[j][local_idx].item()
                    if abs_index in candidate_dict:
                        candidate_dict[abs_index] = max(candidate_dict[abs_index], prob)
                    else:
                        candidate_dict[abs_index] = prob

        all_candidates = sorted(candidate_dict.keys())
        ts_pick_f = filter_picks(all_candidates)
        
        ts_pro_f = []
        for k in range(len(ts_pick_f)):
            ts_pro_f.append(candidate_dict[ts_pick_f[k]])
        
        ts_pick_f_UTC = [wave_starttime + ts*0.01 for ts in ts_pick_f]
        
        ts_label_UTC = s_label[trace_origin_name[i]]
        
        TP, FP, FN = evaluate_picks(ts_pick_f_UTC, ts_label_UTC, ts_pro_f)
        
        s_result[trace_origin_name[i]] = {'TP':TP, 'FP': FP, 'FN': FN}
        
        start_index = end_index
        
    TP_s_number = 0
    FP_s_number = 0
    FN_s_number = 0
    TP_error_s = []

    for i in range(100):
        name = trace_origin_name[i]
        TP = s_result[name]['TP']
        FP = s_result[name]['FP']
        FN = s_result[name]['FN']
        TP_s_number += len(TP)
        FP_s_number += len(FP)
        FN_s_number += len(FN)
        
        for j in range(len(TP)):
            ts_true = TP[j][0]
            ts_pre = TP[j][1]
            ts_error = ts_true - ts_pre
            TP_error_s.append(ts_error)
            
    precision_s = TP_s_number / (TP_s_number + FP_s_number)
    recall_s = TP_s_number / (TP_s_number + FN_s_number)
    F1scores_s = 2 *(precision_s*recall_s)/(precision_s + recall_s)

    s_mae = np.mean(np.abs(TP_error_s))
    s_std = np.std(TP_error_s)
    
    return precision_s, recall_s, F1scores_s, s_mae, s_std

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_s().to(device)
    checkpoint = torch.load('../pretrained_model_parameters/parameters_s.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    trace_test_names = np.load('../sample_data/Japan/window_names.npy')
    window_number = np.load('../sample_data/Japan/window_counts.npy')
    trace_origin_name = np.load('../sample_data/Japan/trace.npy')
    
    origin_path = '../sample_data/Japan/origin_records.hdf5'
    data_paths = '../sample_data/Japan/records_segment.hdf5'

    batch_size = 16

    ts_predict_all = test_model(model, trace_test_names, data_paths, batch_size, device)
    
    precision_s, recall_s, F1scores_s, s_mae, s_std = evaluate_result(window_number, ts_predict_all, origin_path, trace_origin_name)

    print(f"Precision: {precision_s:.4f}")
    print(f"Recall: {recall_s:.4f}")
    print(f"F1 Score: {F1scores_s:.4f}")
    print(f"Mean Absolute Error (s): {s_mae:.4f}")
    print(f"Standard Deviation (s): {s_std:.4f}")