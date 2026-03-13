import torch
from .utils import * 
from .model import LayerNormNet
from .distance_map import *
from .evaluate import *
import pandas as pd
import warnings

import numpy as np

def warn(*args, **kwargs):
    pass
warnings.warn = warn

def infer_pvalue(train_data, test_data, p_value = 1e-5, nk_random = 20, 
                 report_metrics = False, pretrained=True, model_name=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    id_ec_train, ec_id_dict_train = get_ec_id_dict('./data/' + train_data + '.csv')
    id_ec_test, _ = get_ec_id_dict('./data/' + test_data + '.csv')
    # load checkpoints
    # NOTE: change this to LayerNormNet(512, 256, device, dtype) 
    # and rebuild with [python build.py install]
    # if inferencing on model trained with supconH loss
    model = LayerNormNet(512, 128, device, dtype)
    
    if pretrained:
        try:
            checkpoint = torch.load('./data/pretrained/'+ train_data +'.pth', map_location=device)
        except FileNotFoundError as error:
            raise Exception('No pretrained weights for this training data')
    else:
        try:
            checkpoint = torch.load('./data/model/'+ model_name +'.pth', map_location=device)
        except FileNotFoundError as error:
            raise Exception('No model found!')
        
    model.load_state_dict(checkpoint)
    model.eval()
    # load precomputed EC cluster center embeddings if possible
    if train_data == "split70":
        emb_train = torch.load('./data/pretrained/70.pt', map_location=device)
    elif train_data == "split100":
        emb_train = torch.load('./data/pretrained/100.pt', map_location=device)
    else:
        emb_train = model(esm_embedding(ec_id_dict_train, device, dtype))
        
    emb_test = model_embedding_test(id_ec_test, model, device, dtype)
    eval_dist = get_dist_map_test(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)
    seed_everything()
    eval_df = pd.DataFrame.from_dict(eval_dist)
    rand_nk_ids, rand_nk_emb_train = random_nk_model(
        id_ec_train, ec_id_dict_train, emb_train, n=nk_random, weighted=True)
    random_nk_dist_map = get_random_nk_dist_map(
        emb_train, rand_nk_emb_train, ec_id_dict_train, rand_nk_ids, device, dtype)
    ensure_dirs("./results")
    out_filename = "results/" +  test_data
    write_pvalue_choices( eval_df, out_filename, random_nk_dist_map, p_value=p_value)
    # optionally report prediction precision/recall/...
    if report_metrics:
        pred_label = get_pred_labels(out_filename, pred_type='_pvalue')
        pred_probs = get_pred_probs(out_filename, pred_type='_pvalue')
        true_label, all_label = get_true_labels('./data/' + test_data)
        pre, rec, f1, roc, acc = get_eval_metrics(
            pred_label, pred_probs, true_label, all_label)
        print(f'############ EC calling results using random '
        f'chosen {nk_random}k samples ############')
        print('-' * 75)
        print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)} \n'
            f'>>> precision: {pre:.3} | recall: {rec:.3}'
            f'| F1: {f1:.3} | AUC: {roc:.3} ')
        print('-' * 75)  
    

def infer_maxsep(train_data, test_data, report_metrics=False,
                 pretrained=True, model_name=None, gmm=None):
    import os
    import json
    import pickle
    import pandas as pd
    import csv

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32

    id_ec_train, ec_id_dict_train = get_ec_id_dict('./data/' + train_data + '.csv')
    id_ec_test, _ = get_ec_id_dict('./data/' + test_data + '.csv')

    model = LayerNormNet(512, 128, device, dtype)

    if pretrained:
        try:
            checkpoint = torch.load('./data/pretrained/' + train_data + '.pth', map_location=device)
        except FileNotFoundError:
            raise Exception('No pretrained weights for this training data')
    else:
        try:
            checkpoint = torch.load('./data/model/' + model_name + '.pth', map_location=device)
        except FileNotFoundError:
            raise Exception('No model found!')

    model.load_state_dict(checkpoint)
    model.eval()

    if train_data == "split70":
        emb_train = torch.load('./data/pretrained/70.pt', map_location=device)
    elif train_data == "split100":
        emb_train = torch.load('./data/pretrained/100.pt', map_location=device)
    else:
        emb_train = model(esm_embedding(ec_id_dict_train, device, dtype))

    emb_test = model_embedding_test(id_ec_test, model, device, dtype)
    eval_dist = get_dist_map_test(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)

    seed_everything()
    eval_df = pd.DataFrame.from_dict(eval_dist)

    ensure_dirs("./results")
    ensure_dirs("./results/clean_baseline")
    ensure_dirs("./artifacts")
    ensure_dirs("./artifacts/clean_baseline")

    # 1) 保存最原始距离矩阵
    eval_df.to_csv(f"./artifacts/clean_baseline/{test_data}_eval_df.csv")

    # 2) 保持原始 maxsep 输出
    out_filename = "results/" + test_data
    write_max_sep_choices(eval_df, out_filename, gmm=gmm)

    # 3) 复制/另存到 clean_baseline 目录，便于后面统一管理
    pred_label = get_pred_labels(out_filename, pred_type='_maxsep')
    pred_probs = get_pred_probs(out_filename, pred_type='_maxsep')
    true_label, all_label = get_true_labels('./data/' + test_data)

    with open(f"./artifacts/clean_baseline/{test_data}_pred_label.pkl", "wb") as f:
        pickle.dump(pred_label, f)

    with open(f"./artifacts/clean_baseline/{test_data}_pred_probs.pkl", "wb") as f:
        pickle.dump(pred_probs, f)

    with open(f"./artifacts/clean_baseline/{test_data}_true_label.pkl", "wb") as f:
        pickle.dump(true_label, f)

    with open(f"./artifacts/clean_baseline/{test_data}_all_label.pkl", "wb") as f:
        pickle.dump(sorted(list(all_label)), f)

    # 4) 保存逐样本结果
    pred_csv = f"./results/clean_baseline/{test_data}_predictions.csv"
    test_ids = list(id_ec_test.keys())

    print("First 5 test ids:", test_ids[:5])
    print("First 5 true labels:", true_label[:5])
    print("First 5 pred labels:", pred_label[:5])

    with open(pred_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "true_label", "pred_label", "pred_probs"])
        for qid, t, p, prob in zip(test_ids, true_label, pred_label, pred_probs):
            writer.writerow([
                qid,
                ";".join(t),
                ";".join(p),
                ";".join([str(float(x)) for x in prob])
            ])

    if report_metrics:
        pre, rec, f1, roc, acc = get_eval_metrics(
            pred_label, pred_probs, true_label, all_label)

        metrics = {
            "precision_weighted": float(pre),
            "recall_weighted": float(rec),
            "f1_weighted": float(f1),
            "roc_auc_weighted": float(roc),
            "accuracy": float(acc),
            "num_samples": int(len(true_label)),
            "num_labels": int(len(all_label))
        }

        with open(f"./results/clean_baseline/{test_data}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print("############ EC calling results using maximum separation ############")
        print('-' * 75)
        print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)} \n'
              f'>>> precision: {pre:.3} | recall: {rec:.3}'
              f'| F1: {f1:.3} | AUC: {roc:.3} | ACC: {acc:.3}')
        print('-' * 75)

## functions for inference on the fly (without saving esm embeddings)
'''
def infer_maxsep(train_data, test_data, report_metrics = False, 
                 pretrained=True, model_name=None, gmm = None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    id_ec_train, ec_id_dict_train = get_ec_id_dict('./data/' + train_data + '.csv')
    id_ec_test, _ = get_ec_id_dict('./data/' + test_data + '.csv')
    # load checkpoints
    # NOTE: change this to LayerNormNet(512, 256, device, dtype) 
    # and rebuild with [python build.py install]
    # if inferencing on model trained with supconH loss
    model = LayerNormNet(512, 128, device, dtype)
    
    if pretrained:
        try:
            checkpoint = torch.load('./data/pretrained/'+ train_data +'.pth', map_location=device)
        except FileNotFoundError as error:
            raise Exception('No pretrained weights for this training data')
    else:
        try:
            checkpoint = torch.load('./data/model/'+ model_name +'.pth', map_location=device)
        except FileNotFoundError as error:
            raise Exception('No model found!')
            
    model.load_state_dict(checkpoint)
    model.eval()
    # load precomputed EC cluster center embeddings if possible
    if train_data == "split70":
        emb_train = torch.load('./data/pretrained/70.pt', map_location=device)
    elif train_data == "split100":
        emb_train = torch.load('./data/pretrained/100.pt', map_location=device)
    else:
        emb_train = model(esm_embedding(ec_id_dict_train, device, dtype))
        
    emb_test = model_embedding_test(id_ec_test, model, device, dtype)
    eval_dist = get_dist_map_test(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)
    seed_everything()
    eval_df = pd.DataFrame.from_dict(eval_dist)
    ensure_dirs("./results")
    out_filename = "results/" +  test_data
    write_max_sep_choices(eval_df, out_filename, gmm=gmm)
    if report_metrics:
        pred_label = get_pred_labels(out_filename, pred_type='_maxsep')
        pred_probs = get_pred_probs(out_filename, pred_type='_maxsep')
        true_label, all_label = get_true_labels('./data/' + test_data)
        pre, rec, f1, roc, acc = get_eval_metrics(
            pred_label, pred_probs, true_label, all_label)
        print("############ EC calling results using maximum separation ############")
        print('-' * 75)
        print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)} \n'
            f'>>> precision: {pre:.3} | recall: {rec:.3}'
            f'| F1: {f1:.3} | AUC: {roc:.3} ')
        print('-' * 75)
'''
