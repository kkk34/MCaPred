import os
import sys
sys.path.append('../')
import logging
from bin.config import parse_args
import pickle
import pandas as pd
import numpy as np
import torch
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names, build_input_features, combined_dnn_input
from model.MCaPred import MCaPred


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# pre-training
def train(args, X_train, y_train, train_host):

    df_microbe = X_train
    df_labels = y_train
    df_labels['prey'] = df_labels.iloc[:, 1].map(lambda x: 1 if x.lower() == 'tumor' else 0)
    # df_labels['prey'] = df_labels.iloc[:, 1].map(lambda x: 0 if x == 'OtherTypeCancer' else 1)
    # df_labels = df_labels.iloc[:, 1:]

    data = pd.merge(df_microbe, df_labels, how='left')

    if len(train_host) > 0:
        df_hosts = train_host
        data = pd.merge(data, df_hosts, how='left')
    print(data)
    target = [x for x in df_labels.columns if x!="sampleid"]    # select label
    
    dense_features = [x for x in df_microbe.columns if x!="sampleid"]  # selete label
    sparse_features = []
    if len(train_host) > 0:
        sparse_features = [x for x in df_hosts.columns if x!="sampleid"]

        # embedding_dim
        dnn_feature_columns = [SparseFeat(feat, vocabulary_size = data[feat].max()+1, embedding_dim=args.embedding_size)
                                      for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]
    else:
        # Dense features are numeric and do not require embedding
        dnn_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]
        
    feature_names = get_feature_names(dnn_feature_columns)

    train_model_input = {name: data[name] for name in feature_names}
    train_labels = data[target].values
       
    train_model = MCaPred(dnn_feature_columns, dense_num=len(dense_features),
                           num_tasks=args.task_num, num_experts=args.n_expert,
                           dnn_hidden_units=args.hidden_units,
                       tasks=['binary']*args.task_num, device=device)
    # Optimizer
    train_model.compile("adagrad", loss='binary_crossentropy')    

    for epoch in range(2):
        history = train_model.fit(train_model_input, train_labels, batch_size=args.batch_size, epochs=args.n_epoch//2, verbose=1)

    train_model.eval()
    gate_ = train_model.gate_list   
    # torch.save(train_model, args.o+'/train/MCaPred.model')
    torch.save(train_model, args.o + '/pretrain/MCaPred_pre.model')

    # prepare for msi（data）
    x_input = torch.from_numpy(data[sparse_features+dense_features].values).to(device).to(torch.float32)

    # Get the embedded dictionary in the train_model
    embedding_dict = train_model.embedding_dict
    sparse_embedding_list, dense_value_list = train_model.input_from_feature_columns(x_input, dnn_feature_columns, embedding_dict)

    # Creates a list of sparse feature columns
    sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if len(dnn_feature_columns) else []
    feature_index = build_input_features(dnn_feature_columns)

    # Combines the sparse_embedding_list and dense_value_list
    train_dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    # with open(args.o + '/train/dnn_input_train.pickle', 'wb') as f:
    with open(args.o+'/pretrain/dnn_input_train.pickle', 'wb') as f:
        pickle.dump(train_dnn_input, f)

    # save diseases correlation (multiple targets)
    gate_dict = {}
    for i in range(len(target)):
        gate_dict[target[i]] = np.array(gate_[i].detach().cpu())

    df_corr = pd.DataFrame(gate_dict).corr()
    # df_corr.to_csv(args.o + "/train/disease_corr.csv")
    df_corr.to_csv(args.o+"/pretrain/disease_corr.csv")

    # with open(args.o + "/train/disease_name.pickle", "wb") as f:
    with open(args.o+"/pretrain/disease_name.pickle", "wb") as f:
        pickle.dump(target, f)

    return train_model

# fine-tuned
def train_last(args, X_train, y_train, train_host, base_model):
    # model = base_model
    df_microbe = X_train
    df_label = y_train

    # 'Tumor':1，'other':0
    # df_label['prey'] = df_label.iloc[:, 1].map(lambda x: 0 if x == 'OtherTypeCancer' else 1)
    df_label['prey'] = df_label.iloc[:, 1].map(lambda x: 1 if x.lower() == 'tumor' else 0)

    print(df_label)

    data = pd.merge(df_microbe, df_label, how='left')
    print(data)
    if len(train_host) > 0:
        df_hosts = train_host
        data = pd.merge(data, df_hosts, how='left')

    target = [x for x in df_label.columns if x != "sampleid"]  # select label

    dense_features = [x for x in df_microbe.columns if x != "sampleid"]  # select label
    sparse_features = []
    if len(train_host) > 0:
        sparse_features = [x for x in df_hosts.columns if x != "sampleid"]

        dnn_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=args.embedding_size)
                               for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]
    else:

        dnn_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]

    feature_names = get_feature_names(dnn_feature_columns)

    train_model_input = {name: data[name] for name in feature_names}
    train_labels = data[target].values

    train_model = base_model

    print("fine-tuned model")
    for epoch in range(2):
        history = train_model.fit(train_model_input, train_labels, batch_size=args.batch_size, epochs=args.n_epoch // 2, verbose=1)
    train_model.eval()
    gate_ = train_model.gate_list
    torch.save(train_model, args.o+'/train/MCaPred.model')

    train_labels_df = pd.DataFrame(train_labels, columns=target)
    train_labels_df.to_csv(args.o + 'last_train_labels.csv', index=False)
    print(f"Last train labels saved to {args.o}/train/last_train_labels.csv")

    x_input = torch.from_numpy(data[sparse_features + dense_features].values).to(device).to(torch.float32)

    embedding_dict = train_model.embedding_dict
    sparse_embedding_list, dense_value_list = train_model.input_from_feature_columns(x_input, dnn_feature_columns,
                                                                                     embedding_dict)

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if len(dnn_feature_columns) else []
    feature_index = build_input_features(dnn_feature_columns)

    train_dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    with open(args.o + '/train/dnn_input_train.pickle', 'wb') as f:
        pickle.dump(train_dnn_input, f)

    # save diseases correlation
    gate_dict = {}
    for i in range(len(target)):
        gate_dict[target[i]] = np.array(gate_[i].detach().cpu())
    df_corr = pd.DataFrame(gate_dict).corr()
    df_corr.to_csv(args.o + "/train/disease_corr.csv")

    with open(args.o + "/train/disease_name.pickle", "wb") as f:
        pickle.dump(target, f)

    return train_model





    
