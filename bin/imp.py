import os
import sys
sys.path.append('../')
import logging
from bin.config import parse_args

import pickle
import pandas as pd
import numpy as np

import torch
from model.MCaPred_plot import MCaPred
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names, build_input_features, combined_dnn_input

import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_msi(args, X_train, y_train, train_host):
    # df_microbe = pd.read_csv(args.microbe)
    # df_labels = pd.read_csv(args.label)
    df_microbe = X_train
    df_labels = y_train
    # df_labels['prey'] = df_labels.iloc[:, 1].map(lambda x: 1 if x.lower == 'tumor' else 0)
    # df_labels['prey'] = df_labels.iloc[:, 1].map(lambda x: 0 if x == 'OtherTypeCancer' else 1)
    data = pd.merge(df_microbe, df_labels, how='left')
    if len(train_host) > 0:
        df_hosts = train_host
        data = pd.merge(data, df_hosts, how='left')
    print(data)
    
    target = [x for x in df_labels.columns if x!="sampleid"]
    
    dense_features = [x for x in df_microbe.columns if x!="sampleid"]
    sparse_features = []
    if len(train_host) > 0:
        sparse_features = [x for x in df_hosts.columns if x!="sampleid"]
        dnn_feature_columns = [SparseFeat(feat, vocabulary_size = data[feat].max()+1, embedding_dim=args.embedding_size)
                                      for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]
    else:
        dnn_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]
        
    feature_names = get_feature_names(dnn_feature_columns)

    train_model_input = {name: data[name] for name in feature_names}
    train_labels = data[target].values      
    train_model = MCaPred(dnn_feature_columns, num_tasks=args.task_num,
                           dense_num=len(dense_features), sparse_num=len(sparse_features),
                           num_experts=args.n_expert, dnn_hidden_units=args.hidden_units,
                           emb_size = args.embedding_size,
                           tasks=['binary']*args.task_num, device=device)
    train_model.compile("adagrad", loss='binary_crossentropy')    

    for epoch in range(2):
        history = train_model.fit(None, train_labels, args.o+"/train/", batch_size=args.batch_size, epochs=args.n_epoch//2, verbose=1)
    args = parse_args()
    if not os.path.exists(args.o+"/msi/"):
        os.makedirs(args.o+"/msi/")
    torch.save(train_model, args.o+'/msi/msi.model')
        

# def main():
#     args = parse_args()
#     if not os.path.exists(args.o+"/msi/"):
#         os.makedirs(args.o+"/msi/")
#     train_msi(args)
#
# if __name__=='__main__':
#     main()