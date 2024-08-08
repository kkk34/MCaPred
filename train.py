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
    target = [x for x in df_labels.columns if x!="sampleid"]    # 筛选标签
    
    dense_features = [x for x in df_microbe.columns if x!="sampleid"]  # 筛选特征
    sparse_features = []
    if len(train_host) > 0:
        sparse_features = [x for x in df_hosts.columns if x!="sampleid"]

        # feat:特征列名；vocabulary_size:嵌入层输入大小（类别数量)；embedding_dim:是嵌入向量的维度，用于将类别特征映射到连续的低维空间
        dnn_feature_columns = [SparseFeat(feat, vocabulary_size = data[feat].max()+1, embedding_dim=args.embedding_size)
                                      for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]
    else:
        # 密集特征是数值型的，不需要进行嵌入操作
        dnn_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]
        
    feature_names = get_feature_names(dnn_feature_columns)

    train_model_input = {name: data[name] for name in feature_names}
    train_labels = data[target].values
       
    train_model = MCaPred(dnn_feature_columns, dense_num=len(dense_features),
                           num_tasks=args.task_num, num_experts=args.n_expert,
                           dnn_hidden_units=args.hidden_units,
                       tasks=['binary']*args.task_num, device=device)
    #adagrad:自适应学习优化器
    train_model.compile("adagrad", loss='binary_crossentropy')    

    for epoch in range(2):
        history = train_model.fit(train_model_input, train_labels, batch_size=args.batch_size, epochs=args.n_epoch//2, verbose=1)

    train_model.eval()
    gate_ = train_model.gate_list   
    # torch.save(train_model, args.o+'/train/MCaPred.model')
    torch.save(train_model, args.o + '/pretrain/MCaPred_pre.model')

    # prepare for msi（数据）
    x_input = torch.from_numpy(data[sparse_features+dense_features].values).to(device).to(torch.float32)

    # 获取训练好的模型 train_model 中的嵌入字典（embedding_dict）
    embedding_dict = train_model.embedding_dict
    sparse_embedding_list, dense_value_list = train_model.input_from_feature_columns(x_input, dnn_feature_columns, embedding_dict)

    # 创建了一个包含稀疏特征列的列表
    sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if len(dnn_feature_columns) else []
    feature_index = build_input_features(dnn_feature_columns)

    # 这行代码将稀疏特征的嵌入列表 sparse_embedding_list 和密集特征的值列表 dense_value_list 组合成一个用于模型输入的数据结构
    train_dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    # with open(args.o + '/train/dnn_input_train.pickle', 'wb') as f:
    with open(args.o+'/pretrain/dnn_input_train.pickle', 'wb') as f:
        pickle.dump(train_dnn_input, f)

    # save diseases correlation
    gate_dict = {}
    for i in range(len(target)):
        gate_dict[target[i]] = np.array(gate_[i].detach().cpu()) # 列表中获取第 i 个元素，与疾病相关的某种信息或参数。
    # 计算每个疾病的相关性图
    df_corr = pd.DataFrame(gate_dict).corr()
    # df_corr.to_csv(args.o + "/train/disease_corr.csv")
    df_corr.to_csv(args.o+"/pretrain/disease_corr.csv")

    # 可以用于可视化
    # with open(args.o + "/train/disease_name.pickle", "wb") as f:
    with open(args.o+"/pretrain/disease_name.pickle", "wb") as f:
        pickle.dump(target, f)

    return train_model


def train_last(args, X_train, y_train, train_host, base_model):
    # model = base_model
    df_microbe = X_train
    df_label = y_train

    # 将 'Tumor' 替换为 1，其他值替换为 0
    # df_label['prey'] = df_label.iloc[:, 1].map(lambda x: 0 if x == 'OtherTypeCancer' else 1)
    df_label['prey'] = df_label.iloc[:, 1].map(lambda x: 1 if x.lower() == 'tumor' else 0)

    print(df_label)

    data = pd.merge(df_microbe, df_label, how='left')
    print(data)
    if len(train_host) > 0:
        df_hosts = train_host
        data = pd.merge(data, df_hosts, how='left')

    target = [x for x in df_label.columns if x != "sampleid"]  # 筛选标签

    dense_features = [x for x in df_microbe.columns if x != "sampleid"]  # 筛选特征
    sparse_features = []
    if len(train_host) > 0:
        sparse_features = [x for x in df_hosts.columns if x != "sampleid"]

        dnn_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=args.embedding_size)
                               for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]
    else:
        # 密集特征是数值型的，不需要进行嵌入操作
        dnn_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]

    feature_names = get_feature_names(dnn_feature_columns)

    train_model_input = {name: data[name] for name in feature_names}
    train_labels = data[target].values

    train_model = base_model

    print("执行小模型")
    for epoch in range(2):
        history = train_model.fit(train_model_input, train_labels, batch_size=args.batch_size, epochs=args.n_epoch // 2, verbose=1)
    train_model.eval()
    gate_ = train_model.gate_list
    torch.save(train_model, args.o+'/train/MCaPred.model')

    train_labels_df = pd.DataFrame(train_labels, columns=target)
    train_labels_df.to_csv(args.o + 'last_train_labels.csv', index=False)
    print(f"Last train labels saved to {args.o}/train/last_train_labels.csv")

    # prepare for msi（数据）
    x_input = torch.from_numpy(data[sparse_features + dense_features].values).to(device).to(torch.float32)

    # 获取训练好的模型 train_model 中的嵌入字典（embedding_dict）。这个字典包含了模型中嵌入层的参数和嵌入矩阵
    embedding_dict = train_model.embedding_dict
    sparse_embedding_list, dense_value_list = train_model.input_from_feature_columns(x_input, dnn_feature_columns,
                                                                                     embedding_dict)

    # 这行代码创建了一个包含稀疏特征列的列表
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if len(dnn_feature_columns) else []
    feature_index = build_input_features(dnn_feature_columns)

    # 这行代码将稀疏特征的嵌入列表 sparse_embedding_list 和密集特征的值列表 dense_value_list 组合成一个用于模型输入的数据结构
    train_dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    with open(args.o + '/train/dnn_input_train.pickle', 'wb') as f:
        pickle.dump(train_dnn_input, f)

    # save diseases correlation
    gate_dict = {}
    for i in range(len(target)):
        gate_dict[target[i]] = np.array(gate_[i].detach().cpu())  # 列表中获取第 i 个元素，与疾病相关的某种信息或参数。
    # 计算每个疾病的相关性图
    df_corr = pd.DataFrame(gate_dict).corr()
    df_corr.to_csv(args.o + "/train/disease_corr.csv")

    # 可以用于可视化
    with open(args.o + "/train/disease_name.pickle", "wb") as f:
        pickle.dump(target, f)

    return train_model





    
