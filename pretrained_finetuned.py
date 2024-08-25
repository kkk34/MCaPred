# @Author  : hdm
import argparse
import pickle
import os
import joblib
import torch
from sklearn.model_selection import StratifiedKFold
from bin.config import parse_args
import pandas as pd
from bin.train import train
from bin.test import test
from bin.imp import train_msi
from sklearn.preprocessing import LabelEncoder
from bin.train import train_last
from bin.get_msi import get_msi
from sklearn.metrics import plot_roc_curve, roc_curve, auc, roc_auc_score
import sys

def main():
    # Parse command-line arguments
    args = parse_args()

    df_a = pd.read_csv(args.microbe)
    df_b = pd.read_csv(args.label)
    df_b['prey'] = df_b['sample_type'].apply(lambda x: 'tumor' if 'Tumor' in x else 'normal')
    # Merge feature and label
    df_a['primary_site'] = df_b['primary_site']
    df_a['prey'] = df_b['prey']

    # Code the primary_site
    if df_a['primary_site'].dtype != 'int' and df_a['primary_site'].dtype != 'float':
        encoder = LabelEncoder()
        df_a['primary_site'] = encoder.fit_transform(df_a['primary_site'])

        encoder_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

        joblib.dump(encoder_mapping, 'encoder_mapping.pkl')
    print("Encoder Mapping:", encoder_mapping)
    X_train = df_a.drop(columns=['primary_site', 'prey'])

    y_train = df_a[['sampleid', 'prey']]

    train_host = df_a[['sampleid', 'primary_site']]

    # pre-traing
    print("pre-training")
    base_model = train(args, X_train, y_train, train_host)
    # sys.exit()

    # fine tune
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # input data
    # X = pd.read_csv(args.testx)
    # y = pd.read_csv(args.testy)
    # test_data = pd.merge(X, y, how='left')
    test_data = pd.read_csv(args.test)
    train_sets = []
    val_sets = []
    # Five-fold cross-validation
    all_df = []
    for train_index, val_index in skf.split(test_data, test_data['prey']):
        train_set, val_set = test_data.iloc[train_index], test_data.iloc[val_index]
        train_sets.append(train_set)
        val_sets.append(val_set)
    for fold in range(n_splits):
        print(f"Fold {fold + 1}:")
        train_set = train_sets[fold]
        val_set = val_sets[fold]
        X_train1 = train_set.drop(columns=['env_broad_scale', 'env_local_scale', 'prey', 'primary_site'])
        X_val1 = val_set.drop(columns=['env_broad_scale', 'env_local_scale', 'prey', 'primary_site'])
        y_train1 = train_set[['sampleid', 'prey']]
        y_val1 = val_set[['sampleid', 'prey']]
        train_host1 =train_set[['sampleid', 'primary_site']]
        val_host1 = val_set[['sampleid', 'primary_site']]
        # load pre-trained model
        base_model = torch.load('../MCaPred/data/pretrain/MCaPred_pre.model')
        # fine-tuned
        primary_model = train_last(args, X_train1, y_train1,  train_host1, base_model)
        # Predicted
        y_pred = test(args, X_val1, val_host1, primary_model)
        y_true = y_val1

        # label_mapping = {'NAT': 0, 'Tumor': 1}
        y_true['prey'] = y_true.iloc[:, 1].map(lambda x: 1 if x == 'Tumor' else 0)
        y_true = y_true.reset_index(drop=True)
        merged_df = pd.merge(y_pred, y_true, on='sampleid', suffixes=('_pred', '_true'))
        merged_df = pd.merge(merged_df, val_host1, how='left')

        merged_df['Fold'] = fold + 1
        print(merged_df)

        fold_data = merged_df[['sampleid', 'prey_pred', 'prey_true', 'primary_site', 'Fold']]
        all_df.append(fold_data)
        print('all_df:', all_df)
        # To add Importance (MSI) 
        train_msi(args, X_train1, y_train1,  train_host1)
        # To calculate feature Importance
        get_msi(args, X_val1, val_host1, fold+1)
    all_df = pd.concat(all_df, ignore_index=True)
    print(all_df)
    output_filename = os.path.splitext(os.path.basename(args.test))[0] + '_prediction.csv'
    all_df.to_csv(output_filename, index=False)
    # all_df.to_csv(output_filename, index=False)


if __name__ == "__main__":
    main()
