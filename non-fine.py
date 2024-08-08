# @Author  : hdm
import argparse
import pickle
import glob
import joblib
import os
from sklearn.model_selection import StratifiedKFold
from bin.config import parse_args
import pandas as pd
from bin.train import train, train_last
from bin.test import test
from bin.imp import train_msi
from sklearn.preprocessing import LabelEncoder
from bin.get_msi import get_msi
from sklearn.metrics import plot_roc_curve, roc_curve, auc, roc_auc_score
import sys
def non_fine():
    # Parse command-line arguments
    args = parse_args()
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # X = pd.read_csv(args.testx)
    # y = pd.read_csv(args.testy)
    # test_data = pd.merge(X, y, how='left')
    test_data = pd.read_csv(args.test)
    # test_data = glob.glob(args.test + '/*.csv')

    train_sets = []
    val_sets = []

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
        train_host1 = train_set[['sampleid', 'primary_site']]
        val_host1 = val_set[['sampleid', 'primary_site']]
        # base_model = torch.load('/home/hdm/Meta-Spe/bin/data/train/meta_spec.model')
        primary_model = train(args, X_train1, y_train1, train_host1)
        # primary_model = train_last(args, X_train1, y_train1, train_host1, base_model)
        y_pred = test(args, X_val1, val_host1, primary_model)
        # train_msi(args, X_train1, y_train1, train_host1)
        # get_msi(args, X_val1, val_host1, fold + 1)

        y_true = y_val1

        # label_mapping = {'NAT': 0, 'Tumor': 1}
        y_true['prey'] = y_true.iloc[:, 1].map(lambda x: 1 if x.lower() == 'tumor' else 0)
        y_true = y_true.reset_index(drop=True)
        merged_df = pd.merge(y_pred, y_true, on='sampleid', suffixes=('_pred', '_true'))
        merged_df = pd.merge(merged_df, val_host1, how='left')

        merged_df['Fold'] = fold + 1
        print(merged_df)

        fold_data = merged_df[['sampleid', 'prey_pred', 'prey_true', 'primary_site', 'Fold']]
        all_df.append(fold_data)
        print('all_df:', all_df)
    all_df = pd.concat(all_df, ignore_index=True)
    print(all_df)
    output_filename = os.path.splitext(os.path.basename(args.test))[0] + '-nonfine_prediction.csv'
    all_df.to_csv(output_filename, index=False)


if __name__ == "__main__":
    non_fine()

