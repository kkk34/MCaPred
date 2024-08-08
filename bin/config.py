import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import glob
def parse_args():
    parser = argparse.ArgumentParser(description='Meta-Spe')
    
    # Data settings (Replaceable)
    parser.add_argument('--microbe', type=str, default='../data/all-feature-predata.csv')
    # parser.add_argument('--TCGA', type=str, default='../data/TCGA--Breast -- 1 vs all.csv')
    parser.add_argument('--host', type=str, default='')
    # parser.add_argument('--testx', type=str, default='../data/breast-tumor vs nat-allfea.csv')
    parser.add_argument('--test', type=str, default='../data/normal vs tumor/Breast -- tumor vs normal.csv')
    # parser.add_argument('--testy', type=str, default='../data/breast-tumor vs nat-label.csv')
    parser.add_argument('--label', type=str, default='../data/3879_label.csv') 
    parser.add_argument('--o', type=str, default='data')
    # Model settings
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--task_num', type=int, default=1)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--n_expert', type=int, default=9)
    parser.add_argument('--hidden_units', type=tuple, default=(256, 128))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epoch', type=int, default=8)
    # SHAP settings
    parser.add_argument('--is_plot', type=bool, default=True)
    parser.add_argument('--max_plot', type=int, default=30)
    
    return parser.parse_args()


