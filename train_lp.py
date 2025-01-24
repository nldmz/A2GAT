import argparse
import os
import sys
import time
import pickle
import torch
import numpy as np
from tqdm import tqdm
import random
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import average_precision_score,auc,precision_recall_fscore_support
from model import AdaptiveAnchorGAT
from utils import set_random_seed
from calculate import calculate_anchor_points

from torch.utils.data import DataLoader, TensorDataset
def parse_args():
    parser = argparse.ArgumentParser(description='Training Classfier for Link Prediction Task')
    parser.add_argument('--data_root_dir', default='./dataset', type=str, help='root path of datasets')
    parser.add_argument('--model_file', default='models/WikiPedia/model.pt', type=str, help='model path')
    parser.add_argument('--gpu', default='0', type=str, help='gpu number')  # Load model into GPU. Set this rather than CUDA_VISIBLE_DEVICES.
    parser.add_argument('--seed', default=2025, type=int, help='seed')
    parser.add_argument('--operator', default='hadamard', type=str, help='combine operator')
    parser.add_argument('--dim', default=64, type=int, help='embedding dimensionality')


    args = parser.parse_args()

    args.dataset, args.model_name = args.model_file.split('/')[-2:]
    args.input_dir = os.path.join(args.data_root_dir, args.dataset)

    return args

def specific_path(args, path):
    if path in ['train.csr.pickle', 'test.csr.pickle', 'lp.train.npz', 'lp.test.npz']:
        dirname = args.input_dir
    else:
        raise ValueError('unsupported file!')
    return os.path.join(dirname, path)


def gen_indices_and_labels(lp_pos, lp_neg):
    # X
    indices = np.concatenate((lp_pos, lp_neg), axis=0)
    # y
    labels = np.concatenate((np.ones(len(lp_pos), dtype=np.int64), np.zeros(len(lp_neg), dtype=np.int64)))

    return indices, labels


def input_emb(indices, U_emb, V_emb):
    u_emb, v_emb = U_emb[indices[:, 0]], V_emb[indices[:, 1]]
    return u_emb, v_emb


def gen_features(u_emb, v_emb, operator='hadamard'):
    if operator == 'hadamard':
        feat = u_emb * v_emb
    elif operator == 'concat':
        feat = np.concatenate((u_emb, v_emb), axis=-1)
    else:
        raise ValueError('Operator [', operator, '] is not supported!')
    return feat


def gen_input_and_label(lp_pos, lp_neg, U_emb, V_emb, operator='hadamard'):
    X_indices, y = gen_indices_and_labels(lp_pos, lp_neg)
    X_U_emb, X_V_emb = input_emb(X_indices, U_emb, V_emb)
    X = gen_features(X_U_emb, X_V_emb, operator)
    return X, y


def print_seperator():
    print('-' * 20)


if __name__ == '__main__':
    args = parse_args()
    print(args)

    if args.gpu == '-1':
        device = 'cpu'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = 'cuda:0'

    set_random_seed(args.seed, False)

    # load graph (Only num_U & num_V are needed.)
    print('> loading dataset')
    s = time.time()

    with open(os.path.join(args.input_dir, 'train.csr.pickle'), 'rb') as f:
        csr_train = pickle.load(f)
        print('train:', csr_train.shape, csr_train.nnz)
    # with open(os.path.join(args.input_dir, 'test.csr.pickle'), 'rb') as f:
    #     csr_test = pickle.load(f)
    #     print('test:', csr_test.shape, csr_test.nnz)

    num_U, num_V = csr_train.shape

    e = time.time()
    print("loading time: %f s" % (e - s))

    # load embedding
    print_seperator()
    print('> loading model')
    s = time.time()
    n_anchor = calculate_anchor_points(csr_train)
    P = torch.tensor(csr_train.todense(), dtype=torch.float32, device=device)
    model = AdaptiveAnchorGAT(
        embedding_dim=args.dim,
        num_users=csr_train.shape[0],
        num_items=csr_train.shape[1],
        num_anchors=n_anchor,
        anchor_dim=n_anchor // 2,
        adjacency_matrix=P
    ).to(device)
    model.load_state_dict(torch.load(args.model_file))  # Model will be loaded into GPU by default.

    with torch.no_grad():
        idx_U_dataset = TensorDataset(torch.arange(num_U, dtype=torch.long, device=device))

        idx_U_dataloader = DataLoader(idx_U_dataset, batch_size=5000)

        U_emb_list = []
        for batch in idx_U_dataloader:
            batch_idx_U = batch[0].to(device)
            # 对每个批次的idx_U调用get_U_emb
            batch_U_emb = model.get_user_embedding(batch_idx_U)
            U_emb_list.append(batch_U_emb)

        # 将所有的batch拼接起来
        U_emb_list = [emb.cpu().detach().numpy() for emb in U_emb_list]
        U_emb = np.concatenate(U_emb_list, axis=0)
        # U_emb = np.concatenate(U_emb_list, axis=0)

        V_emb = model.get_item_embedding().cpu().detach().numpy()
    print('U_emb:', U_emb.shape)
    print('V_emb:', V_emb.shape)

    e = time.time()
    print("loading time: %f s" % (e - s))

    # load LP dataset
    print_seperator()
    print('> loading LP dataset')
    s = time.time()

    lp_train_path = specific_path(args, 'lp.train.npz')
    lp_test_path = specific_path(args, 'lp.test.npz')

    lp_train = np.load(lp_train_path)
    lp_test = np.load(lp_test_path)

    lp_pos_train, lp_neg_train = lp_train['p'], lp_train['n']
    lp_pos_test, lp_neg_test = lp_test['p'], lp_test['n']

    print('lp_pos_train:', lp_pos_train.shape, 'lp_neg_train:', lp_neg_train.shape)
    print('lp_pos_test:', lp_pos_test.shape, 'lp_neg_test:', lp_neg_test.shape)

    e = time.time()
    print("loading time: %f s" % (e - s))

    # train LP
    print_seperator()
    print('> training LP')

    X_train, y_train = gen_input_and_label(lp_pos_train, lp_neg_train, U_emb, V_emb, args.operator)

    s = time.time()

    lg = LogisticRegression(penalty='l2', C=0.001, random_state=args.seed)
    lg.fit(X_train, y_train)

    e = time.time()
    print("LP training time: %f s" % (e - s))

    # test
    print_seperator()
    print('> predicting')
    s = time.time()

    X_test, y_test = gen_input_and_label(lp_pos_test, lp_neg_test, U_emb, V_emb, args.operator)
    y_pred = lg.predict_proba(X_test)[:, 1]

    e = time.time()
    print("LP predicting time: %f s" % (e - s))

    print('> evaluating')
    s = time.time()

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    average_precision = average_precision_score(y_test, y_pred)
    auc_roc, auc_pr = metrics.auc(fpr, tpr), average_precision

    e = time.time()
    print('link prediction metrics: AUC_ROC : %0.4f, AUC_PR : %0.4f' % (round(auc_roc, 4), round(auc_pr, 4)))
    print("LP evaluating time: %f s" % (e - s))
