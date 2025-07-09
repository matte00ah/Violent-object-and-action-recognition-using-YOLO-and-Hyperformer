# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import os.path as osp
import numpy as np
import pickle
import logging
import h5py
from sklearn.model_selection import train_test_split
import time as tt

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter

import sys

root_path = './'
stat_path = osp.join(root_path, 'statistics')
setup_file = osp.join(stat_path, 'setup.txt')
camera_file = osp.join(stat_path, 'camera.txt')
performer_file = osp.join(stat_path, 'performer.txt')
replication_file = osp.join(stat_path, 'replication.txt')
label_file = osp.join(stat_path, 'label.txt')
skes_name_file = osp.join(stat_path, 'skes_available_name.txt')

denoised_path = osp.join(root_path, 'denoised_data')
raw_skes_joints_pkl = osp.join(denoised_path, 'raw_denoised_joints.pkl')
frames_file = osp.join(denoised_path, 'frames_cnt.txt')

save_path = root_path

if not osp.exists(save_path):
    os.mkdir(save_path)


def remove_nan_frames(ske_name, ske_joints, nan_logger):
    num_frames = ske_joints.shape[0]
    valid_frames = []

    for f in range(num_frames):
        if not np.any(np.isnan(ske_joints[f])):
            valid_frames.append(f)
        else:
            nan_indices = np.where(np.isnan(ske_joints[f]))[0]
            nan_logger.info('{}\t{:^5}\t{}'.format(ske_name, f + 1, nan_indices))

    return ske_joints[valid_frames]

def seq_translation(skes_joints):
    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        if num_bodies == 2:
            missing_frames_1 = np.where(ske_joints[:, :75].sum(axis=1) == 0)[0]
            missing_frames_2 = np.where(ske_joints[:, 75:].sum(axis=1) == 0)[0]
            cnt1 = len(missing_frames_1)
            cnt2 = len(missing_frames_2)

        i = 0  # get the "real" first frame of actor1
        while i < num_frames:
            if np.any(ske_joints[i, :75] != 0):
                break
            i += 1

        origin = np.copy(ske_joints[i, 3:6])  # new origin: joint-2

        for f in range(num_frames):
            if num_bodies == 1:
                ske_joints[f] -= np.tile(origin, 25)
            else:  # for 2 actors
                ske_joints[f] -= np.tile(origin, 50)

        if (num_bodies == 2) and (cnt1 > 0):
            ske_joints[missing_frames_1, :75] = np.zeros((cnt1, 75), dtype=np.float32)

        if (num_bodies == 2) and (cnt2 > 0):
            ske_joints[missing_frames_2, 75:] = np.zeros((cnt2, 75), dtype=np.float32)

        skes_joints[idx] = ske_joints  # Update

    return skes_joints


def frame_translation(skes_joints, skes_name, frames_cnt):
    nan_logger = logging.getLogger('nan_skes')
    nan_logger.setLevel(logging.INFO)
    nan_logger.addHandler(logging.FileHandler("./nan_frames.log"))
    nan_logger.info('{}\t{}\t{}'.format('Skeleton', 'Frame', 'Joints'))

    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        # Calculate the distance between spine base (joint-1) and spine (joint-21)
        j1 = ske_joints[:, 0:3]
        j21 = ske_joints[:, 60:63]
        dist = np.sqrt(((j1 - j21) ** 2).sum(axis=1))

        for f in range(num_frames):
            origin = ske_joints[f, 3:6]  # new origin: middle of the spine (joint-2)
            if (ske_joints[f, 75:] == 0).all():
                ske_joints[f, :75] = (ske_joints[f, :75] - np.tile(origin, 25)) / \
                                      dist[f] + np.tile(origin, 25)
            else:
                ske_joints[f] = (ske_joints[f] - np.tile(origin, 50)) / \
                                 dist[f] + np.tile(origin, 50)

        ske_name = skes_name[idx]
        ske_joints = remove_nan_frames(ske_name, ske_joints, nan_logger)
        frames_cnt[idx] = num_frames  # update valid number of frames
        skes_joints[idx] = ske_joints

    return skes_joints, frames_cnt


def align_frames(skes_joints, frames_cnt):
    """
    Align all sequences with the same frame length.

    """
    num_skes = len(skes_joints)
    max_num_frames = frames_cnt.max()  # 300
    aligned_skes_joints = np.zeros((num_skes, max_num_frames, 150), dtype=np.float32)

    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        if num_bodies == 1:
            aligned_skes_joints[idx, :num_frames] = np.hstack((ske_joints, ske_joints))
            # aligned_skes_joints[idx, :num_frames] = np.hstack((ske_joints, np.zeros_like(ske_joints)))
        else:
            aligned_skes_joints[idx, :num_frames] = ske_joints

    return aligned_skes_joints


def one_hot_vector(labels):
    num_skes = len(labels)
    labels_vector = np.zeros((num_skes, 10))
    for idx, l in enumerate(labels):
        labels_vector[idx, l] = 1

    return labels_vector


def split_train_val(train_indices, method='sklearn', ratio=0.05):
    """
    Get validation set by splitting data randomly from training set with two methods.
    In fact, I thought these two methods are equal as they got the same performance.

    """
    if method == 'sklearn':
        return train_test_split(train_indices, test_size=ratio, random_state=10000)
    else:
        np.random.seed(10000)
        np.random.shuffle(train_indices)
        val_num_skes = int(np.ceil(0.05 * len(train_indices)))
        val_indices = train_indices[:val_num_skes]
        train_indices = train_indices[val_num_skes:]
        return train_indices, val_indices


def split_dataset(skes_joints, label, performer, setup, evaluation, save_path, skes_names_file, YOLO_dir):
    #MODIFIED
    f = open(skes_names_file, 'r')
    skes_names = f.readlines()
    f.close()

    couples = []
    i=0
    for name in skes_names:
        print(f"Processing {name.strip()}...")
        print(f'{i}/{len(skes_names)}\n')
        i+=1
        t = tt.time()
        couple = []
        video_name = name.strip().split('_')[0]
        # if video_name not in ['non-violent-cam1-1', 'non-violent-cam1-2', 'non-violent-cam1-3', 'non-violent-cam1-4']:
        #     continue

        #MODIFIED invece che appendere il tensore effettivo di yolo si aggiunge il percorso assoluto al tensore
        yp = os.path.join(YOLO_dir, video_name + '.npz')
        couple.append(skes_joints[skes_names.index(name)])
        couple.append(yp)
        couples.append(couple)
    
    # train_indices, test_indices = get_indices(performer, setup, evaluation)
    # m = 'sklearn'  # 'sklearn' or 'numpy'
    # Select validation set from training set
    # indices = np.arange(len(couples))
    #train_indices, test_indices = split_train_val(indices, m)
    
    # Bilanciamento dataset con random upsampling
    label = np.array(label, dtype=np.int64)
    positive_idx = np.where(label != 9)[0]  # Violent class
    chosen = np.random.choice(positive_idx, size=np.where(label == 9)[0].shape[0] - len(positive_idx), replace=False)
    # Aggiunta di campioni casuali della classe violenta per bilanciare il dataset
    for c in chosen:
        couples.append(couples[c])  # Aggiunge il campione scelto
    label = np.concatenate((label, label[chosen]), axis=0)

    label = one_hot_vector(label)  # Converti le etichette in one-hot encoding

    # Suddivisione in training e test set
    x_train, x_test, y_train, y_test = train_test_split(couples, label, test_size=0.2, random_state=42, stratify=label)

    train_x = np.array(x_train, dtype=object)
    train_y = np.array(y_train, dtype=object)
    test_x = np.array(x_test, dtype=object)
    test_y = np.array(y_test, dtype=object)

    save_name = save_path + '/FTVD.npz'
    np.savez(save_name, x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y, allow_pickle=True)

    # # Save data into a .h5 file
    # h5file = h5py.File(osp.join(save_path, 'NTU_%s.h5' % (evaluation)), 'w')
    # # Training set
    # h5file.create_dataset('x', data=skes_joints[train_indices])
    # train_one_hot_labels = one_hot_vector(train_labels)
    # h5file.create_dataset('y', data=train_one_hot_labels)
    # # Validation set
    # h5file.create_dataset('valid_x', data=skes_joints[val_indices])
    # val_one_hot_labels = one_hot_vector(val_labels)
    # h5file.create_dataset('valid_y', data=val_one_hot_labels)
    # # Test set
    # h5file.create_dataset('test_x', data=skes_joints[test_indices])
    # test_one_hot_labels = one_hot_vector(test_labels)
    # h5file.create_dataset('test_y', data=test_one_hot_labels)

    # h5file.close()


def get_indices(performer, setup, evaluation='VD-train'):
    test_indices = np.empty(0)
    train_indices = np.empty(0)

    if evaluation == 'VD-train':  # Cross Subject (Subject IDs)
        train_ids = [1]
        test_ids = [1]

        # Get indices of test data
        for idx in test_ids:
            temp = np.where(performer == idx)[0]  # 0-based index
            test_indices = np.hstack((test_indices, temp)).astype(np.int64)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(performer == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(np.int64)
    # else:  # Cross Setup (Setup IDs)
    #     train_ids = [i for i in range(1, 33) if i % 2 == 0]  # Even setup
    #     test_ids = [i for i in range(1, 33) if i % 2 == 1]  # Odd setup

    #     # Get indices of test data
    #     for test_id in test_ids:
    #         temp = np.where(setup == test_id)[0]  # 0-based index
    #         test_indices = np.hstack((test_indices, temp)).astype(np.int64)

    #     # Get indices of training data
    #     for train_id in train_ids:
    #         temp = np.where(setup == train_id)[0]  # 0-based index
    #         train_indices = np.hstack((train_indices, temp)).astype(np.int64)

    test_indices = test_indices[-10:]
    train_indices = train_indices[:-10]
    return train_indices, test_indices


if __name__ == '__main__':
    setup = np.loadtxt(setup_file, dtype=np.int64)  # camera id: 1~32
    performer = np.loadtxt(performer_file, dtype=np.int64)  # subject id: 1~106
    label = np.loadtxt(label_file, dtype=np.int64) - 1  # action label: 0~119

    frames_cnt = np.loadtxt(frames_file, dtype=np.int64)  # frames_cnt
    skes_name = np.loadtxt(skes_name_file, dtype=np.string_)

    with open(raw_skes_joints_pkl, 'rb') as fr:
        skes_joints = pickle.load(fr)  # a list

    skes_joints = seq_translation(skes_joints)

    skes_joints = align_frames(skes_joints, frames_cnt)  # aligned to the same frame length

    evaluations = ['VD-train']
    for evaluation in evaluations:
        yolo_dir = '../raw_data/YOLOWeapon_activation_maps'
        split_dataset(skes_joints, label, performer, setup, evaluation, save_path, skes_name_file, yolo_dir)