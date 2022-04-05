'''
Copyright (c) 2020 NVIDIA
Author: Wentao Yuan
'''

import h5py
import numpy as np
import os
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski

def jitter_pcd(pcd, sigma=0.01, clip=0.05):
    pcd += np.clip(sigma * np.random.randn(*pcd.shape), -1 * clip, clip)
    return pcd

def random_pose(max_angle, max_trans):
    R = random_rotation(max_angle)
    t = random_translation(max_trans)
    return np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)


def random_rotation(max_angle):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.rand() * max_angle
    A = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
    return R


def random_translation(max_dist):
    t = np.random.randn(3)
    t /= np.linalg.norm(t)
    t *= np.random.rand() * max_dist
    return np.expand_dims(t, 1)

def farthest_subsample_points(pointcloud1, pointcloud2, num_subsampled_points=768):
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1)
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud2)
    random_p2 = random_p1 #np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))
    return pointcloud1[idx1, :], pointcloud2[idx2, :]

class TestData(Dataset):
    def __init__(self, path, args):
        super(TestData, self).__init__()
        with h5py.File(path, 'r') as f:
            self.source = f['source'][...]
            self.target = f['target'][...]
            self.transform = f['transform'][...]
        self.n_points = args.n_points
        self.max_angle = args.max_angle / 180 * np.pi
        self.max_trans = args.max_trans
        self.noisy = not args.clean
        self.subsampled = True
        self.num_subsampled_points = args.n_subsampled_points
    def __getitem__(self, index):
        np.random.seed(index)
        pcd1 = self.source[index][:self.n_points]
        pcd2 = self.target[index][:self.n_points]
        transform = self.transform[index]
        pcd1 = pcd1 @ transform[:3, :3].T + transform[:3, 3]
        transform = random_pose(self.max_angle, self.max_trans)
        pose1 = random_pose(np.pi, self.max_trans)
        pose2 = transform @ pose1
        pcd1 = pcd1 @ pose1[:3, :3].T + pose1[:3, 3]
        pcd2 = pcd2 @ pose2[:3, :3].T + pose2[:3, 3]
        R_ab = transform[:3, :3]
        translation_ab = transform[:3, 3]

        if self.subsampled:
            pcd1, pcd2 = farthest_subsample_points(pcd1, pcd2,num_subsampled_points = self.num_subsampled_points)

        return pcd1.T.astype('float32'), pcd2.T.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32')

    def __len__(self):
        return self.transform.shape[0]


class TrainData(Dataset):
    def __init__(self, path, args):
        super(TrainData, self).__init__()
        with h5py.File(path, 'r') as f:
            self.points = f['points'][...]
        self.n_points = args.n_points
        self.max_angle = args.max_angle / 180 * np.pi
        self.max_trans = args.max_trans
        self.noisy = not args.clean
        self.subsampled = True
        self.num_subsampled_points = args.n_subsampled_points

    def __getitem__(self, index):
        pcd1 = self.points[index][:self.n_points]
        pcd2 = self.points[index][:self.n_points]
        transform = random_pose(self.max_angle, self.max_trans)
        pose1 = random_pose(np.pi, self.max_trans)
        pose2 = transform @ pose1
        pcd1 = pcd1 @ pose1[:3, :3].T + pose1[:3, 3]
        pcd2 = pcd2 @ pose2[:3, :3].T + pose2[:3, 3]
        if self.noisy:
            pcd1 = jitter_pcd(pcd1)
            pcd2 = jitter_pcd(pcd2)
        R_ab = transform[:3, :3]
        translation_ab = transform[:3, 3]
        if self.subsampled:
            pcd1, pcd2 = farthest_subsample_points(pcd1, pcd2,num_subsampled_points = self.num_subsampled_points)

        return pcd1.T.astype('float32'), pcd2.T.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32')

    def __len__(self):
        return self.points.shape[0]

