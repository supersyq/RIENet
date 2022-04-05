#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from pointnet2 import pointnet2_utils
import torch.nn.functional as F
from torch.autograd import Variable
from chamfer_loss import *

_EPS = 1e-5
def pairwise_distance_batch(x,y):
    """ 
        pairwise_distance
        Args:
            x: Input features of source point clouds. Size [B, c, N]
            y: Input features of source point clouds. Size [B, c, M]
        Returns:
            pair_distances: Euclidean distance. Size [B, N, M]
    """
    xx = torch.sum(torch.mul(x,x), 1, keepdim = True)#[b,1,n]
    yy = torch.sum(torch.mul(y,y),1, keepdim = True) #[b,1,n]
    inner = -2*torch.matmul(x.transpose(2,1),y) #[b,n,n]
    pair_distance = xx.transpose(2,1) + inner + yy #[b,n,n]
    device = torch.device('cuda')
    zeros_matrix = torch.zeros_like(pair_distance,device = device)
    pair_distance_square = torch.where(pair_distance > 0.0,pair_distance,zeros_matrix)
    error_mask = torch.le(pair_distance_square,0.0)
    pair_distances = torch.sqrt(pair_distance_square + error_mask.float()*1e-16)
    pair_distances = torch.mul(pair_distances,(1.0-error_mask.float()))
    return pair_distances

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k):
    """ 
        knn-graph.
        Args:
            x: Input point clouds. Size [B, 3, N]
            k: Number of nearest neighbors.
        Returns:
            idx: Nearest neighbor indices. Size [B * N * k]
            relative_coordinates: Relative coordinates between nearest neighbors and the center point. Size [B, 3, N, K]
            knn_points: Coordinates of nearest neighbors. Size[B, N, K, 3].
            idx2: Nearest neighbor indices. Size [B, N, k]
    """
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    idx2 = idx
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    knn_points = x.view(batch_size * num_points, -1)[idx, :]
    knn_points = knn_points.view(batch_size, num_points, k, num_dims)#[b, n, k, 3],knn
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)#[b, n, k, 3],central points

    relative_coordinates = (knn_points - x).permute(0, 3, 1, 2)

    return idx, relative_coordinates, knn_points, idx2

class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return x

class Pointer(nn.Module):
    def __init__(self):
        super(Pointer, self).__init__()
        self.conv1 = nn.Conv1d(512, 256, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(256, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        return x

def get_knn_index(x, k):
    """ 
        knn-graph.
        Args:
            x: Input point clouds. Size [B, 3, N]
            k: Number of nearest neighbors.
        Returns:
            idx: Nearest neighbor indices. Size [B * N * k]
            idx2: Nearest neighbor indices. Size [B, N, k]
    """
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    idx2 = idx
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)
    return idx, idx2

def compute_rigid_transformation(src, src_corr, weight):
    """
        Compute rigid transforms between two point sets
        Args:
            src: Source point clouds. Size (B, 3, N)
            src_corr: Pseudo target point clouds. Size (B, 3, N)
            weights: Inlier confidence. (B, 1, N)

        Returns:
            R: Rotation. Size (B, 3, 3)
            t: translation. Size (B, 3, 1)
    """
    src2 = (src * weight).sum(dim = 2, keepdim = True) / weight.sum(dim = 2, keepdim = True)
    src_corr2 = (src_corr * weight).sum(dim = 2, keepdim = True)/weight.sum(dim = 2,keepdim = True)
    src_centered = src - src2
    src_corr_centered = src_corr - src_corr2
    H = torch.matmul(src_centered * weight, src_corr_centered.transpose(2, 1).contiguous())

    R = []

    for i in range(src.size(0)):
        u, s, v = torch.svd(H[i])
        r = torch.matmul(v, u.transpose(1, 0)).contiguous()
        r_det = torch.det(r).item()
        diag = torch.from_numpy(np.array([[1.0, 0, 0],
                                            [0, 1.0, 0],
                                            [0, 0, r_det]]).astype('float32')).to(v.device)
        r = torch.matmul(torch.matmul(v, diag), u.transpose(1, 0)).contiguous()
        R.append(r)

    R = torch.stack(R, dim = 0).cuda()

    t = torch.matmul(-R, src2.mean(dim = 2, keepdim=True)) + src_corr2.mean(dim = 2, keepdim = True)
    return R, t

def get_keypoints(src, src_corr, weight, num_keypoints):
    """
        Compute rigid transforms between two point sets
        Args:
            src: Source point clouds. Size (B, 3, N)
            src_corr: Pseudo target point clouds. Size (B, 3, N)
            weights: Inlier confidence. (B, 1, N)
            num_keypoints: Number of selected keypoints.

        Returns:
            src_topk_idx: Keypoint indices. Size (B, 1, num_keypoints)
            src_keypoints: Keypoints of source point clouds. Size (B, 3, num_keypoints)
            tgt_keypoints: Keypoints of target point clouds. Size (B, 3, num_keypoints)
    """
    src_topk_idx = torch.topk(weight, k = num_keypoints, dim = 2, sorted=False)[1]
    src_keypoints_idx = src_topk_idx.repeat(1, 3, 1)
    src_keypoints = torch.gather(src, dim = 2, index = src_keypoints_idx)
    tgt_keypoints = torch.gather(src_corr, dim = 2, index = src_keypoints_idx)
    return src_topk_idx, src_keypoints, tgt_keypoints
    

class DGCNN(nn.Module):
    def __init__(self, args):
        super(DGCNN, self).__init__()
        self.emb_dims = args.emb_dims
        self.conv1 = nn.Conv2d(3, 32, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(256, self.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(self.emb_dims)
        self.dp = nn.Dropout(p=0.3)

    def forward(self, x):
        """ 
            Simplified DGCNN.
            Args:
                x: Relative coordinates between nearest neighbors and the center point. Size [B, 3, N, K]
            Returns:
                x: Features. Size [B, self.emb_dims, N]
        """
        batch_size, num_dims, num_points, _ = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        return x

class feature_extractor(nn.Module):
    def __init__(self, args):
        super(feature_extractor, self).__init__()
        self.model = DGCNN(args)

    def forward(self, x, k):
        """ 
            feature extraction.
            Args:
                x: Input point clouds. Size [B, 3, N]
                k: Number of nearest neighbors.
            Returns:
                features: Size [B, C, N]
                idx: Nearest neighbor indices. Size [B * N * k]
                knn_points: Coordinates of nearest neighbors Size [B, N, K, 3].
                idx2: Nearest neighbor indices. Size [B, N, k]
        """
        batch_size, num_dims, num_points = x.size()
        idx, relative_coordinates, knn_points, idx2 = get_graph_feature(x,k)
        features = self.model(relative_coordinates)
        return features, idx, knn_points, idx2

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.model1 = nn.Sequential(
            nn.Conv2d(3, args.dim, kernel_size=(3,1), bias=True, padding=(1,0)),
            nn.BatchNorm2d(args.dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(args.dim, args.dim * 2, kernel_size=(3,1), bias=True, padding=(1,0)),
            nn.BatchNorm2d(args.dim * 2),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.model2 = nn.Sequential(
            nn.Conv2d(args.dim * 2, 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.model3 = nn.Sequential(
            nn.Conv2d(args.dim * 2, 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True))

        self.model4 = nn.Sequential(
            nn.Conv1d(16, 8, kernel_size=1, bias=True),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(8, 1, kernel_size=1, bias=True),
            #nn.Tanh(),
        )

        self.tah = nn.Tanh()

    def forward(self, x, y):
        """ 
            Inlier Evaluation.
            Args:
                x: Source neighborhoods. Size [B, N, K, 3]
                y: Pesudo target neighborhoods. Size [B, N, K, 3]
            Returns:
                x: Inlier confidence. Size [B, 1, N]
        """
        b, n, k, _ = x.size()

        x_1x3 = self.model1(x.permute(0,3,2,1)).permute(0,1,3,2)

        y_1x3 = self.model1(y.permute(0,3,2,1)).permute(0,1,3,2) # [b, n, k, 3]-[b, c, k, n]-->[b, c, n, k]
        
        x2 = x_1x3 - y_1x3 # Eq. (5)
        
        x = self.model2(x2) # [b, c, n, k]
        weight = self.model3(x2) # [b, c, n, k] 
        weight = torch.softmax(weight, dim=-1) # Eq. (6)
        x = (x * weight).sum(-1) # [b, c, n]
        x = 1 - self.tah(torch.abs(self.model4(x)))
        return x
