#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from pointnet2 import pointnet2_utils
import torch.nn.functional as F
from torch.autograd import Variable
from util import transform_point_cloud
from chamfer_loss import *
from utils import pairwise_distance_batch, PointNet, Pointer, get_knn_index, Discriminator, feature_extractor, compute_rigid_transformation, get_keypoints

class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.num_keypoints = args.n_keypoints
        self.weight_function = Discriminator(args)
        self.fuse = Pointer()
        self.nn_margin = args.nn_margin
        
    def forward(self, *input):
        """
            Args:
                src: Source point clouds. Size (B, 3, N)
                tgt: target point clouds. Size (B, 3, M)
                src_embedding: Features of source point clouds. Size (B, C, N)
                tgt_embedding: Features of target point clouds. Size (B, C, M)
                src_idx: Nearest neighbor indices. Size [B * N * k]
                k: Number of nearest neighbors.
                src_knn: Coordinates of nearest neighbors. Size [B, N, K, 3]
                i: i-th iteration.
                tgt_knn: Coordinates of nearest neighbors. Size [B, M, K, 3]
                src_idx1: Nearest neighbor indices. Size [B * N * k]
                idx2:  Nearest neighbor indices. Size [B, M, k]
                k1: Number of nearest neighbors.
            Returns:
                R/t: rigid transformation.
                src_keypoints, tgt_keypoints: Selected keypoints of source and target point clouds. Size (B, 3, num_keypoint)
                src_keypoints_knn, tgt_keypoints_knn: KNN of keypoints. Size [b, 3, num_kepoints, k]
                loss_scl: Spatial Consistency loss.
        """
        src = input[0]
        tgt = input[1]
        src_embedding = input[2]
        tgt_embedding = input[3]
        src_idx = input[4]
        k = input[5]
        src_knn = input[6] # [b, n, k, 3]
        i = input[7]
        tgt_knn = input[8] # [b, n, k, 3]
        src_idx1 = input[9] # [b * n * k1]
        idx2 = input[10] #[b, m, k1]
        k1 = input[11]

        batch_size, num_dims_src, num_points = src.size()
        batch_size, _, num_points_tgt = tgt.size()
        batch_size, _, num_points = src_embedding.size()

        ########################## Matching Map Refinement Module ##########################
        distance_map = pairwise_distance_batch(src_embedding, tgt_embedding) #[b, n, m]
        # point-wise matching map
        scores = torch.softmax(-distance_map, dim=2) #[b, n, m]  Eq. (1)
        
        # neighborhood-wise matching map
        src_knn_scores = scores.view(batch_size * num_points, -1)[src_idx1, :]
        src_knn_scores = src_knn_scores.view(batch_size, num_points, k1, num_points) # [b, n, k, m]
        src_knn_scores = pointnet2_utils.gather_operation(src_knn_scores.view(batch_size * num_points, k1, num_points),\
            idx2.view(batch_size, 1, num_points * k1).repeat(1, num_points, 1).view(batch_size * num_points, num_points * k1).int()).view(batch_size,\
                num_points, k1, num_points, k1)[:, :, 1:, :, 1:].sum(-1).sum(2) / (k1-1) # Eq. (2)

        src_knn_scores = self.nn_margin - src_knn_scores
        refined_distance_map = torch.exp(src_knn_scores) * distance_map
        refined_matching_map = torch.softmax(-refined_distance_map, dim=2) # [b, n, m] Eq. (3)

        # pseudo correspondences of source point clouds (pseudo target point clouds)
        src_corr = torch.matmul(tgt, refined_matching_map.transpose(2, 1).contiguous())# [b,3,n] Eq. (4)

        ############################## Inlier Evaluation Module ##############################
        # neighborhoods of pseudo target point clouds
        src_knn_corr = src_corr.transpose(2,1).contiguous().view(batch_size * num_points, -1)[src_idx, :]
        src_knn_corr = src_knn_corr.view(batch_size, num_points, k, num_dims_src)#[b, n, k, 3]

        # edge features of the pseudo target neighborhoods and the source neighborhoods 
        knn_distance = src_corr.transpose(2,1).contiguous().unsqueeze(2) - src_knn_corr #[b, n, k, 3]
        src_knn_distance = src.transpose(2,1).contiguous().unsqueeze(2) - src_knn #[b, n, k, 3]
        
        # inlier confidence
        weight = self.weight_function(knn_distance, src_knn_distance)#[b, 1, n] # Eq. (7)

        # compute rigid transformation 
        R, t = compute_rigid_transformation(src, src_corr, weight) # weighted SVD

        ########################### Preparation for the Loss Function #########################
        # choose k keypoints with highest weights
        src_topk_idx, src_keypoints, tgt_keypoints = get_keypoints(src, src_corr, weight, self.num_keypoints)

        # spatial consistency loss 
        idx_tgt_corr = torch.argmax(refined_matching_map, dim=-1).int() # [b, n]
        identity = torch.eye(num_points_tgt).cuda().unsqueeze(0).repeat(batch_size, 1, 1) # [b, m, m]
        one_hot_number = pointnet2_utils.gather_operation(identity, idx_tgt_corr) # [b, m, n]
        src_keypoints_idx = src_topk_idx.repeat(1, num_points_tgt, 1) # [b, m, num_keypoints]
        keypoints_one_hot = torch.gather(one_hot_number, dim = 2, index = src_keypoints_idx).transpose(2,1).reshape(batch_size * self.num_keypoints, num_points_tgt)
        #[b, m, num_keypoints] - [b, num_keypoints, m] - [b * num_keypoints, m]
        predicted_keypoints_scores = torch.gather(refined_matching_map.transpose(2, 1), dim = 2, index = src_keypoints_idx).transpose(2,1).reshape(batch_size * self.num_keypoints, num_points_tgt)
        loss_scl = (-torch.log(predicted_keypoints_scores + 1e-15) * keypoints_one_hot).sum(1).mean()

        # neighorhood information
        src_keypoints_idx2 = src_topk_idx.unsqueeze(-1).repeat(1, 3, 1, k) #[b, 3, num_keypoints, k]
        tgt_keypoints_knn = torch.gather(knn_distance.permute(0,3,1,2), dim = 2, index = src_keypoints_idx2) #[b, 3, num_kepoints, k]

        src_transformed = transform_point_cloud(src, R, t.view(batch_size, 3))
        src_transformed_knn_corr = src_transformed.transpose(2,1).contiguous().view(batch_size * num_points, -1)[src_idx, :]
        src_transformed_knn_corr = src_transformed_knn_corr.view(batch_size, num_points, k, num_dims_src) #[b, n, k, 3]

        knn_distance2 = src_transformed.transpose(2,1).contiguous().unsqueeze(2) - src_transformed_knn_corr #[b, n, k, 3]
        src_keypoints_knn = torch.gather(knn_distance2.permute(0,3,1,2), dim = 2, index = src_keypoints_idx2) #[b, 3, num_kepoints, k]
        return R, t.view(batch_size, 3), src_keypoints, tgt_keypoints, src_keypoints_knn, tgt_keypoints_knn, loss_scl

class LossFunction(nn.Module):
    def __init__(self, args):
        super(LossFunction, self).__init__()
        self.criterion2 = ChamferLoss()
        self.criterion = nn.MSELoss(reduction='sum')
        self.GAL = GlobalAlignLoss()
        self.margin = args.loss_margin

    def forward(self, *input):
        """
            Compute global alignment loss and neighorhood consensus loss
            Args:
                src_keypoints: Keypoints of source point clouds. Size (B, 3, num_keypoint)
                tgt_keypoints: Keypoints of target point clouds. Size (B, 3, num_keypoint)
                rotation_ab: Size (B, 3, 3)
                translation_ab: Size (B, 3)
                src_keypoints_knn: [b, 3, num_kepoints, k]
                tgt_keypoints_knn: [b, 3, num_kepoints, k]
                k: Number of nearest neighbors.
                src_transformed: Transformed source point clouds. Size (B, 3, N)
                tgt: Target point clouds. Size (B, 3, M)
            Returns:
                neighborhood_consensus_loss
                global_alignment_loss
        """
        src_keypoints = input[0]
        tgt_keypoints = input[1]
        rotation_ab = input[2]
        translation_ab = input[3]
        src_keypoints_knn = input[4]
        tgt_keypoints_knn = input[5]
        k = input[6]
        src_transformed = input[7]
        tgt = input[8]

        batch_size = src_keypoints.size()[0]

        global_alignment_loss = self.GAL(src_transformed.permute(0, 2, 1), tgt.permute(0, 2, 1), self.margin) 
        
        transformed_srckps_forward = transform_point_cloud(src_keypoints, rotation_ab, translation_ab)
        keypoints_loss = self.criterion(transformed_srckps_forward, tgt_keypoints)
        knn_consensus_loss = self.criterion(src_keypoints_knn, tgt_keypoints_knn)
        neighborhood_consensus_loss = knn_consensus_loss/k + keypoints_loss

        return neighborhood_consensus_loss, global_alignment_loss

class LossFunction_kitti(nn.Module):
    def __init__(self, args):
        super(LossFunction_kitti, self).__init__()
        self.criterion2 = ChamferLoss()
        self.criterion = nn.MSELoss(reduction='none')
        self.GAL = GlobalAlignLoss()
        self.margin = args.loss_margin

    def forward(self, *input):
        """
            Compute global alignment loss and neighorhood consensus loss
            Args:
                src_keypoints: Selected keypoints of source point clouds. Size (B, 3, num_keypoint)
                tgt_keypoints: Selected keypoints of target point clouds. Size (B, 3, num_keypoint)
                rotation_ab: Size (B, 3, 3)
                translation_ab: Size (B, 3)
                src_keypoints_knn: [b, 3, num_kepoints, k]
                tgt_keypoints_knn: [b, 3, num_kepoints, k]
                k: Number of nearest neighbors.
                src_transformed: Transformed source point clouds. Size (B, 3, N)
                tgt: Target point clouds. Size (B, 3, M)
            Returns:
                neighborhood_consensus_loss
                global_alignment_loss
        """
        src_keypoints = input[0]
        tgt_keypoints = input[1]
        rotation_ab = input[2]
        translation_ab = input[3]
        src_keypoints_knn = input[4]
        tgt_keypoints_knn = input[5]
        k = input[6]
        src_transformed = input[7]
        tgt = input[8]

        global_alignment_loss = self.GAL(src_transformed.permute(0, 2, 1), tgt.permute(0, 2, 1), self.margin) 
        
        transformed_srckps_forward = transform_point_cloud(src_keypoints, rotation_ab, translation_ab)
        keypoints_loss = self.criterion(transformed_srckps_forward, tgt_keypoints).sum(1).sum(1).mean()
        knn_consensus_loss = self.criterion(src_keypoints_knn, tgt_keypoints_knn).sum(1).sum(1).mean()
        neighborhood_consensus_loss = knn_consensus_loss + keypoints_loss

        return neighborhood_consensus_loss, global_alignment_loss

class RIENET(nn.Module):
    def __init__(self, args):
        super(RIENET, self).__init__()
        self.emb_nn = feature_extractor(args=args)
        self.single_point_embed = PointNet()
        self.forwards = SVDHead(args=args)
        self.iter = args.n_iters 
        if args.dataset == 'kitti':# or args.dataset == 'icl_nuim':
            self.loss = LossFunction_kitti(args)
        else:
            self.loss = LossFunction(args)
        self.list_k1 = args.list_k1
        self.list_k2 = args.list_k2

    def forward(self, *input):
        """ 
            feature extraction.
            Args:
                src = input[0]: Source point clouds. Size [B, 3, N]
                tgt = input[1]: Target point clouds. Size [B, 3, N]
            Returns:
                rotation_ab_pred: Size [B, 3, 3]
                translation_ab_pred: Size [B, 3]
                global_alignment_loss
                consensus_loss
                spatial_consistency_loss
        """

        src = input[0]
        tgt = input[1]
        batch_size, _, _ = src.size()
        rotation_ab_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)
        global_alignment_loss, consensus_loss, spatial_consistency_loss = 0.0, 0.0, 0.0

        for i in range(self.iter):
            src_embedding, src_idx, src_knn, _ = self.emb_nn(src, self.list_k1[i])
            tgt_embedding, _, tgt_knn, _ = self.emb_nn(tgt, self.list_k1[i])

            src_idx1, _ = get_knn_index(src, self.list_k2[i])
            _, tgt_idx = get_knn_index(tgt, self.list_k2[i])

            rotation_ab_pred_i, translation_ab_pred_i, src_keypoints, tgt_keypoints, src_keypoints_knn, tgt_keypoints_knn, spatial_consistency_loss_i\
                = self.forwards(src, tgt, src_embedding, tgt_embedding, src_idx, self.list_k1[i], src_knn, i, tgt_knn,\
                    src_idx1, tgt_idx, self.list_k2[i])

            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ab_pred_i

            src = transform_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)

            neighborhood_consensus_loss_i, global_alignment_loss_i = self.loss(src_keypoints, tgt_keypoints,\
                    rotation_ab_pred_i, translation_ab_pred_i, src_keypoints_knn, tgt_keypoints_knn, self.list_k2[i], src, tgt)

            global_alignment_loss += global_alignment_loss_i
            consensus_loss += neighborhood_consensus_loss_i
            spatial_consistency_loss += spatial_consistency_loss_i
    
        return rotation_ab_pred, translation_ab_pred, global_alignment_loss, consensus_loss, spatial_consistency_loss