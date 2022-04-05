import numpy as np
import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
import random

def generate_rand_rotm(x_lim=5.0, y_lim=5.0, z_lim=180.0):
    '''
    Input:
        x_lim
        y_lim
        z_lim
    return:
        rotm: [3,3]
    '''
    rand_z = np.random.uniform(low=-z_lim, high=z_lim)
    rand_y = np.random.uniform(low=-y_lim, high=y_lim)
    rand_x = np.random.uniform(low=-x_lim, high=x_lim)

    rand_eul = np.array([rand_z, rand_y, rand_x])
    r = Rotation.from_euler('zyx', rand_eul, degrees=True)
    rotm = r.as_matrix()
    return rotm

def generate_rand_trans(x_lim=10.0, y_lim=1.0, z_lim=0.1):
    '''
    Input:
        x_lim
        y_lim
        z_lim
    return:
        trans [3]
    '''
    rand_x = np.random.uniform(low=-x_lim, high=x_lim)
    rand_y = np.random.uniform(low=-y_lim, high=y_lim)
    rand_z = np.random.uniform(low=-z_lim, high=z_lim)

    rand_trans = np.array([rand_x, rand_y, rand_z])

    return rand_trans

def apply_transform(pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts

def calc_error_np(pred_R, pred_t, gt_R, gt_t):
    tmp = (np.trace(pred_R.transpose().dot(gt_R))-1)/2
    if np.abs(tmp) > 1.0:
        tmp = 1.0
    L_rot = np.arccos(tmp)
    L_rot = 180 * L_rot / np.pi
    L_trans = np.linalg.norm(pred_t - gt_t)
    return L_rot, L_trans

def set_seed(seed):
    '''
    Set random seed for torch, numpy and python
    '''
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed) 
        
    torch.backends.cudnn.benchmark=False 
    torch.backends.cudnn.deterministic=True