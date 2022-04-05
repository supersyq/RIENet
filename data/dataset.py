import numpy
import torch.utils.data
import os
import glob
import copy
import six
import numpy as np
import torch
import torch.utils.data
import torchvision

from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
import se_math.se3 as se3
import se_math.so3 as so3
import se_math.mesh as mesh
import se_math.transforms as transforms

"""
The following three functions are defined for getting data from specific database 
"""

def farthest_subsample_points(pointcloud1, pointcloud2, num_subsampled_points=768):
    pointcloud1 = pointcloud1.T
    pointcloud2 = pointcloud2.T
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1)
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud2)
    random_p2 = random_p1 #np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))
    return pointcloud1[idx1, :].T, pointcloud2[idx2, :].T

def find_classes(root):
    """ find ${root}/${class}/* """
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


# get the indexes from given class names
def classes_to_cinfo(classes):
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


# get the whole 3D point cloud paths for a given class
def glob_dataset(root, class_to_idx, ptns):
    """ glob ${root}/${class}/${ptns[i]} """
    root = os.path.expanduser(root)
    samples = []

    # loop all the folderName (class name) to find the class in class_to_idx
    for target in sorted(os.listdir(root)):
        d = os.path.join(root, target)
        if not os.path.isdir(d):
            continue
        # check if it is the class we want
        target_idx = class_to_idx.get(target)
        if target_idx is None:
            continue
        # to find the all point cloud paths in the class folder
        for i, ptn in enumerate(ptns):
            gptn = os.path.join(d, ptn)
            names = glob.glob(gptn)
            for path in sorted(names):
                item = (path, target_idx)
                samples.append(item)
    return samples


# a general class for obtaining the 3D point cloud data from a database
class PointCloudDataset(torch.utils.data.Dataset):
    """ glob ${rootdir}/${classes}/${pattern}
    """

    def __init__(self, rootdir, pattern, fileloader, transform=None, classinfo=None):
        super().__init__()

        if isinstance(pattern, six.string_types):
            pattern = [pattern]

        # find all the class names
        if classinfo is not None:
            classes, class_to_idx = classinfo
        else:
            classes, class_to_idx = find_classes(rootdir)

        # get all the 3D point cloud paths for the class of class_to_idx
        samples = glob_dataset(rootdir, class_to_idx, pattern)
        if not samples:
            raise RuntimeError("Empty: rootdir={}, pattern(s)={}".format(rootdir, pattern))

        self.fileloader = fileloader
        self.transform = transform

        self.classes = classes
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        define the getitem function for Dataloader of torch
        load a 3D point cloud by using a path index
        :param index:
        :return:
        """
        path, target = self.samples[index]
        sample = self.fileloader(path)
        if self.transform is not None:
            # np.random.seed(index)
            sample = self.transform(sample)

        return sample, target

    def split(self, rate):
        """ dateset -> dataset1, dataset2. s.t.
            len(dataset1) = rate * len(dataset),
            len(dataset2) = (1-rate) * len(dataset)
        """
        orig_size = len(self)
        select = np.zeros(orig_size, dtype=int)
        csize = np.zeros(len(self.classes), dtype=int)

        for i in range(orig_size):
            _, target = self.samples[i]
            csize[target] += 1
        dsize = (csize * rate).astype(int)
        for i in range(orig_size):
            _, target = self.samples[i]
            if dsize[target] > 0:
                select[i] = 1
                dsize[target] -= 1

        dataset1 = copy.deepcopy(self)
        dataset2 = copy.deepcopy(self)

        samples1 = list(map(lambda i: dataset1.samples[i], np.where(select == 1)[0]))
        samples2 = list(map(lambda i: dataset2.samples[i], np.where(select == 0)[0]))

        dataset1.samples = samples1
        dataset2.samples = samples2
        return dataset1, dataset2


class Scene7(PointCloudDataset):
    """ [Scene7 PointCloud](https://github.com/XiaoshuiHuang/fmr) """

    def __init__(self, dataset_path, train=1, transform=None, classinfo=None):
        loader = mesh.plyread
        if train > 0:
            pattern = '*.ply'
        elif train == 0:
            pattern = '*.ply'
        else:
            pattern = ['*.ply', '*.ply']
        super().__init__(dataset_path, pattern, loader, transform, classinfo)

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, partition, n_subsampled_points):
        self.dataset = dataset
        self.partition = partition
        self.rot_factor = 4
        self.num_subsampled_points = n_subsampled_points
        self.subsampled = True
        self.gaussian_noise =False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pointcloud, _ = self.dataset[index]

        if self.partition != 'train':
            np.random.seed(index)
        anglex = np.random.uniform() * np.pi / self.rot_factor
        angley = np.random.uniform() * np.pi / self.rot_factor
        anglez = np.random.uniform() * np.pi / self.rot_factor
        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
       
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        

        pointcloud1 = pointcloud.T #[3,n]

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        
        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T #[3,n]
        if self.subsampled:
            pointcloud1, pointcloud2 = farthest_subsample_points(pointcloud1, pointcloud2,
                                                                 num_subsampled_points=self.num_subsampled_points)

        # igt = np.eye(4)
        # igt[:3,:3] = R_ab
        # igt[:3,3] = translation_ab
        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), translation_ab.astype('float32')

def get_categories(args):
    cinfo = None
    if args.categoryfile:
        categories = [line.rstrip('\n') for line in open(args.categoryfile)]
        categories.sort()
        c_to_idx = {categories[i]: i for i in range(len(categories))}
        cinfo = (categories, c_to_idx)
    return cinfo

def get_datasets(args):
    if args.dataset == '7scenes':
        if args.mode == 'train':
            args.categoryfile = './data/categories/7scene_train.txt'
            cinfo = get_categories(args)
            transform = torchvision.transforms.Compose([ \
                transforms.Mesh2Points(), \
                transforms.OnUnitCube(), \
                transforms.Resampler(args.n_points)])

            dataset = Scene7(args.dataset_path, transform=transform, classinfo=cinfo)
            trainset = TransformedDataset(dataset, partition='train',n_subsampled_points=args.n_subsampled_points)

            args.categoryfile = './data/categories/7scene_test.txt'
            cinfo = get_categories(args)

            transform = torchvision.transforms.Compose([ \
                transforms.Mesh2Points(), \
                transforms.OnUnitCube(), \
                transforms.Resampler(args.n_points), \
                ])

            testdata = Scene7(args.dataset_path, transform=transform, classinfo=cinfo)
            testset = TransformedDataset(testdata, partition = 'test',n_subsampled_points=args.n_subsampled_points)

            return trainset, testset
        else:
            # set path and category file for testing
            args.categoryfile = './data/categories/7scene_test.txt'
            cinfo = get_categories(args)

            transform = torchvision.transforms.Compose([ \
                transforms.Mesh2Points(), \
                transforms.OnUnitCube(), \
                transforms.Resampler(args.n_points), \
                ])

            testdata = Scene7(args.dataset_path, transform=transform, classinfo=cinfo)
            testset = TransformedDataset(testdata, partition = 'test',n_subsampled_points=args.n_subsampled_points)
            return testset
