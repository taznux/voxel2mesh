import numpy as np
from skimage import io
from data.data import get_item, sample_outer_surface_in_voxel, sample_to_sample_plus

import sys
import glob
from utils.metrics import jaccard_index, chamfer_weighted_symmetric, chamfer_directed
from utils.utils_common import crop, DataModes, crop_indices, blend
# from utils.utils_mesh import sample_outer_surface, get_extremity_landmarks, voxel2mesh, clean_border_pixels, sample_outer_surface_in_voxel, normalize_vertices 

# from utils import stns
from torch.utils.data import Dataset
import torch
from sklearn.decomposition import PCA
import pickle
import torch.nn.functional as F
from numpy.linalg import norm
import itertools as itr
import torch
from scipy import ndimage
import os
from IPython import embed
import pydicom

class Sample:
    def __init__(self, x, y, atlas=None):
        self.x = x
        self.y = y
        self.atlas = atlas

class SamplePlus:
    def __init__(self, x, y, y_outer=None, x_super_res=None, y_super_res=None, y_outer_super_res=None, shape=None):
        self.x = x
        self.y = y
        self.y_outer = y_outer
        self.x_super_res = x_super_res
        self.y_super_res = y_super_res  
        self.shape = shape

  
class LIDCDataset():

    def __init__(self, data, cfg, mode): 
        self.data = data  

        self.cfg = cfg
        self.mode = mode
 

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx] 
        return get_item(item, self.mode, self.cfg) 

  

class LIDC():




    def pick_surface_points(self, y_outer, point_count):
        idxs = torch.nonzero(y_outer) 
        perm = torch.randperm(len(idxs))

        y_outer = y_outer * 0  
        idxs = idxs[perm[:point_count]]
        y_outer[idxs[:,0], idxs[:,1], idxs[:,2]] = 1
        return y_outer

    def quick_load_data(self, cfg, trial_id):
        # assert cfg.patch_shape == (64, 256, 256), 'Not supported'
        down_sample_shape = cfg.patch_shape

        data_root = cfg.dataset_path
        data = {}
        for i, datamode in enumerate([DataModes.TRAINING, DataModes.TESTING]):
            with open(data_root + '/pre_computed_data_{}_{}.pickle'.format(datamode, "_".join(map(str, down_sample_shape))), 'rb') as handle:
                samples = pickle.load(handle)
                new_samples = sample_to_sample_plus(samples, cfg, datamode)
                data[datamode] = LIDCDataset(new_samples, cfg, datamode) 

        return data

    def pre_process_dataset(self, cfg):
        '''
         :
        '''
 
        data_root = cfg.dataset_path
        samples = glob.glob(f"{data_root}LIDC*s_0*0.npy")
 
        inputs = []
        labels = []

        print('Data pre-processing - LIDC Dataset')
        for sample in samples:
            if 'pickle' not in sample:
                print('.', end='', flush=True)
                
                x = torch.from_numpy(np.load(sample)[0])
                inputs += [x]
                y = torch.from_numpy(np.load(sample.replace("0.npy", "2.npy"))[0])
                labels += [y]

        print('\nSaving pre-processed data to disk')
        np.random.seed(0)
        perm = np.random.permutation(len(inputs)) 
        counts = [perm[:len(inputs)//2], perm[len(inputs)//2:]]
 
        data = {}
        down_sample_shape = cfg.patch_shape


        for i, datamode in enumerate([DataModes.TRAINING, DataModes.TESTING]):

            samples = []
 

            for j in counts[i]: 
                print('.',end='', flush=True)
                x = inputs[j]
                y = labels[j]

                samples.append(Sample(x, y)) 

            with open(data_root + '/pre_computed_data_{}_{}.pickle'.format(datamode, "_".join(map(str, down_sample_shape))), 'wb') as handle:
                pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

            data[datamode] = LIDCDataset(samples, cfg, datamode)
        
        print('Pre-processing complete') 
        return data
 
    def evaluate(self, target, pred, cfg):
        results = {}


        if target.voxel is not None: 
            val_jaccard = jaccard_index(target.voxel, pred.voxel, cfg.num_classes)
            results['jaccard'] = val_jaccard

        if target.mesh is not None:
            target_points = target.points
            pred_points = pred.mesh
            val_chamfer_weighted_symmetric = np.zeros(len(target_points))

            for i in range(len(target_points)):
                val_chamfer_weighted_symmetric[i] = chamfer_weighted_symmetric(target_points[i].cpu(), pred_points[i]['vertices'])

            results['chamfer_weighted_symmetric'] = val_chamfer_weighted_symmetric

        return results

    def update_checkpoint(self, best_so_far, new_value):

        key = 'jaccard'
        new_value = new_value[DataModes.TESTING][key]

        if best_so_far is None:
            return True
        else:
            best_so_far = best_so_far[DataModes.TESTING][key]
            return True if np.mean(new_value) > np.mean(best_so_far) else False



 