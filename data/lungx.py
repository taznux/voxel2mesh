import numpy as np
from data.data import get_item, sample_to_sample_plus
from data.lidc import *

import glob
from utils.metrics import jaccard_index, chamfer_weighted_symmetric
from utils.utils_common import DataModes

import torch
import pickle


  

class LUNGx(LIDC):
    def quick_load_data(self, cfg, trial_id):
        # assert cfg.patch_shape == (64, 256, 256), 'Not supported'
        down_sample_shape = cfg.patch_shape

        data_root = cfg.ext_dataset_path
        data = {}
        for i, datamode in enumerate([DataModes.TRAINING, DataModes.TESTING]):
            with open(data_root + '/pre_computed_data_{}_{}.pickle'.format(datamode, "_".join(map(str, down_sample_shape))), 'rb') as handle:
                samples, sample_pids = pickle.load(handle)
                new_samples = sample_to_sample_plus(samples, cfg, datamode)
                data[datamode] = LIDCDataset(new_samples, sample_pids, cfg, datamode) 

        return data

    def pre_process_dataset(self, cfg):
        data_root = cfg.ext_dataset_path
        samples_train = glob.glob(f"{data_root}CT-Training*s_0*0.npy")
        samples_test = glob.glob(f"{data_root}LUNGx*s_0*0.npy")
 
        pids = []
        inputs = []
        labels = []

        print('Data pre-processing - LUNGx Dataset')
        for sample in samples_train+samples_test:
            if 'pickle' not in sample:
                print('.', end='', flush=True)

                pid = sample.split("/")[-1].split("_")[0]
                pids += [pid]
                x = torch.from_numpy(np.load(sample)[0])
                inputs += [x]
                y = torch.from_numpy(np.load(sample.replace("0.npy", "seg.npy"))) # peak segmenation with nodule area
                y1 = torch.from_numpy(np.load(sample.replace("0.npy", "1.npy"))[0]) # area distortion map
                y2 = torch.from_numpy(np.load(sample.replace("0.npy", "2.npy"))[0]) # nodule segmentation
                y = ((y == 2) + 2*(y == 3)) * (y1 <= 0) # peaks
                y = 3*y2 - y.type(torch.uint8) # apply nodule mask
                
                #y[y==1] = 5 # nodule
                #y[y==2] = 1 # spiculation
                #y[y==3] = 1 # lobulation
                #y[y==4] = 2 # attachment
                #y[y>=5] = 2 # others
                labels += [y]

        print('\nSaving pre-processed data to disk')
        np.random.seed(34234)
        n = len(samples_train)
        m = len(samples_test)
        counts = [range(n), range(n,n+m)]
 
        data = {}
        down_sample_shape = cfg.patch_shape

        for i, datamode in enumerate([DataModes.TRAINING, DataModes.TESTING]):
            samples = []
            sample_pids = []
 
            for j in counts[i]: 
                print('.',end='', flush=True)
                pid = pids[j]
                x = inputs[j]
                y = labels[j]

                samples.append(Sample(x, y)) 
                sample_pids.append(pid)

            with open(data_root + '/pre_computed_data_{}_{}.pickle'.format(datamode, "_".join(map(str, down_sample_shape))), 'wb') as handle:
                pickle.dump((samples, sample_pids), handle, protocol=pickle.HIGHEST_PROTOCOL)

            data[datamode] = LIDCDataset(samples, sample_pids, cfg, datamode)
        
        print('Pre-processing complete') 
        return data
 


 
