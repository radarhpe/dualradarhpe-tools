# """
# Author: trung.l.nguyen@tuni.fi
# Date  : 11-03-2025
# Desc  : This script is for converting 4D Radar DRAE tensor into 4D Radar DZYX tensor
#         for training and testing data with RT-POSE model. Inspired from data processing
#         script of RT-POSE.
# """
import numpy as np
import math
import re
import os
import argparse
import itertools
import time

import sys
# # Get the parent directory of the current file
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

# # Add it to the sys.path
# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)

# from calibration.extrinsic_calib import src2dst_transformation


# path_save_dzyx_c_npy_tail =  'radar/npy_DZYX_real/'
# path_save_fig_tail = 'radar/figure/'
# path_folder_img = 'images/'
# path_mapping_file = 'calib/radarpcl2images_mapping.txt'
# path_folder_tail = 'radar/mat/'

# @cython.boundscheck(False)
# @cython.wraparound(False)
# find index for blinear interpolation
DEG2RAD = 3.14159265 / 180.
RAD2DEG = 180. / 3.14159265
DRAE_SHAPE = [16, 256, 128, 32]
DZYX_CLIP_SHAPE = [16, 16, 48, 196] 

x_min       = 0     #(m)
x_per_bin   = 15.1836 / 256     #(m) 15.1836 / 256
x_max       = 15.1836  #(m) x_max = range size +x_per_bin; 15.1836
x_clip_min  = 0
x_clip_max  = 11.62494375

y_min       = -14.953     #(m) # y_max = np.sin(max_azimuth)*max_range = np.sin(80)*15.1836
y_per_bin   = 14.953*2/128     #(m) #     = 14.953
y_max       = 14.953   #(m)   => y_res = 14.953*2 / 128 = 0.233640625
y_clip_min  = -5.607374999999999
y_clip_max  = 5.607374999999999

z_min       = -5.193    #(m)      z_max = np.sin(max_elevation)*max_range = np.sin(20)*15.1836
z_per_bin   = 5.193*2/32  #(m)           = 5.193
z_max       = 5.193   #(m)     => z_res = 5.193*2 / 32 = 0.3245625
z_clip_min  = -2.5965
z_clip_max  = 2.5965

arr_x = np.arange(x_min,x_max,x_per_bin)
arr_y = np.arange(y_min,y_max,y_per_bin)
arr_y[abs(arr_y) < 1e-10] = 0.0
# print(arr_y.shape)
arr_z = np.arange(z_min,z_max,z_per_bin)

def get_min_max_idx(arr, min_val, max_val):
    min_idx = np.argmin(abs(min_val - arr))
    max_idx = np.argmin(abs(max_val - arr))
    return [min_idx, max_idx]

x_clip_min_idx, x_clip_max_idx = get_min_max_idx(arr_x, x_clip_min, x_clip_max)
y_clip_min_idx, y_clip_max_idx = get_min_max_idx(arr_y, y_clip_min, y_clip_max)
z_clip_min_idx, z_clip_max_idx = get_min_max_idx(arr_z, z_clip_min, z_clip_max)

print(x_clip_min_idx, x_clip_max_idx)
print(y_clip_min_idx, y_clip_max_idx)
print(z_clip_min_idx, z_clip_max_idx)

len_x = len(arr_x[x_clip_min_idx:x_clip_max_idx])
len_y = len(arr_y[y_clip_min_idx:y_clip_max_idx])
len_z = len(arr_z[z_clip_min_idx:z_clip_max_idx])
len_d = 16 # 16

arrAzimuth      = np.arange(0,160,160/128)
arrElevation    = np.arange(0,40,40/32)
arrRange        = np.arange(0,15.1836,15.1836/256)

arrAzimuthRad   = arrAzimuth*DEG2RAD
arrElevationRad = arrElevation*DEG2RAD

#set min/max r,a,e 
r_max = 15.1836
r_min = 0.
r_res = r_max / 256
a_max = 160.*DEG2RAD
a_min = 0.*DEG2RAD
a_res = 160./128*DEG2RAD
e_max = 40.*DEG2RAD
e_min = 0.*DEG2RAD
e_res = 40./32*DEG2RAD

# scan all xyz <=> rae
class RTParsing():
    def __init__(self, T_ti2calib=None):
        st = time.time()
        # Capital letter => idx. Normal letter => measured unit (metre, degree)
        self.XYZ_combs = np.array(list(itertools.product(*[np.arange(x_clip_min_idx, x_clip_max_idx,1), 
                                                    np.arange(y_clip_min_idx, y_clip_max_idx,1),
                                                    np.arange(z_clip_min_idx, z_clip_max_idx,1)])))

        self.xyz_combs = np.array(list(itertools.product(*[arr_x[x_clip_min_idx:x_clip_max_idx], 
                                                    arr_y[y_clip_min_idx:y_clip_max_idx], 
                                                    arr_z[z_clip_min_idx:z_clip_max_idx]])))

        # self.xyz_combs = src2dst_transformation(np.linalg.inv(T_ti2calib), xyz_combs_calib)

        rea_combs = np.zeros((self.xyz_combs.shape[0], self.xyz_combs.shape[1]), dtype=float)
        # r
        rea_combs[:,0] = np.sqrt(self.xyz_combs[:,0]**2 + self.xyz_combs[:,1]**2 + self.xyz_combs[:,2]**2)
        # a
        tmp = self.xyz_combs[:,1]==0
        rea_combs[tmp, 2] = np.pi / 2
        rea_combs[~tmp, 2] = np.arctan(self.xyz_combs[~tmp,0]/self.xyz_combs[~tmp,1])
        rea_combs[rea_combs[:,2]<0, 2] += np.pi
        rea_combs[:,2] = 170*DEG2RAD-rea_combs[:,2]

        # e
        tmp = (self.xyz_combs[:,0]**2 + self.xyz_combs[:,1]**2) == 0
        rea_combs[tmp, 1] = np.pi / 2
        rea_combs[~tmp, 1] = np.arctan(self.xyz_combs[~tmp,2] / np.sqrt(self.xyz_combs[~tmp,0]**2 + self.xyz_combs[~tmp,1]**2))
        rea_combs[:,1] += 20*DEG2RAD

        valid_ind_r = np.logical_and(rea_combs[:,0] >= r_min, rea_combs[:,0] <= r_max)#-r_res)
        valid_ind_e = np.logical_and(rea_combs[:,1] >= e_min, rea_combs[:,1] <= e_max)#-e_res)
        valid_ind_a = np.logical_and(rea_combs[:,2] >= a_min, rea_combs[:,2] <= a_max)#-a_res)
        valid_ind   = np.logical_and(np.logical_and(valid_ind_r, valid_ind_e), valid_ind_a)

        # rae => RAE
        # print(self.xyz_combs[2000:2030,:])
        # print(rea_combs[2000:2030,0])
        self.r_bin_index = (rea_combs[valid_ind,0] // r_res).astype(np.int32)
        self.e_bin_index = (rea_combs[valid_ind,1] // e_res).astype(np.int32) # Adjust back to bin index
        self.a_bin_index = (rea_combs[valid_ind,2] // a_res).astype(np.int32) # Adjust back to bin index

        self.r_bin_next_index = self.r_bin_index + 1
        self.r_bin_next_index[self.r_bin_next_index > 255] = 255
        self.e_bin_next_index = self.e_bin_index + 1
        self.e_bin_next_index[self.e_bin_next_index > 31] = 31
        self.a_bin_next_index = self.a_bin_index + 1
        self.a_bin_next_index[self.a_bin_next_index > 127] = 127

        self.r_index_prev = rea_combs[valid_ind,0] - arrRange[self.r_bin_index]
        self.r_index_next = arrRange[self.r_bin_next_index] - rea_combs[valid_ind,0]
        self.e_index_prev = rea_combs[valid_ind,1] - arrElevationRad[self.e_bin_index]
        self.e_index_next = arrElevationRad[self.e_bin_next_index] - rea_combs[valid_ind,1]
        self.a_index_prev = rea_combs[valid_ind,2] - arrAzimuthRad[self.a_bin_index]
        self.a_index_next = arrAzimuthRad[self.a_bin_next_index] - rea_combs[valid_ind,2]

        # Valid XYZ
        self.XYZ_combs = self.XYZ_combs[valid_ind,:]

        # Adjust XYZ_combs to match clipped bins
        self.XYZ_combs[:,0] -= x_clip_min_idx
        self.XYZ_combs[:,1] -= y_clip_min_idx
        self.XYZ_combs[:,2] -= z_clip_min_idx

        en = time.time()
        print(en-st)

    # cdef DRAE_to_DZYX(path_dataset, file):
    def DRAE_to_DZYX(self, arr4DRadar_inp):
        del_cross = 1/(r_res*a_res*e_res)
        
        #s_file_seq = str(file)
        #path = os.path.join(path_dataset,path_folder_tail, s_file_seq)
        #path_save_dzyx_c_npy = os.path.join(path_dataset,path_save_dzyx_c_npy_tail, s_file_seq)
        ## print('Save path: ',path_save_dzyx_c_npy)
        
        # TODO: Enable this
        #-----loading data-----
        ## name 'matr4' is setting from matlab
        #arr4DRadar = np.fromfile(os.path.join(path), dtype=np.float32)
        arr4DRadar_real = arr4DRadar_inp[:16*256*128*32]
        arr4DRadar_imag = arr4DRadar_inp[16*256*128*32:]
        arr4DRadar = arr4DRadar_real + 1j*arr4DRadar_imag
        arr4DRadar = arr4DRadar.reshape(DRAE_SHAPE, order="F")
        
        # Convert complex to real
        arr4DRadar = np.abs(arr4DRadar)

        # arr4DRadar = np.load(os.path.join(path)) # DRAE
        # print(arr4DRadar)
        #-----loading data-----end
        arr4DRadar = np.transpose(arr4DRadar,(0,1,3,2)) # DREA
        # Creat XYZ  numpy
        # arrZYX = np.ones((len_z,len_y,len_x), dtype='complex64')
        # arrDZYX = np.zeros((len_d,len_z,len_y,len_x), dtype='complex64')
        arrDZYX = np.zeros((len_d,len_z,len_y,len_x), dtype=np.float32)

        # print(x_bin_index[valid_ind])

        # Assigning value
        rea000 = arr4DRadar[:, self.r_bin_index, self.e_bin_index, self.a_bin_index]
        rea001 = arr4DRadar[:, self.r_bin_index, self.e_bin_index, self.a_bin_next_index]
        rea010 = arr4DRadar[:, self.r_bin_index, self.e_bin_next_index, self.a_bin_index]
        rea011 = arr4DRadar[:, self.r_bin_index, self.e_bin_next_index, self.a_bin_next_index]
        rea100 = arr4DRadar[:, self.r_bin_next_index, self.e_bin_index, self.a_bin_index]
        rea101 = arr4DRadar[:, self.r_bin_next_index, self.e_bin_index, self.a_bin_next_index]
        rea110 = arr4DRadar[:, self.r_bin_next_index, self.e_bin_next_index, self.a_bin_index]
        rea111 = arr4DRadar[:, self.r_bin_next_index, self.e_bin_next_index, self.a_bin_next_index]

        # print(rea000.shape,rea001.shape,rea010.shape,rea011.shape,rea100.shape,rea101.shape,
        #       rea110.shape,rea111.shape)
        # print(arrDZYX[:,self.XYZ_combs[:,2], self.XYZ_combs[:,1], self.XYZ_combs[:,0]].shape)

        arrDZYX[:,self.XYZ_combs[:,2], self.XYZ_combs[:,1], self.XYZ_combs[:,0]] = \
            del_cross*(rea000*(self.r_index_next)*(self.e_index_next)*(self.a_index_next)
                        + rea001*(self.r_index_next)*(self.e_index_next)*(self.a_index_prev)
                        + rea010*(self.r_index_next)*(self.e_index_prev)*(self.a_index_next)
                        + rea011*(self.r_index_next)*(self.e_index_prev)*(self.a_index_prev)
                        + rea100*(self.r_index_prev)*(self.e_index_next)*(self.a_index_next)
                        + rea101*(self.r_index_prev)*(self.e_index_next)*(self.a_index_prev)
                        + rea110*(self.r_index_prev)*(self.e_index_prev)*(self.a_index_next)
                        + rea111*(self.r_index_prev)*(self.e_index_prev)*(self.a_index_prev))
        #doing reverse of XYZ axis =Y
        arrDZYX = np.flip(arrDZYX,axis=2)

        # np.save(os.path.join(path_save_dzyx_c_npy),arrDZYX)
        # Save fig
        # print(np.max(arr4DRadar), np.min(arr4DRadar))
        return arrDZYX

