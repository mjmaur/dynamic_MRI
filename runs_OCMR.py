import os
import numpy as np
import logging


import sirf.Gadgetron as pMR

from cil.framework import BlockGeometry, ImageGeometry

from reconstructions import OF_smoothHuber_blockCoordinateDescent 

import sys
import time
import subprocess
import shutil
logging.basicConfig(filename='LOGGING_TEST.log', level=logging.CRITICAL)

np.random.seed(10)


IDX = int(sys.argv[1])

params_list = []

for method in [-2, -1]:
    for acceleration in [1, 4]:
        params_list.append((method, acceleration))
                    
alpha_tv_velocity = 0.075
alpha_tv_rho = 0.075
gamma = 0.1
gamma0 = gamma 
PDE_method = params_list[IDX][0]
sigma_smoothing = 2
epsilonHuber = 0.01
epsilonHuber_velocity = 2.
epsilonHuber_density= 0.01

variables_to_smooth = "both"


n_iter_inner_density = 1400
n_iter_inner_velocity = 3200
n_iter_norm_calculation = 25
n_iter_outer = 200

undersampling_factor = params_list[IDX][1] # 1 for fully sampled data


folder = "runs_OCMR/TEST"
    


folder = folder + "/params_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}/".format(alpha_tv_rho, alpha_tv_velocity, gamma, gamma0, PDE_method, epsilonHuber_density, epsilonHuber_velocity, epsilonHuber, sigma_smoothing, undersampling_factor)

os.makedirs(folder, exist_ok=True)




dtype=np.float32
Nx=256
Ny=208
Nz=0
Nms = 8
channels = Nms
tp_list = None


filename_base = 'fs_0001_1_5T.h5'
#filename_base = 'fs_0002_1_5T.h5'
filename = folder + filename_base

shutil.copyfile(filename_base, filename)

acq_data = pMR.AcquisitionData(filename)
try:
    acq_data = pMR.preprocess_acquisition_data(acq_data)
    acq_data.sort()
except:
    try:
        time.sleep(20)
        subprocess.Popen(["gadgetron", "&"])
        acq_data = pMR.preprocess_acquisition_data(acq_data)
        acq_data.sort()
    except:
        time.sleep(20)
        subprocess.Popen(["gadgetron", "&"])
        acq_data = pMR.preprocess_acquisition_data(acq_data)
        acq_data.sort()


shape_dict = {"channels": channels,
                    "voxel_num_z": Nz,
                    "voxel_num_y": Ny,        
                    "voxel_num_x": Nx}
geom_rho = BlockGeometry(ImageGeometry(**shape_dict, dtype=dtype), ImageGeometry(**shape_dict, dtype=dtype))

geom_base = ImageGeometry(**shape_dict, dtype=dtype)
geom_velocity = BlockGeometry(BlockGeometry(geom_base, geom_base),
                                BlockGeometry(geom_base, geom_base))



phase = acq_data.get_ISMRMRD_info('phase', range(acq_data.shape[0]))
n_spokes = np.max(acq_data.get_ISMRMRD_info('kspace_encode_step_1'))
print("n_spokes = {}; shape = {}".format(n_spokes, acq_data.as_array().shape))


## undersampling
if undersampling_factor == 4 or undersampling_factor == 1:
    k_space_fraction_center = 0.15
    k_space_fraction = 0.25
else:
    raise NotImplementedError
center_ky = np.max(acq_data.get_ISMRMRD_info('kspace_encode_step_1')+1) // 2

idx_center = np.arange(int(center_ky - k_space_fraction_center * n_spokes / 2),
                        int(center_ky + k_space_fraction_center * n_spokes / 2),
                        dtype=int)
idx_total = np.arange(0, n_spokes, dtype=int)
idx_samples1 = np.arange(0, idx_center.min(), dtype=int)
idx_samples2 = np.arange(idx_center.max(), n_spokes, dtype=int)

idx_samples1_prev = np.array([], dtype=int)
idx_samples2_prev = np.array([], dtype=int)

idx_list = []
for i in range(Nms):
    idx_samples_curr1 = np.setdiff1d(idx_samples1, idx_samples1_prev)
    idx_samples_curr2 = np.setdiff1d(idx_samples2, idx_samples2_prev)

    np.random.shuffle(idx_samples_curr1)
    np.random.shuffle(idx_samples_curr2)
    idx_samples1_prev = idx_samples_curr1[:int(n_spokes * (k_space_fraction - k_space_fraction_center) / 2)]
    idx_samples2_prev = idx_samples_curr2[:int(n_spokes * (k_space_fraction - k_space_fraction_center) / 2)]
    
    idx_select = np.append(idx_center, idx_samples1_prev)
    idx_select = np.append(idx_select, idx_samples2_prev)
    
    idx_list.append(idx_select)
    


np.save("/scratch/tmp/m_maur07/cambridge/SIRF_MRI/dynamic_2D_publish" + "/IDX_LIST_SAMPLING_MRI_4.npy", np.array(idx_list))


acq_ms = [0] * Nms
E_ms = [0] * Nms
scaling = 0.5 * 1e+04 ## scaling factor to get images with pixel values of about 1

phase = acq_data.get_ISMRMRD_info('phase', range(acq_data.shape[0]))

idx_list = np.load("/scratch/tmp/m_maur07/cambridge/SIRF_MRI/dynamic_2D_publish" + "/IDX_LIST_SAMPLING_MRI_4.npy") ## load subsampling
acq_data *= scaling
for i in range(Nms):
    
    if undersampling_factor == 1:
        idx = np.argwhere(phase == i)[:, 0]
        idx_CSM = np.argwhere(phase == i)[idx_list[i], 0] 
    else:
        idx = np.argwhere(phase == i)[idx_list[i], 0] 
        idx_CSM = idx
    
    acq_ms[i] = acq_data.get_subset(np.sort(idx_CSM))
    acq_ms[i].sort()
    csm = pMR.CoilSensitivityData()
    csm.smoothness = 100
    csm.calculate(acq_ms[i])
    
    acq_ms[i] = acq_data.get_subset(np.sort(idx))
    acq_ms[i].sort()
    
    E_tmp = pMR.AcquisitionModel(acqs=acq_ms[i], imgs=csm)
    E_tmp.set_coil_sensitivity_maps(csm)
    recon_adj = E_tmp.inverse(acq_ms[i])
    E_ms[i] = pMR.AcquisitionModel(acqs=acq_ms[i], imgs=recon_adj)
    E_ms[i].set_coil_sensitivity_maps(csm)

os.makedirs(folder, exist_ok=True)


if PDE_method == -2: ## FW
    gamma0 = 0
    gamma = 0
    n_iter_outer = 1

if (PDE_method == -1):
        n_iter_outer = 1
        gamma0 = gamma
    
velocity_init = None
rho_init = geom_rho.allocate()

OF_smoothHuber_blockCoordinateDescent(geom_rho,
                                    geom_velocity,
                                    acq_model_list=E_ms,
                                    acq_data_list=acq_ms,
                                    gamma=gamma,
                                    alpha_rho=alpha_tv_rho,
                                    alpha_velocity=alpha_tv_velocity,
                                    n_iter_norm_calculation=n_iter_norm_calculation,
                                    n_iter_outer=n_iter_outer,
                                    n_iter_inner_density=n_iter_inner_density,
                                    n_iter_inner_velocity=n_iter_inner_velocity, 
                                    folder=folder,
                                    gamma0=gamma0,
                                    rho_init=rho_init, 
                                    OF_method=PDE_method, 
                                    velocity_init=velocity_init,
                                    epsilonHuber=epsilonHuber,
                                    epsilonHuber_velocity = epsilonHuber_velocity,
                                    epsilonHuber_density=epsilonHuber_density,
                                    first_update="density",
                                    sigma_smoothing_init=sigma_smoothing,
                                    tp_list=tp_list,
                                    variables_to_smooth=variables_to_smooth)
        
 
            
  