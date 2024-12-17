import time
import numpy as np
import logging
import os
import sys
import subprocess
import shutil

from skimage.metrics import peak_signal_noise_ratio as PSNR


import sirf.Gadgetron as pMR

from cil.framework import BlockGeometry, ImageGeometry

from reconstructions import OF_smoothHuber_blockCoordinateDescent 



logging.basicConfig(filename='LOGGING_TEST.log', level=logging.CRITICAL)

np.random.seed(10)

def compute_PSNR(rho, rho_GT, selection_mask):
   
    METRIC_PSNR_selected = np.zeros(9)
    for tp in range(8):
        v_max = np.abs(rho_GT[tp, selection_mask]).max()
        METRIC_PSNR_selected[tp] = PSNR(np.abs(rho_GT[tp, selection_mask]), np.abs(rho[tp, selection_mask]), data_range=v_max)
    METRIC_PSNR_selected[-1] = np.mean(METRIC_PSNR_selected[:-1])
    
    
    return METRIC_PSNR_selected



def train_function(config, PDE_method):
    
    
    base_folder = config["base_folder"] 
    
    alpha_tv_velocity = config['alpha_tv_velocity']
    alpha_tv_rho = config['alpha_tv_rho']
    gamma = config['gamma']
    gamma0 = config['gamma0_factor'] * gamma
    PDE_method = PDE_method
    sigma_smoothing = config['sigma_smoothing']
    variables_to_smooth = "both"

    n_iter_inner_density = 1400
    n_iter_inner_velocity = 3200
    
    n_iter_norm_calculation = 25
    n_iter_outer = 200
    epsilonHuber = config['epsilonHuber_OF']
    epsilonHuber_velocity = config['epsilonHuber_velocity']
    epsilonHuber_density = config['epsilonHuber_rho']
    real_valued_recon = config['real_valued_recon']
    
    
    folder = config["base_folder_results"] 
    
    folder = folder + config["folder_run"]
    
    os.makedirs(folder, exist_ok=True)
    
    dtype=np.float32
    Nx=256
    Ny=208
    Nz=0
    Nms = 8
    channels = Nms

    ## chose subject and dynamic/non-dynamic GT
    SBJ = "fs_0001" #fs_0002
    DYN = True #False
    SIMULATION = False #True

    if SBJ == "fs_0001":
        filename_base = 'fs_0001_1_5T.h5'
    elif SBJ == "fs_0002":
        filename_base = 'fs_0002_1_5T.h5'
        
    filename = folder + filename_base
    filename_base = base_folder + filename_base
    
    shutil.copyfile(filename_base, filename)
    
    acq_data = pMR.AcquisitionData(filename)
    
    ## sometimes, gadgetron won't start -> try again
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


    
    acq_ms = [0] * Nms
    E_ms = [0] * Nms
    scaling = 0.5 * 1e+04 ## scaling factor to get images with pixel values of about 1
    
    phase = acq_data.get_ISMRMRD_info('phase', range(acq_data.shape[0]))
    
    idx_list = np.load(base_folder + "/IDX_LIST_SAMPLING_MRI_4.npy") ## load subsampling
    acq_data *= scaling
    for i in range(Nms):
        idx = np.argwhere(phase == i)[idx_list[i], 0] 
        idx_CSM = np.argwhere(phase == i)[idx_list[i], 0]
        
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

    
    if SIMULATION:
        if SBJ != "fs_0002":
            raise NotImplementedError
        data_simulation = np.load("recons_simulation_noisy/fs0002_data_subsampled_simulation.npy")
        for i in range(channels):
            acq_ms[i].fill(data_simulation[i])
    

    os.makedirs(folder, exist_ok=True)
    
    
    if (PDE_method == -1) or (PDE_method == 3):
        n_iter_outer = 1
        gamma0 = gamma
    
    if PDE_method == -2: ## FW
        gamma0 = 0
        gamma = 0
        n_iter_outer = 1
        
    if PDE_method == 3: #cheat-OF
        PDE_method = 2
       
        ## load corresponding velocities
        v_Y = np.load("recons_simulation_noisy/v_Y_simulation.npy") 
        v_X = np.load("recons_simulation_noisy/v_X_simulation.npy") 
        
        velocity_init = geom_velocity.allocate()
        velocity_init[1][0].fill(v_Y.imag)
        velocity_init[1][1].fill(v_X.imag)
        velocity_init[0][0].fill(v_Y.real)
        velocity_init[0][1].fill(v_X.real)
        rho_init = geom_rho.allocate()
    else:
        velocity_init = None
    
        rho_init = geom_rho.allocate()

    
    
    rho, iter = OF_smoothHuber_blockCoordinateDescent(geom_rho,
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
                                                real_valued_recon=real_valued_recon,
                                                variables_to_smooth=variables_to_smooth)
    
    
    
    if SBJ == "fs_0001":
        selection_mask = np.load("selection_mask_fs_0001.npy")
        if DYN == True:
            rho_GT = np.load("recon_GT_fs_0001_dyn.npy")
        else:
            rho_GT = np.load("recon_GT_fs_0001.npy")
    elif SBJ == "fs_0002":
        selection_mask = np.load("selection_mask_fs_0002.npy")
        if SIMULATION:
            rho_GT = np.load("recons_simulation_noisy/rho_simulation.npy")
        else:
            if DYN == True:
                rho_GT = np.load("recon_GT_fs_0002_dyn.npy")
            else:
                rho_GT = np.load("recon_GT_fs_0002.npy")
    
    PSNR_val = compute_PSNR(rho[0].as_array() + 1j * rho[1].as_array(), rho_GT, selection_mask)
    np.save(folder + "/PSNR_final.npy", PSNR_val)
    np.save(folder + "/ITER_final.npy", iter)


if __name__ == "__main__":

    param_dict = {
        'alpha_tv_velocity': float(sys.argv[1]),
        'alpha_tv_rho': float(sys.argv[2]),
        'gamma': float(sys.argv[3]),
        'gamma0_factor': float(sys.argv[4]),
        'epsilonHuber_rho': float(sys.argv[5]),
        'epsilonHuber_velocity': float(sys.argv[6]),
        'epsilonHuber_OF': float(sys.argv[7]),
        'sigma_smoothing': float(sys.argv[8]),
        'base_folder' : str(sys.argv[9]),
        'base_folder_results' : str(sys.argv[10]),
        'folder_run' : str(sys.argv[11]),
        'real_valued_recon' : sys.argv[13].lower() == 'true',
    }
    
    train_function(param_dict, float(sys.argv[12]))