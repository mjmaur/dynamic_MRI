import subprocess
import numpy as np
import logging
import os
import time
import subprocess
import shutil

from skopt import Optimizer
from skopt.space import Real



logging.basicConfig(filename='LOGGING_TEST.log', level=logging.CRITICAL)

np.random.seed(10)


def get_folder_name_from_params(alpha_tv_velocity, alpha_tv_rho, gamma, gamma0, PDE_method,
                                epsilonHuber_rho, epsilonHuber_velocity, epsilonHuber_OF, sigma_smoothing):
    return "/params_{}_{}_{}_{}_{}_{}_{}_{}_{}/".format(alpha_tv_rho, alpha_tv_velocity, gamma, gamma0,
                                                        PDE_method, epsilonHuber_rho, epsilonHuber_velocity,
                                                        epsilonHuber_OF, sigma_smoothing)

def get_params_from_run_name(run_name):
    tmp_str = run_name.partition('_')[-1]
    alpha_tv_rho, _, tmp_str = tmp_str.partition('_')
    alpha_tv_rho = float(alpha_tv_rho)

    alpha_tv_velocity, _, tmp_str = tmp_str.partition('_')
    alpha_tv_velocity = float(alpha_tv_velocity)

    gamma, _, tmp_str = tmp_str.partition('_')
    gamma = float(gamma)
    
    gamma0, _, tmp_str = tmp_str.partition('_')
    gamma0 = float(gamma0)

    CE_method, _, tmp_str = tmp_str.partition('_')
    CE_method = float(CE_method)
    
    epsilonHuber_rho, _, tmp_str = tmp_str.partition('_')
    epsilonHuber_rho = float(epsilonHuber_rho)
    
    epsilonHuber_velocity, _, tmp_str = tmp_str.partition('_')
    epsilonHuber_velocity = float(epsilonHuber_velocity)
    
    epsilonHuber_OF, _, tmp_str = tmp_str.partition('_')
    epsilonHuber_OF = float(epsilonHuber_OF)

    sigma_smoothing = float(tmp_str)
    
    return alpha_tv_velocity, alpha_tv_rho, gamma, gamma0, CE_method, epsilonHuber_rho, epsilonHuber_velocity, epsilonHuber_OF, sigma_smoothing


def get_job_status(job_id):
    try:
        result = subprocess.run(['squeue', '--format=%T', '--noheader', '--job', str(job_id)], capture_output=True, text=True, check=True)

        output = result.stdout.strip()
        return output 
    except subprocess.CalledProcessError:
        return "ERROR"



if __name__ == "__main__":
    
    n_parallel_runs = 20
    n_runs_total = 60
    n_initial_points= 20
    real_valued_recon = 'false'
    
    OF_method = 2 
    # -2: FW
    # -1: DT model
    # 2: complex OF model
    # 3: cheat-OF
    
    
    if OF_method == 2: # OF model
        search_space = [
            Real(0.01, 2., prior="log-uniform"), #alpha_tv_velocity
            Real(0.001, 0.5, prior="log-uniform"), #alpha_tv_rho
            Real(0.01, 2., prior="log-uniform"), #gamma
            Real(0.01, 1., prior="log-uniform"), #gamma0_factor
            Real(0.01, 1., prior="log-uniform"), #epsilonHuber_rho
            Real(0.01, 2., prior="log-uniform"), #epsilonHuber_velocity
            Real(0.01, 1., prior="log-uniform"), #epsilonHuber_OF 
            Real(0., 6.) #sigma_smoothing
        ]
        
    elif (OF_method == -1) or (OF_method == 3): #OF_method == -1 -> DT, model considering time derivatives
        #OF_method == 3 -> OFV, OF with known velocity
        search_space = [
            Real(0.001, 1, prior="log-uniform"), #alpha_tv_rho 
            Real(0.001, 2., prior="log-uniform"), #gamma
            Real(0.01, 1., prior="log-uniform"), #epsilonHuber_rho
            Real(0.01, 1., prior="log-uniform") #epsilonHuber_OF 
        ]
        
    elif (OF_method == -2): #Frame-wise reconstruction
        search_space = [
            Real(0.001, 1, prior="log-uniform"), #alpha_tv_rho 
            Real(0.01, 1., prior="log-uniform"), #epsilonHuber_rho
        ]
    else:
        raise NotImplementedError
    
    base_folder = "/scratch/tmp/m_maur07/cambridge/SIRF_MRI/dynamic_2D_publish/"
    path_batch_script_template = "/scratch/tmp/m_maur07/cambridge/SIRF_MRI/dynamic_2D_publish/submit_tuning_template.sh"
    folder_batch_scripts = "/scratch/tmp/m_maur07/cambridge/SIRF_MRI/dynamic_2D_publish/tune_batch_scripts/"
    folder_results = "/scratch/tmp/m_maur07/cambridge/SIRF_MRI/dynamic_2D_publish/tune_results/fs_0001_OF/"
    initial_runs = False
    
    res_list = os.listdir(folder_results)
    initial_params_list = []
    initial_vals_list = []
    for res_dir in res_list:
        try:
            PSNR_val = np.load(folder_results + res_dir + "/PSNR_final.npy")
            initial_vals_list.append(PSNR_val[-1])
            
            alpha_tv_velocity, alpha_tv_rho, gamma, gamma0_factor, CE_method, epsilonHuber_rho, epsilonHuber_velocity, epsilonHuber_OF, sigma_smoothing = get_params_from_run_name(res_dir)
                
            if OF_method == 2.:
                initial_params_list.append([alpha_tv_velocity, alpha_tv_rho, gamma, gamma0_factor, epsilonHuber_rho, epsilonHuber_velocity, epsilonHuber_OF, sigma_smoothing])
            elif (OF_method == -1) or (OF_method == 3):
                initial_params_list.append([alpha_tv_rho, gamma, epsilonHuber_rho, epsilonHuber_OF])
        except:
            pass
    
    print(initial_vals_list)
    print(initial_params_list)
    
    if not initial_runs:
        xi_0 = 0.01
        OPT = Optimizer(search_space,
                        base_estimator="GP",
                        acq_func="EI",
                        n_initial_points=n_initial_points,
                        random_state=10,
                        acq_func_kwargs={'xi': xi_0},
                        acq_optimizer_kwargs={'n_jobs' : -1})
        for param, val in zip(initial_params_list, initial_vals_list):
            print(param)
            OPT.tell(param, -1 * val)
        
    n_runs_finished = 0
    n_running_jobs = 0
    running_jobs_list = []
    counter = 0
    
    if initial_runs: #choose set of initial parameters to evaluate the models (can be omitted to choose initial parameters randomly)
        if OF_method == 2:
            first_params_to_try = []
            alpha_rho_list = [0.01, 0.1]
            alpha_vel_list = [1.0, 0.1]
            eps_list = [0.01, 0.1]
            eps_vel_list = [0.1, 1]
            eps_rho_list = [0.01, 0.1]
            gamma0_factor_list = [0.1, 1.]
            for alpha_rho in alpha_rho_list:
                for alpha_vel in alpha_vel_list:
                    for eps in eps_list:
                        for eps_vel in eps_vel_list:
                            for eps_rho in eps_rho_list:
                                for gamma0_factor in gamma0_factor_list:
                                    first_params_to_try.append([alpha_vel, alpha_rho, 1., gamma0_factor, eps_rho, eps_vel, eps, 3.])
        elif (OF_method == -1) or (OF_method == 3):
            first_params_to_try = []
            alpha_rho_list = [0.01, 0.1]
            eps_list = [0.01, 0.1]
            eps_rho_list = [0.01, 0.1]
            for alpha_rho in alpha_rho_list:
                for eps in eps_list:
                    for eps_rho in eps_rho_list:
                        first_params_to_try.append([alpha_rho, 1., eps_rho,  eps])           
                        
        elif OF_method == -2:
            first_params_to_try = []
            alpha_rho_list = [0.001, 0.01, 0.1]
            eps_rho_list = [0.01, 0.1]
            for alpha_rho in alpha_rho_list:
                for eps_rho in eps_rho_list:
                    first_params_to_try.append([alpha_rho, eps_rho])           
        else:
            raise NotImplementedError
    
    while (counter < n_runs_total):
        if len(running_jobs_list) < n_parallel_runs:
           

            if initial_runs:
                new_params = first_params_to_try
                n_new_params = len(first_params_to_try)
            else:
                n_new_params = n_parallel_runs - len(running_jobs_list)
                new_params = OPT.ask(n_new_params)
            print(new_params)
            for i in range(n_new_params):
            
                shutil.copyfile(path_batch_script_template, folder_batch_scripts + "submit_{}.sh".format(counter))
                
                
                if (OF_method == -1) or (OF_method == 3):
                    folder_run = get_folder_name_from_params(0., #alpha_tv_velocity
                                                            new_params[i][0], #alpha_tv_rho
                                                            new_params[i][1], #gamma
                                                            1., #gamma0_factor
                                                            OF_method,  # OF-methods
                                                            new_params[i][2], #epsilonHuber_rho
                                                            0., ## epsilonHuber_OF
                                                            new_params[i][3], #epsilonHuber_OF
                                                            0.) #sigma_smoothing
                                                            
                    
                    line_to_add = ["python3",
                                "paramTuning_optimization.py",
                                str(0.),  #alpha_tv_velocity
                                str(new_params[i][0]), #alpha_tv_rho
                                str(new_params[i][1]), #gamma
                                str(1.), #gamma0_factor
                                str(new_params[i][2]), #epsilonHuber_rho
                                str(0.),
                                str(new_params[i][2]), #epsilonHuber_OF
                                str(0.),  #sigma_smoothing
                                base_folder, # base_folder
                                folder_results, # base_folder_results
                                folder_run, #folder_run
                                str(OF_method), #PDE-method,
                                real_valued_recon,
                                '>',
                                '/dev/null'
                                ]
                elif (OF_method == -2):
                    folder_run = get_folder_name_from_params(0., #alpha_tv_velocity
                                                            new_params[i][0], #alpha_tv_rho
                                                            0., #gamma
                                                            0., #gamma0_factor
                                                            OF_method,  # OF-methods
                                                            new_params[i][1], #epsilonHuber_rho
                                                            0., ## epsilonHuber velocity
                                                            1., #epsilonHuber_OF
                                                            0.) #sigma_smoothing
                                                            
                    
                    line_to_add = ["python3",
                                "paramTuning_optimization.py",
                                str(0.), #alpha_tv_velocity
                                str(new_params[i][0]), #alpha_tv_rho
                                str(0.), #gamma
                                str(1.), #gamma0_factor
                                str(new_params[i][1]), #epsilonHuber_rho
                                str(0.),
                                str(1.), #epsilonHuber_OF
                                str(0.), #sigma_smoothing
                                base_folder, # base_folder
                                folder_results, # base_folder_results
                                folder_run, #folder_run
                                str(OF_method), #PDE-method,
                                real_valued_recon,
                                '>',
                                '/dev/null'
                                ]
                
                elif OF_method == 2:
                    folder_run = get_folder_name_from_params(new_params[i][0],
                                                             new_params[i][1],
                                                             new_params[i][2],
                                                             new_params[i][3],
                                                             2,
                                                             new_params[i][4],
                                                             new_params[i][5],
                                                             new_params[i][6],
                                                             new_params[i][7])
                    
                    
                    line_to_add = ["python3",
                                "paramTuning_optimization.py",
                                str(new_params[i][0]), #alpha_tv_velocity
                                str(new_params[i][1]), #alpha_tv_rho
                                str(new_params[i][2]), #gamma
                                str(new_params[i][3]), #gamma0_factor
                                str(new_params[i][4]), #epsilonHuber_rho
                                str(new_params[i][5]), #epsilonHuber_velocity
                                str(new_params[i][6]), #epsilonHuber_OF
                                str(new_params[i][7]), #igma_smoothing
                                base_folder, # base_folder
                                folder_results, # base_folder_results
                                folder_run, #folder_run
                                str(2.), #PDE-method
                                real_valued_recon,
                                '>',
                                '/dev/null'
                                ]
                line_to_add = " ".join(line_to_add)
                with open(folder_batch_scripts + "submit_{}.sh".format(counter), 'a') as file:
                    file.write("\n" + line_to_add + "\n")
                
                    
                job_id = subprocess.run(['sbatch', folder_batch_scripts + "submit_{}.sh".format(counter)], stdout=subprocess.PIPE, text=True)
                job_id = job_id.stdout.strip().split()[-1]
                print("job_id = {}".format(job_id))
                running_jobs_list.append([job_id, folder_results + folder_run, folder_batch_scripts + "submit_{}.sh".format(counter), new_params[i], 0])
                counter += 1
                time.sleep(3)
                
                
        if initial_runs:
            break
        
        
        time.sleep(500) # wait some time until check for running/finished jobs
        items_to_keep = []
        for i, elem in enumerate(running_jobs_list):
            job_status = get_job_status(job_id=elem[0])
            print(job_status)
            if job_status == "RUNNING" or job_status == "PENDING":
                items_to_keep.append(i)
            else:
                if os.path.exists(elem[1]+"/PSNR_final.npy"):
                    PSNR_val = np.load(elem[1]+"/PSNR_final.npy")
                    OPT.tell(elem[3], -1 * PSNR_val[-1])
                    print("New value = {}".format(PSNR_val[-1]))
                elif elem[4]<4:##max number of resubmits for crashed jobs
                    try:
                        elem[4] += 1
                        job_id = subprocess.run(['sbatch', folder_batch_scripts + "submit_{}.sh".format(counter)], stdout=subprocess.PIPE, text=True)
                        print("job_id", job_id)
                        job_id = job_id.stdout.strip().split()[-1]
                        elem[0] = job_id
                        items_to_keep.append(i)
                    except:
                        pass
                    
        running_jobs_list = [running_jobs_list[i] for i in items_to_keep]           
            
    