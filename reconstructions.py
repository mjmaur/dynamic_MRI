import numpy as np

from operators import TimeResolvedMRIForwardOperator, TimeDerivative, SelectItems, HuberLoss, OFOperator_density, OFOperator_velocity
from cil.optimisation.operators import GradientOperator, LinearOperator, CompositionOperator
from cil.optimisation.functions import L2NormSquared, BlockFunction, ZeroFunction, LeastSquares, IndicatorBox, OperatorCompositionFunction, ConstantFunction
from cil.optimisation.algorithms import FISTA
from cil.framework import BlockDataContainer
from scipy.ndimage import gaussian_filter




class FISTA_with_early_stopping(FISTA):
    
    def __init__(self, initial, f, g, step_size = None, **kwargs):

        self.y = initial.copy()
        self.y_save = initial.copy()
        self.t = 1
        super().__init__(initial=initial, f=f, g=g, step_size=step_size, **kwargs)
        
        

    def should_stop(self):
        if self.iteration >= 60:
            tmp = (self.x - self.x_old).norm()
            tmp2 = self.x_old.norm()
            tmp /= tmp2
            
            if self.iteration % 10 == 0:
                print("Fista update of solution (relative) = {}".format(tmp))
            
            return tmp < 1e-05  or self.max_iteration_stop_criterion()
        
        else:
            return self.max_iteration_stop_criterion()


def OF_smoothHuber_blockCoordinateDescent(geom_rho, geom_velocity, acq_model_list, acq_data_list,
                                             gamma=1., alpha_rho=0.01, alpha_velocity=0.1, n_iter_norm_calculation=50, n_iter_outer=10, n_iter_inner_density=500, n_iter_inner_velocity=500, 
                                             folder="runs/", gamma0=0.1, rho_init=None, OF_method=2,
                                             velocity_init=None, epsilonHuber=0.01, epsilonHuber_velocity=1., epsilonHuber_density=1.0,
                                             first_update="density", sigma_smoothing_init=2, tp_list=None, real_valued_recon=False,
                                             variables_to_smooth="density"):
    
    
    use_grad_velocity = True
    use_Huber_velocity = True ## else use L2Squared
    use_joint_spatial_TV_on_density = True

    acq_data = BlockDataContainer(*acq_data_list)
    
    MRI_operator = TimeResolvedMRIForwardOperator(acq_model_list, geom_rho, tp_list=tp_list)
    norm_MRI = LinearOperator.PowerMethod(MRI_operator, max_iteration=n_iter_norm_calculation)
    print("norm_MRI = {}".format(norm_MRI))
    
    select_realDensityOp = SelectItems(geom_rho, [0])
    density_op_real = GradientOperator(domain_geometry=select_realDensityOp.range_geometry())
    density_op_real = CompositionOperator(density_op_real, select_realDensityOp)
    norm_density_op_real = LinearOperator.PowerMethod(density_op_real, max_iteration=n_iter_norm_calculation+50)
    
    select_imagDensityOp = SelectItems(geom_rho, [1])
    density_op_imag = GradientOperator(domain_geometry=select_imagDensityOp.range_geometry())
    density_op_imag = CompositionOperator(density_op_imag, select_imagDensityOp)
    norm_density_op_imag = LinearOperator.PowerMethod(density_op_imag, max_iteration=n_iter_norm_calculation+50)



    ## velocity
    timeDerivative_operator = TimeDerivative(geom_rho.get_item(0))
    
    select_v_X_r = CompositionOperator(SelectItems(geom_velocity.get_item(0), [1]), SelectItems(geom_velocity, [0]))
    if use_grad_velocity:
        op_v_X_r = GradientOperator(domain_geometry=select_v_X_r.range_geometry())
        op_v_X_r = CompositionOperator(op_v_X_r, select_v_X_r)
    else:
        op_v_X_r = select_v_X_r
    norm_grad_velocity = LinearOperator.PowerMethod(op_v_X_r, max_iteration=n_iter_norm_calculation)
    
    select_v_X_i = CompositionOperator(SelectItems(geom_velocity.get_item(1), [1]), SelectItems(geom_velocity, [1]))
    if use_grad_velocity:
        op_v_X_i = GradientOperator(domain_geometry=select_v_X_i.range_geometry())
        op_v_X_i = CompositionOperator(op_v_X_i, select_v_X_i)
    else:
        op_v_X_i = select_v_X_i
    norm_grad_velocity = np.max([norm_grad_velocity, LinearOperator.PowerMethod(op_v_X_i, max_iteration=n_iter_norm_calculation)])
    
    select_v_Y_r = CompositionOperator(SelectItems(geom_velocity.get_item(0), [0]), SelectItems(geom_velocity, [0]))
    if use_grad_velocity:
        op_v_Y_r = GradientOperator(domain_geometry=select_v_Y_r.range_geometry())
        op_v_Y_r = CompositionOperator(op_v_Y_r, select_v_Y_r)
    else:
        op_v_Y_r = select_v_Y_r
    norm_grad_velocity = np.max([norm_grad_velocity, LinearOperator.PowerMethod(op_v_Y_r, max_iteration=n_iter_norm_calculation)])
    
    select_v_Y_i = CompositionOperator(SelectItems(geom_velocity.get_item(1), [0]), SelectItems(geom_velocity, [1]))
    if use_grad_velocity:
        op_v_Y_i = GradientOperator(domain_geometry=select_v_Y_i.range_geometry())
        op_v_Y_i = CompositionOperator(op_v_Y_i, select_v_Y_i)
    else:
        op_v_Y_i = select_v_Y_i
    norm_grad_velocity = np.max([norm_grad_velocity, LinearOperator.PowerMethod(op_v_Y_i, max_iteration=n_iter_norm_calculation)])
    
    rho = geom_rho.allocate()
    rho_old = geom_rho.allocate()
    velocity = geom_velocity.allocate()
    velocity_old = geom_velocity.allocate()
    
    ###
    ## initialization
    ###
    if rho_init is not None:
        rho[0].fill(rho_init[0])
        rho[1].fill(rho_init[1])
    if velocity_init is not None:
        velocity.fill(velocity_init)
    
    
    f_velocity_only = ZeroFunction()
    f_density_only = ZeroFunction()
    algo_density = None
    algo_velocity = None
    
    for iter_outer in range(n_iter_outer):     
        
        
        if algo_density is not None:
            rho_old.fill(algo_density.get_output())
        else:
            rho_old.fill(rho)
        
        if algo_velocity is not None:
            velocity_old.fill(algo_velocity.get_output())
        else:
            velocity_old.fill(velocity)
        
        
        if (iter_outer == 0) and (OF_method != -1):
            gamma_iter = gamma0
        else:
            gamma_iter = gamma   
        
        if first_update == "velocity" or iter_outer>0:
            
            timeDerivative_real = timeDerivative_operator.direct(rho[0])
            timeDerivative_imag = timeDerivative_operator.direct(rho[1])
            
            OF_operator = OFOperator_velocity(geom_velocity,
                                            rho.get_item(0),
                                            rho.get_item(1),
                                            fd_method="centered",
                                            OF_method=OF_method)
                
            norm_OF = LinearOperator.PowerMethod(OF_operator, max_iteration=n_iter_norm_calculation)
    
                
            ## compute Lipschitz constant
            if use_Huber_velocity:
                L = 2 * (alpha_velocity * norm_grad_velocity ** 2 / epsilonHuber_velocity + gamma_iter * norm_OF ** 2 / epsilonHuber)
            else:
                L = 2 * (alpha_velocity * norm_grad_velocity ** 2 + gamma_iter * norm_OF ** 2 / epsilonHuber)
            
            
            if use_Huber_velocity:
                fidelity_function_velocity = HuberLoss(epsilon=epsilonHuber_velocity)
            else:
                fidelity_function_velocity = L2NormSquared()
            
            
            if OF_method == 2:
                if use_grad_velocity:
                    f_velocity_only = OperatorCompositionFunction(alpha_velocity * fidelity_function_velocity, op_v_X_r)
                    f_velocity_only += OperatorCompositionFunction(alpha_velocity * fidelity_function_velocity, op_v_Y_r)
                    f_velocity_only += OperatorCompositionFunction(alpha_velocity * fidelity_function_velocity, op_v_X_i)
                    f_velocity_only += OperatorCompositionFunction(alpha_velocity * fidelity_function_velocity, op_v_Y_i)
                else:
                    f_velocity_only = OperatorCompositionFunction(alpha_velocity * L2NormSquared(), op_v_X_r)
                    f_velocity_only += OperatorCompositionFunction(alpha_velocity * L2NormSquared(), op_v_Y_r)
                    f_velocity_only += OperatorCompositionFunction(alpha_velocity * L2NormSquared(), op_v_X_i)
                    f_velocity_only += OperatorCompositionFunction(alpha_velocity * L2NormSquared(), op_v_Y_i)

            huber_function_OF = HuberLoss(epsilon=epsilonHuber,
                                        b=BlockDataContainer(-timeDerivative_real, -timeDerivative_imag))
            f = f_velocity_only + OperatorCompositionFunction(gamma_iter * huber_function_OF, OF_operator)
            f += ConstantFunction(f_density_only(rho))
            if real_valued_recon:
                g = BlockFunction(BlockFunction(ZeroFunction(), ZeroFunction()),
                                  BlockFunction(IndicatorBox(lower=0, upper=0), IndicatorBox(lower=0, upper=0)))
            else:
                g = ZeroFunction()

            algo_velocity = FISTA_with_early_stopping(initial=velocity,
                            f=f,
                            g=g,
                            step_size=0.95 / L,
                            update_objective_interval=10)

            algo_velocity.max_iteration = n_iter_inner_velocity
            
            algo_velocity.run(n_iter_inner_velocity)
            
            
            velocity.fill(algo_velocity.get_output())
            np.save(folder+"/v_Y_imag", velocity.get_item(1).get_item(0).as_array())
            np.save(folder+"/v_X_imag", velocity.get_item(1).get_item(1).as_array())
            np.save(folder+"/v_Y_real", velocity.get_item(0).get_item(0).as_array())
            np.save(folder+"/v_X_real", velocity.get_item(0).get_item(1).as_array())
            np.save(folder+"/velocity_loss_{}".format(iter_outer), algo_velocity.loss)
            
            if sigma_smoothing_init > 0 and (variables_to_smooth == "velocity" or variables_to_smooth == "both"):
                arr_1 = velocity.get_item(1).get_item(0).as_array()
                arr_2 = velocity.get_item(1).get_item(1).as_array()
                arr_3 = velocity.get_item(0).get_item(0).as_array()
                arr_4 = velocity.get_item(0).get_item(1).as_array()
                for i in range(geom_rho.get_item(0).channels):
                    arr_1[i] = gaussian_filter(arr_1[i], sigma_smoothing_init / (iter_outer + 1))
                    arr_2[i] = gaussian_filter(arr_2[i], sigma_smoothing_init / (iter_outer + 1))
                    arr_3[i] = gaussian_filter(arr_3[i], sigma_smoothing_init / (iter_outer + 1))
                    arr_4[i] = gaussian_filter(arr_4[i], sigma_smoothing_init / (iter_outer + 1))
                    
                velocity.get_item(1).get_item(0).fill(arr_1)
                velocity.get_item(1).get_item(1).fill(arr_2)
                velocity.get_item(0).get_item(0).fill(arr_3)
                velocity.get_item(0).get_item(1).fill(arr_4)

            
        
        if first_update == "density" or iter_outer>0:
            if OF_method == -1:
                velocity *= 0.
            OF_operator = OFOperator_density(geom_rho, velocity.get_item(0), velocity.get_item(1), fd_method="centered", OF_method=OF_method)
            norm_OF = LinearOperator.PowerMethod(OF_operator, max_iteration=n_iter_norm_calculation)
            
            L = 2 * (norm_MRI ** 2 + alpha_rho * np.max([norm_density_op_real, norm_density_op_imag]) ** 2 / epsilonHuber_density + gamma_iter * norm_OF ** 2 / epsilonHuber)
            
            huber_function_OF = HuberLoss(epsilon=epsilonHuber)
            huber_function_density = HuberLoss(epsilon=epsilonHuber_density)
            
            f_density_only = LeastSquares(MRI_operator, acq_data)
            if use_joint_spatial_TV_on_density:
                f_density_only += OperatorCompositionFunction(alpha_rho * huber_function_density, density_op_real)
                f_density_only += OperatorCompositionFunction(alpha_rho * huber_function_density, density_op_imag)
            else:
                f_density_only += OperatorCompositionFunction(alpha_rho * BlockFunction(huber_function_density, huber_function_density), density_op_real)
                f_density_only += OperatorCompositionFunction(alpha_rho * BlockFunction(huber_function_density, huber_function_density), density_op_imag)
            f = f_density_only + OperatorCompositionFunction(gamma_iter * huber_function_OF, OF_operator)
            
            f += ConstantFunction(f_velocity_only(velocity))
            
            
            if real_valued_recon:
                g = BlockFunction(ZeroFunction(),
                                  IndicatorBox(lower=0, upper=0))
            else:
                g = ZeroFunction()
            
            algo_density = FISTA_with_early_stopping(initial=rho,
                            f=f,
                            g=g,
                            step_size=0.95 / L,
                            update_objective_interval=10)
           
            algo_density.max_iteration = n_iter_inner_density
            
            algo_density.run(n_iter_inner_density)
            
            
            rho.fill(algo_density.get_output())
            
            if (iter_outer % 70) == 0 or (iter_outer == n_iter_outer-1):
                np.save(folder+"/rho_real_{}".format(iter_outer), rho.get_item(0).as_array())
                np.save(folder+"/rho_imag_{}".format(iter_outer), rho.get_item(1).as_array())
            np.save(folder+"/density_loss_{}".format(iter_outer), algo_density.loss)            
            
            if sigma_smoothing_init > 0 and (variables_to_smooth == "density" or variables_to_smooth == "both"):
                arr_1 = rho.get_item(0).as_array()
                arr_2 = rho.get_item(1).as_array()
                for i in range(geom_rho.get_item(0).channels):
                    arr_1[i] = gaussian_filter(arr_1[i], sigma_smoothing_init / (iter_outer + 1))
                    arr_2[i] = gaussian_filter(arr_2[i], sigma_smoothing_init / (iter_outer + 1))
                    
                rho.get_item(0).fill(arr_1)
                rho.get_item(1).fill(arr_2)
                
        if first_update != "density" and first_update != "velocity":
            raise NotImplementedError
        
        
        ## early stopping of outer iteration
        if iter_outer >= 10:
            tmp1 = (algo_density.get_output() - rho_old).norm()
            tmp2 = (algo_velocity.get_output() - velocity_old).norm()
            tmp3 = rho_old.norm()
            tmp4 = velocity_old.norm()
            
            eps = 0.5 * (tmp1 / tmp3 + tmp2 / tmp4)
            
            if eps < 1e-05:
                np.save(folder+"/rho_real_{}".format(iter_outer), algo_density.get_output().get_item(0).as_array())
                np.save(folder+"/rho_imag_{}".format(iter_outer), algo_density.get_output().get_item(1).as_array())
                break
        
    return algo_density.get_output(), iter_outer



        
