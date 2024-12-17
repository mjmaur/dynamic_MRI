import numpy as np
import logging
import os
lib_dir = os.path.dirname(os.path.realpath(__file__))



from cil.optimisation.functions import Function
from cil.optimisation.operators import LinearOperator
from cil.framework import BlockGeometry, BlockDataContainer

logging.basicConfig(filename='LOGGING_TEST.log', level=logging.CRITICAL)

class HuberLoss(Function):
    def __init__(self, epsilon=0.01, b=None):
        
        self.epsilon = epsilon
        self.b = b
        
        super().__init__(L=1/epsilon)
        
    def __call__(self, x):
        
        if self.b is not None:
            y = x - self.b
        else:
            y = x
        
        if isinstance(y, BlockDataContainer):
            tmp = y[0] ** 2
            for i in range(1,y.shape[0]):
                tmp += y[i] ** 2
            tmp.sqrt(out=tmp)
        else:
            tmp = y.abs()
        tmp_np = tmp.as_array()
        idx = (tmp_np > self.epsilon)
        tmp_np[idx] -= self.epsilon / 2
        tmp_np[np.logical_not(idx)] =  tmp_np[np.logical_not(idx)] ** 2  / (2 * self.epsilon) 
        
        return tmp_np.sum()
    
    def gradient(self, x, out=None):
        return_out = False
        
        if self.b is not None:
            y = x - self.b
        else:
            y = x
        
        if out is None:
            return_out = True
            out = y.copy()
        else:
            out.fill(y)
            
        
        if isinstance(x, BlockDataContainer):
            tmp = out[0] ** 2
            for i in range(1,out.shape[0]):
                tmp += out[i] ** 2
            tmp.sqrt(out=tmp)
            
            tmp_np = tmp.as_array()
            idx = (tmp_np > self.epsilon)
            for i in range(out.shape[0]):
                out_np = out[i].as_array()
                out_np[idx] /= tmp_np[idx]
                out_np[np.logical_not(idx)] /= self.epsilon
                
        else:
            tmp = out.abs()
            tmp_np = tmp.as_array()
            idx = (tmp_np > self.epsilon)
            out_np = out.as_array()
            out_np[idx] /= tmp_np[idx]
            out_np[np.logical_not(idx)] /= self.epsilon
            
        if return_out:
            return out
        
    def convex_conjugate(self, x):
        
        if self.b is not None:
            y = x - self.b
        else:
            y = x
        
        if isinstance(x, BlockDataContainer):
            tmp = y[0] ** 2
            for i in range(1,y.shape[0]):
                tmp += y[i] ** 2
        else:
            tmp = y ** 2
            
        return 0.5 * self.epsilon * tmp.sum()
    
    def proximal_conjugate(self, x, tau, out=None):
        
        if self.b is not None:
            y = x - self.b
        else:
            y = x
        
        
        return_out = False
        if out is None:
            return_out = True
            out = y.copy()
        else:
            out.fill(y)
            
        if isinstance(y, BlockDataContainer):
            tmp = out[0] ** 2
            for i in range(1,out.shape[0]):
                tmp += out[i] ** 2
            tmp.sqrt(out=tmp)
            
            tmp_np = tmp.as_array()
            idx = (tmp_np > (1+tau * self.epsilon))
            for i in range(out.shape[0]):
                out_np = out[i].as_array()
                out_np[idx] *= (1 + tau * self.epsilon) / tmp_np[idx]
                out_np[np.logical_not(idx)] /= (1 + tau * self.epsilon)
                
        else:
            tmp = out.abs()
            tmp_np = tmp.as_array()
            idx = (tmp_np > (1+tau * self.epsilon))
            out_np = out.as_array()
            out_np[idx] *= (1 + tau * self.epsilon) / tmp_np[idx]
            out_np[np.logical_not(idx)] /= (1 + tau * self.epsilon)
        
        if return_out:
            return out
        
        


class SelectItems(LinearOperator):
    
    def __init__(self, domain_geometry, item_list):
        self.item_list = item_list
        if len(item_list)>1:
            range_geometry = BlockGeometry(*[domain_geometry.get_item(item) for item in item_list])
        else:
            range_geometry = domain_geometry.get_item(item_list[0])
        
        super().__init__(domain_geometry=domain_geometry, range_geometry=range_geometry)
        
    def direct(self, x, out=None):
        return_out = False
        if out is None:
            return_out = True
            out = self.range_geometry().allocate()
        
        if len(self.item_list)>1:
            for n, item in enumerate(self.item_list):
                out[n].fill(x[item])
        else:
            out.fill(x[self.item_list[0]])
        
        
        if return_out:
            return out
        
    def adjoint(self, x, out=None):
        return_out = False
        if out is None:
            return_out = True
            out = self.domain_geometry().allocate()
        else:
            out *= 0.
            
        if len(self.item_list)>1:
            for n, item in enumerate(self.item_list):
                out[item].fill(x[n])
        else:
            out[self.item_list[0]].fill(x)
        
        
        if return_out:
            return out
    
       

class TimeResolvedMRIForwardOperator(LinearOperator):
    
    def __init__(self, acq_model_list, domain_geometry_rho, idx_rho_r=0, idx_rho_c=1, tp_list=None):

        self.idx_rho_r = idx_rho_r
        self.idx_rho_c = idx_rho_c
            
        self.acq_model_list = acq_model_list
        self.tp_list = tp_list
        
        self.tmp_list = [acq_model.adjoint(acq_model.range_geometry().allocate()) for acq_model in acq_model_list]
        
        range_geometry = BlockGeometry(*[acq_model.range_geometry() for acq_model in acq_model_list])
        
        super().__init__(domain_geometry=domain_geometry_rho, range_geometry=range_geometry)
        
    def direct(self, x, out=None):
        return_out = False
        if out is None:
            return_out = True
            out = self.range_geometry().allocate()
            
        if self.tp_list is None:
            for tp, acq_model in enumerate(self.acq_model_list):
                self.tmp_list[tp].fill(x.get_item(self.idx_rho_r).get_slice(channel=tp).as_array() + 1j * x.get_item(self.idx_rho_c).get_slice(channel=tp).as_array())
                out.get_item(tp).fill(acq_model.direct(self.tmp_list[tp]))
        else:
            for tp_data, acq_model in enumerate(self.acq_model_list):
                tmp_arr = self.tmp_list[tp_data].as_array()
                tmp_arr *= 0.
                for tp_image in self.tp_list[tp_data]:
                    tmp_arr += (x.get_item(self.idx_rho_r).get_slice(channel=tp_image).as_array() + 1j * x.get_item(self.idx_rho_c).get_slice(channel=tp_image).as_array())
                tmp_arr /= len(self.tp_list[tp_data]) 
                self.tmp_list[tp_data].fill(tmp_arr)
                out.get_item(tp_data).fill(acq_model.direct(self.tmp_list[tp_data]))
        if return_out:
            return out
        
    def adjoint(self, x, out=None):
        return_out = False
        if out is None:
            return_out = True
            out = self.domain_geometry().allocate(0.)
        else:
            out *= 0.
        
        out_r_np = out.get_item(self.idx_rho_r).as_array()
        out_c_np = out.get_item(self.idx_rho_c).as_array()

        if self.tp_list is None:
            for tp, acq_model in enumerate(self.acq_model_list):
                res = acq_model.adjoint(x.get_item(tp)).as_array()
                out_r_np[tp] = res.real
                out_c_np[tp] = res.imag
        else:
            for tp_data, acq_model in enumerate(self.acq_model_list):
                res = acq_model.adjoint(x.get_item(tp_data)).as_array()
                res /= len(self.tp_list[tp_data])
                for tp_image in self.tp_list[tp_data]:
                    out_r_np[tp_image] = res.real
                    out_c_np[tp_image] = res.imag
            
        out.get_item(self.idx_rho_r).fill(out_r_np)
        out.get_item(self.idx_rho_c).fill(out_c_np)

        if return_out:
            return out
         

class TimeDerivative(LinearOperator):
    def __init__(self, domain_geometry):
        
        super().__init__(domain_geometry=domain_geometry, range_geometry=domain_geometry)
        
    def direct(self, x, out=None):
        return_out = False
        if out is None:
            return_out = True
            out = self.range_geometry().allocate()
        
        out_np = out.as_array()
        x_np = x.as_array()
        out_np[-1] = 0
        out_np[:-1] = x_np[1:] -  x_np[:-1]
        
        if return_out:
            return out
        
    def adjoint(self, x, out=None):
        return_out = False
        if out is None:
            return_out = True
            out = self.domain_geometry().allocate()
        
        out_np = out.as_array()
        x_np = x.as_array()
        
        out_np[0] = -x_np[0]
        out_np[-1] = x_np[-2]
        out_np[1:-1] = (x_np[:-2] - x_np[1:-1])
        
        if return_out:
            return out
    

class MyGradientOperator(LinearOperator):

    def __init__(self, domain_geometry, method="centered", boundary_condition="Neumann", dimension=2):
        self.method = method
        self.boundary_condition = boundary_condition
        self.dimension = dimension

        range_geometry = BlockGeometry(*[domain_geometry for _ in range(dimension)])
        super().__init__(domain_geometry=domain_geometry, range_geometry=range_geometry)
    
    def direct(self, x, out=None):
        return_out = False
        if out is None:
            return_out = True
            out = self.range_geometry().allocate()

        x_np = x.as_array()
        if self.dimension == 2:
            if self.boundary_condition == "Neumann":
                # y direction
                if self.method == "centered":
                    arr_np = out[0].as_array()
                    arr_np[-1] = 0
                    arr_np[:-1, 0, :] = 0
                    arr_np[:-1, -1, :] = 0
                    arr_np[:-1, 1:-1, :] = 0.5 * (x_np[:-1,2:, :] - x_np[:-1, :-2, :])
                    
                elif self.method == "forward_grad":
                    arr_np = out[0].as_array()
                    arr_np[-1] = 0
                    arr_np[:-1, -1, :] = 0
                    arr_np[:-1, :-1, :] = x_np[:-1, 1:, :] - x_np[:-1, :-1, :]
                else:
                    raise NotImplementedError
                


                # x direction
                if self.method == "centered":
                    arr_np = out[1].as_array()
                    arr_np[-1] = 0
                    arr_np[:-1, :, 0] = 0
                    arr_np[:-1, :, -1] = 0
                    arr_np[:-1, :, 1:-1] = 0.5 * (x_np[:-1,:, 2:] - x_np[:-1, :, :-2])

                elif self.method == "forward_grad":
                    arr_np = out[1].as_array()
                    arr_np[-1] = 0
                    arr_np[:-1, :, -1] = 0
                    arr_np[:-1, :, :-1] = x_np[:-1, :, 1:] - x_np[:-1, :, :-1]
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
                
        else:
            raise NotImplementedError
        
        if return_out:
            return out
        

    def adjoint(self, x, out=None):
        return_out = False
        if out is None:
            return_out = True
            out = self.domain_geometry().allocate(0.)
        else:
            out *= 0.

        out_np = out.as_array()
        if self.dimension == 2:
            if self.boundary_condition == "Neumann":
                # y component
                if self.method == "centered":
                    arr_np = x[0].as_array()
                    out_np[:-1, 0, :] = -0.5 * arr_np[:-1, 1, :]
                    out_np[:-1, -1, :] = 0.5 * arr_np[:-1, -2, :]
                    out_np[:-1, 1, :] = -0.5 * arr_np[:-1, 2, :]
                    out_np[:-1, -2, :] = 0.5 * arr_np[:-1, -3, :]
                    out_np[:-1, 2:-2, :] = 0.5 * (arr_np[:-1, 1:-3, :] - arr_np[:-1, 3:-1, :])
                    
                elif self.method == "forward_grad":
                    arr_np = x[0].as_array()
                    out_np[:-1, 0, :] = -arr_np[:-1, 0, :]
                    out_np[:-1, -1, :] = arr_np[:-1, -2, :]
                    out_np[:-1, 1:-1, :] = arr_np[:-1, :-2, :] - arr_np[:-1, 1:-1, :]

                else:
                     raise NotImplementedError

                # x component
                if self.method == "centered":
                    arr_np = x[1].as_array()
                    out_np[:-1, :, 0] += -0.5 * arr_np[:-1, :, 1]
                    out_np[:-1, :, -1] += 0.5 * arr_np[:-1, :, -2]
                    out_np[:-1, :, 1] += -0.5 * arr_np[:-1, :, 2]
                    out_np[:-1, :, -2] += 0.5 * arr_np[:-1, :, -3]
                    out_np[:-1, :, 2:-2] += 0.5 * (arr_np[:-1, :, 1:-3] - arr_np[:-1, :, 3:-1])
    
                elif self.method == "forward_grad":
                    arr_np = x[1].as_array()
                    out_np[:-1, :, 0] += -arr_np[:-1, :, 0]
                    out_np[:-1, :, -1] += arr_np[:-1, :, -2]
                    out_np[:-1, :, 1:-1] += arr_np[:-1, :, :-2] - arr_np[:-1, :, 1:-1]

            else:
                raise NotImplementedError
            
        else:
            raise NotImplementedError
        
        if return_out:
            return out


class OFOperator_velocity(LinearOperator):
    
    def __init__(self, domain_geometry, rho_r, rho_c, fd_method="centered", OF_method=2):                      
                               
        self.OF_method = OF_method      
        
        self.rho_r = rho_r
        self.rho_c = rho_c
        
        
        self.gradV = MyGradientOperator(domain_geometry=domain_geometry.get_item(0).get_item(0), method=fd_method, dimension=domain_geometry.shape[0])
        
        self.tmp_gradV = self.gradV.range_geometry().allocate()
        
        self.idx_r = 0
        
        self.idx_c = 1
        
        range_geometry = BlockGeometry(domain_geometry.get_item(0).get_item(0), domain_geometry.get_item(0).get_item(0))
        
        self.index_list = list(range(self.tmp_gradV.shape[0]))
        
        super().__init__(domain_geometry=domain_geometry, range_geometry=range_geometry)
        
        
        
    def direct(self, x, out=None):
        return_out = False
        if out is None:
            return_out = True
            out = self.range_geometry().allocate(0.)     
        else:
            out *= 0.

        
        ## real part of output
        self.gradV.direct(self.rho_r, out=self.tmp_gradV)
        for n in self.index_list: 
            out[self.idx_r].sapyb(1., x[self.idx_r][n], self.tmp_gradV[n], out=out[self.idx_r])
        
        if self.OF_method == 2:
            self.gradV.direct(self.rho_c, out=self.tmp_gradV)
            for n in self.index_list: 
                out[self.idx_r].sapyb(1., x[self.idx_c][n], self.tmp_gradV[n], out=out[self.idx_r])
            
        
        ## complex part
        if self.OF_method == 2:
            self.gradV.direct(self.rho_c, out=self.tmp_gradV)
            self.tmp_gradV *= -1.
            for n in self.index_list: 
                out[self.idx_c].sapyb(1., x[self.idx_r][n], self.tmp_gradV[n], out=out[self.idx_c])
                
        if self.OF_method == 2:
            self.gradV.direct(self.rho_r, out=self.tmp_gradV)
            for n in self.index_list: 
                out[self.idx_c].sapyb(1., x[self.idx_c][n], self.tmp_gradV[n], out=out[self.idx_c])
        
        if return_out:
            return out
        
    def adjoint(self, x, out=None):
        return_out = False
        if out is None:
            return_out = True
            out = self.domain_geometry().allocate(0.)
        else:
            out *= 0.
            
        ## part 1
        # coming from real input
        ## real 
        self.gradV.direct(self.rho_r, out=self.tmp_gradV)
        for n in self.index_list: 
            self.tmp_gradV[n].multiply(x[self.idx_r], out=out[self.idx_r][n])
        
        ## complex
        if self.OF_method == 2:
            self.gradV.direct(self.rho_c, out=self.tmp_gradV)
            #self.tmp_gradV *= -1
            for n in self.index_list:
                self.tmp_gradV[n].multiply(x[self.idx_r], out=out[self.idx_c][n])
        
        ## part 2
        # coming from complex input
        ## real        
        if self.OF_method == 2:
            self.gradV.direct(self.rho_c, out=self.tmp_gradV)
            self.tmp_gradV *= -1
            for n in self.index_list:
                out[self.idx_r][n].sapyb(1., self.tmp_gradV[n], x[self.idx_c], out=out[self.idx_r][n])
                
        ## complex
        if self.OF_method == 2:
            self.gradV.direct(self.rho_r, out=self.tmp_gradV)
            for n in self.index_list:
                out[self.idx_c][n].sapyb(1., self.tmp_gradV[n], x[self.idx_c], out=out[self.idx_c][n])
            
        if return_out:
            return out
  
  
  
class OFOperator_density(LinearOperator):
    
    
    def __init__(self, domain_geometry, v_r, v_c,
                    idx_rho_r = 0, idx_rho_c = 1, fd_method="centered", dtype=np.float64, OF_method=2):
        
        self.OF_method = OF_method
        
        self.idx_rho_r = idx_rho_r
        self.idx_rho_c = idx_rho_c
        
        self.gradV = MyGradientOperator(domain_geometry=domain_geometry.get_item(self.idx_rho_r), method=fd_method, dimension=v_r.shape[0])
        
        self.timeDerivative = TimeDerivative(domain_geometry=domain_geometry.get_item(self.idx_rho_r))        
        
        range_geometry = domain_geometry
        self.idx_r = 0 ## index in output
        self.idx_c = 1
        
   
        self.tmp_range_subset = range_geometry.get_item(0).allocate()
        self.tmp_range_subset2 = range_geometry.get_item(0).allocate()
        self.tmp_domain_subset = domain_geometry.get_item(self.idx_rho_r).allocate()
        self.tmp_gradV = self.gradV.range_geometry().allocate()
        self.tmp_gradV2 = self.tmp_gradV.clone()

        self.v_r = v_r
        self.v_c = v_c
        
        self.index_list = list(range(self.tmp_gradV.shape[0]))
        
        super().__init__(domain_geometry=domain_geometry, range_geometry=range_geometry)
        
    
    def direct(self, x, out=None):
        return_out = False
        if out is None:
            return_out = True
            out = self.range_geometry().allocate(0.)
        else:
            out *= 0. 
        
        
        ## first time derivatives
        self.timeDerivative.direct(x[self.idx_rho_r], out=out[self.idx_r])   
        self.timeDerivative.direct(x[self.idx_rho_c], out=out[self.idx_c])        
        
        ## real -> real (1)        
        self.gradV.direct(x[self.idx_r], out=self.tmp_gradV)
        for n in self.index_list:
            out[self.idx_r].sapyb(1., self.tmp_gradV[n], self.v_r[n], out=out[self.idx_r])
                
        ## complex -> real
        if self.OF_method == 2:
            self.gradV.direct(x[self.idx_c], out=self.tmp_gradV)
            #self.tmp_gradV *= -1
            for n in self.index_list:
                out[self.idx_r].sapyb(1., self.tmp_gradV[n], self.v_c[n], out=out[self.idx_r])
        
        if (self.OF_method == 2):
            ## real -> complex (3)
            self.gradV.direct(x[self.idx_r], out=self.tmp_gradV)
            for n in self.index_list:
                out[self.idx_c].sapyb(1., self.tmp_gradV[n], self.v_c[n], out=out[self.idx_c])
        
        ## complex -> complex (4)
        if (self.OF_method == 2):
            self.gradV.direct(x[self.idx_c], out=self.tmp_gradV)
            self.tmp_gradV *= -1
            for n in self.index_list:
                out[self.idx_c].sapyb(1., self.tmp_gradV[n], self.v_r[n], out=out[self.idx_c])

        
        if return_out:
            return out
        
    def adjoint(self, x, out=None):
        return_out = False
        if out is None:
            return_out = True
            out = self.domain_geometry().allocate(0.)
        else:
            out *= 0. 


        self.timeDerivative.adjoint(x[self.idx_r], out=out[self.idx_rho_r])
        self.timeDerivative.adjoint(x[self.idx_c], out=out[self.idx_rho_c])
        
        ## real -> real (1)
        for n in self.index_list:
            x[self.idx_r].multiply(self.v_r[n], out=self.tmp_gradV[n])
        self.gradV.adjoint(self.tmp_gradV, out=self.tmp_domain_subset)
        out[self.idx_rho_r].add(self.tmp_domain_subset, out=out[self.idx_rho_r])
        
        if self.OF_method == 2:
            ## complex -> real (2)
            for n in self.index_list:
                x[self.idx_r].multiply(self.v_c[n], out=self.tmp_gradV[n])
            self.gradV.adjoint(self.tmp_gradV, out=self.tmp_domain_subset)
            #out[self.idx_rho_c].subtract(self.tmp_domain_subset, out=out[self.idx_rho_c])
            out[self.idx_rho_c].add(self.tmp_domain_subset, out=out[self.idx_rho_c])
            
            ## real -> complex (3)
            for n in self.index_list:
                x[self.idx_c].multiply(self.v_c[n], out=self.tmp_gradV[n])
            self.gradV.adjoint(self.tmp_gradV, out=self.tmp_domain_subset)
            out[self.idx_rho_r].add(self.tmp_domain_subset, out=out[self.idx_rho_r])
        
        ## complex -> complex (4)
        if (self.OF_method == 2):
            for n in self.index_list:
                x[self.idx_c].multiply(self.v_r[n], out=self.tmp_gradV[n])
            self.gradV.adjoint(self.tmp_gradV, out=self.tmp_domain_subset)
            # out[self.idx_rho_c].add(self.tmp_domain_subset, out=out[self.idx_rho_c])
            out[self.idx_rho_c].subtract(self.tmp_domain_subset, out=out[self.idx_rho_c])
            
        
        if return_out:
            return out
