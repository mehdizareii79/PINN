import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
class PINN(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_outputs,
                 num_hidden=None,
                 activation='tanh',
                 pde_weight=1.0,
                 bc_weight=1.0,
                 ic_weight=1.0,
                 reduction='mean',
                 c=1.0,
                 length=1.0,
                 maxtime=1.0,
                 n=1):
        super().__init__()
        self.num_inputs=num_inputs
        self.num_outputs=num_outputs
        self.num_hidden=num_hidden
        self.activation=activation
        self.pde_weight=pde_weight
        self.bc_weight=bc_weight
        self.ic_weight=ic_weight
        self.reduction=reduction
        self.c=c
        self.length=length
        self.maxtime=maxtime
        self.n=n
        self.dropout = nn.Dropout(0.3)
        self.non_linearity = getattr(nn, activation)()
        layer_sizes = [num_inputs]+num_hidden+[num_outputs]
        self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)])
        # Initialize the weights for each layer
        for layer in self.layers:
            if activation == 'ReLU' or 'LeakyRelu':
                nn.init.xavier_uniform_(layer.weight)
            else:
                nn.init.kaiming_uniform_(layer.weight, nonlinearity= activation)


    def forward(self,data : Tensor) -> Tensor:
        for i in range(len(self.layers)-1):
            data = self.layers[i](data)
            data = self.non_linearity(data)
            if i<=1:
                data=self.dropout(data)
        data = self.layers[-1](data)
        return data
    
    def make_collocation(self,
                         num_pde,
                         num_bc,
                         num_ic):

        x = np.random.uniform(0, self.length, num_pde)
        t = np.random.uniform(0, self.maxtime, num_pde)
        pde = np.column_stack((t,x))
        pde = torch.from_numpy(pde.astype(np.float32))

        x = np.random.choice([0, self.length], num_bc)
        t = np.linspace(0, self.maxtime, num_bc)
        bc = np.column_stack((t,x))
        bc = torch.from_numpy(bc.astype(np.float32))

        x =np.linspace(0, self.length, num_ic)
        t = np.zeros(num_ic)
        ic = np.column_stack((t,x))
        ic = torch.from_numpy(ic.astype(np.float32))
        result = {
            'pde_data': pde, # number of train samples for PDE loss
            'bc_data': bc, # number of train samples for BC loss
            'ic_data': ic # number of train samples for IC loss
        }
        return result


    def data_loss(self, data, y):
        y_pred = self.forward(data)
        loss= torch.mean((y-y_pred)**2)
        return loss

    def pde_loss(self, data):

        #loss = torch.tensor(0.0, requires_grad=True)
        #for inputs in data:
        #    hessian = torch.autograd.functional.hessian(self.forward, inputs, create_graph=True)
        #    loss = loss + (hessian[0, 0] - self.c ** 2 * hessian[1, 1]) ** 2
        #loss = loss / len(data)
        data.requires_grad=True
        y = self.forward(data)
        dydx = torch.autograd.grad(y.sum(), data , create_graph=True)
        dydxsum = dydx[0].sum(axis=0)
        dydx1dx = torch.autograd.grad(dydxsum[0], data, create_graph=True)
        dydx2dx = torch.autograd.grad(dydxsum[1], data, create_graph=True)
        utt = dydx1dx[0][:, 0]
        uxx = dydx2dx[0][:, 1]
        loss = torch.mean((utt-self.c**2 * uxx)**2)
        return loss

    def bc_loss(self, data):
        y_pred = self.forward(data)
        loss = torch.mean(y_pred**2)
        return loss

    def ic_loss(self, data):
        data.requires_grad=True
        y_pred = self.forward(data)
        x = data[:, 1].reshape(-1,1) # exclude x from data consisting of (t,x)
        y = torch.sin(self.n*torch.pi/self.length * x) # actual initial condition y=sin(npix/l)
        lossu = torch.mean((y_pred-y)**2)
        dydx = torch.autograd.grad(y_pred.sum(),data,create_graph=True,allow_unused=True)
        ut = dydx[0][:,0].reshape(-1,1)
        lossut = torch.mean((self.n*torch.pi*self.c/self.length * y -ut)**2)
        loss = lossu+lossut
        return loss

    def physics_loss(self,data_pde,data_bc,data_ic):
        loss =  self.pde_weight*self.pde_loss(data_pde) +self.bc_weight*self.bc_loss(data_bc)+self.ic_weight*self.ic_loss(data_ic)
        return loss


