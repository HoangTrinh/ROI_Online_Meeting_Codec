import os
from importlib import import_module
import torch
import torch.nn as nn
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, opt, name , weight):
        super(Model, self).__init__()
        print('Making model...')

        self.cpu = opt.cpu
        self.cudaDevice = opt.cudaDevice
        self.n_GPUs = opt.nGPUs
        self.device = torch.device('cpu' if opt.cpu else 'cuda:'+ str(opt.cudaDevice))

        module = import_module('model.' + name.lower())
        self.model = module.make_model(opt).to(self.device)

        if not opt.cpu and opt.nGPUs > 1:
            self.model = nn.DataParallel(self.model, range(opt.n_GPUs))

        self.load(
            pre_train=weight,
            cpu=opt.cpu
        )

    def forward(self, x):
        target = self.get_model()
        return self.model(x)

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def load(self, pre_train='.', cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if pre_train != '.':
            print('Loading model from {}'.format(pre_train))
            self.get_model().load_state_dict(
                torch.load(pre_train,map_location='cpu', **kwargs),
                strict=False
            )
