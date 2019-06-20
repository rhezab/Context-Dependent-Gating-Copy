import numpy as np
from parameters import *
import model
import sys, os
import pickle


def try_model(save_fn, gpu_id = None):
    # Specify GPU Id as a string
    # Leave blank to use CPU
    try:
        model.main(save_fn, gpu_id)
    except KeyboardInterrupt:
        print('Quit by KeyboardInterrupt.')


###############################################################################
###############################################################################
###############################################################################


mnist_updates = {
    'layer_dims'            : [784, 2000, 2000, 10],
    'n_tasks'               : 100,
    'task'                  : 'mnist',
    'save_dir'              : './savdir/',
    'n_train_batches'       : 3906,
    'drop_keep_pct'         : 0.5,
    'input_drop_keep_pct'   : 1.0,
    'multihead'             : False
    }

cifar_updates = {
    'layer_dims'            : [4096, 1000, 1000, 5],
    'n_tasks'               : 20,
    'task'                  : 'cifar',
    'save_dir'              : './savedir/',
    'n_train_batches'       : 977,
    'input_drop_keep_pct'   : 1.0,
    'drop_keep_pct'         : 0.5,
    'multihead'             : False
    }

imagenet_updates = {
    'layer_dims'            : [4096, 2000, 2000, 10],
    'n_tasks'               : 100,
    'task'                  : 'imagenet',
    'save_dir'              : './savedir/',
    'n_train_batches'       : 977*2,
    'input_drop_keep_pct'   : 1.0,
    'drop_keep_pct'         : 0.5,
    'multihead'             : False
    }

# updates for multi-head network, Cifar and Imagenet only
multi_updates = {'layer_dims':[4096, 1000, 1000, 100], 'multihead': True}
imagenet_multi_updates = {'layer_dims':[4096, 2000, 2000, 1000], 'multihead': True}

# updates for split networks
mnist_split_updates = {'layer_dims':[784, 3665, 3665, 10], 'multihead': False}
cifar_split_updates = {'layer_dims':[4096, 1164, 1164, 5], 'multihead': False}
imagenet_split_updates = {'layer_dims':[4096, 3665, 3665, 10], 'multihead': False}


# training a network on 100 sequential MNIST permutations using synaptic intelligence
# and context-dependent gating (XdG)
def run_mnist_SI_model(gpu_id):
    print('MNIST - Synaptic Stabilization = SI - Gating = 80%')
    update_parameters(mnist_updates)
    update_parameters({'gating_type': 'XdG','gate_pct': 0.8, 'input_drop_keep_pct': 0.8})
    update_parameters({'stabilization': 'pathint', 'omega_c': 0.035, 'omega_xi': 0.01})
    save_fn = 'mnist_SI_XdG.pkl'
    try_model(save_fn, gpu_id)

def run_mnist_SI_model_shorter(gpu_id, gate_type):
    print('MNIST - Synaptic Stabilization = SI - Gating = 80%')
    update_parameters(mnist_updates)
    update_parameters({'gating_type': gate_type,'gate_pct': 0.8, 'input_drop_keep_pct': 0.8})
    update_parameters({'stabilization': 'pathint', 'omega_c': 0.035, 'omega_xi': 0.01})
    update_parameters({'n_train_batches': 101, 'n_tasks': 20, 'layer_dims': [28**2, 1000, 1000, 10]})
    update_parameters({'save_dir': './'})
    save_fn = 'mnist_SI_'+gate_type+'.pkl'
    try_model(save_fn, gpu_id)

def run_mnist_EWC_model_shorter(gpu_id, gate_type):
    print('MNIST - Synaptic Stabilization = EWC - Gating = 80%')
    update_parameters(mnist_updates)
    update_parameters({'gating_type': gate_type,'gate_pct': 0.8, 'input_drop_keep_pct': 0.8})
    update_parameters({'stabilization': 'EWC', 'omega_c': 0.035, 'omega_xi': 0.01})
    update_parameters({'n_train_batches': 10})
    save_fn = 'mnist_EWC_'+gate_type+'.pkl'
    try_model(save_fn, gpu_id)

# training a network on 100 sequential Imagenet tasks using synaptic intelligence
# and context-dependent gating (XdG)
def run_imagenet_SI_model(gpu_id):
    print('ImageNet - Synaptic Stabilization = SI - Gating = 80%')
    update_parameters(imagenet_updates)
    update_parameters({'gating_type': 'XdG','gate_pct': 0.80, 'input_drop_keep_pct': 1.0})
    update_parameters({'stabilization': 'pathint', 'omega_c': 0.75, 'omega_xi': 0.01})
    update_parameters({'train_convolutional_layers': True})
    save_fn = 'imagenet_SI_XdG.pkl'
    try_model(save_fn, gpu_id)
