import numpy as np
from parameters import *
import model
import sys, os
import pickle

if len(sys.argv) > 1:
    GPU_ID = sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
else:
    GPU_ID = None
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

def load_and_replace_parameters(filename):

    data = pickle.load(open(filename, 'rb'))
    data['parameters']['save_fn'] = 'analysis_run_' + filename

    data['parameters']['weight_load_fn'] = filename
    data['parameters']['load_prev_weights'] = True

    update_parameters(data['parameters'])

def try_model(save_fn):
    # To use a GPU, from command line do: python model.py <gpu_integer_id>
    # To use CPU, just don't put a gpu id: python model.py
    try:
        if len(sys.argv) > 1:
            return base_model.main(save_fn, sys.argv[1])
        else:
            return base_model.main(save_fn)
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt.')


def two_tasks():

    task = 'go_antigo'
    update_parameters({'task':task, 'n_tasks':2, 'savetype':10, 'n_hidden':250, 'top_k_neurons':50})
    update_parameters({'architecture':'BIO', 'n_train_batches':501, 'synapse_config':'std_stf'})

    save_fn = task + '_analysis_multistim_BIO'
    update_parameters({'load_from_checkpoint':False, 'weight_cost':0., 'do_k_shot_testing':False})

    update_parameters({'winner_take_all':True})
    return try_model(save_fn+'_with_WTA_v0')


sess, model, stim, x, y, m, g = two_tasks()

print(np.array(sess.run(model.h)).shape)
