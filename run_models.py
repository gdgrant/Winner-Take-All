import numpy as np
from parameters import *
import model
import sys, os
import pickle


def try_model(save_fn):
    # To use a GPU, from command line do: python model.py <gpu_integer_id>
    # To use CPU, just don't put a gpu id: python model.py
    try:
        if len(sys.argv) > 1:
            model.main(save_fn, sys.argv[1])
        else:
            model.main(save_fn)
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt.')

###############################################################################
###############################################################################
###############################################################################

multistim_params   = {'task':'multistim', 'n_tasks':20, 'savetype':1, 'n_hidden':500, 'top_k_neurons':100}
gotask_params      = {'task':'go_tasks', 'n_tasks':2, 'savetype':10, 'n_hidden':250, 'top_k_neurons':50}
go_dly_go_params   = {'task':'go_dly_go', 'n_tasks':2, 'savetype':10, 'n_hidden':250, 'top_k_neurons':50}


BIO_params         = {'architecture':'BIO', 'n_train_batches':8001, 'synapse_config':'std_stf'}
LSTM_params        = {'architecture':'LSTM', 'n_train_batches':2001, 'synapse_config':None}
gotask_batches     = {'n_train_batches':501}

with_WTA_params    = {'winner_take_all':True}
without_WTA_params = {'winner_take_all':False}


def go_BIO():

    # Go task, biological network
    save_fn = 'gotask_BIO'
    update_parameters(gotask_params)
    update_parameters(BIO_params)
    update_parameters(gotask_batches)

    for j in range(1):
        update_parameters(with_WTA_params)
        try_model(save_fn+'_with_WTA_v{}'.format(j))

        update_parameters(without_WTA_params)
        try_model(save_fn+'_without_WTA_v{}'.format(j))


def go_LSTM():

    # Go task, LSTM network
    save_fn = 'gotask_LSTM'
    update_parameters(gotask_params)
    update_parameters(LSTM_params)
    update_parameters(gotask_batches)

    for j in range(1):
        update_parameters(with_WTA_params)
        try_model(save_fn+'_with_WTA_v{}'.format(j))

        update_parameters(without_WTA_params)
        try_model(save_fn+'_without_WTA_v{}'.format(j))


def multistim_BIO():

    # Multistim task, biological network
    save_fn = 'multistim_BIO'
    update_parameters(multistim_params)
    update_parameters(BIO_params)

    for j in range(10,11):
        update_parameters(with_WTA_params)
        try_model(save_fn+'_with_WTA_v{}'.format(j))

        #update_parameters(without_WTA_params)
        #try_model(save_fn+'_without_WTA_v{}'.format(j))


def multistim_LSTM():

    # Multistim task, LSTM network
    save_fn = 'multistim_LSTM'
    update_parameters(multistim_params)
    update_parameters(LSTM_params)

    for j in range(10,11):
        update_parameters(with_WTA_params)
        try_model(save_fn+'_with_WTA_v{}'.format(j))

        update_parameters(without_WTA_params)
        try_model(save_fn+'_without_WTA_v{}'.format(j))


def kshot_testing_multistim_LSTM(with_WTA=True):

    print('Running kshot testing.')

    # Multistim task, LSTM network
    update_parameters(multistim_params)
    update_parameters(LSTM_params)
    update_parameters({'n_train_batches':50001,'use_threshold':True})
    update_parameters({'do_k_shot_testing':True,'load_from_checkpoint':False})

    for task in range(par['n_tasks']):
        update_parameters({'k_shot_task':task})
        save_fn = 'kshot_task{}_multistim_LSTM'.format(task)

        for j in range(1):
            if with_WTA:
                update_parameters({'c_gamma':0.05})
                update_parameters(with_WTA_params)
                try_model(save_fn+'_with_WTA_v{}'.format(j))
            else:
                update_parameters({'c_gamma':0.01})
                update_parameters(without_WTA_params)
                try_model(save_fn+'_without_WTA_v{}'.format(j))

def interleaved():

    print('Running interleaved:')

    # Multistim task, LSTM network
    update_parameters(multistim_params)
    update_parameters(LSTM_params)
    #update_parameters({'weight_distribution':'uniform'})

    save_fn = 'interleaved_analysis_multistim_LSTM'
    update_parameters({'load_from_checkpoint':False, 'weight_cost':0., 'do_k_shot_testing':False})
    update_parameters(with_WTA_params)
    try_model(save_fn+'_with_WTA_v0')


def two_tasks():

    task = 'go_antigo'
    update_parameters({'task':task, 'n_tasks':2, 'n_hidden':250, 'top_k_neurons':50})
    update_parameters(BIO_params)

    save_fn = task + '_analysis_multistim_BIO'
    update_parameters({'load_from_checkpoint':False, 'do_k_shot_testing':False})

    update_parameters(with_WTA_params)
    try_model(save_fn+'_with_WTA_v0')

    #update_parameters(without_WTA_params)
    #try_model(save_fn+'_without_WTA_v0')


def six_tasks():

    task = 'go_set'
    update_parameters({'task':task, 'n_tasks':6, 'savetype':10, 'n_hidden':250, 'top_k_neurons':50})
    update_parameters(LSTM_params)

    save_fn = task + '_analysis_multistim_LSTM'
    update_parameters({'load_from_checkpoint':False, 'weight_cost':0., 'do_k_shot_testing':False})

    if True:
        update_parameters(with_WTA_params)
        try_model(save_fn+'_with_WTA_v0')
    else:
        update_parameters(without_WTA_params)
        try_model(save_fn+'_without_WTA_v0')


def expanded_go_tasks():

    task = 'expanded_go'
    update_parameters({'task':task, 'n_tasks':26, 'savetype':10, 'n_hidden':500, 'top_k_neurons':100})
    update_parameters(BIO_params)

    save_fn = task + '_analysis_multistim_BIO'
    update_parameters({'load_from_checkpoint':False, 'weight_cost':0., 'do_k_shot_testing':False})

    if False:
        update_parameters(with_WTA_params)
        try_model(save_fn+'_with_WTA_v0')
    else:
        update_parameters(without_WTA_params)
        try_model(save_fn+'_without_WTA_v0')


def gamma_c_BIO_sweep():

    # Multistim task, biological network
    save_fn = 'gamma_sweep_multistim_BIO'
    update_parameters(multistim_params)
    update_parameters(BIO_params)
    update_parameters({'n_train_batches':4001})

    gamma_cs = [0.01, 0.02, 0.05, 0.1, 0.2]
    for j in range(10,15):
        for c in range(5):
            update_parameters({'gamma_c':gamma_cs[c]})

            #update_parameters(with_WTA_params)
            #try_model(save_fn+'_with_WTA_c{}_v{}'.format(c, j))

            update_parameters(without_WTA_params)
            try_model(save_fn+'_without_WTA_c{}_v{}'.format(c, j))

####

#go_BIO()
#go_LSTM()
#gamma_c_BIO_sweep()
#multistim_LSTM()
kshot_testing_multistim_LSTM(with_WTA=True)
kshot_testing_multistim_LSTM(with_WTA=False)
#interleaved()
#two_tasks()
#six_tasks()
#expanded_go_tasks()
