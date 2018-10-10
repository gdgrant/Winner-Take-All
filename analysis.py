import numpy as np
import tensorflow as tf
from parameters import *
import sys, os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if len(sys.argv) > 1:
    GPU_ID = sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
else:
    GPU_ID = None
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


def load_and_replace_parameters(filename):
    """ Load parameters from file and plug them into par """

    data = pickle.load(open(filename, 'rb'))
    data['parameters']['save_fn'] = filename[:-4] + 'analysis_run.pkl'

    data['parameters']['weight_load_fn'] = filename
    data['parameters']['load_weights'] = True

    update_parameters(data['parameters'], quiet=True)
    data['parameters'] = par
    return data, data['parameters']['save_fn']


def load_tensorflow_model():
    """ Start the TensorFlow session and build the analysis model """

    import model

    tf.reset_default_graph()
    x, y, m, g, trial_mask, lid = model.get_supervised_placeholders()
    if GPU_ID is not None:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        device = '/gpu:0'
    else:
        sess = tf.Session()
        device = '/cpu:0'

    with tf.device(device):
        mod = model.Model(x, y, m, g, trial_mask, lid)

    load_model_weights(sess)

    return model, sess, mod, x, y, m, g, trial_mask, lid


def load_model_weights(sess):
    """ Load (or reload) TensorFlow weights """
    sess.run(tf.global_variables_initializer())


def network_lesioning(filename):
    """ Lesion individual neurons for linking with gating patterns """

    results, savefile = load_and_replace_parameters(filename)
    model_module, sess, model, x, y, m, g, trial_mask, lid = load_tensorflow_model()

    lesioning_results = np.zeros([par['n_hidden'], par['n_tasks']])
    base_accuracies = np.zeros([par['n_tasks']])

    import stimulus
    stim = stimulus.MultiStimulus()

    for task in range(par['n_tasks']):

        _, stim_in, y_hat, mk, _ = stim.generate_trial(task)
        feed_dict = {x:stim_in, y:y_hat, g:par['gating'][0], m:mk}

        output = sess.run(model.output, feed_dict=feed_dict)
        acc = model_module.get_perf(y_hat, output, mk)

        print('\n'+'-'*60)
        print('Base accuracy for task {}: {}'.format(task, acc))
        base_accuracies[task] = acc

        for n in range(par['n_hidden']):
            print('Lesioning neuron {}/{}'.format(n, par['n_hidden']), end='\r')
            sess.run(model.lesion_neuron, feed_dict={lid:n})
            lesioning_results[n,task] = model_module.get_perf(y_hat, sess.run(model.output, feed_dict=feed_dict), mk)
            load_model_weights(sess)

    return base_accuracies, lesioning_results


def lesioning_analysis():

    accs, lesions = network_lesioning('./LSTM_multi_task/weights/weights_for_multistim_LSTM_with_WTA_gamma0_v0.pkl')

    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.grid()
    ax.scatter(lesions[:,0]-accs[0], lesions[:,1]-accs[1], s=3, c='r')
    ax.set_xlabel('$\\Delta$ Acc Task 0')
    ax.set_ylabel('$\\Delta$ Acc Task 1')
    ax.set_title('Changes in Accuracy after Lesioning Single Neurons')

    plt.savefig('./records/lesioning.png')


def EWC_analysis(filename):
    """ Lesion individual neurons for linking with gating patterns """

    results, savefile = load_and_replace_parameters(filename)
    update_parameters({'stabilization': 'EWC'})

    model_module, sess, model, x, y, m, g, trial_mask, lid = load_tensorflow_model()
    EWC_results = np.zeros(par['n_tasks'])

    import stimulus
    stim = stimulus.MultiStimulus()

    for task in range(par['n_tasks']):
        sess.run(model.reset_big_omega_vars)

        for n in range(par['EWC_fisher_num_batches']):
            name, input_data, _, mk, reward_data = stim.generate_trial(task)
            mk = mk[..., np.newaxis]
            _, big_omegas = sess.run([model.update_big_omega,model.big_omega_var], feed_dict = \
                {x:input_data, target:reward_data, gating:par['gating'][0], mask:mk})

        EWC_results[task] = big_omegas

    return EWC_results

lesioning_analysis()
