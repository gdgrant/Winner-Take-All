import numpy as np
import tensorflow as tf
from parameters import *
import sys, os
import pickle
from itertools import product
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

    accs, lesions = network_lesioning('./weights/lstm_multi_task_with_abs/weights_for_multistim_LSTM_with_WTA_gamma0_v0.pkl')

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
    update_parameters({'batch_size':8})

    model_module, sess, model, x, y, m, g, trial_mask, lid = load_tensorflow_model()
    EWC_results = []

    import stimulus
    stim = stimulus.MultiStimulus()

    import time
    for task in range(par['n_tasks']):
        print('EWC analysis for task {}.'.format(task))
        sess.run(model.reset_big_omega_vars)

        for n in range(par['EWC_fisher_num_batches']):
            print('EWC batch {}'.format(n), end='\r')
            _, stim_in, y_hat, mk, _ = stim.generate_trial(task)
            _, big_omegas = sess.run([model.update_big_omega,model.big_omega_var], feed_dict={x:stim_in, y:y_hat, g:par['gating'][0], m:mk})

        EWC_results.append(big_omegas)

    pickle.dump(EWC_results, open('./records/EWC_results_weights_for_multistim_LSTM_without_WTA_gamma0_v0.pkl', 'wb'))


def render_EWC_results(filename):

    data = pickle.load(open(filename, 'rb'))

    weights = ['Uf', 'Ui', 'Uo', 'Uc']
    figures = {}
    axes = {}
    for w in weights:
        fig, ax = plt.subplots(4,5,figsize=[15,12],sharey=True)
        fig.subplots_adjust(hspace=0.3)

        ax[3,0].set_xlabel('EWC Importance')
        ax[3,0].set_ylabel('Counts')

        for task, task_data in enumerate(data):
            k = [key for key in task_data.keys() if w in key][0]

            # ax[3,4]
            x = task//5
            y = task%5
            ax[x,y].hist(task_data[k].flatten())
            ax[x,y].set_title(task)
            ax[x,y].set_yscale('log')



        plt.suptitle(w)
        plt.savefig('./records/without_WTA_{}.png'.format(w))


def task_variance_analysis(filename, plot=False):

    results, savefile = load_and_replace_parameters(filename)
    model_module, sess, model, x, y, m, g, trial_mask, lid = load_tensorflow_model()

    lesioning_results = np.zeros([par['n_hidden'], par['n_tasks']])
    base_accuracies = np.zeros([par['n_tasks']])

    import stimulus
    stim = stimulus.MultiStimulus()

    task_variance = np.zeros([par['n_tasks'], par['n_hidden']])
    for task in range(par['n_tasks']):

        _, stim_in, y_hat, mk, _ = stim.generate_trial(task)
        feed_dict = {x:stim_in, y:y_hat, g:par['gating'][0], m:mk}

        output, h = sess.run([model.output, model.h], feed_dict=feed_dict)
        acc = model_module.get_perf(y_hat, output, mk)
        h = np.array(h)[par['dead_time']//par['dt']:,:,:]     # [100, 256, 500], or [time, trials, neuron]

        task_variance[task,:] = np.mean(np.mean(np.square(h - np.mean(h, axis=1, keepdims=True)), axis=1), axis=0)

    if plot:
        plt.imshow(task_variance/np.amax(task_variance), aspect='auto', cmap='magma')
        plt.colorbar()
        plt.ylabel('Tasks')
        plt.yticks(np.arange(20))
        plt.xlabel('Neurons')
        plt.xticks(np.arange(500,10))
        plt.title('Normalized Task Variance')
        plt.savefig('./records/task_variance.png')
        plt.clf()
        plt.close()

    return task_variance


def task_variance_rendering(filename):

    import sklearn
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans

    data = pickle.load(open(filename, 'rb'))
    with_WTA = data['with_WTA']
    without_WTA = data['without_WTA']

    for w, w_text in zip([with_WTA, without_WTA], ['with_WTA', 'without_WTA']):
        x = w/np.amax(w)

        kmeans = KMeans(n_clusters=20)
        projection = kmeans.fit_transform(x.T)
        labels = kmeans.fit(x.T).labels_
        prediction = kmeans.predict(x.T)


        plt.title('KMeans for {}'.format(w_text))
        plt.xticks([])
        plt.yticks([])
        plt.scatter(projection[:,0], projection[:,1], c=labels, cmap='tab20')
        plt.savefig('./records/kmeans_attempt_{}.png'.format(w_text))
        plt.clf()
        plt.close()


        projection = TSNE(init='pca', perplexity=75, learning_rate=300, n_iter=10000, method='exact').fit_transform(x.T)  # n_samples x n_features
        plt.title('t-SNE for {}'.format(w_text))
        plt.xticks([])
        plt.yticks([])
        plt.scatter(projection[:,0], projection[:,1], c=labels, cmap='tab20')
        plt.savefig('./records/TSNE_attempt_{}.png'.format(w_text))
        plt.clf()
        plt.close()

        print('--> ' + w_text + ' complete.')




task_variance_rendering('./records/neuronal_task_variance_multistim_LSTM.pkl')

"""
task_variance_with_WTA = task_variance_analysis('./weights/weights_for_multistim_LSTM_with_WTA_input_gamma1_uniform3_v0.pkl')
print('With WTA complete')
task_variance_without_WTA = task_variance_analysis('./weights/weights_for_multistim_LSTM_without_WTA_input_gamma2_uniform3_v0.pkl')
print('Without WTA complete')

pickle.dump({'with_WTA':task_variance_with_WTA,'without_WTA':task_variance_without_WTA}, open('./records/neuronal_task_variance_multistim_LSTM.pkl', 'wb'))
#"""

"""
EWC_analysis('./weights/lstm_multi_task_with_abs/weights_for_multistim_LSTM_without_WTA_gamma0_v0.pkl')
render_EWC_results('./records/EWC_results_weights_for_multistim_LSTM_with_WTA_gamma2_v0.pkl')
render_EWC_results('./records/EWC_results_weights_for_multistim_LSTM_without_WTA_gamma0_v0.pkl')
#"""
