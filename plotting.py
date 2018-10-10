import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def accuracy_curves(task='go_antigo'):
    with_WTA    = pickle.load(open('./savedir/accuracy_'+task+'_analysis_multistim_LSTM_with_WTA_v0.pkl', 'rb'))
    without_WTA = pickle.load(open('./savedir/accuracy_'+task+'_analysis_multistim_LSTM_without_WTA_v0.pkl', 'rb'))
    iters_per_element = 1

    curves = (with_WTA, without_WTA)
    names = ('with WTA', 'without WTA')
    colors = ('b', 'r')

    for a in [0.7, 0.8, 0.9, 0.95]:
        plt.axhline(a, c='k', ls='--')

    for curve, name, color in zip(curves, names, colors):

        xrange = iters_per_element*np.arange(len(curve))
        yrange = np.mean(np.array(curve), axis=1)

        plt.plot(xrange, yrange, c=color, label=name)

    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('WTA Testing for {}'.format(task))
    plt.legend()
    plt.show()


def gating_patterns(task='go_antigo'):

    fns = [fn for fn in os.listdir('./savedir/') if task in fn and 'states' in fn]
    with_WTA    = pickle.load(open('./savedir/'+[fn for fn in fns if 'with_' in fn][0], 'rb'))
    without_WTA = pickle.load(open('./savedir/'+[fn for fn in fns if 'without_' in fn][0], 'rb'))

    # List across tasks
    fig, ax = plt.subplots(1,len(with_WTA)+1)
    for i, task_pattern in enumerate(with_WTA):
        task_pattern = np.array(task_pattern)
        ax[i].imshow(np.sum(task_pattern, axis=1).T, aspect='auto')
        ax[i].set_title('Task {}'.format(i))

    ax[-1].imshow(np.sum(with_WTA[1], axis=1).T - np.sum(with_WTA[0], axis=1).T, aspect='auto')
    ax[-1].set_title('Task 1 - Task 0')

    plt.show()


def plot_gamma_sweep():

    fns = [fn for fn in os.listdir('./savedir/') if 'gamma' in fn and not 'kshot_accuracy' in fn and 'v0' in fn]
    withs = [fn for fn in fns if 'with_' in fn]
    withouts = [fn for fn in fns if 'without_' in fn]

    plt.grid()

    colors = ['maroon', 'firebrick', 'chocolate', 'orange', 'gold']
    for i, (c, w) in enumerate(zip(colors, sorted(withs))):
        print(w)
        w = np.array(pickle.load(open('./savedir/'+w, 'rb')))
        print(w[400,:])
        plt.plot(np.mean(w, axis=1), c=c, label='with, c{}'.format(i))

    print('')
    colors = ['darkblue', 'b', 'blueviolet', 'darkviolet', 'darkmagenta']
    for i, (c, w) in enumerate(zip(colors, sorted(withouts))):
        print(w)
        w = np.array(pickle.load(open('./savedir/'+w, 'rb')))
        print(w[400,:])
        plt.plot(np.mean(w, axis=1), c=c, label='without, c{}'.format(i))

    plt.legend()
    plt.ylim(0,1)
    plt.xlim(0,400)
    plt.show()


plot_gamma_sweep()
