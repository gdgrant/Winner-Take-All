import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from itertools import product

savedir = './savedir/'
fns = [fn for fn in os.listdir(savedir) if 'accuracy' in fn and '__demo__' not in fn]
iters_per_element = 5

def plot_training_curves(fns):

    fig, ax = plt.subplots(2,2, figsize=(12,8))

    for with_WTA, arch in product([True, False], ['BIO', 'LSTM']):
        if with_WTA:
            WTA_fns = [fn for fn in fns if 'with_WTA' in fn]
            if arch == 'BIO':
                color='c'
                p = 0
            elif arch == 'LSTM':
                color='b'
                p = 1

            label = arch + ' with WTA'

        else:
            WTA_fns = [fn for fn in fns if 'without_WTA' in fn]
            if arch == 'BIO':
                color='m'
                p = 0
            elif arch == 'LSTM':
                color='r'
                p = 1

            label = arch + ' without WTA'


        arch_fns = [fn for fn in WTA_fns if arch in fn]

        curves = sorted([pickle.load(open(savedir+fn, 'rb')) for fn in arch_fns], key=lambda x : -len(x))

        aggregate = []
        for i, c in enumerate(curves):
            for j, el in enumerate(c):
                if i == 0:
                    aggregate.append([np.mean(el)])
                else:
                    aggregate[j].append(np.mean(el))

        xrange = iters_per_element*np.arange(len(aggregate))
        yrange = np.array([np.mean(a) for a in aggregate])
        yerror = np.array([np.std(a) for a in aggregate])

        ax[p,0].plot(xrange, yrange, c=color, label=label)
        ax[p,0].fill_between(xrange, yrange-yerror, yrange+yerror, alpha=0.25, edgecolor=color, facecolor='k')

        xpoints = []
        ypoints = []
        for threshold in [0.8, 0.9, 0.95]:
            for i, y in enumerate(yrange):
                if y > threshold:
                    ypoints.append(iters_per_element*i)
                    xpoints.append(-1 if with_WTA else 1)

                    ax[p,1].text(-0.-0.85 if with_WTA else 1.15, iters_per_element*(i-10), str(threshold))

                    break

        ax[p,1].scatter(xpoints, ypoints, c=color, label=label)
        ax[p,1].set_xlim(-2,2)
        if p == 0:
            ax[p,1].set_ylim(0,8000)
        elif p == 1:
            ax[p,1].set_ylim(0,2000)




    for p in [0,1]:
        ax[p,0].legend(loc='lower right')
        ax[p,1].legend(loc='lower right')

        ax[p,0].set_xlabel('Iterations')
        ax[p,0].set_ylabel('Accuracy')
        ax[p,0].set_ylim(0,1)
        ax[p,0].set_xlim(0,8000 if p==0 else 2000)

        ax[p,1].set_ylabel('Iteration')
        ax[p,1].set_xticklabels([])

        ax[p,0].grid()
        ax[p,1].grid()

        #if p == 0:
        #    ax[p,0].set_title('BIO Network', loc='left')
        #else:
        #    ax[p,0].set_title('LSTM Network', loc='left')

    ax[0,1].set_yticks([1000*i for i in range(9)])
    ax[1,1].set_yticks([200*i for i in range(11)])
    ax[1,0].set_xticks([200*i for i in range(11)])
    plt.suptitle('Training Curves and Time-to-Accuracy With and Without Winner-Take-All')
    plt.show()
















plot_training_curves([fn for fn in fns if 'multistim' in fn])







quit()
for fn, includes_WTA in zip(fns, [('with_WTA' in fn) for fn in fns]):
    print('\n\nCurrently analyzing: {}'.format(fn))

    """data = pickle.load(open(savedir+fn, 'rb'))

    print('Acc:',np.mean(data['task_performance']))

    for k in data['grad_correlations'].keys():
        print(k)
        print(data['grad_correlations'][k])"""





    #grads = data['task_gradients']
    #for task in range(2):
    #    for variable in range(6):
    #        print(grads[task][variable].shape)



quit()
