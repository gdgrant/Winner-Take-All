import numpy as np
import pickle
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fns = [fn for fn in os.listdir('./savedir/') if 'goset_BIO' in fn and 'training_accuracy' in fn]

for fn in fns:
    data = np.array(pickle.load(open('./savedir/'+fn, 'rb')))
    plt.plot(10*np.arange(data.shape[0]), np.mean(data,axis=1), label=fn)

plt.title('Go/Anti-Go BIO Training Curves')
plt.xlim(0,500)
plt.ylim(0,1)
plt.legend()
plt.savefig('./records/goset_bio_training_curves.png')
