import numpy as np
import matplotlib.pyplot as plt
import pickle

with_WTA = pickle.load(open('accuracy_with_WTA.pkl', 'rb'))
without_WTA = pickle.load(open('accuracy_without_WTA.pkl', 'rb'))
with_WTA_short = pickle.load(open('accuracy_with_WTA_short_cue.pkl', 'rb'))
without_WTA_short = pickle.load(open('accuracy_without_WTA_short_cue.pkl', 'rb'))

iters_per_element = 5

curves = (with_WTA, without_WTA, with_WTA_short, without_WTA_short)
names = ('with WTA', 'without WTA', 'with WTA (short cue)', 'without WTA (short cue)')
colors = ('b', 'r', 'c', 'm')

for curve, name, color in zip(curves, names, colors):

    xrange = iters_per_element*np.arange(len(curve))
    yrange = np.mean(np.array(curve), axis=1)

    plt.plot(xrange, yrange, c=color, label=name)

for a in [0.7, 0.8, 0.9, 0.95]:
    plt.axhline(a, c='k', ls='--')

plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Preliminary Winner-Take-All Testing')
plt.legend()
plt.show()
