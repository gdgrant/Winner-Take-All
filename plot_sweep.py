import numpy as np
import pickle
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fns = sorted([fn for fn in os.listdir('./savedir/') if 'kshot_accuracy' in fn and 'higher_weight_cost' in fn])

keys = [int([t for t in fn[fn.find('task')+4:fn.find('task')+6].split('_') if t != ''][0]) for fn in fns]
withs = sorted(fns[::2], key=lambda x : keys[::2][fns[::2].index(x)])
withouts = sorted(fns[1::2], key=lambda x : keys[1::2][fns[1::2].index(x)])

acc_withs = np.array([pickle.load(open('./savedir/'+w, 'rb')) for w in withs])
acc_withouts = np.array([pickle.load(open('./savedir/'+w, 'rb')) for w in withouts])


fns = sorted([fn for fn in os.listdir('./savedir/') if 'accuracy_higher_weight_cost_kshot' in fn and not 'kshot_accuracy' in fn and not 'test' in fn])

keys = [int([t for t in fn[fn.find('task')+4:fn.find('task')+6].split('_') if t != ''][0]) for fn in fns]
withs = sorted(fns[::2], key=lambda x : keys[::2][fns[::2].index(x)])
withouts = sorted(fns[1::2], key=lambda x : keys[1::2][fns[1::2].index(x)])

pre_acc_withs = np.array([pickle.load(open('./savedir/'+w, 'rb')) for w in withs])
pre_acc_withouts = np.array([pickle.load(open('./savedir/'+w, 'rb')) for w in withouts])

#print(acc_withs.shape)
#print(acc_withouts.shape)
#print(len(pre_acc_withs[0][-1]))
#print(pre_acc_withouts.shape)
#quit()


plt.plot(np.arange(20), [acc[-1][i] for i, acc in enumerate(pre_acc_withs)], ls='--',  c='c', lw=1)
plt.scatter(np.arange(20), [acc[-1][i] for i, acc in enumerate(pre_acc_withs)], c='c', label='With WTA (Pre)')
plt.scatter(np.arange(20), [np.mean(np.array(acc[-1])[np.arange(20)!=i]) for i, acc in enumerate(pre_acc_withs)], c='g', label='With WTA (Pre others)')

plt.plot(np.arange(20), [acc[i] for i, acc in enumerate(acc_withs)], ls='--', c='b', lw=1)
plt.scatter(np.arange(20), [acc[i] for i, acc in enumerate(acc_withs)], c='b', label='With WTA')

plt.plot(np.arange(20), [acc[-1][i] for i, acc in enumerate(pre_acc_withouts)], ls='--', c='m', lw=1)
plt.scatter(np.arange(20), [acc[-1][i] for i, acc in enumerate(pre_acc_withouts)], c='m', label='Without WTA (Pre)')
plt.scatter(np.arange(20), [np.mean(np.array(acc[-1])[np.arange(20)!=i]) for i, acc in enumerate(pre_acc_withouts)], c='purple', label='Without WTA (Pre others)')

plt.plot(np.arange(20), [acc[i] for i, acc in enumerate(acc_withouts)], ls='--', c='r', lw=1)
plt.scatter(np.arange(20), [acc[i] for i, acc in enumerate(acc_withouts)], c='r', label='Without WTA')

plt.xlabel('Task Number')
plt.ylabel('Accuracy After k-shot Training')
plt.title('WTA Comparison -- k-shot Accuracy Task Sweep\n(5 trials, threshold, higher weight cost)')
plt.legend(loc='lower center', ncol=2)
plt.xticks(np.arange(20))
plt.yticks(np.linspace(0,1,9))
plt.xlim(-1,20)
plt.ylim(0,1)
plt.grid()
plt.savefig('./WTA_with_higher_weight_cost.png')
