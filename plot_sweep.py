import numpy as np
import pickle
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fns = sorted([fn for fn in os.listdir('./savedir/') if 'post_kshot' in fn and 'small' in fn])
keys = [int([t for t in fn[fn.find('task')+4:fn.find('task')+6].split('_') if t != ''][0]) for fn in fns]

withs = sorted(fns[::2], key=lambda x : keys[::2][fns[::2].index(x)])
withouts = sorted(fns[1::2], key=lambda x : keys[1::2][fns[1::2].index(x)])

acc_withs = np.array([pickle.load(open('./savedir/'+w, 'rb')) for w in withs])
acc_withouts = np.array([pickle.load(open('./savedir/'+w, 'rb')) for w in withouts])


fns = sorted([fn for fn in os.listdir('./savedir/') if not 'kshot_accuracy' in fn and 'accuracy_kshot_task' in fn and 'small' in fn])

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

task_limit = 2
plt.figure(figsize=(10,8))

plt.plot(np.arange(20)[:task_limit], [acc[-1][i] for i, acc in enumerate(pre_acc_withs)][:task_limit], ls='--',  c='c', lw=1)
plt.scatter(np.arange(20)[:task_limit], [acc[-1][i] for i, acc in enumerate(pre_acc_withs)][:task_limit], c='c', label='With WTA (Pre)')
plt.scatter(np.arange(20)[:task_limit], [np.mean(np.array(acc[-1])[np.arange(20)!=i]) for i, acc in enumerate(pre_acc_withs)][:task_limit], c='g', label='With WTA (Pre others)')

plt.plot(np.arange(20)[:task_limit], [acc[i] for i, acc in enumerate(acc_withs)][:task_limit], ls='--', c='b', lw=1)
plt.scatter(np.arange(20)[:task_limit], [acc[i] for i, acc in enumerate(acc_withs)][:task_limit], c='b', label='With WTA')
plt.scatter(np.arange(20)[:task_limit], [np.mean(acc[np.arange(20)!=i]) for i, acc in enumerate(acc_withs)][:task_limit], c='mediumseagreen', label='With WTA (Post others)')

plt.plot(np.arange(20)[:task_limit], [acc[-1][i] for i, acc in enumerate(pre_acc_withouts)][:task_limit], ls='--', c='m', lw=1)
plt.scatter(np.arange(20)[:task_limit], [acc[-1][i] for i, acc in enumerate(pre_acc_withouts)][:task_limit], c='m', label='Without WTA (Pre)')
plt.scatter(np.arange(20)[:task_limit], [np.mean(np.array(acc[-1])[np.arange(20)!=i]) for i, acc in enumerate(pre_acc_withouts)][:task_limit], c='purple', label='Without WTA (Pre others)')

plt.plot(np.arange(20)[:task_limit], [acc[i] for i, acc in enumerate(acc_withouts)][:task_limit], ls='--', c='r', lw=1)
plt.scatter(np.arange(20)[:task_limit], [acc[i] for i, acc in enumerate(acc_withouts)][:task_limit], c='r', label='Without WTA')
plt.scatter(np.arange(20)[:task_limit], [np.mean(acc[np.arange(20)!=i]) for i, acc in enumerate(acc_withouts)][:task_limit], c='violet', label='Without WTA (Post others)')

plt.xlabel('Task Number')
plt.ylabel('Accuracy After k-shot Training')
plt.title('WTA Comparison -- k-shot Accuracy Task Sweep\n(5 trials, threshold, optimized gammas)')
plt.legend(loc='lower right', ncol=2)
plt.xticks(np.arange(20))
plt.yticks(np.linspace(0,1,9))
plt.xlim(-1,20)
plt.ylim(0,1)
plt.grid()
plt.savefig('./records/WTA_kshot_learning_sweep.png')
