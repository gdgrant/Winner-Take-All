import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# list of 20 tasks with [time x batch x neurons]
task_type = 'go_dly_go_WTA'
num_top_neurons = 50
num_neurons = 250
num_tasks = 2


# import activities and accuracy file
activities = pickle.load(open('states_go_dly_go_analysis_multistim_LSTM_with_WTA_v0.pkl','rb'))
accuracies = pickle.load(open('accuracy_go_dly_go_analysis_multistim_LSTM_with_WTA_v0.pkl','rb'))


# Plot activities for each task (x10 trials)
for task in range(num_tasks):
    for i in range(10):
        plt.title('Task_'+str(task)+'_trial_'+str(i)+'_n_'+str(top_neurons))
        plt.imshow(np.array(activities[task])[:,i,:],aspect='auto')
        plt.colorbar()
        plt.savefig('./'+task_type+'/task_'+str(task)+'_trial_'+str(i))
        plt.close()

# Getting top 10 neurons from the beginning of the trial -- not really important
top_neurons = np.zeros((num_tasks,num_neurons))
for task in range(num_tasks):
    ind = np.sort(np.array(activities[task])[0,0,:].argsort()[-10:])
    top_neurons[task,ind] = 1
    print(np.sort(np.array(activities[task])[0,0,:].argsort()[-10:]))
plt.imshow(top_neurons, aspect='auto')
plt.savefig('./'+task_type+'/top_10_neurons')
plt.show()

# Getting all active neurons for each task
top_neurons = np.zeros((num_tasks,num_neurons))
for task in range(num_tasks):
    # ind = np.sort(np.array(activities[task])[0,0,:].argsort()[-num_top_neurons:])
    ind = np.where(np.array(activities[task])[0,0,:]>0)[0]
    top_neurons[task,ind] = 1
    print(ind)
plt.imshow(top_neurons, aspect='auto')
plt.savefig('./'+task_type+'/top_k_neurons')
plt.show()
plt.close()

# Plot task accuracies
for i in range(num_tasks):
    plt.plot(np.array(accuracies)[:,i])
plt.savefig('./'+task_type+'/task_accuracies')
plt.close()



# acc_WTA = pickle.load(open('accuracy_go_dly_go_analysis_multistim_LSTM_with_WTA_v0.pkl','rb'))
# acc_without = pickle.load(open('accuracy_go_dly_go_analysis_multistim_LSTM_without_WTA_v0.pkl','rb'))
# plt.plot(np.array(acc_WTA)[:,0],'r')
# plt.plot(np.array(acc_WTA)[:,1],'r')
# plt.plot(np.array(acc_without)[:,0],'g')
# plt.plot(np.array(acc_without)[:,1],'g')
# plt.show()




