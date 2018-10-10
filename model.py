### Authors: Nicolas Y. Masse, Gregory D. Grant

# Required packages
import tensorflow as tf
import numpy as np
import scipy.stats
import pickle
import os, sys, time
from itertools import product

# Model modules
from parameters import *
import stimulus
import AdamOpt

# Match GPU IDs to nvidia-smi command
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class Model:

    """ RNN model for supervised and reinforcement learning training """

    def __init__(self, input_data, target_data, mask, gating, trial_mask, lesion_id):

        # Load input activity, target data, training mask, etc.
        self.input_data         = tf.unstack(input_data, axis=0)
        self.target_data        = tf.unstack(target_data, axis=0)
        self.gating             = tf.reshape(gating, [1,-1])
        self.time_mask          = tf.unstack(mask, axis=0)
        self.trial_mask         = trial_mask
        self.lesion_id          = lesion_id

        # Declare all Tensorflow variables
        self.declare_variables()

        # Build the Tensorflow graph
        self.rnn_cell_loop()

        # Train the model
        self.optimize()


    def declare_variables(self):
        """ Initialize all required variables """

        # All the possible prefixes based on network setup
        lstm_var_prefixes   = ['Wf', 'Wi', 'Wo', 'Wc', 'Uf', 'Ui', 'Uo', 'Uc', 'bf', 'bi', 'bo', 'bc']
        bio_var_prefixes    = ['W_in', 'b_rnn', 'W_rnn']
        rl_var_prefixes     = ['W_pol_out', 'b_pol_out', 'W_val_out', 'b_val_out']
        base_var_prefies    = ['W_out', 'b_out']

        # Add relevant prefixes to variable declaration
        prefix_list = base_var_prefies
        if par['architecture'] == 'LSTM':
            prefix_list += lstm_var_prefixes
        elif par['architecture'] == 'BIO':
            prefix_list += bio_var_prefixes

        if par['training_method'] == 'RL':
            prefix_list += rl_var_prefixes
        elif par['training_method'] == 'SL':
            pass

        # Use prefix list to declare required variables and place them in a dict
        self.var_dict = {}
        with tf.variable_scope('network'):
            for p in prefix_list:
                self.var_dict[p] = tf.get_variable(p, initializer=par[p+'_init'])

        if par['architecture'] == 'BIO':
            # Modify recurrent weights if using EI neurons (in a BIO architecture)
            self.W_rnn_eff = (tf.constant(par['EI_matrix']) @ tf.nn.relu(self.var_dict['W_rnn'])) \
                if par['EI'] else self.var_dict['W_rnn']

        self.var_dict['h_init'] = tf.get_variable('h_init', initializer=par['h_init'], trainable=True)

        self.lesion_gate = tf.get_variable('lesion_gate', initializer=np.float32(np.ones([1,par['n_hidden']])), trainable=False)
        self.lesion_neuron = tf.assign(self.lesion_gate[:,self.lesion_id], tf.zeros_like(self.lesion_gate)[:,self.lesion_id])


    def rnn_cell_loop(self):
        """ Initialize parameters and execute loop through
            time to generate the network outputs """

        # Specify training method outputs
        self.output = []
        self.mask = []
        self.mask.append(tf.constant(np.ones((par['batch_size'], 1), dtype = np.float32)))
        if par['training_method'] == 'RL':
            self.pol_out = self.output  # For interchangeable use
            self.val_out = []
            self.action = []
            self.reward = []
            self.reward.append(tf.constant(np.zeros((par['batch_size'], par['n_val']), dtype = np.float32)))

        # Initialize state records
        self.h      = []
        #self.syn_x  = []
        #self.syn_u  = []

        # Initialize network state
        if par['architecture'] == 'BIO':
            h = self.gating*self.var_dict['h_init']
            c = tf.constant(par['h_init'])
        elif par['architecture'] == 'LSTM':
            h = self.var_dict['h_init']
            c = tf.zeros_like(par['h_init'])
        syn_x = tf.constant(par['syn_x_init'])
        syn_u = tf.constant(par['syn_u_init'])
        mask  = self.mask[0]

        # Loop through the neural inputs, indexed in time
        for rnn_input, target, time_mask in zip(self.input_data, self.target_data, self.time_mask):

            # Compute the state of the hidden layer
            h, c, syn_x, syn_u = self.recurrent_cell(h, c, syn_x, syn_u, rnn_input)

            # Record hidden state
            self.h.append(h)
            #self.syn_x.append(syn_x)
            #self.syn_u.append(syn_u)

            if par['training_method'] == 'SL':
                # Compute outputs for loss
                y = h @ self.var_dict['W_out'] + self.var_dict['b_out']

                # Record supervised outputs
                self.output.append(y)

            elif par['training_method'] == 'RL':
                # Compute outputs for action
                pol_out        = h @ self.var_dict['W_pol_out'] + self.var_dict['b_pol_out']
                action_index   = tf.multinomial(pol_out, 1)
                action         = tf.one_hot(tf.squeeze(action_index), par['n_pol'])

                # Compute outputs for loss
                pol_out        = tf.nn.softmax(pol_out, 1)  # Note softmax for entropy loss
                val_out        = h @ self.var_dict['W_val_out'] + self.var_dict['b_val_out']

                # Check for trial continuation (ends if previous reward was non-zero)
                continue_trial = tf.cast(tf.equal(self.reward[-1], 0.), tf.float32)
                mask          *= continue_trial
                reward         = tf.reduce_sum(action*target, axis=1, keep_dims=True)*mask*tf.reshape(time_mask,[par['batch_size'], 1])

                # Record RL outputs
                self.pol_out.append(pol_out)
                self.val_out.append(val_out)
                self.action.append(action)
                self.reward.append(reward)

            # Record mask (outside if statement for cross-comptability)
            self.mask.append(mask)

        # Reward and mask trimming where necessary
        self.mask = self.mask[1:]
        if par['training_method'] == 'RL':
            self.reward = self.reward[1:]


    def recurrent_cell(self, h, c, syn_x, syn_u, rnn_input):
        """ Using the appropriate recurrent cell
            architecture, compute the hidden state """

        if par['architecture'] == 'BIO':

            # Apply synaptic short-term facilitation and depression, if required
            if par['synapse_config'] == 'std_stf':
                syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_u*syn_x*h
                syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
                syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
                syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
                h_post = syn_u*syn_x*h
            else:
                h_post = h

            # Compute hidden state
            h = self.gating*tf.nn.relu((1-par['alpha_neuron'])*h \
              + par['alpha_neuron']*(rnn_input @ self.var_dict['W_in'] + h_post @ self.W_rnn_eff + self.var_dict['b_rnn']) \
              + tf.random_normal(h.shape, 0, par['noise_rnn'], dtype=tf.float32))
            c = tf.constant(-1.)

        elif par['architecture'] == 'LSTM':

            # Compute LSTM state
            # f : forgetting gate, i : input gate,
            # c : cell state, o : output gate
            f   = tf.sigmoid(rnn_input @ self.var_dict['Wf'] + h @ self.var_dict['Uf'] + self.var_dict['bf'])
            i   = tf.sigmoid(rnn_input @ self.var_dict['Wi'] + h @ self.var_dict['Ui'] + self.var_dict['bi'])
            cn  = tf.tanh(rnn_input @ self.var_dict['Wc'] + h @ self.var_dict['Uc'] + self.var_dict['bc'])
            c   = f * c + i * cn
            o   = tf.sigmoid(rnn_input @ self.var_dict['Wo'] + h @ self.var_dict['Uo'] + self.var_dict['bo'])

            # Compute hidden state
            h = self.gating * o * tf.tanh(c)
            syn_x = tf.constant(-1.)
            syn_u = tf.constant(-1.)

        # Apply lesioning as specified prior
        h *= self.lesion_gate

        # Select top neurons
        if par['winner_take_all']:
            top_k, _ = tf.nn.top_k(h, par['top_k_neurons'])
            drop = tf.where(h < top_k[:,par['top_k_neurons']-1:par['top_k_neurons']], tf.zeros(h.shape, tf.float32), tf.ones(h.shape, tf.float32))
            return drop*h, c, syn_x, syn_u
        else:
            return h, c, syn_x, syn_u


    def optimize(self):
        """ Calculate losses and apply corrections to model """

        # Set up optimizer and required constants
        epsilon = 1e-7
        adam_optimizer = AdamOpt.AdamOpt(tf.trainable_variables(), learning_rate=par['learning_rate'])

        # Make stabilization records
        self.prev_weights = {}
        self.big_omega_var = {}
        reset_prev_vars_ops = []
        aux_losses = []

        # Set up stabilization based on trainable variables
        for var in tf.trainable_variables():
            n = var.op.name

            # Make big omega and prev_weight variables
            self.big_omega_var[n] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            self.prev_weights[n]  = tf.Variable(tf.zeros(var.get_shape()), trainable=False)

            # Don't stabilize value weights/biases
            if not 'val' in n:
                aux_losses.append(par['omega_c'] * \
                    tf.reduce_sum(self.big_omega_var[n] * tf.square(self.prev_weights[n] - var)))

            # Make a reset function for each prev_weight element
            reset_prev_vars_ops.append(tf.assign(self.prev_weights[n], var))

        # Auxiliary stabilization loss
        self.aux_loss = tf.add_n(aux_losses)

        # Spiking activity loss (penalty on high activation values in the hidden layer)
        self.spike_loss = par['spike_cost']*tf.reduce_mean(tf.stack([mask*time_mask*tf.reduce_mean(h) \
            for (h, mask, time_mask) in zip(self.h, self.mask, self.time_mask)]))

        if par['architecture'] == 'BIO':
            if par['EI']:
                self.weight_loss = par['weight_cost']*tf.reduce_mean(tf.nn.relu(self.var_dict['W_rnn']))
            else:
                self.weight_loss = par['weight_cost']*tf.reduce_mean(tf.abs(self.var_dict['W_rnn']))

        elif par['architecture'] == 'LSTM':
            aggregate = tf.reduce_mean(tf.abs(self.var_dict['Uf'])) \
                + tf.reduce_mean(tf.abs(self.var_dict['Ui'])) \
                + tf.reduce_mean(tf.abs(self.var_dict['Uo'])) \
                + tf.reduce_mean(tf.abs(self.var_dict['Uc']))
            self.weight_loss = par['weight_cost']*aggregate

        # Training-specific losses
        if par['training_method'] == 'SL':
            RL_loss = tf.constant(0.)

            # Task loss (cross entropy)
            self.pol_loss = tf.reduce_mean([self.trial_mask*mask*tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, \
                labels=target, dim=1) for y, target, mask in zip(self.output, self.target_data, self.time_mask)])
            sup_loss = self.pol_loss

        elif par['training_method'] == 'RL':
            sup_loss = tf.constant(0.)

            # Collect information from across time
            self.time_mask  = tf.reshape(tf.stack(self.time_mask),(par['num_time_steps'], par['batch_size'], 1))
            self.mask       = tf.stack(self.mask)
            self.reward     = tf.stack(self.reward)
            self.action     = tf.stack(self.action)
            self.pol_out    = tf.stack(self.pol_out)

            # Get the value outputs of the network, and pad the last time step
            val_out = tf.concat([tf.stack(self.val_out), tf.zeros([1,par['batch_size'],par['n_val']])], axis=0)

            # Determine terminal state of the network
            terminal_state = tf.cast(tf.logical_not(tf.equal(self.reward, tf.constant(0.))), tf.float32)

            # Compute predicted value and the advantage for plugging into the policy loss
            pred_val = self.reward + par['discount_rate']*val_out[1:,:,:]*(1-terminal_state)
            advantage = pred_val - val_out[:-1,:,:]

            # Stop gradients back through action, advantage, and mask
            action_static    = tf.stop_gradient(self.action)
            advantage_static = tf.stop_gradient(advantage)
            mask_static      = tf.stop_gradient(self.mask)

            # Policy loss
            self.pol_loss = -tf.reduce_mean(advantage_static*mask_static*self.time_mask*action_static*tf.log(epsilon+self.pol_out))

            # Value loss
            self.val_loss = 0.5*par['val_cost']*tf.reduce_mean(mask_static*self.time_mask*tf.square(val_out[:-1,:,:]-tf.stop_gradient(pred_val)))

            # Entropy loss
            self.entropy_loss = -par['entropy_cost']*tf.reduce_mean(tf.reduce_sum(mask_static*self.time_mask*self.pol_out*tf.log(epsilon+self.pol_out), axis=1))

            # Collect RL losses
            RL_loss = self.pol_loss + self.val_loss - self.entropy_loss

        # Collect loss terms and compute gradients
        total_loss = sup_loss + RL_loss + self.aux_loss + self.spike_loss + self.weight_loss
        self.train_op = adam_optimizer.compute_gradients(total_loss)

        # Determine trial gradients by SGD
        gd_opt = tf.train.GradientDescentOptimizer(learning_rate=1.)
        self.batch_grads = gd_opt.compute_gradients(total_loss)

        # Stabilize weights
        if par['stabilization'] == 'pathint':
            # Zenke method
            self.pathint_stabilization(adam_optimizer)
        elif par['stabilization'] == 'EWC':
            # Kirkpatrick method
            self.EWC()
        else:
            # No stabilization
            pass

        # Make reset operations
        self.reset_prev_vars = tf.group(*reset_prev_vars_ops)
        self.reset_adam_op = adam_optimizer.reset_params()
        self.reset_weights()

        # Make saturation correction operation
        self.make_recurrent_weights_positive()


    def reset_weights(self):
        """ Make new weights, if requested """

        reset_weights = []
        for var in tf.trainable_variables():
            if 'b' in var.op.name:
                # reset biases to 0
                reset_weights.append(tf.assign(var, var*0.))
            elif 'W' in var.op.name:
                # reset weights to initial-like conditions
                new_weight = initialize(var.shape, par['connection_prob'])
                reset_weights.append(tf.assign(var,new_weight))

        self.reset_weights = tf.group(*reset_weights)


    def make_recurrent_weights_positive(self):
        """ Very slightly de-saturate recurrent weights """

        reset_weights = []
        for var in tf.trainable_variables():
            if 'W_rnn' in var.op.name:
                # make all negative weights slightly positive
                reset_weights.append(tf.assign(var,tf.maximum(1e-9, var)))

        self.reset_rnn_weights = tf.group(*reset_weights)


    def pathint_stabilization(self, adam_optimizer):
        """ Synaptic stabilization via the Zenke method """

        # Set up method
        optimizer_task = tf.train.GradientDescentOptimizer(learning_rate =  1.0)
        small_omega_var = {}
        small_omega_var_div = {}

        reset_small_omega_ops = []
        update_small_omega_ops = []
        update_big_omega_ops = []

        # If using reinforcement learning, update rewards
        if par['training_method'] == 'RL':
            self.previous_reward = tf.Variable(-tf.ones([]), trainable=False)
            self.current_reward = tf.Variable(-tf.ones([]), trainable=False)

            reward_stacked = tf.stack(self.reward, axis = 0)
            current_reward = tf.reduce_mean(tf.reduce_sum(reward_stacked, axis = 0))
            self.update_current_reward = tf.assign(self.current_reward, current_reward)
            self.update_previous_reward = tf.assign(self.previous_reward, self.current_reward)

        # Iterate over variables in the model
        for var in tf.trainable_variables():

            # Reset the small omega vars
            small_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            small_omega_var_div[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            reset_small_omega_ops.append(tf.assign(small_omega_var[var.op.name], small_omega_var[var.op.name]*0.0 ) )
            reset_small_omega_ops.append(tf.assign(small_omega_var_div[var.op.name], small_omega_var_div[var.op.name]*0.0 ) )

            # Update the big omega vars based on the training method
            if par['training_method'] == 'RL':
                update_big_omega_ops.append(tf.assign_add( self.big_omega_var[var.op.name], tf.div(tf.abs(small_omega_var[var.op.name]), \
                    (par['omega_xi'] + small_omega_var_div[var.op.name]))))
            elif par['training_method'] == 'SL':
                update_big_omega_ops.append(tf.assign_add( self.big_omega_var[var.op.name], tf.div(tf.nn.relu(small_omega_var[var.op.name]), \
                    (par['omega_xi'] + small_omega_var_div[var.op.name]**2))))

        # After each task is complete, call update_big_omega and reset_small_omega
        self.update_big_omega = tf.group(*update_big_omega_ops)

        # Reset_small_omega also makes a backup of the final weights, used as hook in the auxiliary loss
        self.reset_small_omega = tf.group(*reset_small_omega_ops)

        # This is called every batch
        self.delta_grads = adam_optimizer.return_delta_grads()
        self.gradients = optimizer_task.compute_gradients(self.pol_loss)

        # Update the samll omegas using the gradients
        for (grad, var) in self.gradients:
            if par['training_method'] == 'RL':
                delta_reward = self.current_reward - self.previous_reward
                update_small_omega_ops.append(tf.assign_add(small_omega_var[var.op.name], self.delta_grads[var.op.name]*delta_reward))
                update_small_omega_ops.append(tf.assign_add(small_omega_var_div[var.op.name], tf.abs(self.delta_grads[var.op.name]*delta_reward)))
            elif par['training_method'] == 'SL':
                update_small_omega_ops.append(tf.assign_add(small_omega_var[var.op.name], -self.delta_grads[var.op.name]*grad ))
                update_small_omega_ops.append(tf.assign_add(small_omega_var_div[var.op.name], self.delta_grads[var.op.name]))

        # Make update group
        self.update_small_omega = tf.group(*update_small_omega_ops) # 1) update small_omega after each train!


    def EWC(self):
        """ Synaptic stabilization via the Kirkpatrick method """

        # Set up method
        var_list = [var for var in tf.trainable_variables() if not 'val' in var.op.name]
        epsilon = 1e-6
        fisher_ops = []
        opt = tf.train.GradientDescentOptimizer(learning_rate = 1.0)

        # Sample from logits
        if par['training_method'] == 'RL':
            log_p_theta = tf.stack([mask*time_mask*action*tf.log(epsilon + pol_out) for (pol_out, action, mask, time_mask) in \
                zip(self.pol_out, self.action, self.mask, self.time_mask)], axis = 0)
        elif par['training_method'] == 'SL':
            log_p_theta = tf.stack([mask*time_mask*tf.log(epsilon + output) for (output, mask, time_mask) in \
                zip(self.output, self.mask, self.time_mask)], axis = 0)

        # Compute gradients and add to aggregate
        grads_and_vars = opt.compute_gradients(log_p_theta, var_list = var_list)
        for grad, var in grads_and_vars:
            print(var.op.name)
            fisher_ops.append(tf.assign_add(self.big_omega_var[var.op.name], \
                grad*grad/par['EWC_fisher_num_batches']))

        # Make update group
        self.update_big_omega = tf.group(*fisher_ops)


def supervised_learning(save_fn='test.pkl', gpu_id=None):
    """ Run supervised learning training """

    # Isolate requested GPU
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Reset Tensorflow graph before running anything
    tf.reset_default_graph()

    # Define all placeholders
    x, y, m, g, trial_mask, lid = get_supervised_placeholders()

    # Set up stimulus and accuracy recording
    stim = stimulus.MultiStimulus()
    accuracy_full = []
    accuracy_grid = np.zeros([par['n_tasks'],par['n_tasks']])
    full_activity_list = []

    # Display relevant parameters
    print('\nRunning model with savename: {}'.format(save_fn))
    print_key_info()

    # Start Tensorflow session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8) if gpu_id == '0' else tf.GPUOptions()
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        # Select CPU or GPU
        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(x, y, m, g, trial_mask, lid)

        # Initialize variables and start the timer
        saver = tf.train.Saver(max_to_keep=100)
        sess.run(tf.global_variables_initializer())
        t_start = time.time()
        sess.run(model.reset_prev_vars)

        # Begin training loop, iterating over tasks
        task = 0 # For legacy comptability
        accuracy_record = []
        for i in range(par['n_train_batches']):

            if par['do_k_shot_testing'] and par['load_from_checkpoint']:
                break

            stims = []
            hats = []
            mks = []
            for t in range(par['n_tasks']):
                _, stim_in, hat, mk, _ = stim.generate_trial(t)
                stims.append(stim_in)
                hats.append(hat)
                mks.append(mk)

            stims = np.stack(stims, axis=0)
            hats = np.stack(hats, axis=0)
            mks = np.stack(mks, axis=0)

            base_inds = np.setdiff1d(np.arange(par['n_tasks']), par['k_shot_task']) if \
                par['do_k_shot_testing'] else np.arange(par['n_tasks'])
            inds = np.random.choice(base_inds, size=[par['batch_size']])

            stim_in = np.zeros([par['num_time_steps'], par['batch_size'], par['n_input']])
            y_hat = np.zeros([par['num_time_steps'], par['batch_size'], par['n_output']])
            mk = np.zeros([par['num_time_steps'], par['batch_size']])
            for b in range(par['batch_size']):
                stim_in[:,b,:] = stims[inds[b],:,b,:]
                y_hat[:,b,:] = hats[inds[b],:,b,:]
                mk[:,b] = mks[inds[b],:,b]

            # Put together the feed dictionary
            feed_dict = {x:stim_in, y:y_hat, g:par['gating'][0], m:mk}

            # Run the model using one of the available stabilization methods
            if par['stabilization'] == 'pathint':
                _, _, loss, AL, weight_loss, spike_loss, output, hidden = sess.run([model.train_op, \
                    model.update_small_omega, model.pol_loss, model.aux_loss, \
                    model.weight_loss, model.spike_loss, model.output, model.h], feed_dict=feed_dict)
            elif par['stabilization'] == 'EWC':
                _, loss, AL, output = sess.run([model.train_op, model.pol_loss, \
                    model.aux_loss, model.output], feed_dict=feed_dict)

            # Display network performance
            if i%10 == 0:
                acc = get_perf(y_hat, output, mk)
                print('Iter {} | Accuracy {:5.3f} | Loss {:5.3f} | Weight Loss {:5.3f} | Mean Activity {:5.3f} +/- {:5.3}'.format(\
                    i, acc, loss, weight_loss, np.mean(hidden), np.std(hidden)))

                task_accs = []
                task_grads = []
                task_states = []
                off_task_accs = []
                for t in range(par['n_tasks']):
                    _, stim_in, y_hat, mk, _ = stim.generate_trial(t)
                    output, batch_grads, h = sess.run([model.output, model.batch_grads, model.h], feed_dict={x:stim_in, y:y_hat, g:par['gating'][task], m:mk})

                    perf = get_perf(y_hat, output, mk)
                    if t != par['k_shot_task']:
                        off_task_accs.append(perf)
                    task_accs.append(perf)
                    task_grads.append(batch_grads)
                    task_states.append(h)

                accuracy_record.append(task_accs)
                pickle.dump(accuracy_record, open('./savedir/accuracy_{}.pkl'.format(save_fn), 'wb'))

                print('Task accuracies:', *['| {:5.3f}'.format(el) for el in task_accs])
                print('Trained Tasks Mean:', np.mean(off_task_accs), '\n')

                if par['use_threshold'] and np.mean(off_task_accs) > 0.95:
                    print('Trained tasks 95\% accuracy threshold reached.')
                    break

        print('Saving states, parameters, and weights...')
        pickle.dump(task_states, open('./savedir/states_{}.pkl'.format(save_fn), 'wb'))
        pickle.dump({'parameters':par, 'weights':sess.run(model.var_dict)}, open('./weights/weights_for_'+save_fn+'.pkl', 'wb'))
        print('States, parameters, and weights saved.')

        if par['do_k_shot_testing']:

            print('\nStarting k-shot testing for task {}.'.format(par['k_shot_task']))

            if not par['load_from_checkpoint']:
                saver.save(sess, './checkpoints/{}'.format(save_fn))

            overall_task_accs = []
            for i in range(par['testing_iters']):

                saver.restore(sess, './checkpoints/{}'.format(save_fn))


                _, stim_in, y_hat, mk, _ = stim.generate_trial(par['k_shot_task'])

                masking = np.zeros([par['batch_size'], 1])
                masking[:par['num_shots'],:] = 1
                feed_dict = {x:stim_in, y:y_hat, g:par['gating'][0], m:mk, trial_mask:np.float32(masking)}

                for _ in range(par['shot_reps']):

                    # Run the model using one of the available stabilization methods
                    if par['stabilization'] == 'pathint':
                        _, _, loss, AL, spike_loss, output = sess.run([model.train_op, \
                            model.update_small_omega, model.pol_loss, model.aux_loss, \
                            model.spike_loss, model.output], feed_dict=feed_dict)
                    elif par['stabilization'] == 'EWC':
                        _, loss, AL, output = sess.run([model.train_op, model.pol_loss, \
                            model.aux_loss, model.output], feed_dict=feed_dict)

                task_accs = []
                task_grads = []
                task_states = []
                for t in range(par['n_tasks']):
                    _, stim_in, y_hat, mk, _ = stim.generate_trial(t)
                    output, batch_grads, h = sess.run([model.output, model.batch_grads, model.h], feed_dict={x:stim_in, y:y_hat, g:par['gating'][task], m:mk})
                    task_accs.append(get_perf(y_hat, output, mk))
                    task_grads.append(batch_grads)
                    task_states.append(h)

                print('\nTesting Iter {} | k-shot Trained Task {} Accuracy {:5.3f}'.format(i, par['k_shot_task'], task_accs[par['k_shot_task']]))
                print('Task accuracies:', *['| {:5.3f}'.format(el) for el in task_accs])

                overall_task_accs.append(task_accs)

            overall_task_accs = np.mean(np.array(overall_task_accs), axis=0)
            pickle.dump(overall_task_accs, open('./savedir/kshot_accuracy_{}.pkl'.format(save_fn), 'wb'))
            print('\n-----\nOverall task accuracies:\n   ', *['| {:5.3f}'.format(el) for el in overall_task_accs])

    print('\nModel execution for {} complete. (Supervised)'.format(save_fn))


def reinforcement_learning(save_fn='test.pkl', gpu_id=None):
    """ Run reinforcement learning training """

    # Isolate requested GPU
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Reset Tensorflow graph before running anything
    tf.reset_default_graph()

    # Define all placeholders
    x, target, mask, pred_val, actual_action, \
        advantage, mask, gating = generate_placeholders()

    # Set up stimulus and accuracy recording
    stim = stimulus.MultiStimulus()
    accuracy_full = []
    accuracy_grid = np.zeros([par['n_tasks'],par['n_tasks']])
    full_activity_list = []
    model_performance = {'reward': [], 'entropy_loss': [], 'val_loss': [], 'pol_loss': [], 'spike_loss': [], 'trial': [], 'task': []}
    reward_matrix = np.zeros((par['n_tasks'], par['n_tasks']))

    # Display relevant parameters
    print_key_info()

    # Start Tensorflow session
    with tf.Session() as sess:

        # Select CPU or GPU
        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            # Check order against args unpacking in model if editing
            model = Model(x, target, mask, gating)

        # Initialize variables and start the timer
        sess.run(tf.global_variables_initializer())
        t_start = time.time()
        sess.run(model.reset_prev_vars)

        # Begin training loop, iterating over tasks
        for task in range(par['n_tasks']):
            accuracy_iter = []
            task_start_time = time.time()

            for i in range(par['n_train_batches']):

                # Generate a batch of stimulus data for training
                name, input_data, _, mk, reward_data = stim.generate_trial(task)
                mk = mk[...,np.newaxis]

                # Put together the feed dictionary
                feed_dict = {x:input_data, target:reward_data, mask:mk, gating:par['gating'][task]}

                # Calculate and apply gradients
                if par['stabilization'] == 'pathint':
                    _, _, _, pol_loss, val_loss, aux_loss, spike_loss, ent_loss, h_list, reward_list = \
                        sess.run([model.train_op, model.update_current_reward, model.update_small_omega, model.pol_loss, model.val_loss, \
                        model.aux_loss, model.spike_loss, model.entropy_loss, model.h, model.reward], feed_dict = feed_dict)
                    if i>0:
                        sess.run([model.update_small_omega])
                    sess.run([model.update_previous_reward])
                elif par['stabilization'] == 'EWC':
                    _, _, pol_loss,val_loss, aux_loss, spike_loss, ent_loss, h_list, reward_list = \
                        sess.run([model.train_op, model.update_current_reward, model.pol_loss, model.val_loss, \
                        model.aux_loss, model.spike_loss, model.entropy_loss, model.h, model.reward], feed_dict = feed_dict)

                # Record accuracies
                reward = np.stack(reward_list)
                acc = np.mean(np.sum(reward>0,axis=0))
                accuracy_iter.append(acc)
                if i > 2000:
                    if np.mean(accuracy_iter[-2000:]) > 0.985 or (i>25000 and np.mean(accuracy_iter[-2000:]) > 0.98):
                        print('Accuracy reached threshold')
                        break

                # Display network performance
                if i%500 == 0:
                    print('Iter ', i, 'Task name ', name, ' accuracy', acc, ' aux loss', aux_loss, \
                    'mean h', np.mean(np.stack(h_list)), 'time ', np.around(time.time() - task_start_time))

            # Update big omegaes, and reset other values before starting new task
            if par['stabilization'] == 'pathint':
                big_omegas = sess.run([model.update_big_omega, model.big_omega_var])


            elif par['stabilization'] == 'EWC':
                for n in range(par['EWC_fisher_num_batches']):
                    name, input_data, _, mk, reward_data = stim.generate_trial(task)
                    mk = mk[..., np.newaxis]
                    big_omegas = sess.run([model.update_big_omega,model.big_omega_var], feed_dict = \
                        {x:input_data, target: reward_data, gating:par['gating'][task], mask:mk})

            # Test all tasks at the end of each learning session
            num_reps = 10
            task_activity_list = []
            for task_prime in range(task+1):
                for r in range(num_reps):

                    # make batch of training data
                    name, input_data, _, mk, reward_data = stim.generate_trial(task_prime)
                    mk = mk[..., np.newaxis]

                    reward_list, h = sess.run([model.reward, model.h], feed_dict = {x:input_data, target: reward_data, \
                        gating:par['gating'][task_prime], mask:mk})

                    reward = np.squeeze(np.stack(reward_list))
                    reward_matrix[task,task_prime] += np.mean(np.sum(reward>0,axis=0))/num_reps

                # Record network activity
                task_activity_list.append(h)

            # Aggregate task after testing each task set
            # Each of [all tasks] elements is [tasks tested, time steps, batch size hidden size]
            full_activity_list.append(task_activity_list)

            print('Accuracy grid after task {}:'.format(task))
            print(reward_matrix[task,:])

            results = {'reward_matrix': reward_matrix, 'par': par, 'activity': full_activity_list}
            pickle.dump(results, open(par['save_dir'] + save_fn, 'wb') )
            print('Analysis results saved in', save_fn)
            print('')

            # Reset the Adam Optimizer, and set the previous parameter values to their current values
            sess.run(model.reset_adam_op)
            sess.run(model.reset_prev_vars)
            if par['stabilization'] == 'pathint':
                sess.run(model.reset_small_omega)

    print('\nModel execution complete. (Reinforcement)')


def print_key_info():
    """ Display requested information """

    if par['training_method'] == 'SL':
        key_info = ['synapse_config','spike_cost','weight_cost','entropy_cost','omega_c','omega_xi',\
            'n_hidden','noise_rnn_sd','learning_rate','gating_type', 'gate_pct', 'winner_take_all', 'top_k_neurons']
        key_info = ['training_method', 'architecture', 'synapse_config', 'learning_rate', 'n_hidden', \
            'dt', 'membrane_time_constant', 'winner_take_all', 'top_k_neurons', 'do_k_shot_testing', 'k_shot_task', \
            'num_shots', 'shot_reps', 'testing_iters', 'load_from_checkpoint']

    elif par['training_method'] == 'RL':
        key_info = ['synapse_config','spike_cost','weight_cost','entropy_cost','omega_c','omega_xi',\
            'n_hidden','noise_rnn_sd','learning_rate','discount_rate', 'mask_duration', 'stabilization'\
            ,'gating_type', 'gate_pct','fix_break_penalty','wrong_choice_penalty',\
            'correct_choice_reward','include_rule_signal']

    #print('Key info:')
    print('-'*40)
    for k in key_info:
        print(k.ljust(30), par[k])
    print('-'*40)


def print_reinforcement_results(iter_num, model_performance):
    """ Aggregate and display reinforcement learning results """

    reward = np.mean(np.stack(model_performance['reward'])[-par['iters_between_outputs']:])
    pol_loss = np.mean(np.stack(model_performance['pol_loss'])[-par['iters_between_outputs']:])
    val_loss = np.mean(np.stack(model_performance['val_loss'])[-par['iters_between_outputs']:])
    entropy_loss = np.mean(np.stack(model_performance['entropy_loss'])[-par['iters_between_outputs']:])

    print('Iter. {:4d}'.format(iter_num) + ' | Reward {:0.4f}'.format(reward) +
      ' | Pol loss {:0.4f}'.format(pol_loss) + ' | Val loss {:0.4f}'.format(val_loss) +
      ' | Entropy loss {:0.4f}'.format(entropy_loss))


def get_perf(target, output, mask):
    """ Calculate task accuracy by comparing the actual network output
    to the desired output only examine time points when test stimulus is
    on in another words, when target[:,:,-1] is not 0 """

    output = np.stack(output, axis=0)
    mk = mask*np.reshape(target[:,:,-1] == 0, (par['num_time_steps'], par['batch_size']))

    target = np.argmax(target, axis = 2)
    output = np.argmax(output, axis = 2)

    return np.sum(np.float32(target == output)*np.squeeze(mk))/np.sum(mk)


def append_model_performance(model_performance, reward, entropy_loss, pol_loss, val_loss, trial_num):

    reward = np.mean(np.sum(reward,axis = 0))/par['trials_per_sequence']
    model_performance['reward'].append(reward)
    model_performance['entropy_loss'].append(entropy_loss)
    model_performance['pol_loss'].append(pol_loss)
    model_performance['val_loss'].append(val_loss)
    model_performance['trial'].append(trial_num)

    return model_performance


def get_supervised_placeholders():
    x = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'], par['n_input']], 'stim')
    y = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'], par['n_output']], 'out')
    m = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size']], 'mask')
    g = tf.placeholder(tf.float32, [par['n_hidden']], 'gating')
    trial_mask = tf.placeholder_with_default(np.float32(np.ones([par['batch_size'],1])), [par['batch_size'],1], 'trial_mask')
    lid = tf.placeholder_with_default(np.int32(0), [], 'lid')

    return x, y, m, g, trial_mask, lid


def generate_placeholders():

    mask = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_size'], 1])
    x = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_size'], par['n_input']])  # input data
    target = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_size'], par['n_pol']])  # input data
    pred_val = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_size'], par['n_val'], ])
    actual_action = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_size'], par['n_pol']])
    advantage  = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_size'], par['n_val']])
    gating = tf.placeholder(tf.float32, [par['n_hidden']], 'gating')

    return x, target, mask, pred_val, actual_action, advantage, mask, gating


def main(save_fn='testing', gpu_id=None):

    # Update all dependencies in parameters
    update_dependencies()

    # Identify learning method and run accordingly
    if par['training_method'] == 'SL':
        supervised_learning(save_fn, gpu_id)
    elif par['training_method'] == 'RL':
        reinforcement_learning(save_fn, gpu_id)
    else:
        raise Exception('Select a valid learning method.')


if __name__ == '__main__':

    fn = sys.argv[2] if len(sys.argv) > 2 else 'testing.pkl'

    try:
        if len(sys.argv) > 1:
            main(fn, sys.argv[1])
        else:
            main(fn)
    except KeyboardInterrupt:
        print('Quit by KeyboardInterrupt.')
