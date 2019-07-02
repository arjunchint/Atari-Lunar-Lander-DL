import numpy as np
import gym
import random
# import keras
from keras.models import Sequential, model_from_config, Model,clone_model
from keras.layers import Dense, Activation, Flatten, Lambda, Input, Layer, Dense
from keras.optimizers import Adam
import keras.optimizers as optimizers


import keras.backend 
import tensorflow as tf

import time
import os
import matplotlib.pyplot as plt
from datetime import datetime as dt
from keras.models import load_model


# Set Gym Game to test with
GAME = 'CartPole-v0'
# GAME = 'LunarLander-v2'
EPISODES = 1000
TRIALS = 100

# run on CPU for multiple experiments
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Create env with random seed so to have comparable experiments
env = gym.make(GAME)
np.random.seed(123)
env.seed(123)
action_size = env.action_space.n


# Build 2 hidden layer NN using keras with relu activation funtions, with input layer size of environment and final layer same length as actions in order to decide on actions
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dense(action_size))
model.add(Activation('linear'))
# print(model.summary())


def huber_loss(y_true, y_pred):
	return .5 * keras.backend.square(y_true - y_pred)

# Setup memory structure to remember past sequences of results of env.step 
class memory:
	def __init__(self, mem_length):
		self.mem_length = mem_length
		# Apparently lists are faster than np for this?
		# self.actions = np.zeros(mem_length,dtype=int)
		# self.rewards = np.zeros(mem_length,dtype=float)
		# self.dones = np.zeros(mem_length,dtype=bool)
		self.actions = []
		self.rewards = []
		self.dones = []       
		self.states = []
		self.length=0

	def get_batch(self,batch_size):
		batch_idxs = np.array(random.sample(range(1, self.length - 1), batch_size))+1
		batch_actions = []
		batch_rewards = []
		batch_dones = []       
		batch_states = []
		batch_states1 = []
		for idx in batch_idxs:
			batch_actions.append(self.actions[(idx - 1)%self.mem_length])
			batch_rewards.append(self.rewards[(idx - 1)%self.mem_length])
			batch_dones.append(self.dones[(idx - 1)%self.mem_length])
			batch_states.append([self.states[(idx - 1)%self.mem_length]])
			batch_states1.append([self.states[(idx)%self.mem_length]])
		return batch_actions,batch_rewards,batch_dones,batch_states,batch_states1

	def add(self, state, action, reward, done):
		if self.length <  self.mem_length:
			self.states.append(state)
			self.actions.append(action)
			self.rewards.append(reward)
			self.dones.append(done)            
		else:
			self.states[self.length  % self.mem_length] = state
			self.actions[self.length  % self.mem_length] = action
			self.rewards[self.length  % self.mem_length] = reward
			self.dones[self.length  % self.mem_length] = done
		self.length += 1
	def get_steps(self):
		return self.length

class DQN:
	def __init__(self, state_size, action_size, model,learning_rate=0.002,lr_decay = 1e-05,epsilon_decay = 0.975,epsilon_min = 0.1,epsilon = 1.0,gamma = 0.99,mem_length = 50000, tau = .01):
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_min = epsilon_min
		self.state_size = state_size
		self.action_size = action_size
		self.memory = memory(mem_length)		
		self.epsilon_decay = epsilon_decay
		self.tau = tau

		self.model = model
		self.target = clone_model(model)
		self.target.set_weights(self.model.get_weights())
		self.target.compile(loss=huber_loss, optimizer=Adam(lr=learning_rate, decay = lr_decay))
		self.model.compile(loss=huber_loss, optimizer=Adam(lr=learning_rate, decay = lr_decay))

	def remember(self, state, action, reward, done):
		self.memory.add(state, action, reward, done)

	def act(self, state):
		if np.random.uniform() < self.epsilon:
			return np.random.random_integers(0, self.action_size-1)
		state = np.reshape(state, [1, self.state_size])
		act_values = self.model.predict(np.array([state]))
		return np.argmax(act_values[0]) 

	def replay(self, batch_size):
		batch_actions,batch_rewards,batch_dones,batch_states,batch_states1 = self.memory.get_batch(batch_size)
		
		batch_states1 = np.array(batch_states1)
		batch_rewards = np.array(batch_rewards)
		batch_states1 = np.array(batch_states1)
		batch_state0 = np.array(batch_states)

		# flip targets to compute rewards!
		batch_dones = ~np.array(batch_dones)

		# Vectorize calculations for speed
		target_q_values = self.target.predict_on_batch(batch_states1)
		q_batch = np.max(target_q_values, axis=1).flatten()

		batch_discounted_reward = self.gamma * q_batch
		batch_discounted_reward *= batch_dones
		batch_discounted_reward = batch_rewards + batch_discounted_reward

		target_f = self.model.predict_on_batch(batch_state0)

		for idx, (target, rewards, action) in enumerate(zip(np.zeros((batch_size, self.action_size)), batch_discounted_reward, batch_actions)):
			target_f[idx][action] = rewards

		self.model.train_on_batch(batch_state0, target_f)

		updated_weights=[]
		for tw, sw in zip(self.target.get_weights(), self.model.get_weights()):
			updated_weights.append(self.tau * sw + (1. - self.tau) * tw)
		self.target.set_weights(updated_weights)


if __name__ == "__main__":
	timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
	# relative_folder=GAME+ '_' + timestamp
	# os.mkdir(relative_folder)
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n
	agent = DQN(state_size, action_size,model=model)

	batch_size = 32
	start_time=time.time()
	episodic_rewards = []
	start_gap= 50
	try:
		for episode in range(EPISODES):
			state = env.reset() 
			totes_reward = 0
			done = False      
			# Only run episode for 1000 steps 
			for step in range(1000):
				action = agent.act(state)
				next_state, reward, done, info = env.step(action)
				totes_reward += reward
				agent.remember(state, action, reward, done)
				state = next_state
				if done:
					elapsed_time=time.time()-start_time
					print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),"Trial: ",episode,"/", EPISODES, "Episodic Reward: ",totes_reward,"eps:", agent.epsilon)
					break
				if agent.memory.get_steps() > start_gap:
					agent.replay(batch_size)
			episodic_rewards.append(totes_reward)
			if agent.epsilon > agent.epsilon_min and episode >50:
				agent.epsilon *= agent.epsilon_decay	
	except KeyboardInterrupt:
		pass	
	print('MEAN TRAIN REWARDS: ',sum(episodic_rewards) / float(len(episodic_rewards)))
	
	trial_rewards=[]
	agent.epsilon = -1
	for trial in range(TRIALS):
		state = env.reset()
		state = np.reshape(state, [1, state_size])
		totes_reward = 0
		done = False
		while not done:
			action = agent.act(state)
			next_state, reward, done, _ = env.step(action)
			totes_reward += reward
			agent.remember(state, action, reward, done)
			state = next_state
			if done:
				elapsed_time=time.time()-start_time
				print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),"Trial: ",trial,"/", TRIALS, "Episodic Reward: ",totes_reward,"eps:", agent.epsilon)
				break
		trial_rewards.append(totes_reward)
	print('MEAN TEST REWARDS: ',sum(trial_rewards) / float(len(trial_rewards)))
	# print('TESTING REWARD STEP')
	# model.save_weights('base_weights.h5')

	# Plot results
	plt.figure(1)
	plt.plot(range(0, len(episodic_rewards)), episodic_rewards)
	plt.xlabel('Episode')
	plt.ylabel('Episodic Rewards')

	plt.figure(2)
	plt.plot(range(0, len(trial_rewards)), trial_rewards)
	plt.xlabel('Trial')
	plt.ylabel('Trial Rewards')
	plt.show()
