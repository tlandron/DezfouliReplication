import numpy as np
import matplotlib.pyplot as plt
import random

# t : int, time/number of steps
t = 2000

# i : int, incrementer
i = 0


'''Simulation Parameters' Values'''

# sigma, Ds, alpha, mu, C_u, epsi, k : number^7

sigma = 0.005
mu_N = 5
sigma_N = 0.02
mu_fr = -2
sigma_fr = 0.02
mu_sh = -200
sigma_sh = 0.02
mu_c = 2
sigma_c = 0.02
mu_s = 1
sigma_s = 0.02
mu_l = 15
sigma_l = 0.02

Ds = 15
alpha = 0.2
lambd = 0.0003
C_u = 6
N = 2
epsi = 0.1
k = 7#ts


'''Definition of Rbar values'''

rndGenerator = np.random.RandomState(5)
# R_ : float
R_N = rndGenerator.normal(mu_N, sigma_N**2) #reward of a natural reinforcer
R_sh = rndGenerator.normal(mu_sh, sigma_sh**2) #punishment of the shock
R_fr = rndGenerator.normal(mu_fr, sigma_fr**2) #effect of freezing on the reduction of the shock punishment
R_c = rndGenerator.normal(mu_c, sigma_c**2) #the cocaine reward
R_s = rndGenerator.normal(mu_s, sigma_s**2)
R_l = rndGenerator.normal(mu_l, sigma_l**2)


'''Initialisation of variables '''

# r : number, reward r
r = [R_c] + [None] * t

# r_c : number, reward r under cocaine
r_c = [0] + [None] * t

# Rbar : number, average reward Rbar at time t
###Rbar = [r[0]] + [0] * t What is t0 value of Rbar ?
Rbar = [0] * (t + 1)

# S : dict{'state s':list[dict{action a : tuple(reward r, next_s)}
S = {
	's_0' : [{'F' : (R_fr, 's_0')}, {'PL' : (R_sh, 's_1')}],
	's_1' : [{'' : (R_c, 's_0')}]
	}

#To make it learn the reward value (trial 1)
'''
S = {
	's_0' : [{'PL' : (R_c, 's_0')}]
	}
'''

# list[str] : list of the state really taken
s_t = ['s_0'] + [None] * t

# list[str] : list of the action really performed
a_t = [None] * (t + 1)

# delta : list[float], RPE
delta = [None] * (t + 1)

#delta_c : list[float], RPE under cocaine
delta_c = [None] * (t + 1)

# Q : dict{tuple(state, action): value of reward}
# s, a : str^2, list_action : list[tuple], r : float, next_s : number
Q = dict()
for s, list_action in S.items():
	for action in list_action: ###action = dict()
		[(name_action, (r_action, next_s))] = action.items()
		Q[(s, name_action)] = [r_action] + [None]*t
print('Q=', Q)

# V : list[number] value of state s at time t+1
SubV = list()
for s, list_action in S.items():
	if s == 's_0':
		for action in list_action:
			[(name_action, (r_action, next_s))] = action.items()
			SubV.append(Q[s, name_action][0])
V = [max(SubV)] + [None] * t

# rho : list[number]
rho = [None] * (t + 1)

# kappa : list[number]
kappa = [0] + [None] * t

# PLcount : int, increments each time the agent chooses the action 'PL'
PLcount = [0]

# s_0count : int, increments each time the agent visits the state s_0
s_0count = [1]

# PLfreq : float, computes the probability of pressing th lever (PL)
PLfreq = [0.5]


'''Functions'''

"""
"""


def attribute_values(Arg_name_action, Arg_r_action, Arg_next_s):
	"""
	Attributes the values action, r_action and next_s after the action has been chosen
	"""
	a_t[i] = Arg_name_action
	r[i] = Arg_r_action
	s_t[i + 1] = Arg_next_s



def ifnot_statecondition(current_state, wanted_state, Q, list_action):
	"""
	Enables the continuty of the each state-acion value when it is not selected by the agent and consequently no updated
	"""
	if not current_state == wanted_state: #assigns the i + 1 value of the other actions of the other states ('other-state, other-actions' value)
		for action in list_action:
			[(name_action, (r_action, next_s))] = action.items()
			Q[(s, name_action)][i + 1] = Q[(s, name_action)][i]

def chosen_action(Arg_list_action, Arg_name_action):
	"""
	Return the chosen_action in the case of exploratory action in state s_0
	"""
	if non-explo == False:
		chosen_action = random.choice(Arg_list_action)
		action_condition = Arg_name_action == chosen_action


def read_and_do(todo = attribute_values, state_condition = 's == s_t[i]', S_type_dict = S, Q = Q, Arg_temp_action = None, action_condition = 'True', non_explo = True):
	""" """
	for s, list_action in S_type_dict.items():
		if eval(state_condition):
			if non_explo == False:
				chosen_action = random.choice(list_action)
				[(name_chosen_action, (r_chosen_action, next_s_chosen_action))] = chosen_action.items()
				action_condition = 'name_action == name_chosen_action'

			for action in list_action:
				[(name_action, (r_action, next_s))] = action.items()

				if eval(action_condition): #(action == chosen_action:
					todo(name_action, r_action, next_s)

				if not eval(action_condition): #assigns the i + 1 value of the other actions of the same state ('same-state, other-actions' value)
					Q[(s, name_action)][i + 1] = Q[(s, name_action)][i]

		if not eval(state_condition): #assigns the i + 1 value of the other actions of the other states ('other-state, other-actions' value)
			for action in list_action:
				[(name_action, (r_action, next_s))] = action.items()
				Q[(s, name_action)][i + 1] = Q[(s, name_action)][i]





'''Equations'''

for i in range(0, t):
	print('step, i =', i)
	###Setting a new value to the cocaine reward (following the normal distribution previously defined)
	# R_ : float
	R_N = rndGenerator.normal(mu_N, sigma_N**2) #reward of a natural reinforcer
	R_sh = rndGenerator.normal(mu_sh, sigma_sh**2) #punishment of the shock
	R_fr = rndGenerator.normal(mu_fr, sigma_fr**2) #effect of freezing on the reduction of the shock punishment
	R_c = rndGenerator.normal(mu_c, sigma_c**2) #the cocaine reward
	R_s = rndGenerator.normal(mu_s, sigma_s**2)
	R_l = rndGenerator.normal(mu_l, sigma_l**2)

	non_explo = True


	if s_t[i] == 's_0':		###epsilon-greedy action selection policy
		# temp : float, is part of the epsilon-greedy action selectione (temps = the value that epsilon is compared to)
		temp = random.random()

		if temp > epsi: #choose the action with the highest estimated value (non-exploratory action)

			# non-explo : bool, enables to tell if Rbar has to be updated (only in non-exploratory action)
			non_explo = True

			# temp_dict : dict(), stores the action subdictionaries of the current states for computation a the maximun state-action value
			temp_dict = dict()

			for s, list_action in S.items():
				if s == s_t[i]:
					for action in list_action:
						[(name_action, (r_action, next_s))] = action.items()
						temp_dict[name_action] = Q[(s_t[i], name_action)][i]

			temp_action = max(temp_dict, key = temp_dict.get) # return the name of the action with the maximun state-action value

			read_and_do(state_condition = 's == s_t[i]', Arg_temp_action = temp_action, action_condition = 'name_action == temp_action')


		else: #choose the action randomly (exploratory action)

			non_explo = False #(exploratory action)

			read_and_do(non_explo = False)

	else: # s_t[i] == 's_1'
		read_and_do()


	#computation of PL probability
	#if PLcount[i] is int and s_0count[i] is int:

	if i != 0:
		if s_t[i] == 's_0':
			s_0count.append(1)
	if a_t[i] == 'PL':
		PLcount.append(1)

	PLfreq.append(float(sum(PLcount))/float(sum(s_0count)))


	#Updating V
	SubV = list()
	for s, list_action in S.items():
		if s == s_t[i]:
			for action in list_action:
				[(name_action, (r_action, next_s))] = action.items()
				SubV.append(Q[(s, name_action)][i])
	V[i] = max(SubV)


	#eq. 2.8 // eq. 2.9 not written, simplified version of eq 2.8
	kappa[i + 1] = (1 - lambd) * kappa[i] + lambd * N

	#eq. 2.7
	rho[i] = Rbar[i] + kappa[i]

	#eq. 2.10 ###Under cocaine
	delta_c[i] = max((r[i] + V[i] - Q[(s_t[i], a_t[i])][i] + (Ds - kappa[i])), (Ds - kappa[i])) - Rbar[i]

	#eq. 1.3 : Q(s_t, a_t) <- Q(st, at) + alpha * delta_t
	Q[s_t[i], a_t[i]][i + 1] = Q[s_t[i], a_t[i]][i] + delta_c[i] * alpha

	#eq. 2.3
	delta[i] = r[i] + V[i] - Q[s_t[i], a_t[i]][i] - rho[i]

	#eq. 2.6
	r_c[i] = delta_c[i] - V[i] + Q[s_t[i], a_t[i]][i] + rho[i]

	#eq. 2.2
	if non_explo:
		Rbar[i + 1] = (1 - sigma) * Rbar[i] + sigma * r_c[i]
	else:
		Rbar[i + 1] = Rbar[i]




print('Q[s_t[i], a_t[i]][i] = ', Q[s_t[i], a_t[i]][i])
print('r[i] =', r[i], 'Ds =', Ds, 'kappa[i] =', kappa[i], 'Rbar[i] =', Rbar[i], 'rho[i]', rho[i])

for s, list_action in S.items():
	for action in list_action:
		[(name_action, (r_action, next_s))] = action.items()
		print ('Q[', s, name_action,'] = ', Q[(s, name_action)])

print('PLcount =', PLcount)
print('S_0count =', s_0count)
print('PLfreq =', PLfreq)


'''Graphs'''

'''p = 0
color = ['r-', 'k-', 'b-', 'g-']
for s, list_action in S.items():
	for action in list_action:
		[(name_action, (r_action, next_s))] = action.items()
		plt.plot(range(0, t+1), Q[(s, name_action)], color[p])
		print('Graphs:', (s, name_action),'is in', color[p])
		p += 1
plt.savefig('Write the name of the graph')


plt.plot(s_0count[:-1], PLfreq)
plt.savefig('PLfreq1')'''

""": #assigns the i + 1 value of the other actions of the other states ('other-state, other-actions' value)
for action in list_action:
[(name_action, (r_action, next_s))] = action.items()
Q[(s, name_action)][i + 1] = Q[(s, name_action)][i]"""
