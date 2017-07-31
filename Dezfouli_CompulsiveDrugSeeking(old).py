from dana import *
import numpy as np
import matplotlib.pyplot as plt
import random

'''Instructions'''
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


'''Definition of Rbarvalues'''

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
###Rbar = [r[0]] + [0] * t What is a starting-value of Rbar ?
Rbar = [0] * (t + 1)

# S : dict{'state s':list[dict{action a : tuple(reward r, next_s)}
S = {
	's_0' : [{'F' : (R_fr, 's_0')}, {'PL' : (R_sh, 's_1')}],
	's_1' : [{'' : (R_c, 's_0')}]
	}

# list[str] : list of the state really taken
s_t = ['s_0'] + [None] * t

# delta : list[float], RPE
delta = [None] * (t + 1)

#delta_c : list[float], RPE under cocaine
delta_c = [None] * (t + 1)

# Q : dict{tuple(state, action): value of reward}
# s, a : str^2, list_action : list[tuple], r : float, next_s : number
Q = dict()
for s, list_action in S.items():
	for action in list_action: ###action = dict()
		r_action, next_s = action.values()[0]
		Q[(s, action.keys()[0])] = [r_action] + [None]*t
print('Q=', Q)

# V : list[number] value of state s at time t+1
SubV = list()
for s, list_action in S.items():
	if s == 's_0':
		for action in list_action:
			r_action, next_s = action.values()[0]
			SubV.append(Q[s, action.keys()[0]][0])
V = [max(SubV)] + [None] * t

# rho : list[number]
rho = [None] * (t + 1)

# kappa : list[number]
kappa = [0] + [None] * t



'''Equations'''

for i in range(0, t):

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

		temp = random.random()

		if temp > epsi: #choose the action with the highest estimated value (non-exploratory action)

			non_explo = True
			temp_dict = dict()


			for s, list_action in S.items():
				if s == s_t[i]:
					for action in list_action:
						r_action, next_s = action.values()[0]
						temp_dict[action.keys()[0]] = Q[s_t[i], action.keys()[0]][i]


			temp_action = max(temp_dict, key = temp_dict.get)


			for s, list_action in S.items():
				if s == s_t[i]:
					for action in list_action:
						r_action, next_s = action.values()[0]
						if action.keys()[0] == temp_action:
							r[i] = r_action
							s_t[i + 1] = next_s

							S_ge = s
							A_ge = action.keys()[0]


						if not action.keys()[0] == temp_action:
							Q[(s, action.keys()[0])][i + 1] = Q[(s, action.keys()[0])][i]


				if not s == s_t[i]:
					for action in list_action:
						Q[(s, action.keys()[0])][i + 1] = Q[(s, action.keys()[0])][i]


		else: #choose the action randomly (exploratory action)

			non_explo = False

			for s, list_action in S.items():
				if s == s_t[i]:
					chosen_action = random.choice(list_action)
					r_action, next_s = chosen_action.values()[0]

					r[i] = r_action
					s_t[i + 1] = next_s

					S_ge = s
					A_ge = chosen_action.keys()[0]


					for action in list_action:
						if not action == chosen_action:
							Q[(s, action.keys()[0])][i + 1] = Q[(s, action.keys()[0])][i]

				if not s == s_t[i]:
					for action in list_action:
						Q[(s, action.keys()[0])][i + 1] = Q[(s, action.keys()[0])][i]


	else: # s_t[i] == 's_1'
		for s, list_action in S.items():
			if s == s_t[i]:
				for action in list_action:
					r_action, next_s = action.values()[0]

					r[i] = r_action
					s_t[i + 1] = next_s

					S_ge = s
					A_ge = action.keys()[0]


			if not s == s_t[i]:
				for action in list_action:
					Q[(s, action.keys()[0])][i + 1] = Q[(s, action.keys()[0])][i]

	#Updating V
	SubV = list()
	for s, list_action in S.items():
		if s == s_t[i]:
			for action in list_action:
				r_action, next_s = action.values()[0]

				SubV.append(Q[s, action.keys()[0]][i])

	V[i] = max(SubV)

	###change action.values()[0] with name_action, r_action, next_s = action.items()

	#eq. 2.8 // eq. 2.9 not written, simplified version of eq 2.8
	kappa[i + 1] = (1 - lambd) * kappa[i] + lambd * N

	#eq. 2.7
	rho[i] = Rbar[i] + kappa[i]

	if under_cocaine:
		#eq. 2.10 ###Under cocaine
		delta_c[i] = max((r[i] + V[i] - Q[S_ge, A_ge][i] + (Ds - kappa[i])), (Ds - kappa[i])) - Rbar[i]

	else :
		#eq. 2.3
		delta[i] = r[i] + V[i] - Q[S_ge, A_ge][i] - rho[i]

	#eq. 1.3 : Q(s_t, a_t) <- Q(st, at) + alpha * delta_t
	Q[S_ge, A_ge][i + 1] = Q[S_ge, A_ge][i] + delta_c[i] * alpha

	#eq. 2.3
	#delta[i] = r[i] + V[i] - Q[S_ge, A_ge][i] - rho[i]

	#eq. 2.6
	r_c[i] = delta_c[i] - V[i] + Q[S_ge, A_ge][i] + rho[i]

	#eq. 2.2
	if non_explo:
		Rbar[i + 1] = (1 - sigma) * Rbar[i] + sigma * r_c[i]
	else:
		Rbar[i + 1] = Rbar[i]


print('Q[S_ge, A_ge][i] = ', Q[S_ge, A_ge][i])
print('r[i] =', r[i], 'Ds =', Ds, 'kappa[i] =', kappa[i], 'Rbar[i] =', Rbar[i], 'rho[i]', rho[i])

for s, list_action in S.items():
	for action in list_action:
		print ('Q[', s, action.keys()[0],'] = ', Q[s, action.keys()[0]])

'''Graphs'''

p = 0
ax = list()
color = ['r-', 'k-', 'b-', 'g-']
for s, list_action in S.items():
	for action in list_action:
		plt.plot(range(0, t+1), Q[s, action.keys()[0]], color[p])
		print('Graphs:', (s, action.keys()[0],'is in', color[p]))
		p += 1







plt.savefig('CompDrSeek1')

'''Draft'''

'''r = [0] * 2000
Rbar= [0] * (len(r) + 1)
S = S_0 = [('PL', R, 1)]
V = [0 for s in S]
Q = {(s, a) : 0 for a, r, next_s in s for s in S}

R_ = 0 #???'''

'''#ax=plt.subplot(0, 1000, 2000)
plt.plot(r, range(0, t))
#plt.plot(r)
plt.ylim(0, 500)
plt.ylabel('Value')
plt.xlabel('Number of received rewards drug intake value')

ax=plt.subplot(R, range(0, t))
plt.plot(R[t])
ax.set_ylim(0, 100, 200, 300, 400, 500)
ax.set_ylabel("R")

ax=plt.subplot(r, range5,3,10)
plt.plot(NAc_record[trial_num-1])
ax.set_ylim(-0.3,2)
ax.set_ylabel("VS")'''

'''# Q : dict{tuple(state, action): value of reward}
# s, a : str^2, list_action : list[tuple], r : float, next_s : number
for s, list_action in S.items():
	for action in list_action:
		a, r, next_s = action
		Q[(s, a)] = r

list_action = [] ###tries of defining Q with one line
Q_ = {(s, a) : r for (a, r, next_s) in list_action for (s, (list_action)) in S.items()}
print()
print(Q_)
Q = {(s, a): r for (a, r, next_s) in s for s in S}'''


'''
# Q : dict{[tuple(state, action): value of reward}
Q = dict()
for s, list_action in S.items():
	for action in list_action:
		a, r_of_a = action
		Q[(s, a)] = r_of_a
'''
