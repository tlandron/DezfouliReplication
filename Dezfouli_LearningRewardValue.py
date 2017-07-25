from dana import *
import numpy as np
import matplotlib.pyplot as plt


###IMPORTANT : the fisrt version is coded with 10 steps (t from 0 to 9) !
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


'''Definition of R values'''

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
r = [R_c] + [0] * t

# r_c : number, reward r under cocaine
r_c = [0] * (t + 1)

# R : number, average reward R a time t
#R = [r[0]] + [0] * t
R = [0] * (t + 1)

# S : dict{'state s':list[tuple(action a, reward r, next_s}, state s of the agent  at time t
# tuple(action a, reward r of the action a) = tuple(str, float)
S={'s_0' : [('PL', R_c)]}

# delta : list[float], RPE
delta = [0] * (t + 1)

#delta_c : list[float], RPE under cocaine
delta_c = [0] * (t + 1)

# Q : list[number] 	### a simple way to compute Q in the reward value learning
					### WON'T WORK FOR THE OTHERS !!!
Q = [r[0]] + [0] * t

# V : list[number] value of state s at time t+1
V = [Q[0] for s in S] + [0] * t
#V = [R_c for s in S] + [0] * t
###ATTENTION, t + 1 !
###In the 'learning the reward value' case : V_s(t+1) = R_c

# rho : list[number]
rho = [0] * (t + 1)

# kappa : list[number]
kappa = [0] * (t + 1)



'''Equations'''

for i in range(0, t):
	#print('step', i)
	#print(rndGenerator.randint(5))
	#print(len(r), len(R), len(V))

	###Setting a new value to the cocaine reward (following the normal distribution previously defined)
	R_c = rndGenerator.normal(mu_c, sigma_c**2)
	r[i+1] = R_c

	R_N = rndGenerator.normal(mu_N, sigma_N**2)
	#r[i+1] = R_N

	#eq. 2.1
	#sum_rR += (r[i] - R[i])
	#Q[i] = sum_rR

	###In this case [learning the reward value], there are only one state (S0) and one action ('PL')
	V[i+1] = Q[i]


	#eq. 2.8 // eq. 2.9 not written, simplified version of eq 2.8
	kappa[i + 1] = (1 - lambd) * kappa[i] + lambd * N


	#eq. 2.7 ###Useless ?
	rho[i] = R[i] + kappa[i]


	#eq. 2.10 ###Under cocaine
	delta_c[i] = max((r[i] + V[i + 1] - Q[i] + (Ds - kappa[i])), (Ds - kappa[i])) - R[i]
	print('r[i] =', r[i], 'Q[i] =', Q[i], 'Ds =', Ds, 'kappa[i] =', kappa[i], 'R[i] =', R[i], 'rho[i]', rho[i])
	#print('V[i+1] - Q[i]', V[i+1] - Q[i])

	print('Q[i] =', Q[i])
	#eq. 1.3 : Q(s_t, a_t) <- Q(st, at) + alpha * delta_t
	Q[i + 1] = Q[i] + delta_c[i] * alpha
	#print('delta_c =', delta_c[i], 'alpha =', alpha)

	#eq. 2.3
	delta[i] = r[i] + V[i + 1] - Q[i] - rho[i]

	#eq. 2.4 not written
	#eq. 2.5 not written

	#eq. 2.6
	r_c[i] = delta_c[i] - V[i + 1] + Q[i] + rho[i]
	### Is this eq. not made to assess the results ?

	#eq. 2.2
	R[i + 1] = (1 - sigma) * R[i] + sigma * r_c[i]









'''Graphs'''
plt.plot(range(0, t+1), Q)
plt.axis([0, (len(Q) + 100), 0, (max(Q) + 2)])


plt.show()
plt.savefig('Try6withRhoBis')


'''Draft'''

'''r = [0] * 2000
R = [0] * (len(r) + 1)
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


'''def Q_learning(#Q ou S ?):
	"""
	dict -> dict
	compute the action a in state s reward
	"""
'''


'''
# Q : dict{[tuple(state, action): value of reward}
Q = dict()
for s, list_action in S.items():
	for action in list_action:
		a, r_of_a = action
		Q[(s, a)] = r_of_a
'''
