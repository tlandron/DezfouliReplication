from dana import *
import numpy as np
import matplotlib.pyplot as plt

'''
Simulation Parameters' Values
'''
# sigma, Ds, alpha, mu, C_u, epsi, k : number^7

sigma = 0.005
Ds = 15
alpha = 0.2
mu_N = 5
sigma_N = 0.02
mu_fr = -2
sigma_fr = 0.02
mu_sh = -200
sigma_sh = 0.02
mu_c = 2
sigma_c = 0.02
C_u = 6
lambd = 0.0003
N = 2
mu_s = 1
sigma_s = 0.02
mu_l = 15
sigma_l = 0.02
epsi = 0.1
k = 7#ts


'''
Definition of R values
'''
rndGenerator = np.random.RandomState(1)
# R_ : float
R_N = rndGenerator.normal(mu_N, sigma_N**2) #reward of a natural reinforcer
R_sh = rndGenerator.normal(mu_sh, sigma_sh**2) #punishment of the shock
R_fr = rndGenerator.normal(mu_fr, sigma_fr**2) #effect of freezing on the reduction of the shock punishment
R_c = rndGenerator.normal(mu_c, sigma_c**2) #the cocaine reward
R_s = rndGenerator.normal(mu_s, sigma_s**2)
R_l = rndGenerator.normal(mu_l, sigma_l**2)


'''
Initialisation of variables 
'''

'''r = [0] * 2000
R = [0] * (len(r) + 1)
S = S_0 = [('PL', R, 1)]
V = [0 for s in S]
Q = {(s, a) : 0 for a, r, next_s in s for s in S}'''

R_ = 0 #???

# r : list[number] reward r at time t
r = [0] * 10

# R : list[number] average reward R a time t
R = [0] * (len(r) + 1)

# S : dict{'state s':list[tuple(action a, reward r, next_s} state s of the agent  at time t
# tuple(action a, reward r, next_state) = tuple(str, float, int)
S={	's_0' : [('F', R_fr, 0), ('PL', R_sh, 1)],
	's_1' : [('', R_, 0)]						}

# V : list[number] value of state s at time t
V = [0 for s in S]

# Q :  dict{tuple(state, action): value} value of action a in state s at time t
# s, a : str^2, list_action : list[tuple], r : float, next_s : number
Q = dict()
for s, list_action in S.items():
	for action in list_action:
		a, r, next_s = action
		Q[(s, a)] = r		
print (Q)

'''list_action = [] ###tries of defining Q with one line
Q_ = {(s, a) : r for (a, r, next_s) in list_action for (s, (list_action)) in S.items()}
print()
print(Q_)
Q = {(s, a): r for (a, r, next_s) in s for s in S}'''






'''
Equations
'''
