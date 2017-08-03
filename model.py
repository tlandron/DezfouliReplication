import numpy as np
import matplotlib.pyplot as plt
import random



class History:

    def __init__(self, model):
        self.model = model

        self.Q_history = {sa: [] for sa in self.model.Q.keys()}

        self.V_history = {s: [] for s in self.model.S.keys()}

        # PLcount : int, increments each time the agent chooses the action 'PL'
        self.PLcount = [0]

        # s_0count : int, increments each time the agent visits the state s_0
        self.s_0count = [1]

        # PLfreq : float, computes the probability of pressing th lever (PL)
        self.PLfreq = [0.5]


    def update_Q(self, s_t, a_t):
        self.Q_history[(s_t, a_t)] = (self.model.t, self.model.Q[(s_t, a_t)])

    def update_V(self, s_t, V):
        self.V_history[s_t] = (self.model.t, V)

    def compute_PLfreq(self):
        if i != 0:
            if s_t[i] == 's_0':
                s_0count.append(1)
        if a_t[i] == 'PL':
            PLcount.append(1)

        PLfreq.append(float(sum(PLcount))/float(sum(s_0count)))


class Model:
    """Dezfouli's model"""

    def __init__(self, S,               #action-state graph
                       T     = 10,      #total number of timesteps
                       Ds    = 15,     #dopamine surge
                       alpha = 0.2,    #learning rate
                       lambd = 0.0003, #speed of deviation of kappa_t
                       #C_u   = 6,
                       N     = 2,      #maximum level of deviation
                       epsi  = 0.1,    #epsilon-greedy action selection
                       #k     = 7,     #  (timesteps)
                       sigma = 0.005
                ):

        self.S = S
        self.T = T
        self.Ds = Ds
        self.alpha = alpha
        self.lambd = lambd
        self.N = N
        self.epsi = epsi
        self.sigma = sigma

        self.t = 0 # current timestep

        # list[str] : list of the state really taken
        self.s_t = ['s_0'] + [None] * T

        # r : number, reward r
        self.r = [None] * (T + 1)

        # r_c : number, reward r under cocaine
        self.r_c = [0] + [None] * T

        # Rbar : number, average reward Rbar at time t
        ###Rbar = [r[0]] + [0] * t What is t0 value of Rbar ?
        self.Rbar = [0] * (T + 1)


        # list[str] : list of the action really performed
        self.a_t = [None] * (T + 1)

        # delta : list[float], RPE
        self.delta = [None] * (T + 1)

        #delta_c : list[float], RPE under cocaine
        self.delta_c = [None] * (T + 1)

        # Q : dict{tuple(state, action): value of reward}
        # s, a : str^2, list_action : list[tuple], r : float, next_s : number
        self.Q = {}
        for s, list_action in self.S.items():
            for action in list_action:
                name_action, r_action, next_s = action
                self.Q[(s, name_action)] = 0  # QUESTION: initiate with r_action.value()

        # rho : list[number]
        self.rho = [None] * (T + 1)

        # kappa : list[number]
        self.kappa = [0] + [None] * T


        self.history = History(self)


    def compute_action_and_next_state(self, s):
        """Choose the next action using the epsilon-greedy algorithm, and derive the next state."""

        if len(self.S[s]) >= 2 and random.random() < self.epsi:  # QUESTION: is the len(self.S[s_t]) >= 2 correct?
            exploratory = True
            chosen_action = random.choice(self.S[s])
        else:
            exploratory = False
            value_dict = {}
            for action in self.S[s]:
                name_action, r_action, next_s = action
                value_dict[action] = self.Q[(s, name_action)]

            chosen_action = max(value_dict, key=value_dict.get) # return the name of the action with the maximun state-action value

        self.a_t[self.t]     = chosen_action[0]
        self.r[self.t]       = chosen_action[1]
        self.s_t[self.t + 1] = chosen_action[2]

        return exploratory


    def update(self):
        t = self.t # because it's horrible without it.
        print('step, t =', t)

        exploratory = self.compute_action_and_next_state(self.s_t[t])  # updates, among other things, s_t[t + 1]
        r_t = self.r[t].value()

        #Updating V
        V = max(self.Q[(self.s_t[t], a[0])] for a in self.S[self.s_t[t]])
        self.history.update_V(self.s_t[t], V)

        #eq. 2.8 // eq. 2.9 not written, simplified version of eq 2.8
        self.kappa[t + 1] = (1 - self.lambd) * self.kappa[t] + self.lambd * self.N

        #eq. 2.7
        self.rho[t] = self.Rbar[t] + self.kappa[t]

        #eq. 2.10 ###Under cocaine
        self.delta_c[t] = max((r_t + V - self.Q[(self.s_t[t], self.a_t[t])] + (self.Ds - self.kappa[t])), (self.Ds - self.kappa[t])) - self.Rbar[t]

        #eq. 1.3 : Q(s_t, a_t) <- Q(st, at) + alpha * delta_t
        self.Q[self.s_t[t], self.a_t[t]] = self.Q[self.s_t[t], self.a_t[t]] + self.delta_c[t] * self.alpha
        self.history.update_Q(self.s_t[t], self.a_t[t])

        #eq. 2.3
        self.delta[t] = r_t + V - self.Q[self.s_t[t], self.a_t[t]] - self.rho[t]

        #eq. 2.6
        self.r_c[t] = self.delta_c[t] - V + self.Q[self.s_t[t], self.a_t[t]] + self.rho[t]

        #eq. 2.2
        if exploratory:
            self.Rbar[t + 1] = self.Rbar[t]
        else:
            self.Rbar[t + 1] = (1 - self.sigma) * self.Rbar[t] + self.sigma * self.r_c[t]

        self.t += 1



# print('Q[s_t[i], a_t[i]][i] = ', Q[s_t[i], a_t[i]][i])
# print('r[i] =', r[i], 'Ds =', Ds, 'kappa[i] =', kappa[i], 'Rbar[i] =', Rbar[i], 'rho[i]', rho[i])
#
# for s, list_action in S.items():
#     for action in list_action:
#         [(name_action, (r_action, next_s))] = action.items()
#         print ('Q[', s, name_action,'] = ', Q[(s, name_action)])
#
# print('PLcount =', PLcount)
# print('S_0count =', s_0count)
# print('PLfreq =', PLfreq)



# p = 0
# color = ['r-', 'k-', 'b-', 'g-']
# for s, list_action in S.items():
#     for action in list_action:
#         [(name_action, (r_action, next_s))] = action.items()
#         plt.plot(range(0, t+1), Q[(s, name_action)], color[p])
#         print('Graphs:', (s, name_action),'is in', color[p])
#         p += 1
# plt.savefig('Write the name of the graph')
#
#
# plt.plot(s_0count[:-1], PLfreq)
# plt.savefig('PLfreq1')

""": #assigns the i + 1 value of the other actions of the other states ('other-state, other-actions' value)
for action in list_action:
[(name_action, (r_action, next_s))] = action.items()
Q[(s, name_action)][t + 1] = Q[(s, name_action)][i]"""
