import numpy as np


class Reward:
    """A class to compute stochastic rewards"""

    def __init__(self, mu, sigma, seed=1):
        self.mu = mu
        self.sigma = sigma
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    def value(self):
        """Return the a new value of a reward each time it's called"""
        return self.rng.normal(self.mu, self.sigma**2)



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


R_N  = Reward(mu_N,  sigma_N,  seed=0) #reward of a natural reinforcer
R_sh = Reward(mu_sh, sigma_sh, seed=0) #punishment of the shock
R_fr = Reward(mu_fr, sigma_fr, seed=0) #effect of freezing on the reduction of the shock punishment
R_c  = Reward(mu_c,  sigma_c,  seed=0) #the cocaine reward
R_s  = Reward(mu_s,  sigma_s,  seed=0)
R_l  = Reward(mu_l,  sigma_l,  seed=0)
