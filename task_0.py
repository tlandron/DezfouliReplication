import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot  as plt
import model
import rewards

task_0 = {'s_0' : [('PL', rewards.R_c, 's_0')]}
m = model.Model(task_0, T=2000)

for _ in range(2000):
    m.update()

plt.plot(range(0, m.T), [v for t, v in m.history.V_history['s_0']])
plt.savefig('figures/figure_2.pdf')
