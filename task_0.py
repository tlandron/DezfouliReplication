import model
import rewards

task_0 = {'s_0' : [('PL', rewards.R_c, 's_0')]}
m = model.Model(task_0, T=2000)

for _ in range(2000):
    m.update()
