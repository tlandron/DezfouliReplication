import model
import rewards

task_1 = {'s_0' : [('F', rewards.R_fr, 's_0'), ('PL', rewards.R_sh, 's_1')],
          's_1' : [('', rewards.R_c, 's_0')]
         }

m = model.Model(task_1, T=2000)

for _ in range(2000):
    m.update()
