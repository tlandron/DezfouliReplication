import model
import rewards

task_1 = {'s_0'      : [('C2', 0, 's_1'), ('C1', 0, 's_1prime')],
          's_1'      : [('', 0, 's_2'), ('', 0, 's_3'), ('', 0, 's_4'), ('', 0, 's_5'), ('', 0, 's_6'), ('', 0, 's_7'), ('', rewards.R_l, 's_0')] # for k = 7
          's_1prime' : [('', rewards.R_s, 's_0')]
         }
for i in range(k):
    task_1['s_{}'.format(i)] =

m = model.Model(task_2, T=2000)

for _ in range(2000):
    m.update()
