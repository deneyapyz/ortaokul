import gym
import numpy as np 
ortam = gym.make('FrozenLake-v1', is_slippery=False)
Q = np.zeros([ortam.observation_space.n,ortam.action_space.n])
eta = .628
gma = .9
dongu = 5000
odul_list = []
for i in range(dongu):
    s = ortam.reset()
    odul_tumu = 0
    d = False
    j = 0
    while j < 99:
        ortam.render()
        j+=1
        a = np.argmax(Q[s,:] + np.random.randn(1,ortam.action_space.n)*(1./(i+1)))
        s1,o,d,_ = ortam.step(a)
        Q[s,a] = Q[s,a] + eta*(o + gma*np.max(Q[s1,:]) - Q[s,a])
        odul_tumu += o
        s = s1
        if d == True:
            break
    odul_list.append(odul_tumu)
    ortam.render()
print("Elde edilen Q Tablosu:")
print(Q)
print("Ortalama ödül değeri:" + str(sum(odul_list)/dongu)) 
