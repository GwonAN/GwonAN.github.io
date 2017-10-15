#Next homework
#1.x: -100~100 range : increase data that has to 1000 10000 ... 10 0000
#2.test with noise X 2  value 
#3. initial value test and many trial 
#4. learning rate test and trial


import matplotlib.pyplot as plt
import numpy as np

x= [float(x/100.0) for x in range(-100,100,1)]
a=1
b=2
noise = np.random.normal()
y = [a*xi+b+noise for xi in x]
print(y)

def grad_hw(x,y,a,b):
    l=0; ga=0; gb=0;
    for k in range(len(x)):
        mk=x[k]*a+b
        ek=y[k]-mk
        l+= ek**2
        ga+=2*ek*(-x[k]) #gradient result
        gb+=2*ek*(-1)    #gradient result
    return l,ga,gb

#fitting
a=0; b=0;loss=[];lr=0.0001; #initial value
for iter in range(100):
    l,ga,gb = grad_hw(x,y,a,b)
    loss.append(l)

    print('iter %d : l=%e a= %e b=%e' %(iter,l,a,b))
    #update a,b
    a=a-ga*(lr) #learning rate (lr) =0.001
    b=b-gb*(lr) #learning rate (lr) =0.001
print(x)
plt.plot(loss)
plt.show()
