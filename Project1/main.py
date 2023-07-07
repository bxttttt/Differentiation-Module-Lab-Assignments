import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

import numpy as np
from scipy.optimize import curve_fit

def logistic_increase_function(t ,K ,P0 ,r):

    t0 =1
    exp_value=np. exp(r*(t- t0) )
    return (K*exp_value*P0) / (K+ ( exp_value-1)* P0)


t=[1,2,4,5,8,10,11,17,20,22,23,25,26,28,29,32,34,37,38,40]
t =np . array(t)
P=[5.29,5.71,6.63,7.09,8.61,9.51,10.21,12.97,14.31,15.17,15.45,16.15,16.61,16.91,17.19,17.82,18.18,18.62,18.74,18.95]
P=np . array(P)

# 用最小二乘法估计拟合
popt, pcov = curve_fit(logistic_increase_function, t, P)


# 获取popt里面是拟合系数
print("K:capacity  P0:initial_value   r:increase_rate   t:time")
print(popt)

# 拟合后预测的P值
P_predict = logistic_increase_function(t,popt[0],popt[1],popt[2])

# 未来预测
future=[1,2,4,5,8,10,11,17,20,22,23,25,26,28,29,32,34,37,38,40,42,44,46,48,50,52 ]
future=np.array(future)
future_predict=logistic_increase_function(future,popt[0],popt[1],popt[2])

print("第五十二周销售额：")
print(future_predict[-1])
# 绘图
plot1 = plt.plot(t, P, 's',label="exact")
plot2 = plt.plot(t, P_predict, 'r',label='predict')
plt.xlabel('周次')
plt.ylabel('周销售额')

plt.legend(loc=0)  #   定legend的位置右下角

print(logistic_increase_function(np.array(28),popt[0],popt[1],popt[2]))
print(logistic_increase_function(np.array(29),popt[0],popt[1],popt[2]))
plt.show()