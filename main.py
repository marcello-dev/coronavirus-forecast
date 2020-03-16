import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
#import matplotlib

#matplotlib.use('nbagg')
#%matplotlib inline


print('hello')

url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
df = pd.read_csv(url)

print(df)

df = df.loc[:, ['data', 'totale_casi']]

FMT = '%Y-%m-%d %H:%M:%S'
date = df['data']
df['data'] = date.map(lambda x: (datetime.strptime(x, FMT) - datetime.strptime("2020-01-01 00:00:00", FMT)).days)

print('data')
print(df)


def logistic_model(x, a, b, c):
    return c/(1+np.exp(-(x-b)/a))


x = list(df.iloc[:, 0])
y = list(df.iloc[:, 1])
fit = curve_fit(logistic_model, x, y, p0=[2, 100, 20000])

print('covariance matrix')
print(fit)
print()
a = fit[0][0]
b = fit[0][1]
c = fit[0][2]

print('a=', a)
print('b=', b)
print('c=', c)

errors = [np.sqrt(fit[1][i][i]) for i in [0, 1, 2]]

print(errors)

sol = int(fsolve(lambda x : logistic_model(x, a, b, c) - int(c),b))

print('solution:', sol)


def exponential_model(x, a, b, c):
    return a*np.exp(b*(x-c))


exp_fit = curve_fit(exponential_model, x, y, p0=[1, 1, 1])

print('exponential function')
print(exp_fit)
print()
exp_a = exp_fit[0][0]
exp_b = exp_fit[0][1]
exp_c = exp_fit[0][2]

print('a=', exp_a)
print('b=', exp_b)
print('c=', exp_c)

print('plot')

pred_x = list(range(max(x), sol))
print('pred_x=', pred_x)

plt.rcParams['figure.figsize'] = [7, 7]
plt.rc('font', size=14)
# Real data
plt.scatter(x,y,label="Real data",color="red")
# Predicted logistic curve
plt.plot(x+pred_x, [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in x+pred_x], label="Logistic model" )
# Predicted exponential curve
plt.plot(x+pred_x, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2]) for i in x+pred_x], label="Exponential model" )
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of infected people")
plt.ylim((min(y)*0.9,c*1.1))

plt.savefig("mygraph.png")
plt.show()

y_pred_logistic = [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in x]
y_pred_exp =  [exponential_model(i,exp_fit[0][0], exp_fit[0][1], exp_fit[0][2]) for i in x]
print('error logistic=', mean_squared_error(y,y_pred_logistic))
print('error exponential', mean_squared_error(y,y_pred_exp))


