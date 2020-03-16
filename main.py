import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
df = pd.read_csv(url)

print('Raw data:')
print(df)

df = df.loc[:, ['data', 'totale_casi']]

FMT = '%Y-%m-%d %H:%M:%S'
date = df['data']
df['data'] = date.map(lambda x2: (datetime.strptime(x2, FMT) - datetime.strptime("2020-01-01 00:00:00", FMT)).days)

print('Days and Total cases:')
print(df)


def logistic_model(x3, a3, b3, c3):
    return c3/(1+np.exp(-(x3-b3)/a3))


x = list(df.iloc[:, 0])
y = list(df.iloc[:, 1])

fit = curve_fit(logistic_model, x, y, p0=[2, 100, 20000])

a = fit[0][0]
b = fit[0][1]
c = fit[0][2]

print('Infection speed (a):', round(a,2))

first_january = datetime.strptime('2020/01/01', "%Y/%m/%d")
infection_peak_date = first_january + timedelta(days=int(b))
print('Peak of infection:', datetime.strftime(infection_peak_date, "%Y/%m/%d"))
print('Total infected people at the end (c):', round(c,1))

errors = [np.sqrt(fit[1][i][i]) for i in [0, 1, 2]]

print('Errors for a, b and c:',errors)

sol = int(fsolve(lambda x : logistic_model(x, a, b, c) - int(c),b))

infection_end_date = first_january + timedelta(days=int(sol))
print('Final day of infection:', datetime.strftime(infection_end_date, "%Y/%m/%d"))

pred_x = list(range(max(x), sol))
plt.rcParams['figure.figsize'] = [7, 7]
plt.rc('font', size=14)
# Real data
plt.scatter(x,y,label="Real data",color="red")
# Predicted logistic curve
plt.plot(x+pred_x, [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in x+pred_x], label="Logistic model" )

plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of infected people")
plt.ylim((min(y)*0.9,c*1.1))

today_date = datetime.today().strftime('%Y-%m-%d')
filename = 'plot-' + today_date + '.png'
plt.savefig('plots/'+filename, bbox_inches="tight")

y_pred_logistic = [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in x]
print('error logistic regression model:', mean_squared_error(y,y_pred_logistic))

