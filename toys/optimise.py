import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta, lognorm, chisquare

import matplotlib.pyplot as plt

a = b = 0.5

x = np.linspace(beta.ppf(0.01, a, b),
                beta.ppf(0.99, a, b), 100)
x_trim = x[2:-3] + 0.3

y1 = beta.pdf(x, 0.5, 0.5) + 20*lognorm.pdf(x*2, 1)

y2 = np.random.uniform(0, 0.5, (100,)) + y1
y2 = y2[2:-3]

new_x_trim = np.sort(np.random.uniform(x_trim[0], x_trim[-1], int(len(x_trim)/2)))
new_y2 = np.interp(new_x_trim, x_trim, y2)

def min_chisq(ht, x_exp, y_exp, x_th, y_th):
    new_x_exp = x_exp + ht
    new_y_exp = np.interp(x_th, new_x_exp, y_exp)
    values = np.vstack([new_y_exp, y_th])
    a, b = values[:, ~np.isnan(values).any(axis=0)]
    return chisquare(a, b)[0]


res = minimize(min_chisq, [0.], args=(new_x_trim, new_y2, x, y1),
               bounds=(np.array([-0.5, 0.5])*np.ptp(x_trim),))

fig, ax = plt.subplots()

ax.plot(x, y1, label='Theory')
ax.plot(new_x_trim, new_y2, label='Experiment')
ax.plot(new_x_trim + res.x, new_y2, label='Shifted Experiment')

plt.legend()

plt.show()
