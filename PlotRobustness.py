import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# calculate the result without history information
def CalH0(x):
    lam = x / 100
    if lam < 1 / 3:
        mu = 1 - 2 * lam
    else:
        mu = (1 - lam)**2 / (1 + lam)
    return mu


# calculate the lower bound in Thomas' work
def CalLowerBound(x):
    lam = x / 100
    # Calculate the lower bound
    if lam < 1 / 3:
        mu_l = 1 - 2 * lam
    else:
        mu_l = (1 - lam)**2 / (4 * lam)
    return mu_l


# calculate the upper bound in Thomas' work
def CalUpperBound(x):
    lam = x / 100
    # Initialize variables
    y = 2
    v_1 = -float('inf')
    # Calculate the lower bound
    while True:
        cur_v_1 = CalUpperBoundAssist(y, lam)
        if cur_v_1 - v_1 > 1e-4:
            v_1 = cur_v_1
            y = y + 1
        else:
            break
    mu_u = 1 / (1 - v_1)
    return mu_u


# assist to calculate the upper bound in Thomas' work
def CalUpperBoundAssist(y, lam):
    if lam == 1 / 2:
        v_1 = ((1 / 2)**(y - 1) + y - 3 / 2) / ((1 / 2)**y - 1 / 2)
    else:
        v_1 = ((2 - 4 * lam) * (lam * (1 - lam))**(y - 1) + 2 *
               lam * (1 - lam)**(y - 1) - lam**(y - 1)) / ((1 - 2 *
               lam) * (lam**(y - 1) - 1) * (1 - lam)**(y - 1))
    return v_1


# Import training results
os.chdir('Results/Convergence')
with open('H1_R1_N100_M200.csv', "r") as file1:
    reader1 = csv.reader(file1)
    row1 = next(reader1)
    H1_R1_N100_M200 = [float(i) for i in row1]
with open('H3_R10_N200_M200.csv', "r") as file2:
    reader2 = csv.reader(file2)
    row2 = next(reader2)
    H3_R10_N200_M200 = [float(i) for i in row2]
with open('H9_R10_N300_M400.csv', "r") as file3:
    reader3 = csv.reader(file3)
    row3 = next(reader3)
    H10_R10_N200_M200 = [float(i) for i in row3]
os.chdir('..')


# Calculate bounds
originLowerBound = []
originUpperBound = []
H0 = []
for i in range(100):
    originLowerBound.append(CalLowerBound(i))
    originUpperBound.append(CalUpperBound(i))
    H0.append(CalH0(i))


# # plot results
# xAxis = np.arange(0.0, 1.0, 0.01)
# fig = plt.figure(figsize=(12, 9))
# ax = fig.gca()
# ax.plot(xAxis, originLowerBound, '--', c='#2C3E50', linewidth=1.5)
# ax.plot(xAxis, originUpperBound, '--', c='#7F8C8D', linewidth=1.5)
# ax.plot(xAxis, H0, '#E74C3C', linewidth=2)
# ax.plot(xAxis, H1_R1_N100_M200, '#F39C12', linewidth=2)
# ax.plot(xAxis, H3_R10_N200_M200, '#27AE60', linewidth=2)
# ax.plot(xAxis, H10_R10_N200_M200, '#2980B9', linewidth=2)
# ax.tick_params(labelsize=16)
# ax.legend(['Original Lower Bound',
#            'Original Upper Bound',
#            '$H = 1$',
#            '$H = 2$',
#            '$H = 4$',
#            '$H = 10$'])
# ax.xaxis.set_minor_locator(MultipleLocator(0.1))
# ax.yaxis.set_minor_locator(MultipleLocator(0.1))
# ax.grid(which='minor', axis='both', linestyle='--')
# plt.xlabel('$\lambda$', fontsize=16)
# plt.ylabel('$\mu^*$', fontsize=16)
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.show()

# plot original bounds
xAxis = np.arange(0.0, 1.0, 0.01)
fig = plt.figure(figsize=(6, 5))
ax = fig.gca()
ax.plot(xAxis, H0, c='#9B59B6', linewidth=1.5)
ax.plot(xAxis, originLowerBound, c='#2C3E50', linewidth=1.5)
ax.plot(xAxis, originUpperBound, c='#7F8C8D', linewidth=1.5)
ax.tick_params(labelsize=16)
ax.legend(['New Lower Bound',
           'Original Lower Bound',
           'Original Upper Bound'])
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.grid(which='minor', axis='both', linestyle='--')
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('$\mu^*$', fontsize=16)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()

# plot new bounds
xAxis = np.arange(0.0, 1.0, 0.01)
fig = plt.figure(figsize=(6, 5))
ax = fig.gca()
ax.plot(xAxis, originLowerBound, c='#2C3E50', linewidth=1.5)
ax.plot(xAxis, originUpperBound, c='#7F8C8D', linewidth=1.5)
ax.tick_params(labelsize=16)
ax.legend(['Original Lower Bound',
           'Original Upper Bound'])
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.grid(which='minor', axis='both', linestyle='--')
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('$\mu^*$', fontsize=16)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()
