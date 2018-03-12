import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# calculate the result without history information
def CalNewLowerBound(x):
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


# calculate moving average of results
def MovingAverage(array, windowSize):
    # N = np.ceil((windowSize - 1) / 2)
    N = int((windowSize - 1) / 2)
    T = len(array)
    newArray = array[:]
    for i in range(T):
        newArray[i] = np.mean(array[max(i - N, 0):min(i + N, T - 1)])
    return newArray


# Define parameters
timeSlotNum = 500
windowSize = 21

# Import training results
os.chdir('Results/Convergence')
# \lambda = 0.2
with open('Conv_lam0.2_H1_R1_N32_M500.csv', "r") as file:
    reader = csv.reader(file)
    row = next(reader)
    Conv_02_H1 = [float(i) for i in row]
    Conv_02_H1_MA = MovingAverage(Conv_02_H1, windowSize)
with open('Conv_lam0.2_H2_R2_N32_M500.csv', "r") as file:
    reader = csv.reader(file)
    row = next(reader)
    Conv_02_H2 = [float(i) for i in row]
    Conv_02_H2_MA = MovingAverage(Conv_02_H2, windowSize)
with open('Conv_lam0.2_H4_R4_N32_M500.csv', "r") as file:
    reader = csv.reader(file)
    row = next(reader)
    Conv_02_H4 = [float(i) for i in row]
    Conv_02_H4_MA = MovingAverage(Conv_02_H4, windowSize)
with open('Conv_lam0.2_H10_R10_N32_M500.csv', "r") as file:
    reader = csv.reader(file)
    row = next(reader)
    Conv_02_H10 = [float(i) for i in row]
    Conv_02_H10_MA = MovingAverage(Conv_02_H10, windowSize)
with open('Conv_lam0.2_H20_R20_N32_M500.csv', "r") as file:
    reader = csv.reader(file)
    row = next(reader)
    Conv_02_H20 = [float(i) for i in row]
    Conv_02_H20_MA = MovingAverage(Conv_02_H20, windowSize)
# \lambda = 0.4
with open('Conv_lam0.4_H1_R1_N32_M500.csv', "r") as file:
    reader = csv.reader(file)
    row = next(reader)
    Conv_04_H1 = [float(i) for i in row]
    Conv_04_H1_MA = MovingAverage(Conv_04_H1, windowSize)
with open('Conv_lam0.4_H2_R2_N32_M500.csv', "r") as file:
    reader = csv.reader(file)
    row = next(reader)
    Conv_04_H2 = [float(i) for i in row]
    Conv_04_H2_MA = MovingAverage(Conv_04_H2, windowSize)
with open('Conv_lam0.4_H4_R4_N32_M500.csv', "r") as file:
    reader = csv.reader(file)
    row = next(reader)
    Conv_04_H4 = [float(i) for i in row]
    Conv_04_H4_MA = MovingAverage(Conv_04_H4, windowSize)
with open('Conv_lam0.4_H10_R10_N32_M500.csv', "r") as file:
    reader = csv.reader(file)
    row = next(reader)
    Conv_04_H10 = [float(i) for i in row]
    Conv_04_H10_MA = MovingAverage(Conv_04_H10, windowSize)
with open('Conv_lam0.4_H20_R20_N32_M500.csv', "r") as file:
    reader = csv.reader(file)
    row = next(reader)
    Conv_04_H20 = [float(i) for i in row]
    Conv_04_H20_MA = MovingAverage(Conv_04_H20, windowSize)
# \lambda = 0.5
with open('Conv_lam0.5_H1_R1_N32_M500.csv', "r") as file:
    reader = csv.reader(file)
    row = next(reader)
    Conv_05_H1 = [float(i) for i in row]
    Conv_05_H1_MA = MovingAverage(Conv_05_H1, windowSize)
with open('Conv_lam0.5_H2_R2_N32_M500.csv', "r") as file:
    reader = csv.reader(file)
    row = next(reader)
    Conv_05_H2 = [float(i) for i in row]
    Conv_05_H2_MA = MovingAverage(Conv_05_H2, windowSize)
with open('Conv_lam0.5_H4_R4_N32_M500.csv', "r") as file:
    reader = csv.reader(file)
    row = next(reader)
    Conv_05_H4 = [float(i) for i in row]
    Conv_05_H4_MA = MovingAverage(Conv_05_H4, windowSize)
with open('Conv_lam0.5_H10_R10_N32_M500.csv', "r") as file:
    reader = csv.reader(file)
    row = next(reader)
    Conv_05_H10 = [float(i) for i in row]
    Conv_05_H10_MA = MovingAverage(Conv_05_H10, windowSize)
with open('Conv_lam0.5_H20_R20_N32_M500.csv', "r") as file:
    reader = csv.reader(file)
    row = next(reader)
    Conv_05_H20 = [float(i) for i in row]
    Conv_05_H20_MA = MovingAverage(Conv_05_H20, windowSize)
# \lambda = 0.8
with open('Conv_lam0.7_H1_R1_N32_M500.csv', "r") as file:
    reader = csv.reader(file)
    row = next(reader)
    Conv_07_H1 = [float(i) for i in row]
    Conv_07_H1_MA = MovingAverage(Conv_07_H1, windowSize)
with open('Conv_lam0.7_H2_R2_N32_M500.csv', "r") as file:
    reader = csv.reader(file)
    row = next(reader)
    Conv_07_H2 = [float(i) for i in row]
    Conv_07_H2_MA = MovingAverage(Conv_07_H2, windowSize)
with open('Conv_lam0.7_H4_R4_N32_M500.csv', "r") as file:
    reader = csv.reader(file)
    row = next(reader)
    Conv_07_H4 = [float(i) for i in row]
    Conv_07_H4_MA = MovingAverage(Conv_07_H4, windowSize)
with open('Conv_lam0.7_H10_R10_N32_M500.csv', "r") as file:
    reader = csv.reader(file)
    row = next(reader)
    Conv_07_H10 = [float(i) for i in row]
    Conv_07_H10_MA = MovingAverage(Conv_07_H10, windowSize)
with open('Conv_lam0.7_H20_R20_N32_M500.csv', "r") as file:
    reader = csv.reader(file)
    row = next(reader)
    Conv_07_H20 = [float(i) for i in row]
    Conv_07_H20_MA = MovingAverage(Conv_07_H20, windowSize)
os.chdir('..')


# Calculate bounds
newLowerBound_02 = np.tile([[CalNewLowerBound(20)]], (timeSlotNum, 1))
originLowerBound_02 = np.tile([[CalLowerBound(20)]], (timeSlotNum, 1))
originUpperBound_02 = np.tile([[CalUpperBound(20)]], (timeSlotNum, 1))
newLowerBound_04 = np.tile([[CalNewLowerBound(40)]], (timeSlotNum, 1))
originLowerBound_04 = np.tile([[CalLowerBound(40)]], (timeSlotNum, 1))
originUpperBound_04 = np.tile([[CalUpperBound(40)]], (timeSlotNum, 1))
newLowerBound_05 = np.tile([[CalNewLowerBound(50)]], (timeSlotNum, 1))
originLowerBound_05 = np.tile([[CalLowerBound(50)]], (timeSlotNum, 1))
originUpperBound_05 = np.tile([[CalUpperBound(50)]], (timeSlotNum, 1))
newLowerBound_07 = np.tile([[CalNewLowerBound(70)]], (timeSlotNum, 1))
originLowerBound_07 = np.tile([[CalLowerBound(70)]], (timeSlotNum, 1))
originUpperBound_07 = np.tile([[CalUpperBound(70)]], (timeSlotNum, 1))

# plot results when \lambda = 0.2
xAxis = np.arange(0, timeSlotNum, 1)
fig = plt.figure(figsize=(10, 5))
ax = fig.gca()
ax.plot(xAxis, newLowerBound_02, '--', c='#9B59B6', linewidth=1.5)
ax.plot(xAxis, originLowerBound_02, '--', c='#2C3E50', linewidth=1.5)
ax.plot(xAxis, originUpperBound_02, '--', c='#7F8C8D', linewidth=1.5)
ax.plot(xAxis, Conv_02_H1_MA, c='#E74C3C', linewidth=1.5)
ax.plot(xAxis, Conv_02_H2_MA, c='#F1C40F', linewidth=1.5)
ax.plot(xAxis, Conv_02_H4_MA, c='#F39C12', linewidth=1.5)
ax.plot(xAxis, Conv_02_H10_MA, c='#27AE60', linewidth=1.5)
ax.plot(xAxis, Conv_02_H20_MA, c='#2980B9', linewidth=1.5)
ax.plot(xAxis, Conv_02_H1, '--', c='#E74C3C', linewidth=0.75)
ax.plot(xAxis, Conv_02_H2, '--', c='#F1C40F', linewidth=0.75)
ax.plot(xAxis, Conv_02_H4, '--', c='#F39C12', linewidth=0.75)
ax.plot(xAxis, Conv_02_H10, '--', c='#27AE60', linewidth=0.75)
ax.plot(xAxis, Conv_02_H20, '--', c='#2980B9', linewidth=0.75)
ax.tick_params(labelsize=16)
ax.legend(['New Lower Bound',
           'Original Lower Bound',
           'Original Upper Bound',
           '$H = 1$',
           '$H = 2$',
           '$H = 4$',
           '$H = 10$',
           '$H = 20$'])
ax.xaxis.set_minor_locator(MultipleLocator(50))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.grid(which='minor', axis='both', linestyle='--')
plt.xlabel('Iteration Times', fontsize=16)
plt.ylabel('$\mu^*$', fontsize=16)
plt.xlim([0, timeSlotNum])
# plt.ylim([0, 1])
plt.show()

# plot results when \lambda = 0.4
xAxis = np.arange(0, timeSlotNum, 1)
fig = plt.figure(figsize=(10, 5))
ax = fig.gca()
ax.plot(xAxis, newLowerBound_04, '--', c='#9B59B6', linewidth=1.5)
ax.plot(xAxis, originLowerBound_04, '--', c='#2C3E50', linewidth=1.5)
ax.plot(xAxis, originUpperBound_04, '--', c='#7F8C8D', linewidth=1.5)
ax.plot(xAxis, Conv_04_H1_MA, c='#E74C3C', linewidth=1.5)
ax.plot(xAxis, Conv_04_H2_MA, c='#F1C40F', linewidth=1.5)
ax.plot(xAxis, Conv_04_H4_MA, c='#F39C12', linewidth=1.5)
ax.plot(xAxis, Conv_04_H10_MA, c='#27AE60', linewidth=1.5)
ax.plot(xAxis, Conv_04_H20_MA, c='#2980B9', linewidth=1.5)
ax.plot(xAxis, Conv_04_H1, '--', c='#E74C3C', linewidth=0.75)
ax.plot(xAxis, Conv_04_H2, '--', c='#F1C40F', linewidth=0.75)
ax.plot(xAxis, Conv_04_H4, '--', c='#F39C12', linewidth=0.75)
ax.plot(xAxis, Conv_04_H10, '--', c='#27AE60', linewidth=0.75)
ax.plot(xAxis, Conv_04_H20, '--', c='#2980B9', linewidth=0.75)
ax.tick_params(labelsize=16)
ax.legend(['New Lower Bound',
           'Original Lower Bound',
           'Original Upper Bound',
           '$H = 1$',
           '$H = 2$',
           '$H = 4$',
           '$H = 10$',
           '$H = 20$'])
ax.xaxis.set_minor_locator(MultipleLocator(50))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.grid(which='minor', axis='both', linestyle='--')
plt.xlabel('Iteration Times', fontsize=16)
plt.ylabel('$\mu^*$', fontsize=16)
plt.xlim([0, timeSlotNum])
# plt.ylim([0, 1])
plt.show()

# plot results when \lambda = 0.5
xAxis = np.arange(0, timeSlotNum, 1)
fig = plt.figure(figsize=(10, 5))
ax = fig.gca()
ax.plot(xAxis, newLowerBound_05, '--', c='#9B59B6', linewidth=2)
ax.plot(xAxis, originLowerBound_05, '--', c='#2C3E50', linewidth=2)
ax.plot(xAxis, originUpperBound_05, '--', c='#7F8C8D', linewidth=2)
ax.plot(xAxis, Conv_05_H1_MA, c='#E74C3C', linewidth=1.5)
ax.plot(xAxis, Conv_05_H2_MA, c='#F1C40F', linewidth=1.5)
ax.plot(xAxis, Conv_05_H4_MA, c='#F39C12', linewidth=1.5)
ax.plot(xAxis, Conv_05_H10_MA, c='#27AE60', linewidth=1.5)
ax.plot(xAxis, Conv_05_H20_MA, c='#2980B9', linewidth=1.5)
ax.plot(xAxis, Conv_05_H1, '--', c='#E74C3C', linewidth=0.75)
ax.plot(xAxis, Conv_05_H2, '--', c='#F1C40F', linewidth=0.75)
ax.plot(xAxis, Conv_05_H4, '--', c='#F39C12', linewidth=0.75)
ax.plot(xAxis, Conv_05_H10, '--', c='#27AE60', linewidth=0.75)
ax.plot(xAxis, Conv_05_H20, '--', c='#2980B9', linewidth=0.75)
ax.tick_params(labelsize=16)
ax.legend(['New Lower Bound',
           'Original Lower Bound',
           'Original Upper Bound',
           '$H = 1$',
           '$H = 2$',
           '$H = 4$',
           '$H = 10$',
           '$H = 20$'])
ax.xaxis.set_minor_locator(MultipleLocator(50))
ax.yaxis.set_minor_locator(MultipleLocator(0.01))
ax.grid(which='minor', axis='both', linestyle='--')
plt.xlabel('Iteration Times', fontsize=16)
plt.ylabel('$\mu^*$', fontsize=16)
plt.xlim([0, timeSlotNum])
# plt.ylim([0, 1])
plt.show()

# plot results when \lambda = 0.7
xAxis = np.arange(0, timeSlotNum, 1)
fig = plt.figure(figsize=(10, 5))
ax = fig.gca()
ax.plot(xAxis, newLowerBound_07, '--', c='#9B59B6', linewidth=1.5)
ax.plot(xAxis, originLowerBound_07, '--', c='#2C3E50', linewidth=1.5)
ax.plot(xAxis, originUpperBound_07, '--', c='#7F8C8D', linewidth=1.5)
ax.plot(xAxis, Conv_07_H1_MA, c='#E74C3C', linewidth=1.5)
ax.plot(xAxis, Conv_07_H2_MA, c='#F1C40F', linewidth=1.5)
ax.plot(xAxis, Conv_07_H4_MA, c='#F39C12', linewidth=1.5)
ax.plot(xAxis, Conv_07_H10_MA, c='#27AE60', linewidth=1.5)
ax.plot(xAxis, Conv_07_H20_MA, c='#2980B9', linewidth=1.5)
ax.plot(xAxis, Conv_07_H1, '--', c='#E74C3C', linewidth=0.75)
ax.plot(xAxis, Conv_07_H2, '--', c='#F1C40F', linewidth=0.75)
ax.plot(xAxis, Conv_07_H4, '--', c='#F39C12', linewidth=0.75)
ax.plot(xAxis, Conv_07_H10, '--', c='#27AE60', linewidth=0.75)
ax.plot(xAxis, Conv_07_H20, '--', c='#2980B9', linewidth=0.75)
ax.tick_params(labelsize=16)
ax.legend(['New Lower Bound',
           'Original Lower Bound',
           'Original Upper Bound',
           '$H = 1$',
           '$H = 2$',
           '$H = 4$',
           '$H = 10$',
           '$H = 20$'])
ax.xaxis.set_minor_locator(MultipleLocator(50))
ax.yaxis.set_minor_locator(MultipleLocator(0.01))
ax.grid(which='minor', axis='both', linestyle='--')
plt.xlabel('Iteration Times', fontsize=16)
plt.ylabel('$\mu^*$', fontsize=16)
plt.xlim([0, timeSlotNum])
# plt.ylim([0, 1])
plt.show()
