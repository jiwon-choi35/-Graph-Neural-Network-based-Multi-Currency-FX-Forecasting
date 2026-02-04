from matplotlib import pyplot as plt
import numpy as np
import csv

def exponential_smoothing(series, alpha):

    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result
  
def plot_exponential_smoothing(series, alphas, attack):
 
    plt.figure(figsize=(17, 8))
    for alpha in alphas:
        plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))
    plt.plot(series.values, "c", label = "Actual")
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title("Exponential Smoothing - "+attack)
    plt.grid(True);

def double_exponential_smoothing(series, alpha, beta):

    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)
    return result

def plot_double_exponential_smoothing(series, alphas, betas, attack):
     
    plt.figure(figsize=(17, 8))
    for alpha in alphas:
        for beta in betas:
            plt.plot(double_exponential_smoothing(series, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))
    plt.plot(series, label = "Actual")
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title("Double Exponential Smoothing - "+attack)
    plt.grid(True)
    plt.show()

#The below script performs double exponential smoothing for the data
alpha=float(0.1)
beta=float(0.3)
file_name='data/data.txt'
fin = open(file_name)
rawdat = np.loadtxt(fin, delimiter='\t')
print(rawdat)
print(rawdat.shape)


smoothed=[]
i=0
for r in rawdat.transpose():
    smoothed.append(double_exponential_smoothing(r, alpha,beta))
smoothed = list(map(list, zip(*smoothed)))


with open("data/sm_data.csv", "w",newline="") as f:
    writer = csv.writer(f)
    writer.writerows(smoothed [:-1]) #do not include the last extra row


