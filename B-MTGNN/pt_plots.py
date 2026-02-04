import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable
import sys
import csv
from collections import defaultdict
from matplotlib import pyplot
import random

pyplot.rcParams['savefig.dpi'] = 1200


def exponential_smoothing(series, alpha):

    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

def consistent_name(name):
    name_map = {
        'kr_fx': 'Korean Won',
        'us_Trade Weighted Dollar Index': 'US Dollar Index',
        'jp_fx': 'Japanese Yen',
        'us_fx': 'US FX'
    }
    if name in name_map: return name_map[name]
    
    if not name.isupper():
        words=name.split(' ')
        result=''
        for i,word in enumerate(words):
            if len(word)<=2: #e.g., "of"
                result+=word
            else:
                result+=word[0].upper()+word[1:]
            
            if i<len(words)-1:
                result+=' '

        return result
    

    words= name.split(' ')
    result=''
    for i,word in enumerate(words):
        if len(word)<=3 or '/' in word or word=='MITM' or word =='SIEM':
            result+=word
        else:
            result+=word[0]+(word[1:].lower())
        
        if i<len(words)-1:
            result+=' '
        
    return result


#negative values (due to smoothing) are changed to 0
def zero_negative_curves(data, forecast, s):

    a = data[:, index[s]]
    f= forecast[:,index[s]]
    for i in range(a.shape[0]):
        if a[i]<0:
            a[i]=0
    for i in range(f.shape[0]):
        if f[i]<0:
            f[i]=0
    return data, forecast
           

      

#plots past data and forecast of a single pertinent technology node s
def plot_forecast(data,forecast,confidence,s,index,col):
    #pyplot.style.use("seaborn-dark")
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    #Plot the forecast
    d=torch.cat((data[:,index[s]],forecast[0:1,index[s]]),dim=0)#connect the past to future in the plot
    f=forecast[:,index[s]]
    c=confidence[:,index[s]]

    s_name=consistent_name(s)
    ax.plot(range(len(d)),d,'-', color='red',label=s_name,linewidth = 1)#past
    ax.plot(range(len(d)-1, (len(d)+len(f))-1),f,'-',color= 'red',linewidth=1)#future
    ax.fill_between(range(len(d)-1, (len(d)+len(f))-1),f - c, f + c, color= 'red', alpha=0.4)
      
    # Updated timeframe: 2011-01 to 2025-12 (Actual) + 2026-01 to 2028-12 (Forecast)
    years = [str(year) for year in range(2011, 2030)]
    ticks = [i * 12 for i in range(len(years))]
    ax.set_xticks(ticks, years)

    ax.set_ylabel("Exchange Rate",fontsize=15)
    pyplot.yticks(fontsize=13)
    ax.axis('tight')
    ax.grid(True)
    pyplot.xticks(rotation=90,fontsize=13)
    pyplot.title(s_name, y=1.03,fontsize=18)

    fig = pyplot.gcf()
    fig.set_size_inches(10, 7) 

    #save and show the forecast
    images_dir = 'model/Bayesian/forecast/pt_plots/'
    save_fn = s_name.replace(' ', '_').replace('/', '_')
    pyplot.savefig(images_dir + save_fn + '.png', bbox_inches="tight")
    pyplot.savefig(images_dir + save_fn + ".pdf", bbox_inches = "tight", format='pdf')
    pyplot.show(block=False)
    pyplot.pause(5)
    pyplot.close()




#given data file, returns the list of column names and dictionary of the format (column name,column index)
def create_columns(file_name):

    col_name=[]
    col_index={}

    # Read the CSV file of the dataset
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        # Read the first row
        col_name = [c for c in next(reader)]
        if 'Date' in col_name[0]:
            col_name= col_name[1:]
        
        for i,c in enumerate(col_name):
            col_index[c]=i
        
        return col_name,col_index


#builds the attacks and pertinent technologies graph
def build_graph(file_name):
    # Initialise an empty dictionary with default value as an empty list
    graph = defaultdict(list)

    # Read the graph CSV file
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        # Iterate over each row in the CSV file
        for row in reader:
            # Extract the key node from the first column
            key_node = row[0]
            # Extract the adjacent nodes from the remaining columns
            adjacent_nodes =  [node for node in row[1:] if node]#does not include empty columns
            
            # Add the adjacent nodes to the graph dictionary
            graph[key_node].extend(adjacent_nodes)
    print('Graph loaded with',len(graph),'attacks...')
    return graph




#This script forecasts the future of the graph, up to 3 years in advance


data_file='./data/sm_data.txt'
model_file='model/Bayesian/o_model.pt'
nodes_file='data/data.csv'
graph_file='data/graph.csv'


#read the data
fin = open(data_file)
rawdat = np.loadtxt(fin, delimiter='\t')
n, m = rawdat.shape

#load column names and dictionary of (column name, index)
col,index=create_columns(nodes_file)


#build the graph in the format {attack:list of pertinent technologies}
graph=build_graph(graph_file)

# Log-Return Transformation (Matching Training Logic)
min_val = np.min(rawdat, axis=0)
shift = np.zeros(m)
for i in range(m):
    if min_val[i] <= 0:
        shift[i] = abs(min_val[i]) + 1.0

shifted_rawdat = rawdat + shift
log_rawdat = np.log(shifted_rawdat)
returns = np.diff(log_rawdat, axis=0)

# Standardization (Normalize=3 style)
dat = np.zeros(returns.shape)
means = np.zeros(m)
stds = np.zeros(m)

for i in range(m):
    means[i] = np.mean(returns[:, i])
    stds[i] = np.std(returns[:, i])
    if stds[i] == 0: stds[i] = 1
    dat[:, i] = (returns[:, i] - means[i]) / stds[i]
print('data shape:',dat.shape)

#preparing last part of the data to be used for the forecast
P=10 #look back
X= torch.from_numpy(dat[-P:, :]) #look back 10 months
X = torch.unsqueeze(X,dim=0)
X = torch.unsqueeze(X,dim=1)
X = X.transpose(2,3)
X = X.to(torch.float)



#load the model
model=None
with open(model_file, 'rb') as f:
    model = torch.load(f, weights_only=False)


# Bayesian estimation
num_runs = 10

# Create a list to store the outputs
outputs = []


# Use model to predict next time step
for _ in range(num_runs):
    with torch.no_grad():
        output = model(X)  
        y_pred = output[-1, :, :,-1].clone()#36x142
    outputs.append(y_pred)

# Stack the outputs along a new dimension
outputs = torch.stack(outputs)


Y=torch.mean(outputs,dim=0)
variance = torch.var(outputs, dim=0)#variance
std_dev = torch.std(outputs, dim=0)#standard deviation
# Calculate 95% confidence interval
z=1.96
confidence=z*std_dev/torch.sqrt(torch.tensor(num_runs))

# 1. Denormalize
scale_tensor = torch.from_numpy(stds).float().to(Y.device)
mean_tensor = torch.from_numpy(means).float().to(Y.device)

Y_denorm = Y * scale_tensor + mean_tensor
confidence_denorm = confidence * scale_tensor 

# 2. Price Reconstruction (Actual FX Rates)
shift_t = torch.from_numpy(shift).float().to(Y.device)
Y_price = []
current_p = torch.from_numpy(rawdat[-1]).float().to(Y.device)

for t in range(Y_denorm.shape[0]):
    # Formula: P_t = (P_{t-1} + shift) * exp(ret_t) - shift
    current_p = (current_p + shift_t) * torch.exp(Y_denorm[t]) - shift_t
    Y_price.append(current_p)
Y_price = torch.stack(Y_price)

# Confidence interval in price space
v_factor = 5.0 # Visibility multiplier: 오차율 보정치 강제 추가
confidence_price = (Y_price * confidence_denorm) * v_factor

# Final Plotting Data (Actual Prices)
dat_final = torch.from_numpy(rawdat).float().to(Y.device)
Y_final = Y_price
confidence_final = confidence_price

print('output shape:', Y_final.shape)

#----------------------------------------------------------------------------------------------------#
#Plotting:
#combine data (past + future)
all_data = torch.cat((dat_final, Y_final), dim=0)

#smoothing
smoothed_dat = torch.stack(exponential_smoothing(all_data, 0.1))
smoothed_confidence = torch.stack(exponential_smoothing(confidence_final, 0.1))

#plot specific target currencies only
target_nodes = ['us_Trade Weighted Dollar Index', 'kr_fx', 'jp_fx']
for node_name in target_nodes:
    if node_name in index:
        plot_forecast(smoothed_dat[:-36,], smoothed_dat[-36:,], smoothed_confidence, node_name, index, col)
