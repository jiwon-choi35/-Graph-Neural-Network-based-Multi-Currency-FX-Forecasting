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
plot_mode = 1 # 0: Exchange(환율), 1: Change(변화량), 2: Minmax, 3: Zscore
mode_names = ["Exchange", "Change", "Minmax", "Zscore"]


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
            if len(word)<=2: 
                result+=word
            else:
                result+=word[0].upper()+word[1:]

            if i<len(words)-1:
                result+=' '

        return result
    
    words= name.split(' ')
    result=''
    for i,word in enumerate(words):
        if len(word)<=3 or '/' in word:
            result+=word
        else:
            result+=word[0]+(word[1:].lower())
        
        if i<len(words)-1:
            result+=' '
        
    return result

#returns the closest curve cc to a given curve c in a list of forecasted curves, where cc is strictly larger than c
def getClosestCurveLarger(c,forecast,confidence, target, comparisons, col):
    d=999999999
    cc=None
    cc_conf=None
    for j in range(forecast.shape[1]):
        f= forecast[:,j]
        f_conf=confidence[:,j]
        if not col[j] in comparisons and not col[j]==target: #exclude irrelevant curves
            continue 
        if torch.mean(f) <= torch.mean(c):
            continue #must be larger
        if torch.mean(f)-torch.mean(c)<d:
            d=torch.mean(f)-torch.mean(c)
            cc=f.clone()
            cc_conf=f_conf.clone()
    return cc,cc_conf

#returns closest curve cc to given curve c in a list of forecasted curves, where cc is strictly smaller than c
def getClosestCurveSmaller(c,forecast,confidence,target, comparisons, col):
    d=999999999
    cc=None
    cc_conf=None
    for j in range(forecast.shape[1]):
        f= forecast[:,j]
        f_conf=confidence[:,j]
        if not col[j] in comparisons and not col[j]==target: #exclude irrelevant curves
            continue 
        if torch.mean(f) >= torch.mean(c):
            continue #must be smaller
        if torch.abs(torch.mean(f)-torch.mean(c))<d:
            d=torch.abs(torch.mean(f)-torch.mean(c))
            cc=f.clone()
            cc_conf=f_conf.clone()
    return cc,cc_conf

#negative values (due to smoothing or artifacts) are changed to 0 (optional for FX, but good safety)
def clip_negative_curves(data, forecast, target, comparisons, index):
    # FX prices shouldn't be negative.
    # Check target
    t_idx = index[target]
    data[:, t_idx] = torch.clamp(data[:, t_idx], min=0)
    forecast[:, t_idx] = torch.clamp(forecast[:, t_idx], min=0)

    # Check comparisons
    for s in comparisons:
        s_idx = index[s]
        data[:, s_idx] = torch.clamp(data[:, s_idx], min=0)
        forecast[:, s_idx] = torch.clamp(forecast[:, s_idx], min=0)
        
    return data, forecast
           
        

#plots forecast of target and relevant comparisons trends.
def plot_forecast(data,forecast,confidence,target,comparisons,index,col,alarming=False):


    # data,forecast = clip_negative_curves(data, forecast, target, comparisons, index) # Removed for Returns plotting

    
    colours = ["RoyalBlue", "Crimson", "DarkOrange", "MediumPurple", "MediumVioletRed",
          "DodgerBlue", "Indigo", "coral", "hotpink", "DarkMagenta",
          "SteelBlue", "brown", "MediumAquamarine", "SlateBlue", "SeaGreen",
          "MediumSpringGreen", "DarkOliveGreen", "Teal", "OliveDrab", "MediumSeaGreen",
          "DeepSkyBlue", "MediumSlateBlue", "MediumTurquoise", "FireBrick",
          "DarkCyan", "violet", "MediumOrchid", "DarkSalmon", "DarkRed"]

    
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.75])


    #Plot the forecast of target
    counter=0
    d=torch.cat((data[:,index[target]],forecast[0:1,index[target]]),dim=0)#connect the past to future in the plot
    f=forecast[:,index[target]]
    c=confidence[:,index[target]]
    a=consistent_name(target)
    ax.plot(range(len(d)),d,'-', color=colours[counter],label=a,linewidth = 2)
    ax.plot(range(len(d)-1, (len(d)+len(f))-1),f,'-', color=colours[counter],linewidth=2)
    ax.fill_between(range(len(d)-1, (len(d)+len(f))-1),f - c, f + c, color=colours[counter], alpha=0.5)
    f_target=f.clone()
    counter+=1

    #Filter comparisons if alarming is set (keeping logic generic: if comparison > target)
    if alarming:
        for s in list(comparisons):
            f=forecast[:,index[s]]
            if torch.mean(f)>= torch.mean(f_target): 
                comparisons.remove(s)

    #Plot the forecast of the comparisons
    for s in comparisons:
        d=torch.cat((data[:,index[s]],forecast[0:1,index[s]]),dim=0)#connect the past to future in the plot
        f=forecast[:,index[s]]
        c=confidence[:,index[s]]
        s_name=consistent_name(s)
        ax.plot(range(len(d)),d,'-', color=colours[counter],label=s_name,linewidth = 1)
        ax.plot(range(len(d)-1, (len(d)+len(f))-1),f,'-', color=colours[counter],linewidth=1)
        ax.fill_between(range(len(d)-1, (len(d)+len(f))-1),f - c, f + c, color=colours[counter], alpha=0.35)
        
        # Highlight gap
        if torch.mean(f_target) > torch.mean(f):
            cc,cc_conf=getClosestCurveLarger(f,forecast,confidence,target, comparisons, col)
            ax.fill_between(range(len(d)-1, (len(d)+len(f))-1),cc-cc_conf, f+c, color=colours[counter], alpha=0.2)
        else:
            cc,cc_conf=getClosestCurveSmaller(f,forecast,confidence,target,comparisons,col)
            ax.fill_between(range(len(d)-1, (len(d)+len(f))-1),cc+cc_conf, f-c,  color=colours[counter], alpha=0.2)

        counter+=1

    start_year = 2011
    years = range(start_year, start_year + 20)  # Enough to cover data + forecast
    x_labels = [str(y) for y in years]
    # Ticks at 0, 12, 24... corresponding to Jan of each year
    x_ticks = [i * 12 for i in range(len(x_labels))]
    ax.set_xticks(x_ticks, x_labels)

    ax.set_ylabel(mode_names[plot_mode], fontsize=15)
    pyplot.yticks(fontsize=13)
    ax.legend(loc="upper left",prop={'size': 10}, bbox_to_anchor=(1, 1.03))
    ax.axis('tight')
    #ax.set_ylim(-5, 5)
    ax.grid(True)
    pyplot.xticks(rotation=90,fontsize=13)
    pyplot.title(consistent_name(a) + " (" + mode_names[plot_mode] + ")", y=1.03,fontsize=18)

    fig = pyplot.gcf()
    fig.set_size_inches(10, 7) 

    #save and show the forecast
    images_dir = 'model/Bayesian/forecast/plots/'
    os.makedirs(images_dir, exist_ok=True)
    save_fn = consistent_name(a).replace(' ', '_').replace('/', '_') + "_" + mode_names[plot_mode]
    pyplot.savefig(images_dir + save_fn + '.png', bbox_inches="tight")
    pyplot.savefig(images_dir + save_fn + ".pdf", bbox_inches = "tight", format='pdf')
    pyplot.show(block=False)
    pyplot.pause(5)
    pyplot.close()


#saves the numerical forecast to text file as well as past data of each node
def save_data(data, forecast, confidence, variance, col, target_nodes=None):
    # write the data and forecast
    file_dir = 'model/Bayesian/forecast/data/'
    os.makedirs(file_dir, exist_ok=True)
    
    for i in range(data.shape[1]):
        if target_nodes is not None and col[i] not in target_nodes:
            continue
        d= data[:,i]
        f= forecast[:,i]
        c=confidence[:,i]
        v=variance[:,i]
        name=col[i]
        
        save_name = consistent_name(name) + "_" + mode_names[plot_mode] + ".txt"
        with open(file_dir+save_name, 'w') as ff:
            ff.write('Data: '+str(d.tolist())+'\n')
            ff.write('Forecast: '+str(f.tolist())+'\n')
            ff.write('95% Confidence: '+str(c.tolist())+'\n')
            ff.write('Variance: '+str(v.tolist())+'\n') # Note: Variance is from returns
    ff.close()

#saves the forecasted trend's gap between target and its relevant comparisons
def save_difference(forecast, target, comparisons, index):
    # write the data and forecast
    file_dir = 'model/Bayesian/forecast/gap/'
    os.makedirs(file_dir, exist_ok=True)
    
    with open(file_dir+consistent_name(target)+'_'+mode_names[plot_mode]+'_gap.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the list as a row
        writer.writerow(['Comparison','2026','2027','2028']) # Generic Headers
        table=[]
        a=forecast[:,index[target]].tolist()
        # Aggregating by 12 months for 3 years
        a_reduced= [sum(a[i:i+12]) / 12 for i in range(0, min(len(a), 36), 12)] # 기간 수정: min(len(a), 36)
        
        for s in comparisons:
            row=[consistent_name(s)]
            f=forecast[:,index[s]].tolist()
            f_reduced= [sum(f[i:i+12]) / 12 for i in range(0, min(len(f), 36), 12)] # 기간 수정: min(len(f), 36)
            
            gap=[x - y for x, y in zip(a_reduced, f_reduced)]
            row.extend(gap)
            table.append(row)
        sorted_table = sorted(table, key=lambda row: sum(row[1:])) # sort by sum of gaps
        for row in sorted_table:
            writer.writerow(row)
    

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


#builds the graph
def build_graph(file_name):
    graph = defaultdict(list)
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key_node = row[0]
            adjacent_nodes =  [node for node in row[1:] if node]
            graph[key_node].extend(adjacent_nodes)
    print('Graph loaded with',len(graph),'targets...')
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

#build the graph 
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
        y_pred = output[-1, :, :,-1].clone()# 36x142
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

# Confidence interval in price space (Pointwise for visibility)
v_factor = 5.0 # Visibility multiplier: 오차율 보정치 강제 추가
confidence_price = (Y_price * confidence_denorm) * v_factor

# 3. Choose Data and Apply Plot Scaling
if plot_mode == 0: # Price
    Y_final = Y_price
    confidence_final = confidence_price
    dat_final = torch.from_numpy(rawdat).float().to(Y.device)
elif plot_mode == 1: # Return
    Y_final = Y_denorm
    confidence_final = confidence_denorm
    dat_final = torch.from_numpy(returns).float().to(Y.device)
else:
    # Scale based on Price trends. not Change
    prices_past = torch.from_numpy(rawdat).float().to(Y.device)
    prices_future = Y_price
    all_prices = torch.cat((prices_past, prices_future), dim=0)
    
    if plot_mode == 2: # Minmax
        min_v = all_prices.min(dim=0)[0]
        max_v = all_prices.max(dim=0)[0]
        denom = max_v - min_v + 1e-8
        all_scaled = (all_prices - min_v) / denom
        confidence_final = confidence_price / denom
    else: # Zscore
        mean_v = all_prices.mean(dim=0)
        std_v = all_prices.std(dim=0) + 1e-8
        all_scaled = (all_prices - mean_v) / std_v
        confidence_final = confidence_price / std_v
        
    dat_final = all_scaled[:-36, :]
    Y_final = all_scaled[-36:, :]

print('output shape:', Y_final.shape)

#----------------------------------------------------------------------------------------------------#
#Plotting:
#save the data to desk
target_nodes=list(graph.keys())
save_data(dat_final, Y_final, confidence_final, variance, col, target_nodes=target_nodes)

#combine data (past + future) for smoothing
all_data = torch.cat((dat_final, Y_final), dim=0)

#smoothing
smoothed_dat = torch.stack(exponential_smoothing(all_data, 0.1))
smoothed_confidence = torch.stack(exponential_smoothing(confidence_final, 0.1))

main_targets = list(graph.keys())
#plot all forecasted nodes in the graph as groups of plots. 
for target in main_targets:
    # comparisons = all other main targets
    comparisons = [t for t in main_targets if t != target]
    # Confidence: smoothed_confidence
    plot_forecast(smoothed_dat[:-36,], smoothed_dat[-36:,], smoothed_confidence, target, comparisons, index, col)   # 기간 수정
    save_difference(smoothed_dat[-36:,], target, comparisons, index)    # 기간 수정
