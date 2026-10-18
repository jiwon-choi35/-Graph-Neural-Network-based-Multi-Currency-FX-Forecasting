import torch
import sys
import os
import time
import random

# Ensure scripts directory is in path for module imports
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from layer import *
from util import DataLoaderS

fixed_seed=123

# Modified for Exchange Rate Forecasting
class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=15, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=1, out_dim=8, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)


        self.idx = torch.arange(self.num_nodes).to(device)
        


    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))



        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx) # this line computes the adjacency matrix adaptively by calling the function forward in the gc
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A
        
        # print('Forward...')
        # time.sleep(1)
        # print(adp[4])
        # time.sleep(3)
        # #sys.exit()

        # col=DataLoaderS.col
        # for i in range(adp.shape[0]):
        #     print('connections to node '+col[i]+': [',end='')
        #     counter=0
        #     for j in range(adp.shape[1]):
        #         if adp[i,j].item()>0:
        #             print(col[j],end='')
        #             if j<adp.shape[1]-1:
        #                 print(', ', end='')
        #             counter+=1
        #         if j==adp.shape[1]-1:
        #             print('] total=',counter)
        # sys.exit()

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))

        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training) 
                    
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x
    

    @torch.no_grad()
    def mc_predict(self, input, idx=None, mc_samples: int = 30, quantiles=(0.1, 0.9)):
        """Monte-Carlo dropout uncertainty for multi-horizon forecast.

        Returns:
            mean:   [B, H, N, 1]
            lower:  [B, H, N, 1]
            upper:  [B, H, N, 1]
        """
        # preserve mode
        was_training = self.training
        try:
            # enable dropout
            self.train()
            outs = []
            for _ in range(int(mc_samples)):
                outs.append(self.forward(input, idx))
            outs = torch.stack(outs, dim=0)  # [S, B, H, N, 1]
            mean = outs.mean(dim=0)
            q_lo, q_hi = quantiles
            lower = torch.quantile(outs, q_lo, dim=0)
            upper = torch.quantile(outs, q_hi, dim=0)
            return mean, lower, upper
        finally:
            self.train(was_training)

    @torch.no_grad()
    def plot_forecast(self,
                      input,
                      idx=None,
                      node_names=None,
                      target_channel: int = 0,
                      mc_samples: int = 30,
                      quantiles=(0.1, 0.9),
                      title: str = "Exchange Rates",
                      save_path: str = "forecast.png",
                      show: bool = True):
        """Generate a 'paper-like' plot: history + forecast with shaded interval.

        - Uses the first sample in the batch.
        - X-axis uses step index (0..L-1 for history, L..L+H-1 for forecast).
        """
        import matplotlib.pyplot as plt

        mean, lower, upper = self.mc_predict(input, idx=idx, mc_samples=mc_samples, quantiles=quantiles)

        # Shapes
        # input: [B, C, N, L]
        # mean : [B, H, N, 1]
        x_hist = input[0, target_channel].detach().cpu().numpy()  # [N, L]
        y_mean = mean[0, :, :, 0].detach().cpu().numpy().T       # [N, H]
        y_lo   = lower[0, :, :, 0].detach().cpu().numpy().T      # [N, H]
        y_hi   = upper[0, :, :, 0].detach().cpu().numpy().T      # [N, H]

        N, L = x_hist.shape
        H = y_mean.shape[1]

        if node_names is None:
            node_names = [f"Node{i}" for i in range(N)]

        # Plot
        plt.figure(figsize=(14, 7))
        t_hist = list(range(L))
        t_fore = list(range(L, L + H))

        for i in range(N):
            plt.plot(t_hist, x_hist[i], label=str(node_names[i]))
            plt.plot(t_fore, y_mean[i], linestyle="--")
            plt.fill_between(t_fore, y_lo[i], y_hi[i], alpha=0.15)

        plt.title(title)
        plt.ylabel("Trend")
        plt.xlabel("Time")
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        if show:
            plt.show()
        else:
            plt.close()
        return save_path