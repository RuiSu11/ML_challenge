"""ProtMIMO model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtMIMO_CNN(nn.Module):
    def __init__(self, label_len, embed_dim, max_len, cnn_channels, cnn_kernels):
        super().__init__()
        
        # independent input embedding layers
        self.Embed0 = nn.Embedding(num_embeddings=label_len,embedding_dim=embed_dim)
        self.Embed1 = nn.Embedding(num_embeddings=label_len,embedding_dim=embed_dim)
        self.Embed2 = nn.Embedding(num_embeddings=label_len,embedding_dim=embed_dim)
        
        # shared CNN layers
        # The CNN architecture and hyperparameters were used from DeepDTA(https://arxiv.org/pdf/1801.10193.pdf)
        self.Conv0 = nn.Conv1d(in_channels=embed_dim, out_channels=cnn_channels[0], kernel_size=cnn_kernels[0], padding=0)
        self.Conv1 = nn.Conv1d(in_channels=cnn_channels[0], out_channels=cnn_channels[1], kernel_size=cnn_kernels[1], padding=0)
        self.Conv2 = nn.Conv1d(in_channels=cnn_channels[1], out_channels=cnn_channels[2], kernel_size=cnn_kernels[2], padding=0)
        self.Maxpool = nn.MaxPool1d(max_len-cnn_kernels[0]-cnn_kernels[1]-cnn_kernels[2]+3)
        self.ReLu = nn.ReLU()
        
        # independent output fully connected layers
        self.fc0 = nn.Linear(cnn_channels[2],1)
        self.fc1 = nn.Linear(cnn_channels[2],1)
        self.fc2 = nn.Linear(cnn_channels[2],1)
    
    def forward(self, pro0, pro1, pro2):
        
        x0 = self.Embed0(pro0)
        x1 = self.Embed1(pro1)
        x2 = self.Embed2(pro2)
        
        xs = [x0,x1,x2]
        ys = []
        for x in xs:
            x = x.permute(0,2,1)
            x = self.Conv0(x)
            x = self.ReLu(x)
            x = self.Conv1(x)
            x = self.ReLu(x)
            x = self.Conv2(x)
            x = self.ReLu(x)
            x = self.Maxpool(x).squeeze(2)
            ys.append(x)
        
        y0 = self.fc0(ys[0]).squeeze(1)
        y1 = self.fc1(ys[1]).squeeze(1)
        y2 = self.fc2(ys[2]).squeeze(1)
        
        return [y0,y1,y2]
