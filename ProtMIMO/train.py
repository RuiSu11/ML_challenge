"""Training ProtMIMOOracle for Fluorescence data."""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr 
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import ProtMIMO_CNN
from data_utils import *

device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data
        optimizer.zero_grad()
        y0, y1, y2 = model(data['pro0'].to(device), data['pro1'].to(device), data['pro2'].to(device))
        output = torch.concat((y0,y1,y2)).to(device)
        loss = loss_fn(output, torch.concat((data['y0'], data['y1'], data['y2'])).float().to(device))
        loss.backward()
        optimizer.step()

def predict(model, device, loader, return_std=False):
    # return the std of the three predicitons when needed

    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    model.eval()
    preds = torch.Tensor()
    trues = torch.Tensor()
    pred_stds = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            #data = data
            y0, y1, y2 = model(data['pro'].to(device), data['pro'].to(device), data['pro'].to(device))
            output = (y0+y1+y2)/3
            preds = torch.cat((preds, output.cpu()), 0)
            trues = torch.cat((trues, data['y'].cpu()), 0)
            if(return_std):
              stds = torch.std(torch.stack((y0,y1,y2)),dim=0)
              pred_stds = torch.cat((pred_stds, stds.cpu()), 0)
    if(return_std):
      return trues.numpy().flatten(), preds.numpy().flatten(), pred_stds.numpy().flatten()
    else:
      return trues.numpy().flatten(),preds.numpy().flatten()



if __name__ == "__main__":
    try:
        train_df, val_df, test_df = get_gfp_dfs()
    except FileNotFoundError:
        if not os.path.exists("fluorescence.tar.gz"):
            os.system(
                "wget http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/fluorescence.tar.gz"
            )  # Note: This data can also be downloaded by searching the link above in a browser.
        os.system("tar xzf fluorescence.tar.gz")
        train_df, val_df, test_df = get_gfp_dfs()


    BATCH_SIZE = 50
    LR = 1e-4
    L2_weight = 1e-5    # L2 regularization
    NUM_EPOCHS = 33     # Since we have 3 inputs, 33 epochs is equivalent to 100 epochs for single-in-single-out NNs
    
    # The CNN architecture and hyperparameters were used from DeepDTA(https://arxiv.org/pdf/1801.10193.pdf)
    model = ProtMIMO_CNN(label_len=len(GFP_ALPHABET), 
                         embed_dim=128, 
                         max_len=237, 
                         cnn_channels=[32,64,96], 
                         cnn_kernels=[4,8,12])

    # Implement your training and evaluation loop here.
    train_data = MyTrainDataset(train_df)
    val_data = MyTestDataset(val_df)
    test_data = MyTestDataset(test_df)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    cuda_name = "cuda:0"
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    df_results = pd.DataFrame(columns = ['# of epoch', 'mse', 'pearson correlation', 'spearman rho'])
    torch.save(model.state_dict(), 'checkpoints/model_checkpoint.pt')
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    best_mse = 1000
    best_epoch = -1
    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch+1)
        torch.save(model.state_dict(), 'results/model_epoch'+str(epoch+1)+'.pt')
        T,P = predict(model, device, val_loader)
        rmse = mean_squared_error(T, P)
        pearson_correlation = pearsonr(T, P)
        spearman_rho = spearmanr(T,P)
        df_results.loc[len(df_results)] = [epoch+1, rmse, pearson_correlation, spearman_rho]
        if rmse<best_mse:
            torch.save(model.state_dict(), 'results/best_model.pt')
            best_epoch = epoch+1
            best_mse = rmse
            print('rmse improved at epoch ', best_epoch, '; best_mse:', best_mse)
        else:
            print('current rmse:',rmse,'No improvement since epoch ', best_epoch, '; best_mse:', best_mse)
    df_results.to_csv('results/training_history.csv')