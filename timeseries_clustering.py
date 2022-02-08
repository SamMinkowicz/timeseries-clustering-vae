# Set which gpu to use
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from vrae.vrae import VRAE
from vrae.utils import *
from vrae.utils_VAE import *
from vrae import utils_sm
import numpy as np
import torch
import pickle

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

import plotly
from torch.utils.data import DataLoader, TensorDataset
import time


# Download dir
dload = r'/home/sam/timeseries-clustering-vae/model_dir'
data_dir = r'/media/storage/sam/anipose_out'

# Hyper parameters
seq_len = 250
hidden_size = 256
hidden_layer_depth = 3
latent_length = 16
batch_size = 32
learning_rate = 0.00002
n_epochs = 350
dropout_rate = 0.0
optimizer = 'Adam' # options: ADAM, SGD
cuda = True # options: True, False
print_every = 10
val_every = 1000
clip = True # options: True, False
max_grad_norm=5
loss = 'MSELoss' # options: SmoothL1Loss, MSELoss
block = 'LSTM' # options: LSTM, GRU
output = False
reduction = 'mean'

# Load data and preprocess

X = utils_sm.load_training_data(data_dir, batch_size=batch_size,
                                seq_len=seq_len, trim=True)
dataset = TensorDataset(torch.from_numpy(X))

num_features = X.shape[2]

# ### Initialize VRAE object
#
# VRAE inherits from `sklearn.base.BaseEstimator` and overrides `fit`, `transform` and `fit_transform` functions, similar to sklearn modules

from vrae.vrae import VRAE
vrae = VRAE(sequence_length=seq_len,
            number_of_features = num_features,
            hidden_size = hidden_size,
            hidden_layer_depth = hidden_layer_depth,
            latent_length = latent_length,
            batch_size = batch_size,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            dropout_rate = dropout_rate,
            optimizer = optimizer,
            cuda = cuda,
            print_every=print_every,
            val_every = val_every,
            clip=clip,
            max_grad_norm=max_grad_norm,
            loss = loss,
            block = block,
            dload = dload,
            output = output,
            reduction = reduction)

# ### Fit the model onto dataset

# Cross validation:
# train_loss, train_mse, val_mse = crossval(dataset, vrae, batch_size, k = 5)
# with open(dload+'/meanrecloss_loss_mse', "wb") as fh:
#     pickle.dump([train_loss, train_mse, val_mse], fh)

# Training model:
vrae.fit(dataset, None)

# ### Plot loss and MSE
plt.figure(figsize=(8,4.5))
plt.semilogy(vrae.all_loss, color = 'r', alpha = 0.5, label = 'loss')
plt.semilogy(vrae.recon_loss, color = 'b', alpha = 0.5, label = 'recon mse')
plt.semilogy(vrae.kl_loss, color = 'k', alpha = 0.5, label = 'KL')
plt.legend()
plt.savefig(f'model_loss_{time.strftime("%Y%m%d-%H%M%S")}')

# train_loss, train_mse, val_mse

# for ii in range(5):
# #     plt.plot(train_loss[ii], color = 'k', alpha = 0.5, linewidth=1)
#     plt.plot(train_mse[ii], color = 'r', alpha = 0.5, linewidth=1)
#     plt.plot(np.arange(5, 601, 5), val_mse[ii], color = 'b', alpha = 0.5, linewidth=1)
# plt.figure(figsize = (4.5, 4.5))
# for ii in range(5):
#     diff = [y - x for x,y in zip(val_mse[ii],val_mse[ii][1:])]
#     plt.plot(np.arange(10, 601, 5), diff, color = 'b', alpha = 0.5, linewidth=1)
#     plt.xlim([150, 500])
#     plt.ylim([-5,5])
# plt.plot([150, 600], [0,0], color = 'k', linewidth = 1)
# plt.plot([350, 350], [-5, 5], color = 'k', linewidth = 1)
# plt.title('mean recon mse, z=16, 5-fold cv')

# ### Transform the input timeseries to encoded latent vectors

#If the latent vectors have to be saved, pass the parameter `save`
# z_run = vrae.transform(train_dataset, save = True, filename = 'z_run_e2_b32_z16_output.pkl')
z_run = vrae.transform(dataset, save = False)
z_run.shape

# ### Save / load the model
model_name = f'/vrae_b{batch_size}_z{latent_length}_mean_{n_epochs}epoch.pth'
with open(dload+model_name, "wb") as fh:
    pickle.dump(vrae, fh)

# with open(dload+'/vrae_b32_z8_mean_350epoch.pth', "rb") as fh:
#     vrae = pickle.load(fh)

# %%
vrae.save('.'+model_name)

# %%
# vrae.load(dload+'/vrae_b32_z16_300epoch.pth')
# with open(dload+'/z_run_e57_b32_z16_output.pkl', 'rb') as fh:
#     z_run = pickle.load(fh)

# %% [markdown]
# ## Reconstruction

# %%
reconstruction = recon(vrae, X)

# %%
plot_recon_long(X, reconstruction)

# %%
plot_recon_single(X, reconstruction, idx = None)

# %%
corr_mean, mse_mean, mean_mean = plot_recon_metrics(X, reconstruction, verbose=False, plot=False)

# %%
corr_all, R2_all, uqi, term1, term2, term3 = recon_metrics2(X, reconstruction)
plt.figure(figsize=(8,5))
plt.plot(corr_all, label = 'corr')
# plt.plot(corr_all**2, label = 'corr^2', color = 'r', alpha = 0.5)
plt.plot(R2_all, label = 'R2', color = 'b', alpha = 0.5)
# plt.plot(uqi, label = 'uqi')
# plt.plot(term1, label = 'term1-corr', color = 'y', alpha = 0.5)
plt.plot(term2, label = 'term2-luminance')
plt.plot(term3, label = 'term3-contrast')
plt.legend()
plt.xlabel('# channel')

# %%
print(list(corr_mean))
print(list(mse_mean))
print(list(mean_mean))

# %% [markdown]
# ### Convert PC back to channel traces

# %%
# recon_channel = pca_inverse(X_pca, reconstruction)

# %%
# plot_recon_feature(X_train_ori, recon_channel, idx = None)

# %%
# corr_mean, mse_mean, mean_mean = plot_recon_metrics(X_train_ori, recon_channel, verbose=False, plot=False)

# %% [markdown]
# ## Reconstruction of test dataset

# %%
# testing_file = ['20201020_Pop_Cage_006']
# X_test, y_test = load_data(direc = 'data', dataset="EMG", all_file = testing_file,
#                          do_pca = False, single_channel = None,
#                          batch_size = batch_size, seq_len = seq_len, pca_component = 6)

# %%
# Uncomment if using pca

# recon_test = recon(vrae, X_test)
# recon_channel_test = pca_inverse(test_pca, recon_test)

# %%
# plot_recon_feature(X_test, recon_test, idx = None)
# plot_recon_feature(X_test_ori, recon_channel_test, idx = None)

# %%
# corr_mean, mse_mean, mean_mean = plot_recon_metrics(X_test, recon_test, x_lim = [0, 2000])
# corr_mean, mse_mean, mean_mean = plot_recon_metrics(X_test_ori, recon_channel_test, verbose=False, plot=False)

# %%
# print(list(corr_mean))
# print(list(mse_mean))
# print(list(mean_mean))

# %% [markdown]
# ## Visualize latent space using PCA and tSNE

# %%
# bhvs = {'crawling': np.array([0]),
#         'high picking treats': np.array([1]),
#         'low picking treats': np.array([2]),
#         'pg': np.array([3]),
#         'sitting still': np.array([4]),
#         'grooming': np.array([5]),
#         'no_behavior': np.array([-1])}

# inv_bhvs = {int(v): k for k, v in bhvs.items()}

# # %%
# test_dataset = TensorDataset(torch.from_numpy(X_test))
# z_run_test = vrae.transform(test_dataset, save = False)
# z_run_all = np.vstack((z_run, z_run_test))
# y_all = np.vstack((y, y_test))

# # %%
# visualize(z_run = z_run, y = y, inv_bhvs = inv_bhvs, one_in = 4)


