import os
import torch
import matplotlib.pyplot as plt

from vrae import utils_sm
from utils_VAE import *

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

from torch.utils.data import DataLoader, TensorDataset


# Download dir
model_dir = r'/home/sam/timeseries-clustering-vae/model_dir'
data_dir = r'/media/storage/sam/anipose_out'

# Hyper parameters
latent_length = 16
batch_size = 32
n_epochs = 350
seq_len = 250

# load the model
model_name = f'vrae_b{batch_size}_z{latent_length}_mean_{n_epochs}epoch.pth'
model_path = os.path.join(model_dir, model_name)
vrae = torch.load(model_path)

# Load training data
X = utils_sm.load_training_data(data_dir, batch_size=batch_size,
                                seq_len=seq_len, trim=True)
dataset = TensorDataset(torch.from_numpy(X))

# ## Reconstruction

reconstruction = recon(vrae, X)

plot_recon_long(X, reconstruction)

plot_recon_single(X, reconstruction, idx = None)

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

