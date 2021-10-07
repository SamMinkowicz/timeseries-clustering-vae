import numpy as np
import matplotlib.pyplot as plt
# from vrae.vrae import VRAE
from torch import from_numpy
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset

def load_one_dataset(direc = 'data', dataset="EMG", filename = "."):
    """
    Load one dataset.
    """
    datadir = direc + '/' + dataset + '/' + filename
    data = np.loadtxt(datadir, delimiter=',')
    data = np.concatenate((data, ), axis=0)
    data = np.expand_dims(data, -1)
    
    return data[:, 1:, :], data[:, 0, :]


def load_data(direc = 'data', dataset="EMG", all_file = [], do_pca = False, single_channel = None, 
             batch_size = 32, seq_len = 10, pca_component = 6):
    """
    Load all dataset and preprocess.
    
    all_file: list of all the filenames to load. Each have a segments * (seq length * channel amount) * 1 ndarray.
    do_pca: preform PCA on training data. If true, return original and transformed data, and PCA object.
            If False, return original data.
    single_channel: a list of channels to load, channel starts with 1. If None, load all channels.
    seq_length: should be the same as when converting to vrae format.
    
    Return:
    X_train: data in segments * seq length * features ndarray.
    y_train: labels in segments * 1 ndarray.
    X_train_ori: X_train before doing PCA.
    X_pca: PCA object.
    
    """
    
    # Load every dataset in list
    X_train = []
    y_train = []
    for file in all_file:
        X_train_small, y_train_small = load_one_dataset(direc = 'data', dataset="EMG", filename = file)
        X_train.append(X_train_small)
        y_train.append(y_train_small)
        print(f'Loading {file}, X shape {X_train_small.shape}, y shape {y_train_small.shape}', end = '')
        print(f', has label {np.unique(y_train_small)}')
    
    # Concatenate into one np array
    if len(all_file) == 1:
        X_train = X_train[0]
        y_train = y_train[0]
    else:
        X_train = np.concatenate(X_train, axis = 0)
        y_train = np.concatenate(y_train, axis = 0)
        
    # Cut the last several segments to match batch size
    X_train = X_train.reshape(X_train.shape[0], seq_len, -1)
    num_seg = (X_train.shape[0] // batch_size) * batch_size
    X_train = X_train[:num_seg, :, :]
    y_train = y_train[:num_seg, :]
    
    # Check
    if do_pca and single_channel:
        raise ValueError("Don't do both pca and single channel.")
    
    # Doing pca
    if do_pca:
        print('Doing PCA')
        # Copy original training data
        X_train_ori = np.copy(X_train)
        
        # Explained variance
        temp = X_train.reshape(-1, X_train.shape[2])
        X_pca = PCA(n_components=15).fit(temp)
        print(f'Explained variance ratio: {np.cumsum(X_pca.explained_variance_ratio_)}')
        
        # Need to specify n_components inside pca for later reconstruction
        X_pca = PCA(n_components = pca_component).fit(temp)
        X_train = X_pca.transform(temp)
        X_train = X_train.reshape(-1, seq_len, pca_component)
    
    # Extract single channels
    if single_channel:
        print(f'Extracting channels {single_channel}')
        single_channel = np.array(single_channel)-1
        X_train = X_train[:, :, single_channel]
        #X_train = np.expand_dims(X_train, -1)
    
    print(f'Dataset shape: {X_train.shape}')
    print(f'Label: {np.unique(y_train)}, shape: {y_train.shape}')
    
    if do_pca:
        return X_train, y_train, X_train_ori, X_pca
    
    return X_train, y_train


def recon(model, dataset):
    """
    Pass dataset through vrae to get a reconstruction.
    
    model: trained vrae model.
    dataset: original data in segments * seq length * features ndarray.
    
    return:
    reconstruction: segments * seq length * features ndarray.
    """
    
    # torch_data = TensorDataset(torch.from_numpy(dataset))
    torch_data = TensorDataset(from_numpy(dataset))
    reconstruction = model.reconstruct(torch_data)
    reconstruction = reconstruction.transpose((1,0,2))
    
    return reconstruction


def plot_recon_feature(dataset, reconstruction, idx = None):
    """
    Plot the original and reconstructed feature of one segment.
    
    dataset: original data in segments * seq length * features ndarray.
    reconstruction: reconstructed data in segments * seq length * features ndarray.
    idx: Index of segment to plot. If None, randomly choose one.
    """
    
    num_seq = dataset.shape[0]
    num_features = dataset.shape[2]
    num_rows = -(-num_features//5)
    
    # reconstruction = recon(model, dataset)
    
    if idx:
        idx = idx-1
    else:
        idx = np.random.choice(num_seq, 1)[0]-1
    
    fig, axs = plt.subplots(num_rows, 5, figsize = (20, num_rows*5))
    for ii in range(num_features):
        ori = dataset[idx, :, ii]
        rec = reconstruction[idx, :, ii]
        axs[ii//5, ii%5].plot(ori, color = 'black')
        axs[ii//5, ii%5].plot(rec, color = 'red')
        axs[ii//5, ii%5].set_title(f'Feature #{ii+1}')
    fig.suptitle(f'Ori and rec of sequesce # {idx+1}', size = 20)
    plt.show()

    
def plot_recon_metrics(dataset, reconstruction, x_lim = None, verbose = True, plot = True):
    """
    Plot correlation, mse, mean.
    
    dataset: original data in segments * seq length * features ndarray.
    reconstruction: reconstructed data in the same shape as dataset.
    
    """
    
    # # Use only data in x_lim range.
    # if x_lim:
    #     dataset = dataset[x_lim[0]:x_lim[1], :]
    #     reconstruction = reconstruction[x_lim[0]:x_lim[1], :]
    
    num_seq = dataset.shape[0]
    num_features = dataset.shape[2]
    
    # Compute mse for each seq
    mse_all = ((dataset-reconstruction)**2).mean(axis=1)
    
    # Compute mean for each seq
    mean_all = np.mean(dataset, axis = 1)
    
    # Compute corr for each seq
    corr_all = []
    for ii in range(num_features):
        corr_channel = []
        for jj in range(num_seq):
            corr_seq = np.corrcoef(reconstruction[jj, :, ii], dataset[jj, :, ii])[0,1]
            corr_channel.append(corr_seq)
        corr_channel = np.array(corr_channel)
        corr_all.append(corr_channel)
    corr_all = np.array(corr_all).transpose()
    
    # Plot corr, mse, mean
    if plot:
        if num_features == 1:
            fig, axs = plt.subplots(2, 1, figsize = (20, 12))
        else:
            fig, axs = plt.subplots(num_features, 1, figsize = (20, num_features*6))

        times = np.max(mse_all, axis = 0)
        for ii in range(num_features):
            axs[ii].plot(corr_all[:,ii]*times[ii]/3, color = 'r', label = 'corr')
            axs[ii].plot(mse_all[:, ii], color = 'y', label = 'mse')
            axs[ii].plot(mean_all[:, ii], color = 'dimgray', label = 'mean', alpha = 0.7)
            axs[ii].set_title(f'# {ii+1}, mean corr = {np.mean(corr_all[:,ii]):.4f}, ' \
                              f'mean mse = {np.mean(mse_all[:, ii]):.4f}, ' \
                              f'mean = {np.mean(mean_all[:, ii]):.4f}')
            if x_lim:
                axs[ii].set_xlim(x_lim)
            axs[ii].legend()
    
    # Print the mean of corr, mse, mean    
    if verbose:
        corr_mean = np.mean(corr_all, axis = 0)
        mse_mean = np.mean(mse_all, axis = 0)
        mean_mean = np.mean(mean_all, axis = 0)
        
        for jj in range(num_features):
            print(f'Channel {jj+1}, corr = {corr_mean[jj]:.4f}, '\
                  f'mse = {mse_mean[jj]:4f}, mean = {mean_mean[jj]:.4f}.')
    
    return corr_mean, mse_mean, mean_mean

        
def pca_inverse(PCA_obj, reconstruction):
    """
    Convert reconstructed PCs back to channels.
    
    PCA_obj: fitted PCA.
    reconstruction: in segments * seq length * features ndarray.
    """
    
    seq_len = reconstruction.shape[1]
    num_features = reconstruction.shape[2]
    
    reconstruction = reconstruction.reshape(-1, num_features)
    recon_channel = PCA_obj.inverse_transform(reconstruction)
    recon_channel = recon_channel.reshape(-1, seq_len, 15)
    
    return recon_channel


def visualize(z_run, y, inv_bhvs, one_in = 4, perplexity=80, n_iter=3000):
    """
    Visualize latent space using PCA and tSNE
    
    z_run: latent values, n_segments * lateng length ndarray.
    y: label of each segment.
    inv_bhvs: dict from label to behavior classes.
    one_in: default use one in every 4 segments.
    perplexity, n_iter: for tSNE.
    """
    
    z_run_down = z_run[::one_in, :]
    label = y[::one_in, :]
    
    # z_run_pca = TruncatedSVD(n_components=2).fit_transform(z_run_down)
    z_run_pca = PCA(n_components=2).fit_transform(z_run_down)
    z_run_tsne = TSNE(perplexity=perplexity, min_grad_norm=1E-12, n_iter=n_iter).fit_transform(z_run_down)
    
    all_colors = ['b','g','r','c','m','y','darkgrey']
    fig, axs = plt.subplots(1,2, figsize=(20,10))

    for ii in np.unique(label):
        ii = int(ii)

        x_pca = z_run_pca[:,0].reshape(-1,1)[label == ii]
        y_pca = z_run_pca[:,1].reshape(-1,1)[label == ii]
        axs[0].scatter(x_pca, y_pca, c=all_colors[ii], marker='.', label = inv_bhvs[ii], linewidths=None)

        x_tsne = z_run_tsne[:,0].reshape(-1,1)[label == ii]
        y_tsne = z_run_tsne[:,1].reshape(-1,1)[label == ii]
        axs[1].scatter(x_tsne, y_tsne, c=all_colors[ii], marker='.', label = inv_bhvs[ii], linewidths=None)

    axs[0].set_title('PCA on z_run')
    axs[1].set_title('tSNE on z_run')
    axs[0].legend()
    axs[1].legend()
    plt.show()
