import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex
import copy
from torch import from_numpy
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset, Subset
from seaborn import clustermap
from sklearn.mixture import GaussianMixture


def compare_label_tSNE(dp, label1, inv1, name1, label2, inv2, name2,
                       one_in = 4, perplexity=80, n_iter=3000):
    """
    Plot tSNE of data, coloring with two set of labels.
    
    dp: num_datapoints by num_features ndarray.
    label1, label2: num_datapoints by 1 ndarray, containing int labels for each datapoint.
                    If more than 10 classes, don't plot legend.
    inv1, inv2: inverse label, dict with labels in key and class names in value, used for legend.
                If None, labeled with int label value.
    name1, name2: string for figure title.
    one_in: default use one in every 4 dp for tSNE.
    
    """
    
    # Downsample datapoints
    dp_down = dp[::one_in, :]
    label1 = label1.reshape(label1.shape[0],1)
    label2 = label2.reshape(label2.shape[0],1)
    label1 = label1[::one_in, :]
    label2 = label2[::one_in, :]
    
    # Fit tSNE
    dp_tsne = TSNE(perplexity=perplexity, min_grad_norm=1E-12, n_iter=n_iter).fit_transform(dp_down)
    
    # Plot
    # all_colors = ['b','g','r','c','m','y','darkgrey']
    fig, axs = plt.subplots(1,2, figsize=(20,10))
    
    # If inverse label not given, make a dict with the same int label as each key and velue
    if not inv1:
        inv1 = {int(k): k for k in np.unique(label)}
        
    # Set color
    # if not color1:
    #     color1 = all_colors
    color1 = ['b','g','r','c','m','y','darkgrey']
    
    # Plot first one    
    for ii in np.unique(label1):
        ii = int(ii)
        x_tsne = dp_tsne[:,0].reshape(-1,1)[label1 == ii]
        y_tsne = dp_tsne[:,1].reshape(-1,1)[label1 == ii]
        axs[0].scatter(x_tsne, y_tsne, c=color1[ii], marker='.', label = inv1[ii], linewidths=None)
        # if len(np.unique(label1)) < 11:
        axs[0].legend()
    
    # Same
    if not inv2:
        inv2 = {int(k): k for k in np.unique(label2)}
    
    # Set colors
    # if not color2:
    #     color2 = all_colors
    cmap = get_cmap('nipy_spectral')
    num = np.arange(0, 1, 1/(np.max(label2)+1))
    color2 = []
    for ii in range(np.max(label2)+1):
        c = to_hex(cmap(num[ii]))
        color2.append(c)    
        
    for jj in np.unique(label2):
        jj = int(jj)
        x_tsne = dp_tsne[:,0].reshape(-1,1)[label2 == jj]
        y_tsne = dp_tsne[:,1].reshape(-1,1)[label2 == jj]
        axs[1].scatter(x_tsne, y_tsne, c=color2[jj], marker='.', label = inv2[jj], linewidths=None)
        # if len(np.unique(label2)) < 11:
        axs[1].legend()

    axs[0].set_title(name1)
    axs[1].set_title(name2)
    plt.show()
    
    
def plot_confusion_matrix(label1, label2, tick_label, sum_by = 'col', h_clustering = True, figsize = (18,6)):
    """
    Plot confusion matric, where the color is the probability of label2 belongs to label1.
    
    ##### Need to change the xy tick labels #####
    
    label1, label2: num_datapoints by 1 ndarray, containing int labels for each datapoint.
                    If more than 10 classes, don't plot legend.
    tick_label is only given when label1 is true label and as bhvs.keys().
    """
    
    label1 = label1.reshape(label1.shape[0])
    label2 = label2.reshape(label2.shape[0])
    
    n1 = np.unique(label1).shape[0]
    n2 = np.unique(label2).shape[0]
    confusion = np.zeros((n1, n2))
    
    for ii in range(n1):
        class1 = np.unique(label1)[ii]
        for jj in range(n2):
            class2 = np.unique(label2)[jj]
            nn = sum(np.logical_and((label1 == class1), (label2 == class2)))
            confusion[ii, jj] = nn
    if sum_by == 'row':        
        confusion_prob = confusion.T / np.sum(confusion, axis=1)
        confusion_prob = confusion_prob.T
    elif sum_by == 'col':
        confusion_prob = confusion / np.sum(confusion, axis=0)
    #plt.matshow(confusion_prob[:, :])
    if h_clustering:
        ax = clustermap(confusion_prob, row_cluster=False, xticklabels=1, figsize=figsize)
        ax.ax_cbar.set_position((0.15, .2, .03, .4))
        if tick_label:
            ax.ax_heatmap.set_yticklabels(tick_label, rotation=0)
    else:
        prob = np.argmax(confusion_prob, axis = 0)
        idx = np.argsort(prob)
        plt.matshow(confusion_prob[:, idx][:, ::-1])
        
        
def gmm_bic(data, num_cluster, num_iteration = 5, next = 5, fit = True, plot = True):
    """
    Find the cluster number for gmm using BIC. If the BIC for next several cluster numbers after nn are all higher, then 
    choose nn as cluster number. Fit gmm if fit==True.
    
    data: n_sample * n_feature ndarray
    num_cluster: list of int [start, end].
    num_iteration: int, calculate num_iteration yimes for each cluster number.
    nest: int, number of #cluster after nn that are higher than nn, for nn to be selected.
    fit: if true, fit gmm on data, and return labels.
    plot: if true, plot BIC scores.
    
    return:
    all_bic: list of bic scores for all cluster number
    all_bic_err: list of bic scores std for all cluster number
    cluster: the cluster number to use
    gmm_labels: gmm labels fitted using selected cluster. If fit=False, return None.
    """
    
    # Store the output
    all_bic = []
    all_bic_err = []
    gmm_labels = None
    
    if num_cluster[0] < 1:
        raise ValueError('Cluster number starts from 1')
    else:
        start = num_cluster[0]
        end = num_cluster[1]
        
    # Number of clusters whose bic is larger then the smallest one
    count = 0
    
    # [smallest bic, # clusters]
    curr_bic = [None, None]
    
    # For each cluster number to test
    for nn in range(start, end+1):
        # print(nn)
        bic_temp = []
        for ii in range(num_iteration):
            gmm = GaussianMixture(n_components=nn, max_iter=1000)
            gmm.fit(data)
            bic_temp.append(gmm.bic(data))
        bic_mean = np.mean(bic_temp)
        bic_err = np.std(bic_temp)
        all_bic.append(bic_mean)
        all_bic_err.append(bic_err)
	
	# If not the first one in loop
        if curr_bic[0]:
            # If current bic is larger than the smallest one, count + 1
            if bic_mean > curr_bic[0]:
                count += 1
            # If current bic is smaller then the smallest one, reset count to 0 and count from current nn
            else:
                curr_bic[0] = bic_mean
                curr_bic[1] = nn
                count = 0
        # If the first one in loop
        else:
            #bic_previous = bic_mean
            curr_bic[0] = bic_mean
            curr_bic[1] = nn
            
        if count == next or nn == end:
            
            if count == next:
                print(f'Number of clusters: {curr_bic[1]}.')
            if nn == end:
                print(f'Number of clusters not found, use {curr_bic[1]}.')
            
            if plot:
                #print(start, nn+1, list(range(start, nn+1)), all_bic, all_bic_err)
                plt.errorbar(range(start, nn+1), all_bic, yerr=all_bic_err, label='BIC')
                plt.title("BIC Score")
                plt.xticks(range(start, nn+1))
                plt.xlabel("# clusters")
                plt.ylabel("Score")
                plt.show()
            
            if fit:
                gmm_labels = GaussianMixture(n_components=curr_bic[1], random_state=111, max_iter=1000).fit_predict(data)
                gmm_labels = np.reshape(gmm_labels, (gmm_labels.shape[0], 1))
                
            return all_bic, all_bic_err, curr_bic[1], gmm_labels
