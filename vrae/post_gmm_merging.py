"""
Adapted from https://github.com/AllenInstitute/drcme/blob/master/drcme/post_gmm_merging.py
Use function merge_gmm
"""

"""
The :mod:`drcme.post_gmm_merging` module contains functions for merging
Gaussian mixture model components together based on an entropy criterion
as in `Baudry et al. (2010)
<https://www.tandfonline.com/doi/abs/10.1198/jcgs.2010.08111>`_.
"""

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import scipy
import os.path
import joblib
from scipy.cluster import hierarchy
# import argschema as ags
import logging
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from kneed import KneeLocator

#class MergeParameters(ags.ArgSchema):
#    project = ags.fields.String(default="T301")
#    gmm_types = ags.fields.List(ags.fields.String, default=["diag", "diag", "diag"])


def log_p(x):
    return np.log(np.maximum(x, 1e-300))


def entropy_merges(results_dir, project="T301", gmm_type="diag", piecewise_components=2):
    data = pd.read_csv(os.path.join("..", "dev", results_dir, "sparse_pca_components_{:s}.csv".format(project)), index_col=0)
    gmm = joblib.load(os.path.join("..", "dev", results_dir, "best_gmm_{:s}.pkl".format(project)))
    my_gmm = gmm[gmm_type]
    combo = data.values
    tau = my_gmm.predict_proba(combo)
    labels = my_gmm.predict(combo)
    K_bic = max(labels) + 1

    return entropy_combi(tau_labels, K_bic, piecewise_components=piecewise_components)


def entropy_specific_merges(tau, labels, K_bic, clusters_to_merge):
    """Merge set of specified clusters by entropy criterion

    Parameters
    ----------
    tau : array
        Cluster membership probabilities
    labels : array
        Cluster assignments
    K_bic : int
        Number of original clusters (before merging)
    clusters_to_merge : array
        Set of clusters to merge

    Returns
    -------
    merge_info : dict
        Contains information about what was merged: entropies at each
        merge point ("entropies"), sequence of merges by indices as the
        matrix changes size ("merge_sequence"), sequence of merges by
        original column name ("merges_by_name"), original cluster number
        ("K_bic"), sequence of cluster numbers ("Ks"), number of
        cumulative points merged ("cumul_merges")
    new_labels : array
        Labels after merging
    tau_merged : array
        Cluster membership probabilities after merging
    merge_matrix : array
        Matrix of relationships between original clusters and merged clusters
    """
    entropy = -np.sum(np.multiply(tau, log_p(tau)))
    prior_merges = []
    merges_by_names = []
    entropies = [entropy]
    Ks = [K_bic]
    n_to_merge = len(clusters_to_merge)
    n_merge_tracker = [0]
    orig_col_names = np.arange(K_bic)

    for K in range(K_bic, K_bic - n_to_merge, -1):
        merge_matrix = np.identity(K_bic, dtype=int)
        merge_col_names = orig_col_names.copy()
        for merger in prior_merges:
            i, j = merger
            merge_matrix[:, i] = merge_matrix[:, i] | merge_matrix[:, j]
            mask = np.ones(merge_matrix.shape[1], dtype=bool)
            mask[j] = False
            merge_matrix = merge_matrix[:, mask]
            merge_col_names = merge_col_names[mask]
        labels = np.argmax(np.dot(tau, merge_matrix), axis=1)

        ent_current = np.inf
        candidate_cols = np.arange(K)[np.isin(merge_col_names, clusters_to_merge)]
        logging.debug("candidates left")
        logging.debug(merge_col_names[np.isin(merge_col_names, clusters_to_merge)])
        remaining_cols = np.arange(K)[~np.isin(merge_col_names, clusters_to_merge)]
        for i in remaining_cols:
            for j in candidate_cols:
                new_merge_matrix = merge_matrix.copy()
                new_merge_matrix[:, i] = merge_matrix[:, i] | merge_matrix[:, j]
                mask = np.ones(K, dtype=bool)
                mask[j] = False
                new_merge_matrix = new_merge_matrix[:, mask]
                tau_m = np.dot(tau, new_merge_matrix)
                ent = -np.sum(np.multiply(tau_m, log_p(tau_m)))
                if ent < ent_current:
                    ent_current = ent
                    merger = (i, j)
                    n_merged = np.sum(labels == i) + np.sum(labels == j)
        prior_merges.append(merger)
        merges_by_names.append([merge_col_names[i].item() for i in merger])
        entropies.append(ent_current)
        Ks.append(K-1)
        n_merge_tracker.append(n_merged)

    merge_info = {
        "entropies": entropies,
        "merge_sequence": prior_merges,
        "merges_by_names": merges_by_names,
        "K_bic": K_bic,
        "Ks": Ks,
        "cumul_merges": np.cumsum(n_merge_tracker)
    }

    merge_matrix = np.identity(K_bic, dtype=int)
    for merger in prior_merges:
        i, j = merger
        merge_matrix[:, i] = merge_matrix[:, i] | merge_matrix[:, j]
        mask = np.ones(merge_matrix.shape[1], dtype=bool)
        mask[j] = False
        merge_matrix = merge_matrix[:, mask]

    tau_merged = np.dot(tau, merge_matrix)
    new_labels = np.argmax(tau_merged, axis=1)

    return merge_info, new_labels, tau_merged, merge_matrix


def entropy_combi(tau, labels, K_bic, stop_method = 'kneed', piecewise_components=2, stop_at = 1):
    """Merge clusters by entropy criterion and piecewise fit

    Parameters
    ----------
    tau : array
        Cluster membership probabilities
    labels : array
        Cluster assignments
    K_bic : int
        Number of original clusters (before merging)
    stop_method : 'kneed' or 'piecewise'
        Method used to find the stop merging point. If 'kneed', piecewise fit params are ignored.
    piecewise_components : {2, 3}, optional
        Number of components of linear piecewise fit

    Returns
    -------
    merge_info : dict
        Contains information about what was merged: entropies at each
        merge point ("entropies"), sequence of merges by indices as the
        matrix changes size ("merge_sequence"), sequence of merges by
        original column name ("merges_by_name"), original cluster number
        ("K_bic"), sequence of cluster numbers ("Ks"), number of
        cumulative points merged ("cumul_merges")
    new_labels : array
        Labels after merging
    tau_merged : array
        Cluster membership probabilities after merging
    merge_matrix : array
        Matrix of relationships between original clusters and merged clusters
    """
    entropy = -np.sum(np.multiply(tau, log_p(tau)))
    prior_merges = []
    entropies = [entropy]
    Ks = [K_bic]
    n_merge_tracker = [0]
    if K_bic <= 3:
        print("Too few clusters to assess merging")
        merge_info = {
            "entropies": entropies,
            "cumul_merges": None,
            "best_fits": None,
#             "fit2": None,
            "cp": None,
            "merge_sequence": None,
            "K_bic": K_bic,
        }
        return merge_info, labels, tau, None

    for K in range(K_bic, 1, -1):
        
        ### Merge original clusters following previous results 
        # A matrix to indicate which pairs of clusters are merged at previous step
        merge_matrix = np.identity(K_bic, dtype=int)
        # For each pair of merged clusters, replace one of the corresponding column with the sum of the two col
        for merger in prior_merges:
            i, j = merger
            merge_matrix[:, i] = merge_matrix[:, i] | merge_matrix[:, j]
            mask = np.ones(merge_matrix.shape[1], dtype=bool)
            mask[j] = False
            merge_matrix = merge_matrix[:, mask]
        # Update labels
        labels = np.argmax(np.dot(tau, merge_matrix), axis=1)
        
        ### Find the two clusters to merge at current step
        # Store current lowest entropy
        ent_current = np.inf
        # For each pair of clusters
        for i in range(K-1):
            for j in range(i + 1, K):
                # The same step as above, but in a copy, to see the entropy after merging
                new_merge_matrix = merge_matrix.copy()
                new_merge_matrix[:, i] = merge_matrix[:, i] | merge_matrix[:, j]
                mask = np.ones(K, dtype=bool)
                mask[j] = False
                new_merge_matrix = new_merge_matrix[:, mask]
                tau_m = np.dot(tau, new_merge_matrix)
                ent = -np.sum(np.multiply(tau_m, log_p(tau_m)))
                # If the entropy decrease, store the current entropy, the two clusters to merge, number of points in teh two clusters
                # Repeat for all the pairs to find the lowest entropy after merge
                if ent < ent_current:
                    ent_current = ent
                    merger = (i, j)
                    n_merged = np.sum(labels == i) + np.sum(labels == j)
        prior_merges.append(merger)

        entropies.append(ent_current)
        Ks.append(K-1)
        n_merge_tracker.append(n_merged)

    cumul_merges = np.cumsum(n_merge_tracker)
    
    if stop_method == 'piecewise':
        best_fits, cp = fit_piecewise(cumul_merges, entropies, piecewise_components)
        stop_point = cp[stop_at-1]
        merge_info = {
            "entropies": entropies,
            "cumul_merges": cumul_merges,
            "best_fits": best_fits,
#             "fit2": fit2,
            "cp": cp,
            "merge_sequence": prior_merges[:stop_point],
            "K_bic": K_bic,
        }
    elif stop_method == 'kneed':
        kneedle = KneeLocator(cumul_merges, entropies, S=5.0, curve='convex', direction='decreasing')
        stop_point = np.where(cumul_merges == kneedle.knee)[0][0]
        merge_info = {
            "entropies": entropies,
            "cumul_merges": cumul_merges,
#            "best_fits": best_fits,
#             "fit2": fit2,
#            "cp": cp,
            "stop_point": stop_point,
            "merge_sequence": prior_merges[:stop_point],
            "K_bic": K_bic,
        }
        
    ### The final merge matrix
    merge_matrix = np.identity(K_bic, dtype=int)
    for merger in prior_merges[:stop_point]:
        i, j = merger
        merge_matrix[:, i] = merge_matrix[:, i] | merge_matrix[:, j]
        mask = np.ones(merge_matrix.shape[1], dtype=bool)
        mask[j] = False
        merge_matrix = merge_matrix[:, mask]

    tau_merged = np.dot(tau, merge_matrix)
    new_labels = np.argmax(tau_merged, axis=1)

    return merge_info, new_labels, tau_merged, merge_matrix


def fit_piecewise(cumul_merges, entropies, n_parts):
    """ Fit entropy vs cumulative merge number with linear piecewise function

    Parameters
    ----------
    cumul_merges : array
        Number of cumulative samples merged at each merge step. Length
        must be `n_parts` + 2 or more and same length as `entropies`.
    entropies : array
        Entropy values at each merge step. Length must be
        `n_parts` + 2 or more and same length as `cumul_merges`.
    n_parts : {2, 3}
        Number of components for linear fit

    Returns
    -------
    best_fits : tuple
        Fit parameters for each component. Length of tuple equals
        `n_parts`
    cp : tuple
        Locations of change point(s) for linear fits. Length of tuple
        equals `n_parts` - 1
    """
    total_err = np.inf

    if len(entropies) < n_parts + 2:
        logging.info("Not enough clusters for piecewise fit")
        return (None,), (0,)

    if n_parts == 2:
        for c in range(1, len(entropies)-1):
            x1 = cumul_merges[:c + 1]
            x2 = cumul_merges[c:]
            y1 = entropies[:c + 1]
            y2 = entropies[c:]

            A1 = np.vstack([x1, np.ones(len(x1))]).T
            A2 = np.vstack([x2, np.ones(len(x2))]).T

            fit1 = np.linalg.lstsq(A1, y1, rcond=-1)
            fit2 = np.linalg.lstsq(A2, y2, rcond=-1)

            err = fit1[1] + fit2[1]

            if err < total_err:
                total_err = err
                best_fits = (fit1[0], fit2[0])
                cp = (c,)
    elif n_parts == 3:
        for c in range(1, len(entropies) - 3):
            for d in range(c + 1, len(entropies)-1):
                x1 = cumul_merges[:c + 1]
                x2 = cumul_merges[c:d + 1]
                x3 = cumul_merges[d:]
                y1 = entropies[:c + 1]
                y2 = entropies[c:d + 1]
                y3 = entropies[d:]

                A1 = np.vstack([x1, np.ones(len(x1))]).T
                A2 = np.vstack([x2, np.ones(len(x2))]).T
                A3 = np.vstack([x3, np.ones(len(x3))]).T

                fit1 = np.linalg.lstsq(A1, y1, rcond=-1)
                fit2 = np.linalg.lstsq(A2, y2, rcond=-1)
                fit3 = np.linalg.lstsq(A3, y3, rcond=-1)

                err = fit1[1] + fit2[1] + fit3[1]

                if err < total_err:
                    total_err = err
                    best_fits = (fit1[0], fit2[0], fit3[0])
                    cp = (c, d)
    else:
        raise("Wrong value for n_parts")

    return best_fits, cp


def order_new_labels(new_labels, tau_merged, data):
    """Reorder cluster labels by similarity of centroids

    Parameters
    ----------
    new_labels : array
        Cluster labels for samples
    tau_merged : array
        Cluster membership probability matrix to be reordered
    data : array
        Data matrix

    Returns
    -------
    new_labels_reorder : list
        Cluster labels for samples after labels have been reordered
    tau_merged : array
        Cluster membership probability matrix following new cluster order
    new_labels_reorder_dict : dict
        Mapping from previous labels (keys) to newly orderered labels (values)
    leaves : array
        Original labels as leaves of hierarchical clustering of centroids
    """
    uniq_labels = np.unique(new_labels)
    uniq_labels = uniq_labels[~np.isnan(uniq_labels)]
    n_cl = len(uniq_labels)
    centroids = np.zeros((n_cl, data.shape[1]))
    for i in range(n_cl):
        centroids[i, :] = np.mean(data[new_labels == i, :], axis=0)
    Z = hierarchy.linkage(centroids, method="ward")
    D = hierarchy.dendrogram(Z, no_plot=True)
    leaves = np.array(D["leaves"])
    new_labels_reorder_dict = {d: i for i, d in enumerate(leaves)}
    tau_merged = tau_merged[:, leaves]
    new_labels_reorder = [new_labels_reorder_dict[d] if not np.isnan(d) else d for d in new_labels]

    return new_labels_reorder, tau_merged, new_labels_reorder_dict, leaves
    

def plot_fit_piecewise(merge_info, piecewise_components, stop_at):

    cumul_merges = np.array(merge_info['cumul_merges'])
    entropies = np.array(merge_info['entropies'])
    best_fits = merge_info['best_fits']

    plt.figure(figsize=(7,5))
    plt.plot(cumul_merges, entropies, marker = 'o', color = 'b', markersize=4, linestyle='None')
    
    if piecewise_components == 2:
        stop = merge_info['cp'][0]
        plt.plot(cumul_merges[stop], entropies[stop], marker = 'o', color = 'r', markersize=4)
        plt.axvline(cumul_merges[stop], color = 'r', label = 'Stop point')
        
        x1 = cumul_merges[:stop + 1]
        x2 = cumul_merges[stop:]
        
        y1_hat = x1 * best_fits[0][0] + best_fits[0][1]
        y2_hat = x2 * best_fits[1][0] + best_fits[1][1]
        
        plt.plot(x1, y1_hat, color = 'k', label = 'piecewise fit')
        plt.plot(x2, y2_hat, color = 'k')
        
    elif piecewise_components == 3:
        stop1 = merge_info['cp'][0]
        stop2 = merge_info['cp'][1]
        plt.plot(cumul_merges[[stop1, stop2]], entropies[[stop1, stop2]], 
            marker = 'o', color = 'r', markersize=4, linestyle='None')
        plt.axvline(cumul_merges[merge_info['cp'][stop_at-1]], color = 'r', label = 'Stop point')
        
        x1 = cumul_merges[ : stop1 + 1]
        x2 = cumul_merges[stop1 : stop2 + 1]
        x3 = cumul_merges[stop2 : ]
        
        y1_hat = x1 * best_fits[0][0] + best_fits[0][1]
        y2_hat = x2 * best_fits[1][0] + best_fits[1][1]
        y3_hat = x3 * best_fits[2][0] + best_fits[2][1]
        
        plt.plot(x1, y1_hat, color = 'k', label = 'piecewise fit')
        plt.plot(x2, y2_hat, color = 'k')
        plt.plot(x3, y3_hat, color = 'k')
        
    plt.xlabel('# of cumulative merged points')
    plt.ylabel('Entropy after each merge')
    plt.legend()

def plot_kneed(merge_info):

    cumul_merges = np.array(merge_info['cumul_merges'])
    entropies = np.array(merge_info['entropies'])
    stop = merge_info['stop_point']

    plt.figure(figsize=(7,5))
    plt.plot(cumul_merges, entropies, marker = 'o', color = 'b', markersize=4, linestyle='None')    
    plt.axvline(cumul_merges[stop], color = 'r', label = 'Stop point')
    
    plt.xlabel('# of cumulative merged points')
    plt.ylabel('Entropy after each merge')
    plt.legend()


def merge_gmm(data, n_components, stop_method = 'kneed', piecewise_components=2, stop_at = 1, plot = True):
    """
    Merge clusters given by GMM.
    
    data: n_sample * n_feature nparray.
    n_components: number of clusters in gmm.
    stop_method: "piecewise" or "kneed", the method used to find the stop merging point.
        If 'kneed', piecewise_components and stop_at are ignored.
    piecewise_components: Number of components for linear fit, {2,3}.
    stop_at: stop merging at 1st or 2nd changing point in fitting.
    plot: if True, plot entropies to cumulative merged points, plot stop points.

    return:
    merge_info:
    gmm_labels, merged_labels: label before and after merging
    gmm_prob, merged_prob: probability of each sample belongs to each cluster
    merge_matrix: (# clusters before merge) * (# clusters after merge)
    """

    gmm = GaussianMixture(n_components, random_state=111, max_iter=1000)
    gmm_labels = gmm.fit_predict(data)
    # gmm_labels = np.reshape(gmm_labels, (gmm_labels.shape[0], 1))
    gmm_prob = gmm.predict_proba(data)
    
    merge_info, merged_labels, merged_prob, merge_matrix = entropy_combi(gmm_prob, gmm_labels, n_components, stop_method, piecewise_components, stop_at)
    
    try:
        stop = merge_info['cp'][stop_at-1]
    except:
        stop = merge_info['stop_point']
        
    print(f'Before merge: {n_components} clusters, after merge {stop} times: {np.max(merged_labels)+1} clusters.')
    
    if plot:
        if stop_method == 'kneed':
            plot_kneed(merge_info)
        elif stop_method == 'piecewise':
            plot_fit_piecewise(merge_info, piecewise_components, stop_at)

    return merge_info, gmm_labels, merged_labels, gmm_prob, merged_prob, merge_matrix
    

#def main():
#    module = ags.ArgSchemaParser(schema_type=MergeParameters)
#    project = module.args["project"]
#    gmm_types = module.args["gmm_types"]
#
#    sub_dirs = [s.format(project) for s in ["all_{:s}", "exc_{:s}", "inh_{:s}"]]
#    piecewise_components = [2, 2, 3]
#    for sub_dir, gmm_type, pw_comp in zip(sub_dirs, gmm_types, piecewise_components):
#        print("merging for ", sub_dir, "with", gmm_type)
#        merge_info, new_labels, tau_merged, _ = entropy_merges(sub_dir, project, gmm_type=gmm_type, piecewise_components=pw_comp)
#        print(merge_info)
#        data = pd.read_csv(os.path.join(sub_dir, "sparse_pca_components_{:s}.csv".format(project)), index_col=0)
#        new_labels, tau_merged, _, _ = order_new_labels(new_labels, tau_merged, data)
#
#        np.savetxt(os.path.join(sub_dir, "post_merge_proba.txt"), tau_merged)
#        np.save(os.path.join(sub_dir, "post_merge_cluster_labels.npy"), new_labels)
#        df = pd.read_csv(os.path.join(sub_dir, "all_tsne_coords_{:s}.csv".format(project)))
#        df["clustering_3"] = new_labels
#        df.to_csv(os.path.join(sub_dir, "all_tsne_coords_{:s}_plus.csv".format(project)))


#if __name__ == "__main__":
#    main()
