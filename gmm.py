import os
import pickle
import numpy as np
from vrae import utils_sm, utils_GMM
from vrae.post_gmm_merging import merge_gmm
from sklearn.mixture import GaussianMixture


# load latent vector
model_dir = r"E:\sam\timeseries-clustering-vae\model_dir\model3"

params = None
latent_vector = None
for f in os.scandir(model_dir):
    if f.name.startswith("params"):
        with open(f.path, "rb") as params_f:
            params = pickle.load(params_f)
    if f.name.startswith("z_run"):
        with open(f.path, "rb") as latent_f:
            latent_vector = pickle.load(latent_f)


# all_bic, all_bic_err, n_cluster, gmm_labels = utils_GMM.gmm_bic(
#     latent_vector, [10, 60], num_iteration=5, fit=False, plot=True
# )
print("kneed")
merge_info, gmm_labels, merged_labels, gmm_prob, merged_prob, merge_matrix = merge_gmm(
    latent_vector, 51, "kneed", fig_filename="kneed_gmm_merge"
)
np.savez(
    os.path.join(model_dir, "kneed_gmm_merge"),
    merge_info=np.array(list(merge_info.items())),
    gmm_labels=gmm_labels,
    merged_labels=merged_labels,
    gmm_prob=gmm_prob,
    merged_prob=merged_prob,
    merge_matrix=merge_matrix,
)
print("piecewise")
merge_info, gmm_labels, merged_labels, gmm_prob, merged_prob, merge_matrix = merge_gmm(
    latent_vector, 51, "piecewise", 3, fig_filename="piecewise_gmm_merge"
)
np.savez(
    os.path.join(model_dir, "piecewise_gmm_merge"),
    merge_info=np.array(list(merge_info.items())),
    gmm_labels=gmm_labels,
    merged_labels=merged_labels,
    gmm_prob=gmm_prob,
    merged_prob=merged_prob,
    merge_matrix=merge_matrix,
)

# gmm_labels = GaussianMixture(
#     n_components=51, random_state=111, max_iter=1000
# ).fit_predict(latent_vector)
# gmm_labels = np.reshape(gmm_labels, (gmm_labels.shape[0], 1))

# fake_labels = np.zeros((gmm_labels.shape))
# utils_GMM.compare_label_tSNE(
#     latent_vector, fake_labels, None, "fake", gmm_labels, None, "GMM labels"
# )
