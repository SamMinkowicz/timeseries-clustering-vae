import torch
from torch.utils.data import TensorDataset
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from vrae import utils_sm, utils_VAE
from vrae.vrae import VRAE

# load model
model_dir = r"E:\sam\timeseries-clustering-vae\model_dir\model2"
data_dir = r"E:\sam\anipose_out"

params = None
for f in os.scandir(model_dir):
    if f.name.startswith("params"):
        with open(f.path, "rb") as params_f:
            params = pickle.load(params_f)

assert params is not None

vrae = VRAE(
    sequence_length=params["seq_len"],
    number_of_features=params["n_features"],
    hidden_size=params["hidden_size"],
    hidden_layer_depth=params["hidden_layer_depth"],
    latent_length=params["latent_length"],
    batch_size=params["batch_size"],
    learning_rate=params["learning_rate"],
    n_epochs=params["n_epochs"],
    dropout_rate=params["dropout_rate"],
    optimizer=params["optimizer"],
    cuda=params["cuda"],
    print_every=params["print_every"],
    val_every=params["val_every"],
    clip=params["clip"],
    max_grad_norm=params["max_grad_norm"],
    loss=params["loss"],
    block=params["block"],
    dload=params["model_dir"],
    output=params["output"],
    reduction=params["reduction"],
)

for f in os.scandir(model_dir):
    if f.name.startswith("vrae"):
        vrae.load(f.path)

# load training data
X = utils_sm.load_training_data(
    data_dir,
    batch_size=params["batch_size"],
    seq_len=params["seq_len"],
    window_slide=params["window_slide"],
    trim=params["trim"],
)
print(f"Training data shape: {X.shape}")

# plot reconstruction of training data
reconstruction = utils_VAE.recon(vrae, X)

utils_VAE.plot_recon_long(
    X, reconstruction, save_filename=os.path.join(model_dir, "recon_long.png")
)
utils_VAE.plot_recon_single(
    X, reconstruction, save_filename=os.path.join(model_dir, "recon_single.png")
)

# plot reconstruction metrics
corr_mean, mse_mean, mean_mean = utils_VAE.plot_recon_metrics(
    X, reconstruction, verbose=False, plot=False
)
corr_all, R2_all, uqi, term1, term2, term3 = utils_VAE.recon_metrics2(X, reconstruction)
plt.figure(figsize=(8, 5))
plt.plot(corr_all, label="corr")
plt.plot(corr_all**2, label="corr^2", color="r", alpha=0.5)
plt.plot(R2_all, label="R2", color="b", alpha=0.5)
plt.plot(uqi, label="uqi")
plt.plot(term1, label="term1-corr", color="y", alpha=0.5)
plt.plot(term2, label="term2-luminance")
plt.plot(term3, label="term3-contrast")
plt.legend()
plt.xlabel("# channel")
plt.title("Reconstruction metrics")
plt.savefig(os.path.join(model_dir, "recon_metrics.png"))
plt.close()

# plot reconstruction of test data

# visualize the latent space
z_run = vrae.transform(TensorDataset(torch.from_numpy(X)))
fake_labels = np.zeros((X.shape[0]))
fake_behaviors = {0: "all"}
utils_VAE.visualize(
    z_run,
    fake_labels,
    fake_behaviors,
    save_filename=os.path.join(model_dir, "latent_space.png"),
)
