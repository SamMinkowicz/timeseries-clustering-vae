# Set which gpu to use
import os

GPU_TO_USE = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_TO_USE}"

from vrae.vrae import VRAE
from vrae import utils_sm
import torch
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset
import time
from datetime import datetime
import numpy as np

# Download dir
model_dir = r"/home/sam/timeseries-clustering-vae/model_dir"
data_dir = r"/media/storage/sam/anipose_out"

model_dir = r"E:\sam\timeseries-clustering-vae\model_dir\model"
data_dir = r"E:\sam\anipose_out"

model_i = 0
while os.path.exists(f"{model_dir}{model_i}"):
    model_i += 1

model_dir = f"{model_dir}{model_i}"

# Hyper parameters
seq_len = 200
window_slide = 5  # options: int >= 0
hidden_size = 256
hidden_layer_depth = 3
latent_length = 24
batch_size = 32
learning_rate = 0.00002
n_epochs = 1000
dropout_rate = 0.0
optimizer = "Adam"  # options: ADAM, SGD
cuda = True  # options: True, False
print_every = 50
val_every = 1000
clip = True  # options: True, False
max_grad_norm = 5
loss = "MSELoss"  # options: SmoothL1Loss, MSELoss
block = "LSTM"  # options: LSTM, GRU
output = False
reduction = "mean"
trim = True
checkpoint_every = 50
pad = True
only_freq = True
omega0 = 5
num_periods = 15
sampling_freq = 125
max_f = 30
min_f = 1

# Load training data
X = utils_sm.load_training_data(
    data_dir,
    batch_size=batch_size,
    seq_len=seq_len,
    window_slide=window_slide,
    trim=trim,
    pad=pad,
    only_freq=True,
    omega0=omega0,
    num_periods=num_periods,
    sampling_freq=sampling_freq,
    max_f=max_f,
    min_f=min_f,
    gpu=GPU_TO_USE,
)
print(f"Training data shape: {X.shape}")
dataset = TensorDataset(torch.from_numpy(X))

n_features = X.shape[2]

# ### Initialize VRAE object
#
# VRAE inherits from `sklearn.base.BaseEstimator` and overrides `fit`, `transform` and `fit_transform` functions, similar to sklearn modules

from vrae.vrae import VRAE

vrae = VRAE(
    sequence_length=seq_len,
    number_of_features=n_features,
    hidden_size=hidden_size,
    hidden_layer_depth=hidden_layer_depth,
    latent_length=latent_length,
    batch_size=batch_size,
    learning_rate=learning_rate,
    n_epochs=n_epochs,
    dropout_rate=dropout_rate,
    optimizer=optimizer,
    cuda=cuda,
    print_every=print_every,
    val_every=val_every,
    clip=clip,
    max_grad_norm=max_grad_norm,
    loss=loss,
    block=block,
    dload=model_dir,
    output=output,
    reduction=reduction,
    checkpoint_every=checkpoint_every,
)

# ### Fit the model onto dataset

# Cross validation:
# train_loss, train_mse, val_mse = crossval(dataset, vrae, batch_size, k = 5)
# with open(dload+'/meanrecloss_loss_mse', "wb") as fh:
#     pickle.dump([train_loss, train_mse, val_mse], fh)

# Training model:
time_training_started = time.strftime("%m%d%Y-%H%M%S")

# Save training hyperparameters
print("Saving training hyper-parameters...")
filename = f"params_{time_training_started}.pickle"
hyper_params_dict = {
    "model_dir": model_dir,
    "data_dir": data_dir,
    "seq_len": seq_len,
    "window_slide": window_slide,
    "hidden_size": hidden_size,
    "hidden_layer_depth": hidden_layer_depth,
    "latent_length": latent_length,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "n_epochs": n_epochs,
    "dropout_rate": dropout_rate,
    "optimizer": optimizer,
    "cuda": cuda,
    "print_every": print_every,
    "val_every": val_every,
    "clip": clip,
    "max_grad_norm": max_grad_norm,
    "loss": loss,
    "block": block,
    "output": output,
    "reduction": reduction,
    "n_segments": X.shape[0],
    "n_features": n_features,
    "trim": trim,
    "pad": pad,
    "only_freq": True,
    "omega0": omega0,
    "numPeriods": num_periods,
    "samplingFreq": sampling_freq,
    "maxF": max_f,
    "minF": min_f,
    "notes": "",
}
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
utils_sm.save_hyperparams(os.path.join(model_dir, filename), hyper_params_dict)

print_fmt = "%X"
started = time.strftime(print_fmt)
print(f"Starting training: {started}")
vrae.fit(
    train_dataset=dataset,
    val_dataset=None,
    checkpoint_filename_suffix=str(time_training_started),
)
ended = time.strftime(print_fmt)
print(f"Finished training: {ended}")
train_time = datetime.strptime(ended, print_fmt) - datetime.strptime(started, print_fmt)
print(f"Total train time: {str(train_time)}")
print(f"Avg time per epoch: {np.round(train_time.seconds / n_epochs, 2)} seconds")

# ### Plot loss and MSE
plt.figure(figsize=(8, 4.5))
plt.semilogy(vrae.all_loss, color="r", alpha=0.5, label="loss")
plt.semilogy(vrae.recon_loss, color="b", alpha=0.5, label="recon mse")
plt.semilogy(vrae.kl_loss, color="k", alpha=0.5, label="KL")
plt.legend()

plt.savefig(os.path.join(model_dir, f"train_loss_{time_training_started}"))

# plots for cross validation
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
# If the latent vectors have to be saved, pass the parameter `save`
print("Obtaining latent vectors for the training data...")
z_run = vrae.transform(
    dataset, save=True, filename=f"z_run_{time_training_started}.pkl"
)
# z_run = vrae.transform(dataset, save = False)
print(f"latent vector shape: {z_run.shape}")

# ### Save model
print("Saving trained model...")
vrae.save(f"vrae_{time_training_started}.pth")
