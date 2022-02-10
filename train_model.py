# Set which gpu to use
import os
from turtle import shape

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from vrae.vrae import VRAE
from vrae import utils_sm
import torch
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset
import time

# Download dir
model_dir = r"/home/sam/timeseries-clustering-vae/model_dir"
data_dir = r"/media/storage/sam/anipose_out"

model_dir = r"E:\sam\timeseries-clustering-vae\model_dir\model1"
data_dir = r"E:\sam\anipose_out"

# Hyper parameters
seq_len = 250
window_slide = 10  # options: int >= 0
hidden_size = 256
hidden_layer_depth = 3
latent_length = 16
batch_size = 32
learning_rate = 0.00002
n_epochs = 60
dropout_rate = 0.0
optimizer = "Adam"  # options: ADAM, SGD
cuda = True  # options: True, False
print_every = 5
val_every = 1000
clip = True  # options: True, False
max_grad_norm = 5
loss = "MSELoss"  # options: SmoothL1Loss, MSELoss
block = "LSTM"  # options: LSTM, GRU
output = False
reduction = "mean"
trim = True

# Load training data
X = utils_sm.load_training_data(
    data_dir,
    batch_size=batch_size,
    seq_len=seq_len,
    window_slide=window_slide,
    trim=trim,
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
)

# ### Fit the model onto dataset

# Cross validation:
# train_loss, train_mse, val_mse = crossval(dataset, vrae, batch_size, k = 5)
# with open(dload+'/meanrecloss_loss_mse', "wb") as fh:
#     pickle.dump([train_loss, train_mse, val_mse], fh)

# Training model:
time_training_started = time.strftime("%m%d%Y-%H%M%S")
print(f'Starting training: {time.strftime("%X")}')
vrae.fit(train_dataset=dataset, val_dataset=None)
print(f'Finished training: {time.strftime("%X")}')

# ### Plot loss and MSE
plt.figure(figsize=(8, 4.5))
plt.semilogy(vrae.all_loss, color="r", alpha=0.5, label="loss")
plt.semilogy(vrae.recon_loss, color="b", alpha=0.5, label="recon mse")
plt.semilogy(vrae.kl_loss, color="k", alpha=0.5, label="KL")
plt.legend()

if not os.path.exists(model_dir):
    os.mkdir(model_dir)
plt.savefig(os.path.join(model_dir, f"model_loss_{time_training_started}"))

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
z_run = vrae.transform(
    dataset, save=True, filename=f"z_run_{time_training_started}.pkl"
)
# z_run = vrae.transform(dataset, save = False)
print(f"latent vector shape: {z_run.shape}")

# ### Save model
vrae.save(f"vrae_{time_training_started}.pth")

# Save training hyperparameters
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
}
utils_sm.save_hyperparams(os.path.join(model_dir, filename), hyper_params_dict)
