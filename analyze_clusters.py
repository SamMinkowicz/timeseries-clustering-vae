from cgi import test
import os
import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from vrae import utils_sm
from vrae.vrae import VRAE

MODEL_DIR = r"E:\sam\timeseries-clustering-vae\model_dir\model3"


def load_params(model_dir):
    # load training params
    params = None
    for f in os.scandir(model_dir):
        if f.name.startswith("params"):
            with open(f.path, "rb") as params_f:
                params = pickle.load(params_f)

    return params


def load_gmm_results(model_dir, merge_method):
    merge_methods = ["kneed", "piecewise"]
    if merge_method not in merge_methods:
        raise ValueError(f"merge method must be one of {merge_methods}")

    # load gmm labels
    gmm_merge_result = np.load(
        os.path.join(model_dir, f"{merge_method}_gmm_merge.npz"), allow_pickle=True
    )
    # merge_info = dict(gmm_merge_result["merge_info"])
    # gmm_labels = gmm_merge_result["gmm_labels"]
    # merged_labels = gmm_merge_result["merged_labels"]
    # gmm_prob = gmm_merge_result["gmm_prob"]
    # merged_prob = gmm_merge_result["merged_prob"]
    # merge_matrix = gmm_merge_result["merge_matrix"]
    return gmm_merge_result


# if the window stride < window length then
# convert gmm labels to labels per frame
def per_frame_labels(labels, params):
    seq_len = params["seq_len"]
    window_slide = params["window_slide"]
    if window_slide == seq_len:
        return labels

    if window_slide < seq_len:
        # there is 1 label for each segment so gmm_labels is the same shape as params['n_segments']
        # ignoring Trim: n_segments = ((n_frames - seq_len) / window_slide) + 1
        n_frames = ((params["n_segments"] - 1) * window_slide) + seq_len
        per_frame = np.zeros((n_frames))
        for i in range(labels.shape[0]):
            per_frame[i * seq_len : (i * seq_len) + seq_len] = labels[i]

        # the prediction for a given frame is taken from the prediction from when the window centered over that frame
        # without padding, we have to skip the first and last window_length/2 frames
        # we will make the window_slide == 1 when testing/predicting on new data

    print(1)


def get_bout_lengths(labels):
    """
    Let a bout be some number of consecutive frames with the same cluster.
    Count the length of each bout for each cluster.
    Return a dictionary mapping clusters to a list of bout lengths
    """
    counts = {label: [] for label in np.unique(labels)}
    for frame, label in enumerate(np.squeeze(labels)):
        if frame == 0:
            previous_label = label
            count = 1
            continue

        if label == previous_label:
            count += 1
        else:
            counts[previous_label].append(count)
            previous_label = label
            count = 1

    counts[label].append(count)

    return counts


def get_bout_frames(labels):
    """
    Let a bout be some number of consecutive frames with the same cluster.
    Count the length of each bout for each cluster.
    Return dictionary mapping clusters to a list of lists of
    start and end frame indices for each bout
    """
    frames = {label: [] for label in np.unique(labels)}
    for frame, label in enumerate(np.squeeze(labels)):
        if frame == 0:
            previous_label = label
            frames[label].append([frame])
            continue

        if label != previous_label:
            frames[previous_label][-1].append(frame - 1)
            frames[label].append([frame])
            previous_label = label

    frames[label][-1].append(frame)

    return frames


def get_gmm_clusters(model_dir):
    # load data
    gmm_labels_file = os.path.join(model_dir, "inference_gmm_labels.npy")
    if os.path.exists(gmm_labels_file):
        return np.load(gmm_labels_file)

    DATA_DIR = r"E:\sam\anipose_out"
    params = load_params(model_dir)

    assert params is not None

    X = utils_sm.load_training_data(
        DATA_DIR,
        batch_size=params["batch_size"],
        seq_len=params["seq_len"],
        window_slide=1,
        trim=True,
        pad=True,
    )
    print(f"Data shape: {X.shape}")

    # load model
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

    # transform data
    latent_vector = vrae.transform(TensorDataset(torch.from_numpy(X)))

    # get labels
    N_COMPONENTS = 51
    gmm_labels = GaussianMixture(
        n_components=N_COMPONENTS, random_state=111, max_iter=1000
    ).fit_predict(latent_vector)
    gmm_labels = np.reshape(gmm_labels, (gmm_labels.shape[0], 1))
    np.save(gmm_labels_file, gmm_labels)
    return gmm_labels


def plot_bout_lengths(bouts, n_rows, n_cols):
    """
    plot distribution of bout lengths. Plots bout lengths for all clusters together
    and then each separate.
    """
    # plot distribution of cluster bout lengths
    # all_bout_lengths = np.hstack(bouts.values())
    # plt.hist(all_bout_lengths, bins=500)
    # plt.title("all bout lengths")
    # plt.xlabel("Bout length")
    # plt.show()

    # plot distribution of cluster bout lengths separately for each cluster
    cluster = 0
    n_clusters = len(bouts.keys()) - 1

    fig, axs = plt.subplots(n_rows, n_cols)
    for row in range(n_rows):
        for col in range(n_cols):
            if cluster > n_clusters:
                break
            axs[row, col].hist(bouts[cluster])
            cluster += 1
    plt.tight_layout()
    plt.show()


def plot_motif_usage(labels, out_filename=None):
    """plot the usage % for each motif"""
    labels = np.squeeze(labels)
    unique_labels = np.unique(labels)
    total_frames = labels.size
    usage = [
        100 * np.round(np.count_nonzero(labels == label) / total_frames, 2)
        for label in unique_labels
    ]
    usage = np.array(usage)
    plt.plot(usage[np.argsort(-usage)])
    plt.xlabel("Motif")
    plt.ylabel("Usage percent")

    if out_filename:
        plt.savefig(out_filename)
        plt.close()
    else:
        plt.show()


def test_animation():
    data_dir = r"E:\sam\anipose_out"

    for f in os.scandir(data_dir):
        if f.name.endswith("npy"):
            limbs_file = f.name.replace("npy", "pickle")
            limbs_file = limbs_file.replace("pose", "limbs")
            limbs_path = os.path.join(data_dir, limbs_file)

            if not os.path.exists(limbs_path):
                continue

            poses = np.load(f.path)
            with open(limbs_path, "rb") as f:
                limb_dict = pickle.load(f)
            break

    connected = [
        ("snout", "l_eye"),
        ("l_eye", "r_eye"),
        ("snout", "r_eye"),
        ("l_eye", "l_ear"),
        ("r_eye", "r_ear"),
        ("l_paw", "l_wrist"),
        ("l_wrist", "l_elbow"),
        ("l_elbow", "l_shoulder"),
        ("l_shoulder", "l_hindpaw"),
        ("r_paw", "r_wrist"),
        ("r_wrist", "r_elbow"),
        ("r_elbow", "r_shoulder"),
        ("r_shoulder", "r_hindpaw"),
    ]

    class PoseAnimation:
        def __init__(self, ax, limits) -> None:
            self.ax = ax

            self.ax.set_xlabel("X")
            self.ax.set_xlabel("Y")
            self.ax.set_xlabel("Z")

            self.ax.set_xlim(limits[:2])
            self.ax.set_ylim(limits[2:4])
            self.ax.set_zlim(limits[4:])

            self.lines = [
                self.ax.plot([], [], [], color=f"C{i}", linewidth=3)[0]
                for i, pair in enumerate(connected)
            ]

        def __call__(self, j):
            for line, pair in zip(self.lines, connected):
                line.set_data(
                    [poses[limb_dict[pair[i]], 0, j] for i in range(2)],
                    [poses[limb_dict[pair[i]], 1, j] for i in range(2)],
                )
                line.set_3d_properties(
                    [poses[limb_dict[pair[i]], 2, j] for i in range(2)],
                )
            return self.lines

    n_frames = 20

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    limits = [
        np.min(poses[:, 0]),
        np.max(poses[:, 0]),
        np.min(poses[:, 1]),
        np.max(poses[:, 1]),
        np.min(poses[:, 2]),
        np.max(poses[:, 2]),
    ]

    pose_anim = PoseAnimation(ax, limits)
    anim = animation.FuncAnimation(
        fig, pose_anim, frames=n_frames, interval=100, blit=True
    )
    # plt.show()
    out_filename = r"E:\sam\timeseries-clustering-vae\test_anim.gif"
    anim.save(out_filename, animation.PillowWriter(fps=20))
    plt.close()


def plot_cluster_example(poses, limb_dict, out_path):
    """
    save an animated plot of a bout
    poses will be in shape limbs x coordinates x time
    """
    connected = [
        ("snout", "l_eye"),
        ("l_eye", "r_eye"),
        ("snout", "r_eye"),
        ("l_eye", "l_ear"),
        ("r_eye", "r_ear"),
        ("l_paw", "l_wrist"),
        ("l_wrist", "l_elbow"),
        ("l_elbow", "l_shoulder"),
        ("l_shoulder", "l_hindpaw"),
        ("r_paw", "r_wrist"),
        ("r_wrist", "r_elbow"),
        ("r_elbow", "r_shoulder"),
        ("r_shoulder", "r_hindpaw"),
    ]

    class PoseAnimation:
        def __init__(self, ax, limits, n_frames) -> None:
            self.ax = ax

            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")

            self.ax.set_xlim(limits[:2])
            self.ax.set_ylim(limits[2:4])
            self.ax.set_zlim(limits[4:])
            self.ax.set_title(f"{n_frames} frames")

            self.lines = [
                self.ax.plot([], [], [], color=f"C{i}", linewidth=3)[0]
                for i, pair in enumerate(connected)
            ]

        def __call__(self, j):
            for line, pair in zip(self.lines, connected):
                line.set_data(
                    [poses[limb_dict[pair[i]], 0, j] for i in range(2)],
                    [poses[limb_dict[pair[i]], 1, j] for i in range(2)],
                )
                line.set_3d_properties(
                    [poses[limb_dict[pair[i]], 2, j] for i in range(2)],
                )
            return self.lines

    n_frames = poses.shape[2]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    limits = [
        np.min(poses[:, 0]),
        np.max(poses[:, 0]),
        np.min(poses[:, 1]),
        np.max(poses[:, 1]),
        np.min(poses[:, 2]),
        np.max(poses[:, 2]),
    ]

    pose_anim = PoseAnimation(ax, limits, n_frames)
    anim = animation.FuncAnimation(fig, pose_anim, frames=n_frames, blit=True)
    anim.save(out_path, animation.PillowWriter(fps=30))
    plt.close()


def plot_cluster_examples(labels, n_plots, out_dir, raw=True):
    """plot n example animations of each cluster"""
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # load the pose data
    data_dir = r"E:\sam\anipose_out"
    if raw:
        data_dir = os.path.join(data_dir, "raw")

    limb_dict = None
    pose_file_prefix = "pose" if not raw else "pose_raw"
    all_poses = []
    for f in os.scandir(data_dir):
        if f.name.endswith("npy"):
            limbs_file = f.name.replace("npy", "pickle")
            limbs_file = limbs_file.replace(pose_file_prefix, "limbs")
            limbs_path = os.path.join(data_dir, limbs_file)

            if not os.path.exists(limbs_path):
                continue

            all_poses.append(np.load(f.path))

            # only need to get this once
            if limb_dict:
                continue

            with open(limbs_path, "rb") as f:
                limb_dict = pickle.load(f)

    # this is now limbs x coordinates x frames
    poses = np.dstack(all_poses)
    all_bout_frames = get_bout_frames(labels)

    for cluster in np.unique(labels):
        print(f"cluster: {cluster}")
        for bout in range(n_plots):
            print(f"bout: {bout}")

            # check whether there are this many bouts for the given cluster
            if bout >= len(all_bout_frames[cluster]):
                break

            # get the pose data for the given bout
            bout_frames = all_bout_frames[cluster][bout]

            # skip if there is only one frame in the bout
            if bout_frames[0] == bout_frames[1]:
                print(f"Only one frame in cluster {cluster} bout: {bout}. Skipping...")
                continue
            bout_pose = poses[:, :, bout_frames[0] : bout_frames[1]]

            # pass the data and out_filename to plot_cluster_example
            gif_filename = f"anim_cluster{cluster}_bout{bout}.gif"
            if raw:
                gif_filename = gif_filename.replace(".gif", "_raw.gif")
            plot_cluster_example(
                bout_pose, limb_dict, os.path.join(out_dir, gif_filename)
            )


plot_cluster_examples(
    get_gmm_clusters(MODEL_DIR),
    5,
    os.path.join(MODEL_DIR, "gmm_cluster_gifs"),
    raw=True,
)
plot_cluster_examples(
    get_gmm_clusters(MODEL_DIR),
    5,
    os.path.join(MODEL_DIR, "gmm_cluster_gifs"),
    raw=False,
)


def plot_all_bout_lengths():
    gmm_labels = get_gmm_clusters(MODEL_DIR)
    bouts = get_bout_lengths(gmm_labels)
    plot_bout_lengths(bouts, 11, 5)

    kneed_merged = load_gmm_results(MODEL_DIR, "kneed")["merged_labels"]
    kneed_bouts = get_bout_lengths(kneed_merged)
    plot_bout_lengths(kneed_bouts, 6, 4)

    piece_merged = load_gmm_results(MODEL_DIR, "piecewise")["merged_labels"]
    piece_bouts = get_bout_lengths(piece_merged)
    plot_bout_lengths(piece_bouts, 8, 5)


def plot_all_usage():
    """
    plot the usage percent for each motif for the gmm labels
    and for the labels after the two types of merging
    """
    gmm_labels = get_gmm_clusters(MODEL_DIR)
    kneed_merged = load_gmm_results(MODEL_DIR, "kneed")["merged_labels"]
    piece_merged = load_gmm_results(MODEL_DIR, "piecewise")["merged_labels"]
    labels = [gmm_labels, kneed_merged, piece_merged]
    filenames = [
        os.path.join(MODEL_DIR, filename)
        for filename in [
            "gmm_labels_usage",
            "kneed_labels_usage",
            "piecewise_labels_usage",
        ]
    ]

    for i in range(3):
        plot_motif_usage(labels[i], filenames[i])


# if __name__ == "__main__":
