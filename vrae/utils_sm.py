import os
import numpy as np
import pickle


def compute_limb_distance(poses, limb1, limb2, limb_dict):
    """compute the euclidean distance between the two limbs"""
    return np.linalg.norm(
        poses[limb_dict[limb1]] - poses[limb_dict[limb2]],
        axis=0
    )


def load_single_file(pose_path, limb_dict_path, seq_len):
    # load the np array
    # now is limbs x coordinates x timepoints
    raw_data = np.load(pose_path)

    n_limbs, n_coordinates, n_timepoints = raw_data.shape
    n_features = n_limbs*n_coordinates

    # now is timepoints x n_features
    x_train = raw_data.reshape((n_features, n_timepoints)).T

    # add frame difference feature
    difference = np.diff(x_train, axis=1)
    # pad with the last difference so will match the original shape
    difference = np.hstack((difference, difference[:, -1][:, None]))
    x_train = np.hstack((x_train, difference))
    n_features *= 2

    # add distance features
    with open(limb_dict_path, 'rb') as f:
        limb_dict = pickle.load(f)

    pairs = [('l_paw', 'l_eye'), ('l_paw', 'l_ear'),
             ('r_paw', 'r_eye'), ('r_paw', 'r_ear')]
    n_features += len(pairs)
    distances = np.empty((n_timepoints, len(pairs)))

    for i, pair in enumerate(pairs):
        distances[:, i] = compute_limb_distance(
            raw_data, pair[0], pair[1], limb_dict)

    x_train = np.hstack((x_train, distances))

    # finally is (timepoints / seq_length) x seq_length x n_features
    n_segments = n_timepoints // seq_len
    return x_train.reshape((n_segments, seq_len, n_features))


def load_training_data(data_dir, batch_size=32, seq_len=250,
                       trim=True):
    """
    Load dataset and preprocess.

    data will be put in form segments * seq_length * features.
    segments = data total time / seq_length
    features will be organized like: limb1_x, limb1_y, limb1_z, lib2_x, ...,
    followed by other features like paw ear distances

    seq_length: should be the same as when converting to vrae format. this is
                the window size that's used to window the data over time.
                Sequence length chosen at the first elbow of the
                autocorrelation of the keypoints over time. Was ~2 seconds
                with frame rate of 125/s so we choose 250.

    trim: if true, remove the extra part that can't be divided by batch size.

    Return:
    x_train: data in segments x seq length x features ndarray.
    """
    # TODO ensure matches batch size

    # get all the files
    all_x_train = []
    for f in os.scandir(data_dir):
        if f.name.endswith('npy'):
            limbs_file = f.name.replace('npy', 'pickle')
            limbs_file = limbs_file.replace('pose', 'limbs')
            limbs_path = os.path.join(data_dir, limbs_file)
            if not os.path.exists(limbs_path):
                continue
            all_x_train.append(
                load_single_file(f.path, limbs_path, seq_len))

    if all_x_train:
        return np.vstack(all_x_train)
    

if __name__ == '__main__':
    x_t = load_training_data(
        r'/home/sam/anipose_out')
    if x_t is not None:
        print(x_t.shape)
        # print(x_t[0, 0, :])
