import numpy as np
import pickle


def compute_limb_distance(poses, limb1, limb2, limb_dict):
    """compute the euclidean distance between the two limbs"""
    return np.linalg.norm(
        poses[limb_dict[limb1]] - poses[limb_dict[limb2]],
        axis=0
    )


def load_training_data(data_path, limb_dict_path, batch_size=32, seq_len=250,
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
    # TODO combine input data from multiple files
    # TODO add neighboring frame difference feature

    # load the np array
    # now is limbs x coordinates x timepoints
    raw_data = np.load(data_path)

    n_limbs, n_coordinates, n_timepoints = raw_data.shape
    n_features = n_limbs*n_coordinates

    # now is timepoints x n_features
    x_train = raw_data.reshape((n_features, n_timepoints)).T

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
    x_train = x_train.reshape((n_segments, seq_len, n_features))

    return x_train


if __name__ == '__main__':
    x_t = load_training_data(
        r'E:\sam\anipose4\poses_2021_9_4.npy',
        r'E:\sam\anipose4\limb_dict.pickle')
    print(x_t.shape)
    print(x_t[0, 0, :])
