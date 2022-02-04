import numpy as np


def load_training_data(data_path, batch_size=32, seq_len=250,
                       trim=True):
    """
    Load all dataset and preprocess.

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
    x_train: data in segments * seq length * features ndarray.
    """
    # load the np array
    # now is 15x3x45000
    raw_data = np.load(data_path)

    # TODO add other features

    n_limbs, n_coordinates, n_timepoints = raw_data.shape
    n_features = n_limbs*n_coordinates

    # next 45000x45
    x_train = raw_data.reshape((n_features, n_timepoints)).T

    # finally 180x250x45
    n_segments = n_timepoints // seq_len
    x_train = x_train.reshape((n_segments, seq_len, n_features))

    return x_train


if __name__ == '__main__':
    x_t = load_training_data(r'E:\sam\anipose4\poses_2021_9_4.npy')
    print(x_t.shape)
    print(x_t[0, 0, :])
