# %%
import os
import random

import numpy as np
from numpy.testing import assert_array_almost_equal
import torch
from sklearn.model_selection import train_test_split


def build_uniform_P(size, noise):
    """ The noise matrix flips any class to any other with probability
    noise / (#class - 1).
    """

    assert (noise >= 0.) and (noise <= 1.)

    P = np.float64(noise) / np.float64(size - 1) * np.ones((size, size))
    np.fill_diagonal(P, (np.float64(1) - np.float64(noise)) * np.ones(size))

    diag_idx = np.arange(size)
    P[diag_idx, diag_idx] = P[diag_idx, diag_idx] + 1.0 - P.sum(0)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def build_pair_p(size, noise):
    assert (noise >= 0.) and (noise <= 1.)
    P = (1.0 - np.float64(noise)) * np.eye(size)
    for i in range(size):
        P[i, i - 1] = np.float64(noise)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def noisify(y, p_minus, p_plus=None, random_state=0):
    """ Flip labels with probability p_minus.
    If p_plus is given too, the function flips with asymmetric probability.
    """

    assert np.all(np.abs(y) == 1)

    m = y.shape[0]
    new_y = y.copy()
    coin = np.random.RandomState(random_state)

    if p_plus is None:
        p_plus = p_minus

    # This can be made much faster by tossing all the coins and completely
    # avoiding the loop. Although, it is not simple to write the asymmetric
    # case then.
    for idx in np.arange(m):
        if y[idx] == -1:
            if coin.binomial(n=1, p=p_minus, size=1) == 1:
                new_y[idx] = -new_y[idx]
        else:
            if coin.binomial(n=1, p=p_plus, size=1) == 1:
                new_y[idx] = -new_y[idx]

    return new_y


def generate_noise_labels(train_labels, num_classes, data_type, flip_rate, seed, train_data=None):
    np.random.seed(seed)
    train_labels = np.array(train_labels)
    if len(train_labels) <= 0 or flip_rate == 0:
        return train_labels
    if data_type == 'uniform':
        T_diag = np.eye(num_classes, num_classes) * (1 - flip_rate)
        T = T_diag + (1 - np.eye(num_classes, num_classes)) * (flip_rate / (num_classes - 1))
    elif data_type == 'pair':
        T_diag = np.eye(num_classes, num_classes) * flip_rate
        T = np.eye(num_classes, num_classes) * (1 - flip_rate)
        T = T + np.roll(T_diag, -1, axis=0)
    else:
        raise ValueError('Invalid data generate strategy')
    assert_array_almost_equal(T.sum(axis=1), np.ones(T.shape[1]))

    noise_label = np.array(train_labels.copy()) + num_classes
    for i in range(0, num_classes):
        noise_label[noise_label == (i + num_classes)] = np.random.choice(
            a=np.arange(num_classes),
            size=noise_label[noise_label == (i + num_classes)].__len__(),
            replace=True, p=T[i])
    print('Actual noise %.2f' % (noise_label != train_labels).mean())
    return noise_label


def noisify_with_P(y_train, nb_classes, noise, random_state=None, noise_type='uniform'):
    if noise > 0.0:
        if noise_type == 'uniform':
            print('Uniform noise')
            P = build_uniform_P(nb_classes, noise)
        elif noise_type == 'pair':
            print('Pair noise')
            P = build_pair_p(nb_classes, noise)
        else:
            print('Noise type have implemented')
        # seed the random numbers with #run
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    else:
        P = np.eye(nb_classes)

    return y_train, P


def get_train_val_test(nnodes, val_size=0.15, test_size=0.8, stratify=None, seed=None):
    """This setting follows nettack/mettack, where we split the nodes
    into 10% training, 10% validation and 80% testing data

    Parameters
    ----------
    nnodes : int
        number of nodes in total
    val_size : float
        size of validation set
    test_size : float
        size of test set
    stratify :
        data is expected to split in a stratified fashion. So stratify should be labels.
    seed : int or None
        random seed

    Returns
    -------
    idx_train :
        node training indices
    idx_val :
        node validation indices
    idx_test :
        node test indices
    """

    assert stratify is not None, 'stratify cannot be None!'

    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(nnodes)
    train_size = 1 - val_size - test_size
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test


def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


# https://zhuanlan.zhihu.com/p/417153633
def init_gpuseed(seed, device, benchmark=True, deterministic=True):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    torch.manual_seed(seed)
