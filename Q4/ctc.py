import numpy as np
import tensorflow as tf
from typing import List


def extend_labels(labels: List[int], blank: int=0 ) -> List[int]:
    '''
    Adds blank label between each label and at the beginning and end.

    :param labels List[int]:
        Labels as their indices to be extended
    :param blank int:
        The integer index of the blank label

    :return List[int]:
        List of extended labels with blank in between each label and at the beginning and end
    '''

    extended = [blank]
    for l in labels:
        extended.append(l)
        extended.append(blank)
    return extended


def forward_pass(log_probs: np.ndarray, extended_labels: List[int]) -> np.ndarray:
    '''
    Compute the forward pass of the CTC algorithm.

    :param log_probs np.ndarray:
        Log probabilities of the input sequence
    :param extended_labels List[int]:
        List of extended labels with blank in between each label and at the beginning and end

    :return np.ndarray:
        Forward pass matrix alpha
    '''

    T, V = log_probs.shape
    S = len(extended_labels)

    alpha = -np.inf * np.ones((T, S))

    alpha[0, 0] = log_probs[0, extended_labels[0]]  # blank at s=0
    if S > 1:
        alpha[0, 1] = log_probs[0, extended_labels[1]]

    for t in range(1, T):
        for s in range(S):
            log_sum = alpha[t - 1, s]
            if s > 0:
                log_sum = np.logaddexp(log_sum, alpha[t - 1, s - 1])
            if s > 1 and extended_labels[s] != extended_labels[s - 2]:
                log_sum = np.logaddexp(log_sum, alpha[t - 1, s - 2])
            alpha[t, s] = log_sum + log_probs[t, extended_labels[s]]

    return alpha


def backward_pass(log_probs: np.ndarray, extended_labels: List[int]) -> np.ndarray:
    '''
    Compute the backward pass of the CTC algorithm.

    :param log_probs:
        log probabilities of the input sequence
    :param extended_labels List[int]:
        List of extended labels with blank in between each label and at the beginning and end

    :return np.ndarray:
        Backward pass matrix beta
    '''

    T, V = log_probs.shape
    S = len(extended_labels)

    beta = -np.inf * np.ones((T, S))
    beta[T - 1, S - 1] = log_probs[T - 1, extended_labels[S - 1]]
    if S > 1:
        beta[T - 1, S - 2] = log_probs[T - 1, extended_labels[S - 2]]

    for t in range(T - 2, -1, -1):
        for s in range(S - 1, -1, -1):
            log_sum = beta[t + 1, s]
            if s < S - 1:
                log_sum = np.logaddexp(log_sum, beta[t + 1, s + 1])
            if s < S - 2 and extended_labels[s] != extended_labels[s + 2]:
                log_sum = np.logaddexp(log_sum, beta[t + 1, s + 2])
            beta[t, s] = log_sum + log_probs[t, extended_labels[s]]

    return beta


def ctc_loss_custom(log_probs: np.ndarray, extended: List[int]) -> float:
    '''
    Compute the CTC loss using the forward and backward pass.

    :param log_probs np.ndarray:
        log probabilities of the input sequence
    :param extended_labels List[int]:
        List of extended labels with blank in between each label and at the beginning and end

    :return float:
        CTC loss
    '''

    alpha = forward_pass(log_probs, extended)
    beta = backward_pass(log_probs, extended)

    S = len(extended)

    if S == 1:
        ll = alpha[-1, 0]
    else:
        ll = np.logaddexp(alpha[-1, S - 1], alpha[-1, S - 2])
    return -ll


def compute_tf_ctc_loss(logits: np.ndarray, labels: List[int], blank: int = 0) -> float:
    '''
    Compute the CTC loss using TensorFlow's CTC loss function.

    :param logits np.ndarray:
        raw logits of nn output
    :param labels:
        List of indices of the labels
    :param blank:
        Index of the blank label

    :return float:
        CTC loss computed using TensorFlow
    '''

    T, V = logits.shape
    tf_logits = tf.convert_to_tensor(logits.reshape(T, 1, V), dtype=tf.float32)

    indices = np.array([[0, i] for i in range(len(labels))], dtype=np.int64)
    values = np.array(labels, dtype=np.int32)
    dense_shape = np.array([1, len(labels)], dtype=np.int64)
    tf_labels = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    label_length = [len(labels)]
    logit_length = [T]

    tf_loss = tf.nn.ctc_loss(
        labels=tf_labels,
        logits=tf_logits,
        label_length=label_length,
        logit_length=logit_length,
        blank_index=blank,
        logits_time_major=True
    )
    tf_loss_val = tf.reduce_mean(tf_loss).numpy()

    return tf_loss_val

def get_char_indices_and_add_one(input_string: str) -> List[int]:
    '''
    Get indices of characters in the input string and make a list from their indices plus one.

    :param input_string:
        Input string

    :return List[int]:
        List of indices of characters in the input string plus one
    '''

    indices = [i + 1 for i in range(len(input_string))]
    return indices

if __name__ == "__main__":

    label = 'cat'
    labels = get_char_indices_and_add_one(label)
    extended_label = extend_labels(labels, blank=0)

    T, V = 10, len(labels) + 1
    np.random.seed(42)
    raw_logits = np.random.randn(T, V)
    log_probabilities = tf.nn.log_softmax(raw_logits, axis=-1).numpy()

    custom_loss_val = ctc_loss_custom(log_probabilities, extended_label)
    tf_ctc_loss_val = compute_tf_ctc_loss(raw_logits, labels, blank=0)

    assert np.isclose(custom_loss_val, tf_ctc_loss_val, atol=1e-5), "CTC Losses do not match."
