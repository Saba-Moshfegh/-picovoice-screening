import numpy as np


def extend_labels(labels, blank=0):
    """
    Insert the blank token between every label symbol and at the boundaries.
    E.g. if labels = [1, 2, 3] and blank=0,
         extended_labels = [0, 1, 0, 2, 0, 3, 0].
    """
    extended = [blank]
    for l in labels:
        extended.append(l)
        extended.append(blank)
    return np.array(extended, dtype=np.int32)


def forward_pass(log_probs, extended_labels):
    """
    Compute the forward variable alpha for the CTC algorithm.

    Args:
        log_probs: (T x V) array of log probabilities, where T is the sequence length
                   and V is the vocabulary size (including blank).
        extended_labels: 1D array of the extended label sequence (length S).

    Returns:
        alpha: 2D array (T x S), alpha[t, s] is the total log-prob of all paths
               ending in extended_labels[s] at time t.
    """
    T, V = log_probs.shape
    S = len(extended_labels)

    # Initialize alpha
    alpha = -np.inf * np.ones((T, S))

    # Initialization at t=0
    alpha[0, 0] = log_probs[0, extended_labels[0]]  # blank at s=0
    if S > 1:
        alpha[0, 1] = log_probs[0, extended_labels[1]]  # first actual label if it differs from blank

    # Recurrence
    for t in range(1, T):
        for s in range(S):
            # s can stay the same or come from s-1 or s-2 (if allowed)
            # Because the extended label sequence has blanks in between,
            # we need to handle transitions carefully.

            # alpha[t-1, s]
            log_sum = alpha[t - 1, s]

            # from s-1 (always possible)
            if s - 1 >= 0:
                log_sum = np.logaddexp(log_sum, alpha[t - 1, s - 1])

            # from s-2 (if not repeating label)
            # only if extended_labels[s] != extended_labels[s-2], i.e. no immediate symbol repetition
            if s - 2 >= 0 and extended_labels[s] != extended_labels[s - 2]:
                log_sum = np.logaddexp(log_sum, alpha[t - 1, s - 2])

            # Multiply (add in log domain) with current probability
            alpha[t, s] = log_sum + log_probs[t, extended_labels[s]]

    return alpha


def backward_pass(log_probs, extended_labels):
    """
    Compute the backward variable beta for the CTC algorithm.

    Args:
        log_probs: (T x V) array of log probabilities.
        extended_labels: 1D array of extended label sequence (length S).

    Returns:
        beta: 2D array (T x S), beta[t, s] is the total log-prob of all paths
              starting in extended_labels[s] at time t.
    """
    T, V = log_probs.shape
    S = len(extended_labels)

    # Initialize beta
    beta = -np.inf * np.ones((T, S))

    # Initialization at t = T-1
    beta[T - 1, S - 1] = log_probs[T - 1, extended_labels[S - 1]]
    if S > 1:
        beta[T - 1, S - 2] = log_probs[T - 1, extended_labels[S - 2]]

    # Recurrence backward in time
    for t in range(T - 2, -1, -1):
        for s in range(S - 1, -1, -1):
            # transitions to s, s+1, s+2
            log_sum = beta[t + 1, s]

            if s + 1 < S:
                log_sum = np.logaddexp(log_sum, beta[t + 1, s + 1])

            if s + 2 < S and extended_labels[s] != extended_labels[s + 2]:
                log_sum = np.logaddexp(log_sum, beta[t + 1, s + 2])

            beta[t, s] = log_sum + log_probs[t, extended_labels[s]]

    return beta


def ctc_loss_and_gradient(log_probs, labels, blank=0):
    """
    Compute the CTC loss and gradient w.r.t. the log_probs (forward + backward).

    Args:
        log_probs: (T x V) array of log probabilities (already log-softmaxed).
        labels: 1D array of the original label sequence (no blanks).
        blank: The index used for the blank symbol.

    Returns:
        ctc_loss: Scalar float, the CTC loss for this sequence.
        grad: (T x V) array, gradient of the loss w.r.t. each log probability.
    """
    # 1. Extend labels by inserting blanks
    extended_labels = extend_labels(labels, blank=blank)

    # 2. Compute forward and backward variables
    alpha = forward_pass(log_probs, extended_labels)
    beta = backward_pass(log_probs, extended_labels)

    T, V = log_probs.shape
    S = len(extended_labels)

    # 3. Compute total log-prob of all valid paths (denominator for posteriors)
    #    This is sum in the linear domain, but we keep it in log domain:
    #    p = alpha[T-1, S-1] + alpha[T-1, S-2] in linear domain => logaddexp in log domain
    #    (When S=1, we only have S-1=0 index, so handle carefully.)
    if S == 1:
        log_likelihood = alpha[-1, 0]
    else:
        log_likelihood = np.logaddexp(alpha[-1, S - 1], alpha[-1, S - 2])

    ctc_loss = -log_likelihood  # negative log-likelihood

    # 4. Gradient computation
    #    For each time t and label v, the gradient is:
    #       grad[t, v] = p(t,v) - p*(t,v),
    #    where p(t,v) is the posterior of v at time t.
    #
    #    We can compute posterior(t,s) = exp(alpha[t,s] + beta[t,s] - log_likelihood).
    #    Then accumulate those posteriors to get the total posterior for each symbol v.

    grad = np.zeros_like(log_probs)

    # Posterior for each extended label index (t, s)
    posterior = np.exp(alpha + beta - log_likelihood)  # shape (T, S)

    # Accumulate posterior for each symbol at time t
    for s in range(S):
        symbol = extended_labels[s]
        grad[np.arange(T), symbol] -= posterior[:, s]  # subtract because of the derivative of -log

    # Because the log_probs are log-softmax, the gradient for the “predicted symbol” is:
    # grad[t, v] = exp(log_probs[t, v]) - sum_of_posterior_if_v_is_label
    # Here we have subtracted the posterior portion. Next, we add p(v|x_t) for each t,v:
    grad += np.exp(log_probs)

    return ctc_loss, grad


if __name__ == "__main__":
    # Example usage:

    # Suppose we have T time steps, V=symbols ('_'=blank, 1,2,3=labels).
    # We'll create random log_probs for demonstration (must be log-softmaxed).
    np.random.seed(42)
    T, V = 6, 4
    raw_logits = np.random.randn(T, V)

    # Convert raw logits to log-softmax:
    log_probs = raw_logits - np.logaddexp.reduce(raw_logits, axis=1, keepdims=True)

    # Our target label is, say, [1, 2].  (No blank index in "labels"; that is inserted automatically.)
    labels = np.array([1, 2], dtype=np.int32)

    # Compute CTC loss and gradients
    loss, grad = ctc_loss_and_gradient(log_probs, labels, blank=0)

    print("CTC Loss:", loss)
    print("Gradient shape:", grad.shape)
    print("Gradient:\n", grad)