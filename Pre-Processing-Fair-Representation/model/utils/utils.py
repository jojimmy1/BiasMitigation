from numba.decorators import jit  # Import the Just-In-Time compiler (jit) decorator from Numba to accelerate function execution
import numpy as np  # Import NumPy for array operations

# Function to compute distances between each data point and the prototypes
@jit
def distances(X, v, alpha, N, P, k):
    dists = np.zeros((N, P))  # Initialize a matrix to store distances
    for i in range(N):  # Loop over all data points
        for p in range(P):  # Loop over all features (dimensions)
            for j in range(k):  # Loop over all prototypes
                # Compute squared distance between data point and prototype, weighted by alpha
                dists[i, j] += (X[i, p] - v[j, p]) * (X[i, p] - v[j, p]) * alpha[p]
    return dists  # Return the computed distance matrix

# Function to compute soft cluster assignments (M_nk matrix) based on distances
@jit
def M_nk(dists, N, k):
    M_nk = np.zeros((N, k))  # Initialize the membership matrix (soft assignments)
    exp = np.zeros((N, k))  # Initialize matrix to store exponentiated distances
    denom = np.zeros(N)  # Denominator for normalization (sum of exponentials)
    for i in range(N):  # Loop over data points
        for j in range(k):  # Loop over prototypes
            exp[i, j] = np.exp(-1 * dists[i, j])  # Compute the exponentiated negative distance
            denom[i] += exp[i, j]  # Accumulate the denominator
        for j in range(k):  # Normalize to get soft assignments
            if denom[i]:  # Avoid division by zero
                M_nk[i, j] = exp[i, j] / denom[i]  # Normalize by the sum of exponentials
            else:
                M_nk[i, j] = exp[i, j] / 1e-6  # Small constant for stability
    return M_nk  # Return soft assignments matrix

# Function to compute the average membership for each cluster
@jit
def M_k(M_nk, N, k):
    M_k = np.zeros(k)  # Initialize a vector for cluster membership totals
    for j in range(k):  # Loop over clusters
        for i in range(N):  # Loop over data points
            M_k[j] += M_nk[i, j]  # Sum the memberships for each cluster
        M_k[j] /= N + 1e-10  # Normalize by the number of data points (to avoid division by zero)
    return M_k  # Return the average membership for each cluster

# Function to compute the reconstructed data points and reconstruction loss
@jit
def x_n_hat(X, M_nk, v, N, P, k, result):
    x_n_hat = np.zeros((N, P))  # Initialize the matrix for reconstructed data
    L_x = 0.0  # Initialize reconstruction loss
    for i in range(N):  # Loop over data points
        for p in range(P):  # Loop over features
            for j in range(k):  # Loop over prototypes
                # Weighted sum of prototypes to reconstruct the data points
                x_n_hat[i, p] += M_nk[i, j] * v[j, p]
            if not result:  # If we are not just returning the result (training phase), compute the loss
                L_x += (X[i, p] - x_n_hat[i, p]) ** 2  # Squared reconstruction error
    return x_n_hat, L_x  # Return reconstructed data and reconstruction loss

# Function to compute predicted target values and classification loss
@jit
def yhat(M_nk, y, w, N, k, result):
    yhat = np.zeros(N)  # Initialize predictions
    L_y = 0.0  # Initialize classification loss
    for i in range(N):  # Loop over data points
        for j in range(k):  # Loop over prototypes
            yhat[i] += M_nk[i, j] * w[j]  # Weighted sum of prototype labels to get predictions
        # Clip predictions between (0, 1) to avoid numerical issues
        yhat[i] = max(1e-6, min(0.999, yhat[i]))
        if not result:  # If we are in the training phase, compute classification loss
            L_y += -y[i] * np.log(yhat[i]) - (1 - y[i]) * np.log(1 - yhat[i])  # Binary cross-entropy loss
    return yhat, L_y  # Return predictions and classification loss

# Main function for the Latent Fair Representation (LFR) model
def lfr(params, data_sensitive, data_nonsensitive, y_sensitive,
        y_nonsensitive, k=10, A_x=1e-4, A_y=0.1, A_z=1000, results=0):
    lfr.iters += 1  # Increment iteration counter (used for logging/debugging)

    # Extract dimensions from sensitive and nonsensitive data
    Ns, P = data_sensitive.shape  # Ns = number of sensitive group data points, P = number of features
    Nns, _ = data_nonsensitive.shape  # Nns = number of nonsensitive group data points

    # Extract model parameters from the input 'params' array
    alpha0 = params[:P]  # Alpha values for the nonsensitive group (feature weights)
    alpha1 = params[P:2 * P]  # Alpha values for the sensitive group (feature weights)
    w = params[2 * P: (2 * P) + k]  # Prototype weights for the classification task
    v = np.matrix(params[(2 * P) + k:]).reshape((k, P))  # Prototypes (latent representation)

    # Compute distances between data points and prototypes for both groups
    dists_sensitive = distances(data_sensitive, v, alpha1, Ns, P, k)
    dists_nonsensitive = distances(data_nonsensitive, v, alpha0, Nns, P, k)

    # Compute soft cluster assignments (M_nk) for both groups
    M_nk_sensitive = M_nk(dists_sensitive, Ns, k)
    M_nk_nonsensitive = M_nk(dists_nonsensitive, Nns, k)

    # Compute average cluster memberships (M_k) for both groups
    M_k_sensitive = M_k(M_nk_sensitive, Ns, k)
    M_k_nonsensitive = M_k(M_nk_nonsensitive, Nns, k)

    # Compute fairness loss (L_z) based on the difference in cluster memberships
    L_z = 0.0
    for j in range(k):
        L_z += abs(M_k_sensitive[j] - M_k_nonsensitive[j])  # Difference in memberships contributes to fairness loss

    # Compute reconstruction loss for both groups
    x_n_hat_sensitive, L_x1 = x_n_hat(data_sensitive, M_nk_sensitive, v, Ns, P, k, results)
    x_n_hat_nonsensitive, L_x2 = x_n_hat(data_nonsensitive, M_nk_nonsensitive, v, Nns, P, k, results)
    L_x = L_x1 + L_x2  # Total reconstruction loss

    # Compute classification loss for both groups
    yhat_sensitive, L_y1 = yhat(M_nk_sensitive, y_sensitive, w, Ns, k, results)
    yhat_nonsensitive, L_y2 = yhat(M_nk_nonsensitive, y_nonsensitive, w, Nns, k, results)
    L_y = L_y1 + L_y2  # Total classification loss

    # Compute the overall objective function (combination of reconstruction, classification, and fairness losses)
    criterion = A_x * L_x + A_y * L_y + A_z * L_z

    # Log the current iteration and criterion value every 250 iterations
    if lfr.iters % 250 == 0:
        print(lfr.iters, criterion)

    # If we are in the evaluation phase, return predictions and soft assignments
    if results:
        return yhat_sensitive, yhat_nonsensitive, M_nk_sensitive, M_nk_nonsensitive
    else:
        return criterion  # Otherwise, return the objective value for optimization

# Initialize the iteration counter for the LFR function
lfr.iters = 0
