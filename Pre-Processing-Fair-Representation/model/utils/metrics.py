from sklearn.metrics import confusion_matrix  # Import confusion matrix from scikit-learn to evaluate classification performance.

# Function to calculate accuracy
def accuracy(y_true, y_pred):
    # Compute 1 minus the mean absolute difference between the true and predicted labels.
    # This allows handling continuous predictions as well as binary predictions.
    return 1 - (abs(y_true - y_pred)).mean()

# Function to calculate discrimination between privileged and unprivileged groups
def discrimination(y_real, y_pred, SensitiveCat, privileged, unprivileged):
    # Filter predictions for the privileged group based on the sensitive attribute (SensitiveCat)
    y_priv = y_pred[y_real[SensitiveCat] == privileged]
    
    # Filter predictions for the unprivileged group based on the sensitive attribute (SensitiveCat)
    y_unpriv = y_pred[y_real[SensitiveCat] == unprivileged]
    
    # Compute the absolute difference in mean predictions between the two groups
    return abs(y_priv.mean() - y_unpriv.mean())

# Function to calculate prediction consistency based on the similarity of nearby data points
def consistency(X, y_pred, k=5):
    from sklearn.neighbors import NearestNeighbors  # Import NearestNeighbors to find the k nearest neighbors.
    
    # Fit a NearestNeighbors model to the feature matrix X using the ball_tree algorithm
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(X)
    
    y = 0  # Initialize a variable to accumulate consistency scores.
    N = X.shape[0]  # Get the number of samples (N) from X.
    
    # Loop over all data points in X
    for i in range(N):
        # Find the k+1 nearest neighbors for data point i (including itself)
        distances, indices = nbrs.kneighbors(X[i, :].reshape(1, -1))
        
        # The first index corresponds to the data point itself, so we remove it from consideration.
        # Calculate the absolute difference between the prediction for the data point and the average prediction for its neighbors.
        y += abs(y_pred.iloc[i] - y_pred.iloc[indices.tolist()[0][1:]].sum())
    
    # Return the consistency score, normalized by the number of data points and the number of neighbors (k).
    return 1 - y / (N * k)

# Function to compute the difference in equal opportunity between privileged and unprivileged groups
def DifferenceEqualOpportunity(y_pred, y_real, SensitiveCat, outcome, privileged, unprivileged, labels):
    '''
    ABS Difference in True Positive Rate between the two groups (privileged and unprivileged).
    :param y_pred: Predicted labels.
    :param y_real: Real labels (ground truth).
    :param SensitiveCat: Name of the sensitive feature (e.g., race or gender).
    :param outcome: Name of the outcome feature (e.g., target variable).
    :param privileged: The privileged group value (e.g., male).
    :param unprivileged: The unprivileged group value (e.g., female).
    :param labels: Labels for the confusion matrix (e.g., [0, 1]).
    :return: Absolute difference in True Positive Rate (TPR) between privileged and unprivileged groups.
    '''
    
    # Get predictions and real labels for the privileged group
    y_priv = y_pred[y_real[SensitiveCat] == privileged]
    y_real_priv = y_real[y_real[SensitiveCat] == privileged]
    
    # Get predictions and real labels for the unprivileged group
    y_unpriv = y_pred[y_real[SensitiveCat] == unprivileged]
    y_real_unpriv = y_real[y_real[SensitiveCat] == unprivileged]
    
    # Compute confusion matrix for the privileged group and extract TN, FP, FN, TP
    TN_priv, FP_priv, FN_priv, TP_priv = confusion_matrix(y_real_priv[outcome], y_priv, labels=labels).ravel()
    
    # Compute confusion matrix for the unprivileged group and extract TN, FP, FN, TP
    TN_unpriv, FP_unpriv, FN_unpriv, TP_unpriv = confusion_matrix(y_real_unpriv[outcome], y_unpriv, labels=labels).ravel()
    
    # Compute and return the absolute difference in True Positive Rate (TPR) between the unprivileged and privileged groups
    return abs(TP_unpriv / (TP_unpriv + FN_unpriv) - TP_priv / (TP_priv + FN_priv))

# Function to compute the difference in average odds between privileged and unprivileged groups
def DifferenceAverageOdds(y_pred, y_real, SensitiveCat, outcome, privileged, unprivileged, labels):
    '''
    Mean absolute difference in True Positive Rate and False Positive Rate of the two groups (privileged and unprivileged).
    :param y_pred: Predicted labels.
    :param y_real: Real labels (ground truth).
    :param SensitiveCat: Name of the sensitive feature (e.g., race or gender).
    :param outcome: Name of the outcome feature (e.g., target variable).
    :param privileged: The privileged group value (e.g., male).
    :param unprivileged: The unprivileged group value (e.g., female).
    :param labels: Labels for the confusion matrix (e.g., [0, 1]).
    :return: Mean absolute difference in TPR and FPR between the two groups.
    '''
    
    # Get predictions and real labels for the privileged group
    y_priv = y_pred[y_real[SensitiveCat] == privileged]
    y_real_priv = y_real[y_real[SensitiveCat] == privileged]
    
    # Get predictions and real labels for the unprivileged group
    y_unpriv = y_pred[y_real[SensitiveCat] == unprivileged]
    y_real_unpriv = y_real[y_real[SensitiveCat] == unprivileged]
    
    # Compute confusion matrix for the privileged group and extract TN, FP, FN, TP
    TN_priv, FP_priv, FN_priv, TP_priv = confusion_matrix(y_real_priv[outcome], y_priv, labels=labels).ravel()
    
    # Compute confusion matrix for the unprivileged group and extract TN, FP, FN, TP
    TN_unpriv, FP_unpriv, FN_unpriv, TP_unpriv = confusion_matrix(y_real_unpriv[outcome], y_unpriv, labels=labels).ravel()
    
    # Compute the mean absolute difference in False Positive Rate (FPR) and True Positive Rate (TPR) between the two groups
    return 0.5 * (abs(FP_unpriv / (FP_unpriv + TN_unpriv) - FP_priv / (FP_priv + TN_priv)) + 
                  abs(TP_unpriv / (TP_unpriv + FN_unpriv) - TP_priv / (TP_priv + FN_priv)))
