from abc import ABC  # Importing Abstract Base Class module to define abstract classes
import numpy as np  # Importing NumPy for numerical operations
import scipy.optimize as optim  # Importing optimization module from SciPy
from model.utils.utils import *  # Importing utility functions from the custom 'utils' module

# Define the LFR class (Learning Fair Representations) that inherits from the ABC class
class LFR(ABC):
    """Learning fair representations is a pre-processing technique that finds a
       latent representation which encodes the data well but obfuscates information
       about protected attributes [R. Zemel, et al.]."""
    
    # Constructor method, initializes the LFR model with various parameters
    def __init__(self, sensitive_feature, privileged_class, unprivileged_class, seed, output_feature, parameter):
        self.sensitive_f = sensitive_feature  # Name of the sensitive feature (e.g., gender)
        self.priv_class = privileged_class  # The privileged class (e.g., male = 1)
        self.unpriv_class = unprivileged_class  # The unprivileged class (e.g., female = 0)
        self.outupt_f = output_feature  # The target output feature for classification
        self.seed = seed  # Random seed for reproducibility
        self.k = parameter['k']  # Number of prototypes or clusters
        self.Ax = parameter['Ax']  # Parameter for reconstruction loss term
        self.Ay = parameter['Ay']  # Parameter for classification loss term
        self.Az = parameter['Az']  # Parameter for fairness loss term
        self.max_iter = parameter['max_iter']  # Maximum number of iterations for optimization
        self.max_fun = parameter['max_fun']  # Maximum function evaluations for optimization
        self.w = None  # Placeholder for model weights
        self.prototypes = None  # Placeholder for learned prototypes
        self.learned_model = None  # Placeholder for the learned model

    # Method to fit the model using training data X and target labels y
    def fit(self, X, y):
        print(">>>>>>>>fit_called<<<<<<<<<<")
        np.random.seed(self.seed)  # Set the random seed for reproducibility

        # Identify indices for privileged and unprivileged groups based on the sensitive feature
        idx_priv = np.where(y[self.sensitive_f].values == self.priv_class)[0]  # Privileged group indices
        idx_unpriv = np.where(y[self.sensitive_f].values == self.unpriv_class)[0]  # Unprivileged group indices
        
        # Separate data for privileged and unprivileged groups
        data_priv = X[idx_priv, :]  # Data for privileged group
        data_unpriv = X[idx_unpriv, :]  # Data for unprivileged group

        # Store the number of features (dimensions) in the dataset
        self.features_dim = X.shape[1]

        # Target labels for privileged and unprivileged groups
        Y_priv = y[self.outupt_f].values[idx_priv]  # Target for privileged group
        Y_unpriv = y[self.outupt_f].values[idx_unpriv]  # Target for unprivileged group

        # Initialize model parameters with random values
        parameter_init = np.random.uniform(size=self.features_dim * 2 + self.k + self.features_dim * self.k)  # Random initialization of parameters (weights and prototypes)

        # Define bounds for the optimization problem
        Bound = []  # Bounds for each parameter
        for i, k2 in enumerate(parameter_init):
            if i < self.features_dim * 2 or i >= self.features_dim * 2 + self.k:
                Bound.append((None, None))  # No bounds for some parameters
            else:
                Bound.append((0, 1))  # Bound others between 0 and 1 (e.g., prototypes)

        # Use a bounded optimization algorithm to learn the latent fair representation (LFR)
        self.learned_proto = optim.fmin_l_bfgs_b(lfr,  # Optimizes the latent fair representation function
                                                 x0=parameter_init,  # Initial parameters
                                                 epsilon=1e-5,  # Tolerance for optimization
                                                 args=(data_priv, data_unpriv, Y_priv, Y_unpriv, self.k, self.Ax, self.Ay, self.Az),  # Arguments to pass to the objective function
                                                 bounds=Bound,  # Parameter bounds
                                                 approx_grad=True,  # Approximate the gradient
                                                 maxfun=self.max_fun,  # Maximum function evaluations
                                                 maxiter=self.max_iter)[0]  # Maximum number of iterations

    # Method to transform the dataset based on the learned fair representation
    def transform(self, X, y, threshold=0.5):
        print(">>>>>>>>>>Transform_called<<<<<<<<<<")
        Y = y.copy()  # Create a copy of the target labels

        # Identify indices for privileged and unprivileged groups
        idx_priv = np.where(y[self.sensitive_f].values == self.priv_class)[0]
        idx_unpriv = np.where(y[self.sensitive_f].values == self.unpriv_class)[0]

        # Separate data and target labels for both groups
        data_priv = X[idx_priv, :]
        data_unpriv = X[idx_unpriv, :]
        Y_priv = y[self.outupt_f].values[idx_priv]
        Y_unpriv = y[self.outupt_f].values[idx_unpriv]

        # Compute the latent fair representation for both groups using the learned prototypes
        Y_hat_p, Y_hat_unp, M_nk_priv, M_nk_unpriv = lfr(self.learned_proto, data_priv, data_unpriv, Y_priv, Y_unpriv, results=1)

        # Update target values for both groups based on the predicted values (Y_hat_p and Y_hat_unp)
        Y[self.outupt_f].iloc[idx_priv] = Y_hat_p.reshape(-1)
        Y[self.outupt_f].iloc[idx_unpriv] = Y_hat_unp.reshape(-1)

        # Prepare the transformed data in a new feature space (latent space)
        X_pred = np.zeros((X.shape[0], M_nk_priv.shape[1]))  # Initialize an empty matrix to store transformed data
        X_pred[idx_priv, :] = M_nk_priv  # Assign transformed data for privileged group
        X_pred[idx_unpriv, :] = M_nk_unpriv  # Assign transformed data for unprivileged group

        # Return the transformed data and updated target labels
        return X_pred, Y

    # Placeholder method for fitting and transforming the data in one step (not implemented here)
    def fit_transform(self, X, y=None, **fit_params):
        pass  # This method can be extended if needed
