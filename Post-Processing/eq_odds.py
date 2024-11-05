import cvxpy as cvx  # Import the CVXPY library for convex optimization.
import numpy as np  # Import NumPy for numerical operations.
from collections import namedtuple  # Import namedtuple for creating simple classes.

# Define a Model class using namedtuple to store predictions and labels.
class Model(namedtuple('Model', 'pred label')):
    def logits(self):
        # Compute raw logits from predicted probabilities, clipping to prevent overflow.
        raw_logits = np.clip(np.log(self.pred / (1 - self.pred)), -100, 100)
        return raw_logits

    def num_samples(self):
        # Return the number of samples in the predictions.
        return len(self.pred)

    def base_rate(self):
        """
        Calculate the base rate, which is the percentage of samples belonging to the positive class (label = 1).
        """
        return np.mean(self.label)

    def accuracy(self):
        # Calculate overall accuracy by averaging accuracies across samples.
        return self.accuracies().mean()

    def precision(self):
        # Calculate precision as the mean of true positives out of predicted positives.
        return (self.label[self.pred.round() == 1]).mean()

    def recall(self):
        # Calculate recall as the mean of true positives out of actual positives.
        return (self.label[self.label == 1].round()).mean()

    def tpr(self):
        """
        Calculate the True Positive Rate (TPR) - the proportion of actual positives correctly identified.
        """
        return np.mean(np.logical_and(self.pred.round() == 1, self.label == 1))

    def fpr(self):
        """
        Calculate the False Positive Rate (FPR) - the proportion of actual negatives incorrectly identified as positives.
        """
        return np.mean(np.logical_and(self.pred.round() == 1, self.label == 0))

    def tnr(self):
        """
        Calculate the True Negative Rate (TNR) - the proportion of actual negatives correctly identified.
        """
        return np.mean(np.logical_and(self.pred.round() == 0, self.label == 0))

    def fnr(self):
        """
        Calculate the False Negative Rate (FNR) - the proportion of actual positives incorrectly identified as negatives.
        """
        return np.mean(np.logical_and(self.pred.round() == 0, self.label == 1))

    def fn_cost(self):
        """
        Calculate the generalized cost of false negatives.
        This is computed as the mean predicted probability for true positive instances.
        """
        return 1 - self.pred[self.label == 1].mean()

    def fp_cost(self):
        """
        Calculate the generalized cost of false positives.
        This is computed as the mean predicted probability for true negative instances.
        """
        return self.pred[self.label == 0].mean()

    def accuracies(self):
        # Return a boolean array indicating whether each prediction is correct.
        return self.pred.round() == self.label

    def eq_odds(self, othr, mix_rates=None):
        """
        Adjust predictions to achieve equalized odds with another model (othr).
        If mix_rates are provided, they will be used; otherwise, they will be optimized.
        """
        has_mix_rates = not (mix_rates is None)
        if not has_mix_rates:
            # If no mixing rates are provided, compute optimal mixing rates.
            mix_rates = self.eq_odds_optimal_mix_rates(othr)

        sp2p, sn2p, op2p, on2p = tuple(mix_rates)  # Unpack mixing rates.

        # Adjust predictions for self model.
        self_fair_pred = self.pred.copy()
        self_pp_indices, = np.nonzero(self.pred.round())  # Indices of predicted positives.
        self_pn_indices, = np.nonzero(1 - self.pred.round())  # Indices of predicted negatives.
        np.random.shuffle(self_pp_indices)  # Shuffle predicted positives.
        np.random.shuffle(self_pn_indices)  # Shuffle predicted negatives.

        # Flip predictions for some predicted negatives to achieve equalized odds.
        n2p_indices = self_pn_indices[:int(len(self_pn_indices) * sn2p)]
        self_fair_pred[n2p_indices] = 1 - self_fair_pred[n2p_indices]
        p2n_indices = self_pp_indices[:int(len(self_pp_indices) * (1 - sp2p))]
        self_fair_pred[p2n_indices] = 1 - self_fair_pred[p2n_indices]

        # Adjust predictions for the other model.
        othr_fair_pred = othr.pred.copy()
        othr_pp_indices, = np.nonzero(othr.pred.round())
        othr_pn_indices, = np.nonzero(1 - othr.pred.round())
        np.random.shuffle(othr_pp_indices)
        np.random.shuffle(othr_pn_indices)

        # Flip predictions for the other model to achieve equalized odds.
        n2p_indices = othr_pn_indices[:int(len(othr_pn_indices) * on2p)]
        othr_fair_pred[n2p_indices] = 1 - othr_fair_pred[n2p_indices]
        p2n_indices = othr_pp_indices[:int(len(othr_pp_indices) * (1 - op2p))]
        othr_fair_pred[p2n_indices] = 1 - othr_fair_pred[p2n_indices]

        fair_self = Model(self_fair_pred, self.label)  # Create adjusted model for self.
        fair_othr = Model(othr_fair_pred, othr.label)  # Create adjusted model for the other.

        if not has_mix_rates:
            return fair_self, fair_othr, mix_rates  # Return adjusted models and mixing rates.
        else:
            return fair_self, fair_othr  # Return adjusted models only.

    def eq_odds_optimal_mix_rates(self, othr):
        """
        Optimize mixing rates to achieve equalized odds with another model (othr).
        The method uses convex optimization to minimize discrepancies in error rates between the two models.
        """
        sbr = float(self.base_rate())  # Base rate for self model.
        obr = float(othr.base_rate())  # Base rate for other model.

        # Define optimization variables for mixing rates.
        sp2p = cvx.Variable(1)  # Mixing rate for self positive to positive.
        sp2n = cvx.Variable(1)  # Mixing rate for self positive to negative.
        sn2p = cvx.Variable(1)  # Mixing rate for self negative to positive.
        sn2n = cvx.Variable(1)  # Mixing rate for self negative to negative.

        op2p = cvx.Variable(1)  # Mixing rate for other positive to positive.
        op2n = cvx.Variable(1)  # Mixing rate for other positive to negative.
        on2p = cvx.Variable(1)  # Mixing rate for other negative to positive.
        on2n = cvx.Variable(1)  # Mixing rate for other negative to negative.

        # Calculate expected false positive and false negative rates.
        sfpr = self.fpr() * sp2p + self.tnr() * sn2p  # Self false positive rate.
        sfnr = self.fnr() * sn2n + self.tpr() * sp2n  # Self false negative rate.
        ofpr = othr.fpr() * op2p + othr.tnr() * on2p  # Other false positive rate.
        ofnr = othr.fnr() * on2n + othr.tpr() * op2n  # Other false negative rate.
        error = sfpr + sfnr + ofpr + ofnr  # Total error to minimize.

        # Create constraints for the optimization problem.
        constraints = [
            sp2p == 1 - sp2n,  # Mixing rates must sum to 1.
            sn2p == 1 - sn2n,
            op2p == 1 - op2n,
            on2p == 1 - on2n,
            sp2p <= 1,  # Ensure mixing rates are between 0 and 1.
            sp2p >= 0,
            sn2p <= 1,
            sn2p >= 0,
            op2p <= 1,
            op2p >= 0,
            on2p <= 1,
            on2p >= 0,
            # Ensure equalized odds conditions.
            (sn2p * (1 - self.pred.mean()) + sp2p * self.pred.mean()) / sbr ==
            (on2p * (1 - othr.pred.mean()) + op2p * othr.pred.mean()) / obr,
        ]

        # Define and solve the optimization problem.
        prob = cvx.Problem(cvx.Minimize(error), constraints)
        prob.solve()

        # Return the optimized mixing rates as a numpy array.
        res = np.array([sp2p.value, sn2p.value, op2p.value, on2p.value])
        return res

    def __repr__(self):
        # String representation of the model's performance metrics.
        return '\n'.join([
            'Accuracy:\t%.3f' % self.accuracy(),
            'F.P. cost:\t%.3f' % self.fp_cost(),
            'F.N. cost:\t%.3f' % self.fn_cost(),
            'Base rate:\t%.3f' % self.base_rate(),
            'Avg. score:\t%.3f' % self.pred.mean(),
        ])


"""
Demo execution
"""
if __name__ == '__main__':
    import pandas as pd  # Import Pandas for data handling.
    import sys  # Import sys for command-line argument handling.

    # Read the input filename from command-line arguments.
    data_filename = sys.argv[1]
    
    # Load the validation set scores from the specified CSV file.
    test_and_val_data = pd.read_csv(data_filename)

    # Assuming a binary demographic feature is present, separate the groups.
    # Here, 'score' represents predicted probabilities and 'label' represents true labels.
    scores_group1 = test_and_val_data[test_and_val_data['group'] == 1]['score'].values
    labels_group1 = test_and_val_data[test_and_val_data['group'] == 1]['label'].values
    scores_group0 = test_and_val_data[test_and_val_data['group'] == 0]['score'].values
    labels_group0 = test_and_val_data[test_and_val_data['group'] == 0]['label'].values

    # Create model instances for each group and for validation/test data.
    model_group1 = Model(scores_group1, labels_group1)
    model_group0 = Model(scores_group0, labels_group0)

    # Calculate optimal mixing rates for achieving equalized odds.
    fair_model_group1, fair_model_group0, mix_rates = model_group1.eq_odds(model_group0)

    # Output results for the original and adjusted models.
    print("Original Model Group 1:")
    print(model_group1)
    print("\nAdjusted Model Group 1:")
    print(fair_model_group1)

    print("\nOriginal Model Group 0:")
    print(model_group0)
    print("\nAdjusted Model Group 0:")
    print(fair_model_group0)

    # Print mixing rates for equalized odds.
    print("\nMixing Rates:")
    print(mix_rates)
