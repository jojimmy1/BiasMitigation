import numpy as np  # Import NumPy for numerical operations
from collections import namedtuple  # Import namedtuple for creating a simple class

# Define a Model class using namedtuple for easy access to prediction and label attributes
class Model(namedtuple('Model', 'pred label')):
    def logits(self):
        """
        Calculate the raw logits from predicted probabilities.
        Clipping ensures stability by limiting values within a specific range.
        """
        raw_logits = np.clip(np.log(self.pred / (1 - self.pred)), -100, 100)
        return raw_logits

    def num_samples(self):
        """Return the number of samples in the model."""
        return len(self.pred)

    def base_rate(self):
        """
        Calculate the base rate (the percentage of samples belonging to the positive class).
        """
        return np.mean(self.label)

    def accuracy(self):
        """Calculate the overall accuracy of the model."""
        return self.accuracies().mean()

    def precision(self):
        """
        Calculate the precision of the model.
        Precision is the ratio of true positives to the total predicted positives.
        """
        return (self.label[self.pred.round() == 1]).mean()

    def recall(self):
        """
        Calculate the recall of the model.
        Recall is the ratio of true positives to the total actual positives.
        """
        return (self.label[self.label == 1].round()).mean()

    def tpr(self):
        """
        Calculate the true positive rate (TPR).
        TPR is the same as recall; it measures the proportion of actual positives correctly identified.
        """
        return np.mean(np.logical_and(self.pred.round() == 1, self.label == 1))

    def fpr(self):
        """
        Calculate the false positive rate (FPR).
        FPR measures the proportion of actual negatives incorrectly identified as positives.
        """
        return np.mean(np.logical_and(self.pred.round() == 1, self.label == 0))

    def tnr(self):
        """
        Calculate the true negative rate (TNR).
        TNR measures the proportion of actual negatives correctly identified.
        """
        return np.mean(np.logical_and(self.pred.round() == 0, self.label == 0))

    def fnr(self):
        """
        Calculate the false negative rate (FNR).
        FNR measures the proportion of actual positives incorrectly identified as negatives.
        """
        return np.mean(np.logical_and(self.pred.round() == 0, self.label == 1))

    def fn_cost(self):
        """
        Calculate the generalized false negative cost.
        This measures the average predicted probability for positive samples.
        """
        return 1 - self.pred[self.label == 1].mean()

    def fp_cost(self):
        """
        Calculate the generalized false positive cost.
        This measures the average predicted probability for negative samples.
        """
        return self.pred[self.label == 0].mean()

    def accuracies(self):
        """Return a boolean array indicating whether each prediction matches the actual label."""
        return self.pred.round() == self.label

    def calib_eq_odds(self, other, fp_rate, fn_rate, mix_rates=None):
        """
        Adjust the model predictions to achieve equalized odds between two groups.
        Mixing rates determine how much of each model's predictions to blend.
        """
        if fn_rate == 0:
            self_cost = self.fp_cost()  # False positive cost for self
            other_cost = other.fp_cost()  # False positive cost for the other model
            self_trivial_cost = self.trivial().fp_cost()  # Trivial model's false positive cost
            other_trivial_cost = other.trivial().fp_cost()  # Trivial model's false positive cost
        elif fp_rate == 0:
            self_cost = self.fn_cost()  # False negative cost for self
            other_cost = other.fn_cost()  # False negative cost for the other model
            self_trivial_cost = self.trivial().fn_cost()  # Trivial model's false negative cost
            other_trivial_cost = other.trivial().fn_cost()  # Trivial model's false negative cost
        else:
            self_cost = self.weighted_cost(fp_rate, fn_rate)  # Weighted cost for self
            other_cost = other.weighted_cost(fp_rate, fn_rate)  # Weighted cost for the other model
            self_trivial_cost = self.trivial().weighted_cost(fp_rate, fn_rate)  # Trivial weighted cost for self
            other_trivial_cost = other.trivial().weighted_cost(fp_rate, fn_rate)  # Trivial weighted cost for the other model

        # Determine if the other model's costs are greater than self's
        other_costs_more = other_cost > self_cost

        # Calculate mixing rates based on cost differences
        self_mix_rate = (other_cost - self_cost) / (self_trivial_cost - self_cost) if other_costs_more else 0
        other_mix_rate = 0 if other_costs_more else (self_cost - other_cost) / (other_trivial_cost - other_cost)

        # Create new classifiers by mixing predictions based on calculated rates
        self_indices = np.random.permutation(len(self.pred))[:int(self_mix_rate * len(self.pred))]
        self_new_pred = self.pred.copy()  # Create a copy of the original predictions
        self_new_pred[self_indices] = self.base_rate()  # Set mixed predictions to the base rate
        calib_eq_odds_self = Model(self_new_pred, self.label)  # Create new model for self

        other_indices = np.random.permutation(len(other.pred))[:int(other_mix_rate * len(other.pred))]
        other_new_pred = other.pred.copy()  # Create a copy of the other model's predictions
        other_new_pred[other_indices] = other.base_rate()  # Set mixed predictions to the base rate
        calib_eq_odds_other = Model(other_new_pred, other.label)  # Create new model for the other

        # Return the new models and mixing rates
        if mix_rates is None:
            return calib_eq_odds_self, calib_eq_odds_other, (self_mix_rate, other_mix_rate)
        else:
            return calib_eq_odds_self, calib_eq_odds_other

    def trivial(self):
        """
        Create a trivial classifier that returns the base rate for all predictions.
        This is used as a baseline for evaluating the model's performance.
        """
        base_rate = self.base_rate()
        pred = np.ones(len(self.pred)) * base_rate  # Set all predictions to base rate
        return Model(pred, self.label)

    def weighted_cost(self, fp_rate, fn_rate):
        """
        Calculate the weighted cost based on false positive and false negative rates.
        This provides a more flexible measure of model performance based on given rates.
        """
        norm_const = float(fp_rate + fn_rate) if (fp_rate != 0 and fn_rate != 0) else 1
        res = fp_rate / norm_const * self.fp_cost() * (1 - self.base_rate()) + \
            fn_rate / norm_const * self.fn_cost() * self.base_rate()
        return res

    def __repr__(self):
        """String representation of the model's performance metrics."""
        return '\n'.join([
            'Accuracy:\t%.3f' % self.accuracy(),
            'F.P. cost:\t%.3f' % self.fp_cost(),
            'F.N. cost:\t%.3f' % self.fn_cost(),
            'Base rate:\t%.3f' % self.base_rate(),
            'Avg. score:\t%.3f' % self.pred.mean(),
        ])


"""
Demo
"""
if __name__ == '__main__':
    """
    To run the demo:

    ```
    python calib_eq_odds.py <path_to_model_predictions.csv> <cost_constraint>
    ```

    `<cost_constraint>` defines the cost constraint to match for the groups. It can be:
    - `fnr` - match false negatives across groups
    - `fpr` - match false positives across groups
    - `weighted` - match a weighted combination of false positives and false negatives

    `<path_to_model_predictions.csv>` should contain the following columns for the VALIDATION set:

    - `prediction` (a score between 0 and 1)
    - `label` (ground truth - either 0 or 1)
    - `group` (group assignment - either 0 or 1)

    Try the following experiments, which were performed in the paper:
    ```
    python calib_eq_odds.py data/income.csv fnr
    python calib_eq_odds.py data/health.csv weighted
    python calib_eq_odds.py data/criminal_recidivism.csv fpr
    ```
    """
    import pandas as pd  # Import Pandas for data manipulation
    import sys  # Import sys for command-line argument handling

    # Check for the correct number of command-line arguments
    if not len(sys.argv) == 3:
        raise RuntimeError('Invalid number of arguments')

    # Get the cost constraint from the command-line argument
    cost_constraint = sys.argv[2]
    if cost_constraint not in ['fnr', 'fpr', 'weighted']:
        raise RuntimeError('cost_constraint (arg #2) should be one of fnr, fpr, weighted')

    # Set false negative and false positive rates based on the cost constraint
    if cost_constraint == 'fnr':
        fn_rate = 1
        fp_rate = 0
    elif cost_constraint == 'fpr':
        fn_rate = 0
        fp_rate = 1
    elif cost_constraint == 'weighted':
        fn_rate = 1
        fp_rate = 1

    # Load the validation set scores from the provided CSV file
    data_filename = sys.argv[1]
    test_and_val_data = pd.read_csv(data_filename)

    # Randomly split the data into validation and test sets
    order = np.random.permutation(len(test_and_val_data))
    val_indices = order[0::2]  # Select every second index for validation
    test_indices = order[1::2]  # Select the other indices for testing
    val_data = test_and_val_data.iloc[val_indices]  # Create validation set
    test_data = test_and_val_data.iloc[test_indices]  # Create test set

    # Create model objects for each group, using validation and test data
    group_0_val_data = val_data[val_data['group'] == 0]
    group_1_val_data = val_data[val_data['group'] == 1]
    group_0_test_data = test_data[test_data['group'] == 0]
    group_1_test_data = test_data[test_data['group'] == 1]

    # Instantiate models for the validation and test datasets
    group_0_val_model = Model(group_0_val_data['prediction'].to_numpy(), group_0_val_data['label'].to_numpy())
    group_1_val_model = Model(group_1_val_data['prediction'].to_numpy(), group_1_val_data['label'].to_numpy())
    group_0_test_model = Model(group_0_test_data['prediction'].to_numpy(), group_0_test_data['label'].to_numpy())
    group_1_test_model = Model(group_1_test_data['prediction'].to_numpy(), group_1_test_data['label'].to_numpy())

    # Find mixing rates for equalized odds models
    _, _, mix_rates = Model.calib_eq_odds(group_0_val_model, group_1_val_model, fp_rate, fn_rate)

    # Apply the mixing rates to the test models
    calib_eq_odds_group_0_test_model, calib_eq_odds_group_1_test_model = Model.calib_eq_odds(
        group_0_test_model, group_1_test_model, fp_rate, fn_rate, mix_rates)

    # Print results of the test models and their calibrated versions
    print('Original group 0 model:\n%s\n' % repr(group_0_test_model))
    print('Original group 1 model:\n%s\n' % repr(group_1_test_model))
    print('Equalized odds group 0 model:\n%s\n' % repr(calib_eq_odds_group_0_test_model))
    print('Equalized odds group 1 model:\n%s\n' % repr(calib_eq_odds_group_1_test_model))
