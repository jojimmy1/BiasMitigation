from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from model.utils.metrics import *
from model.utils.dataloader import dataloader
from model.LFR import LFR
import matplotlib.pyplot as plt
import numpy as np
import random

# Set random seeds for reproducibility
random.seed(42)  # Fixes the seed for Python's random module to ensure consistent results
np.random.seed(42)  # Fixes the seed for NumPy's random module for consistency


sensitive_features = {'credit': 'V5', 
                      'student':  'Family_Income',
                      'kiva': 'Borrower_genders', 
}


if __name__ == '__main__':
    dataSet = "student"
    sensitive_feature = sensitive_features[dataSet]
    # Load and preprocess the dataset using a custom dataloader
    # Dataloader returns the dataset, target labels, numerical, and categorical features

    data = dataloader(dataSet, sensitive_feature) 
    dataset, target, numvars, categorical = data  # Unpack the returned values
    
    # print(dataset)
    # print(target)
    

    # Split the dataset into training and testing sets with stratified sampling to maintain target distribution
    x_train, x_test, y_train, y_test = train_test_split(dataset,
                                                        target,
                                                        test_size=0.1,
                                                        random_state=42,
                                                        stratify=target)
    
    
    
    # Remove the sensitive feature ('gender') from the classification target
    classification = target.columns.to_list()  # Get a list of target columns
    classification.remove(sensitive_feature)  # Remove the sensitive feature
    classification = classification[0]  # Select the primary classification target feature
    
    # Create a pipeline to standardize numerical features
    numeric_transformer = Pipeline(
        steps=[('scaler', StandardScaler())])  # Standardizes numerical data by removing the mean and scaling to unit variance

    # Create a pipeline to one-hot encode categorical features
    categorical_transformer = Pipeline(
        steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])  # Transforms categorical variables into binary vectors

    # Combine numerical and categorical transformations
    transformations = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numvars),  # Apply numeric transformation to numeric columns
            ('cat', categorical_transformer, categorical)])  # Apply categorical transformation to categorical columns

    # Build a pipeline that preprocesses both numerical and categorical data
    pipeline = Pipeline(steps=[('preprocessor', transformations)])  
    dict_all_result_in_grid = {}  # Initialize an empty dictionary to store grid search results

    # Apply the transformation pipeline to the training data
    x_train = pipeline.fit_transform(x_train)

    # Set parameters for the LFR fairness transformation model
    parameters = {'k': 10, 'Ax': 0.001, 'Ay': 0.1, 'Az': 10.0, 'max_iter': 150000, 'max_fun': 150000}

    # Instantiate the LFR (Learning Fair Representations) model
    lfr = LFR(sensitive_feature=sensitive_feature, privileged_class=1, unprivileged_class=0, seed=42,
              output_feature=classification, parameter=parameters)

    # Train the LFR model on the transformed training data
    lfr.fit(X=x_train, y=y_train)

    # Transform the training data into fair representations using LFR
    Z_train, y_trainLFR = lfr.transform(X=x_train, y=y_train)

    # Initialize arrays to store results for accuracy and fairness metrics
    bal_acc_arr_transf = []
    deo_arr_transf = []
    dao_arr_transf = []
    FairDEO = []
    FairDAO = []

    # Define a range of thresholds to evaluate the trade-off between accuracy and fairness
    thresholds = np.linspace(0.01, 0.99, 100)

    # Initialize and train an SVM classifier with a linear kernel on the transformed training data
    svc = SVC(kernel='linear')
    svc.fit(Z_train, y_train[classification])

    # Apply the transformation pipeline to the test data
    x_test = pipeline.transform(x_test)

    # Transform the test data using the LFR model
    Z_test, y_testLFR = lfr.transform(X=x_test, y=y_test)

    # Predict the classification labels for the test set using the trained SVM
    y_pred = svc.predict(Z_test)

    # Calculate accuracy of the SVM predictions on the test set
    ACC = accuracy_score(y_pred, y_test[classification])

    # Loop through each threshold and evaluate the accuracy and fairness metrics
    for thresh in thresholds:
        Y_pred = y_trainLFR.copy()  # Copy the training labels after LFR transformation
        Y_pred[classification] = np.array(Y_pred[classification] > thresh).astype(np.float64)  # Apply threshold to predictions
        ACC = accuracy_score(y_train[classification], Y_pred[classification])  # Calculate accuracy
        DEO = DifferenceEqualOpportunity(Y_pred[classification], y_train, sensitive_feature, classification, 1, 0, [0, 1])  # Equal opportunity difference
        DAO = DifferenceAverageOdds(Y_pred[classification], y_train, sensitive_feature, classification, 1, 0, [0, 1])  # Average odds difference
        bal_acc_arr_transf.append(ACC)
        deo_arr_transf.append(DEO)
        dao_arr_transf.append(DAO)
        FairDEO.append(ACC * (1 - DEO))  # Weighted fairness with DEO
        FairDAO.append(ACC * (1 - DAO))  # Weighted fairness with DAO

    # Plot the trade-off between accuracy and fairness metrics for the training set
    plt.title("tradeOff Accuracy-Fairness for different thresholds (TRAIN)")
    plt.plot(thresholds, bal_acc_arr_transf, marker='.')
    plt.plot(thresholds, deo_arr_transf, marker='.')
    plt.plot(thresholds, dao_arr_transf, marker='.')
    plt.plot(thresholds, FairDEO, marker='.')
    plt.plot(thresholds, FairDAO, marker='.')
    plt.ylim(0, 1)
    plt.legend(["ACC", "DEO", "DAO", "F_DEO", "F_DAO"])
    plt.xlabel("threshold")
    plt.show()

    # Clear arrays to store test results
    bal_acc_arr_transf = []
    deo_arr_transf = []
    dao_arr_transf = []
    FairDEO = []
    FairDAO = []

    # Evaluate accuracy and fairness metrics on the test data
    for thresh in thresholds:
        Y_pred = y_testLFR.copy()
        Y_pred[classification] = np.array(Y_pred[classification] > thresh).astype(np.float64)
        ACC = accuracy_score(y_test[classification], Y_pred[classification])
        DEO = DifferenceEqualOpportunity(Y_pred[classification], y_test, sensitive_feature, classification, 1, 0, [0, 1])
        DAO = DifferenceAverageOdds(Y_pred[classification], y_test, sensitive_feature, classification, 1, 0, [0, 1])
        bal_acc_arr_transf.append(ACC)
        deo_arr_transf.append(DEO)
        dao_arr_transf.append(DAO)
        FairDEO.append(ACC * (1 - DEO))
        FairDAO.append(ACC * (1 - DAO))

    # Plot the trade-off between accuracy and fairness metrics for the test set
    plt.title("tradeOff Accuracy-Fairness for different thresholds (TEST)")
    plt.plot(thresholds, bal_acc_arr_transf, marker='.')
    plt.plot(thresholds, deo_arr_transf, marker='.')
    plt.plot(thresholds, dao_arr_transf, marker='.')
    plt.plot(thresholds, FairDEO, marker='.')
    plt.plot(thresholds, FairDAO, marker='.')
    plt.ylim(0, 1)
    plt.legend(["ACC", "DEO", "DAO", "F_DEO", "F_DAO"])
    plt.xlabel("threshold")
    plt.show()

    # Print accuracy and fairness metrics based on Zemel's method
    print("Next the metrics used by zemel, et al.")
    print("Accuracy: {}".format(accuracy(y_test[classification], y_testLFR[classification])))
    print("Discrimination: {}".format(discrimination(
        y_test, y_testLFR[classification], sensitive_feature, 1, 0)))
    print("Consistency: {}".format(consistency(x_test, y_testLFR[classification], k=5)))
