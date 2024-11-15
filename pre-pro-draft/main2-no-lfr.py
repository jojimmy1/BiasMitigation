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

random.seed(42)
np.random.seed(42)


if __name__=='__main__':
    # sensitive_feature = 'Gender'
    sensitive_feature = 'gender'
    DF ='german' #or 'adult'
    # DF ='STUDENT'
    data = dataloader(DF, sensitive_feature =sensitive_feature) # else adult
    dataset, target, numvars, categorical = data
    # Split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(dataset,
                                                        target,
                                                        test_size=0.1,
                                                        random_state=42,
                                                        stratify=target)
    classification = target.columns.to_list()
    classification.remove(sensitive_feature)
    classification = classification[0]
    # We create the preprocessing pipelines for both numeric and categorical data.
    numeric_transformer = Pipeline(
        steps=[('scaler', StandardScaler())])

    categorical_transformer = Pipeline(
        steps=[('onehot', OneHotEncoder(handle_unknown='ignore',sparse=False))])

    transformations = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numvars),
            ('cat', categorical_transformer, categorical)])

    pipeline = Pipeline(steps=[('preprocessor', transformations)])
    dict_all_result_in_grid = {}
    x_train = pipeline.fit_transform(x_train)
    parameters = {'k': 10, 'Ax': 0.001, 'Ay': 0.1, 'Az': 10.0, 'max_iter': 150000, 'max_fun': 150000}

    # lfr = LFR(sensitive_feature=sensitive_feature, privileged_class=1, unprivileged_class=0, seed=42,
    #           output_feature=classification, parameter=parameters)
    # lfr.fit(X=x_train, y=y_train)
    # Z_train, y_trainLFR = lfr.transform(X=x_train, y=y_train)


    bal_acc_arr_transf = []
    deo_arr_transf = []
    dao_arr_transf = []
    FairDEO = []
    FairDAO = []
    thresholds = np.linspace(0.01, 0.99, 100)
    svc = SVC(kernel='linear')
    svc.fit(x_train, y_train[classification])

    x_test = pipeline.transform(x_test)
    # Z_test, y_testLFR = lfr.transform(X=x_test, y=y_test)

    y_pred = svc.predict(x_test)

    y_train_pred = svc.predict(x_train)
    print(y_pred.shape)
    print(y_test.shape)
    print(y_train_pred.shape)
    print(y_train_pred[:10])
    print(classification)
    ACC = accuracy_score(y_pred, y_test[classification])
    print("Accuracy: {}".format(ACC))
    for thresh in thresholds:
        Y_pred = y_train.copy()
        Y_pred[classification] = np.array(Y_pred[classification] > thresh).astype(np.float64)
        ACC = accuracy_score(y_train[classification], Y_pred[classification])
        DEO = DifferenceEqualOpportunity(Y_pred[classification], y_train, sensitive_feature, classification, 1, 0, [0, 1])
        DAO = DifferenceAverageOdds(Y_pred[classification], y_train, sensitive_feature, classification, 1, 0, [0, 1])
        bal_acc_arr_transf.append(ACC)
        deo_arr_transf.append(DEO)
        dao_arr_transf.append(DAO)
        FairDEO.append(ACC * (1 - DEO))
        FairDAO.append(ACC * (1 - DAO))

    plt.title("tradeOff Accuracy-Fairness for different thresholds (TRAIN)")
    plt.plot(thresholds, bal_acc_arr_transf, marker='.')
    plt.plot(thresholds, deo_arr_transf, marker='.')
    plt.plot(thresholds, dao_arr_transf, marker='.')
    plt.plot(thresholds, FairDEO, marker='.')
    plt.plot(thresholds, FairDAO, marker='.')
    plt.ylim(0, 1)
    plt.legend(["ACC", "DEO", "DAO", "F_DEO", "F_DAO"])
    plt.xlabel("threshold")
    # plt.show()
    # save the plot instead of showing it
    plt.savefig('tradeOff Accuracy-Fairness for different thresholds (TRAIN).png')
    plt.clf()
    bal_acc_arr_transf = []
    deo_arr_transf = []
    dao_arr_transf = []
    FairDEO = []
    FairDAO = []

    print(Y_pred[classification][:10])
    print(Y_pred.shape)
    print(Y_pred.columns)
    print(Y_pred[classification].shape)
    for thresh in thresholds:
        Y_pred = y_test.copy()
        Y_pred[classification] = np.array(Y_pred[classification] > thresh).astype(np.float64)
        ACC = accuracy_score(y_test[classification], Y_pred[classification])
        DEO = DifferenceEqualOpportunity(Y_pred[classification], y_test, sensitive_feature, classification, 1, 0, [0, 1])
        DAO = DifferenceAverageOdds(Y_pred[classification], y_test, sensitive_feature, classification, 1, 0, [0, 1])
        bal_acc_arr_transf.append(ACC)
        deo_arr_transf.append(DEO)
        dao_arr_transf.append(DAO)
        FairDEO.append(ACC * (1 - DEO))
        FairDAO.append(ACC * (1 - DAO))

    plt.title("tradeOff Accuracy-Fairness for different thresholds (TEST)")
    plt.plot(thresholds, bal_acc_arr_transf, marker='.')
    plt.plot(thresholds, deo_arr_transf, marker='.')
    plt.plot(thresholds, dao_arr_transf, marker='.')
    plt.plot(thresholds, FairDEO, marker='.')
    plt.plot(thresholds, FairDAO, marker='.')
    plt.ylim(0, 1)
    plt.legend(["ACC", "DEO", "DAO", "F_DEO", "F_DAO"])
    plt.xlabel("threshold")
    # plt.show()
    # save the plot instead of showing it
    plt.savefig('tradeOff Accuracy-Fairness for different thresholds (TEST).png')
    print("Next the metrics used by zemel, et al.")
    print("Accuracy: {}".format(accuracy(y_test[classification], y_test[classification])))
    print("Discrimination: {}".format(discrimination(
        y_test, y_test[classification], sensitive_feature,1,0)))
    print("Consistency: {}".format(consistency(x_test, y_test[classification], k=5)))
