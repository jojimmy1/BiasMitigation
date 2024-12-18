from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
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
    # sensitive_feature = 'Family_Income'
    sensitive_feature = 'borrower_genders'
    # DF ='german' #or 'adult'
    # DF ='STUDENT'
    DF ='KIVA'
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

    # categorical_transformer = Pipeline(
    #     steps=[('onehot', OneHotEncoder(handle_unknown='ignore',sparse=False))])
    
    categorical_transformer = Pipeline(
        steps=[('scaler', StandardScaler())])

    transformations = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numvars),
            ('cat', categorical_transformer, categorical)])

    pipeline = Pipeline(steps=[('preprocessor', transformations)])
    dict_all_result_in_grid = {}
    x_train = pipeline.fit_transform(x_train)
    parameters = {'k': 10, 'Ax': 0.001, 'Ay': 0.1, 'Az': 10.0, 'max_iter': 150000, 'max_fun': 150000}

    lfr = LFR(sensitive_feature=sensitive_feature, privileged_class=1, unprivileged_class=0, seed=42,
              output_feature=classification, parameter=parameters)
    lfr.fit(X=x_train, y=y_train)
    Z_train, y_trainLFR = lfr.transform(X=x_train, y=y_train)

    bal_acc_arr_transf = []
    deo_arr_transf = []
    dao_arr_transf = []
    FairDEO = []
    FairDAO = []
    thresholds = np.linspace(0.01, 0.99, 100)
    
    # Using Logistic Regression instead of SVC
    log_reg = LogisticRegression(max_iter=10000, solver='liblinear')
    # iterate over Z_train, set NaN, infinity or a value too large for dtype('float64') to 0. For label encoding
    Z_train = np.nan_to_num(Z_train)
    log_reg.fit(Z_train, y_train[classification])

    x_test = pipeline.transform(x_test)
    Z_test, y_testLFR = lfr.transform(X=x_test, y=y_test)

    # iterate over Z_test, set NaN, infinity or a value too large for dtype('float64') to 0. For label encoding
    Z_test = np.nan_to_num(Z_test) 
    y_pred = log_reg.predict(Z_test)

    ACC = accuracy(y_test[classification], y_pred)
    DEO = DifferenceEqualOpportunity(y_pred, y_test, sensitive_feature, classification, 1, 0, [0, 1])
    DAO = DifferenceAverageOdds(y_pred, y_test, sensitive_feature, classification, 1, 0, [0, 1])
    cv_score = CVScore(y_pred, y_test, sensitive_feature, classification, 1, 0, [0, 1])
    FairDEO_val = ACC * (1 - DEO)
    FairDAO_val = ACC * (1 - DAO)
    print("Accuracy: {}".format(ACC))
    print("DEO: {}".format(DEO))
    print("DAO: {}".format(DAO))
    print("CV score: {}".format(cv_score))
    print("FairDEO: {}".format(FairDEO_val))
    print("FairDAO: {}".format(FairDAO_val))


    # for thresh in thresholds:
    #     Y_pred = y_trainLFR.copy()
    #     Y_pred[classification] = np.array(Y_pred[classification] > thresh).astype(np.float64)
    #     ACC = accuracy_score(y_train[classification], Y_pred[classification])
    #     DEO = DifferenceEqualOpportunity(Y_pred[classification], y_train, sensitive_feature, classification, 1, 0, [0, 1])
    #     DAO = DifferenceAverageOdds(Y_pred[classification], y_train, sensitive_feature, classification, 1, 0, [0, 1])
    #     bal_acc_arr_transf.append(ACC)
    #     deo_arr_transf.append(DEO)
    #     dao_arr_transf.append(DAO)
    #     FairDEO.append(ACC * (1 - DEO))
    #     FairDAO.append(ACC * (1 - DAO))

    # plt.title("tradeOff Accuracy-Fairness for different thresholds (TRAIN)")
    # plt.plot(thresholds, bal_acc_arr_transf, marker='.')
    # plt.plot(thresholds, deo_arr_transf, marker='.')
    # plt.plot(thresholds, dao_arr_transf, marker='.')
    # plt.plot(thresholds, FairDEO, marker='.')
    # plt.plot(thresholds, FairDAO, marker='.')
    # plt.ylim(0, 1)
    # plt.legend(["ACC", "DEO", "DAO", "F_DEO", "F_DAO"])
    # plt.xlabel("threshold")
    # # plt.show()
    # # save the plot instead of showing it
    # plt.savefig('tradeOff Accuracy-Fairness for different thresholds (TRAIN).png')
    # plt.clf()
    # bal_acc_arr_transf = []
    # deo_arr_transf = []
    # dao_arr_transf = []
    # FairDEO = []
    # FairDAO = []

    # for thresh in thresholds:
    #     Y_pred = y_testLFR.copy()
    #     Y_pred[classification] = np.array(Y_pred[classification] > thresh).astype(np.float64)
    #     ACC = accuracy_score(y_test[classification], Y_pred[classification])
    #     DEO = DifferenceEqualOpportunity(Y_pred[classification], y_test, sensitive_feature, classification, 1, 0, [0, 1])
    #     DAO = DifferenceAverageOdds(Y_pred[classification], y_test, sensitive_feature, classification, 1, 0, [0, 1])
    #     bal_acc_arr_transf.append(ACC)
    #     deo_arr_transf.append(DEO)
    #     dao_arr_transf.append(DAO)
    #     FairDEO.append(ACC * (1 - DEO))
    #     FairDAO.append(ACC * (1 - DAO))

    # plt.title("tradeOff Accuracy-Fairness for different thresholds (TEST)")
    # plt.plot(thresholds, bal_acc_arr_transf, marker='.')
    # plt.plot(thresholds, deo_arr_transf, marker='.')
    # plt.plot(thresholds, dao_arr_transf, marker='.')
    # plt.plot(thresholds, FairDEO, marker='.')
    # plt.plot(thresholds, FairDAO, marker='.')
    # plt.ylim(0, 1)
    # plt.legend(["ACC", "DEO", "DAO", "F_DEO", "F_DAO"])
    # plt.xlabel("threshold")
    # # plt.show()
    # # save the plot instead of showing it
    # plt.savefig('tradeOff Accuracy-Fairness for different thresholds (TEST).png')
    # print("Next the metrics used by zemel, et al.")
    # print("Accuracy: {}".format(accuracy(y_test[classification], y_testLFR[classification])))
    # print("Discrimination: {}".format(discrimination(
    #     y_test, y_testLFR[classification], sensitive_feature,1,0)))
    # print("Consistency: {}".format(consistency(x_test, y_testLFR[classification], k=5)))
