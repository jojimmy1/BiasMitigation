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
import pandas as pd

random.seed(42)
np.random.seed(42)

if __name__=='__main__':
    # Load data
    df = pd.read_csv("data/student.csv", sep=',')
    df = df.dropna()
    needed_feature = ['Hours_Studied','Attendance','Parental_Involvement','Access_to_Resources','Extracurricular_Activities','Sleep_Hours','Previous_Scores', 'Tutoring_Sessions' ,'Family_Income','Teacher_Quality','Gender','Exam_Score']
    # drop columns not in needed_feature
    df = df.drop(columns=df.columns.difference(needed_feature))
    # drop gender
    df = df.drop(columns=['Gender'])
    # if exam score is greater than 67, then 1, else 0
    df['Exam_Score'] = df['Exam_Score'].apply(lambda x: 1 if x > 67 else 0)
    df = df.sample(n=1000, random_state=42)
    # Reset the index of the DataFrame after all filtering and modifications, and drop the old index
    df = df.reset_index(drop=True)

    # Define features and labels
    X = df.drop(columns=['Exam_Score'])
    y = df['Exam_Score']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Preprocessing for numeric and categorical features
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first')

    # Use ColumnTransformer to apply the transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Create a pipeline with preprocessing and logistic regression model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression(max_iter=10000, solver='liblinear', random_state=42))])

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f'Training accuracy: {train_acc:.2f}')
    print(f'Test accuracy: {test_acc:.2f}')


    # # Split data into train and test
    # x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=['Exam_Score']),
    #                                                     df['Exam_Score'],
    #                                                     test_size=0.1,
    #                                                     random_state=42,
    #                                                     stratify=df['Exam_Score'])
    
    # # Use logistic regression to predict the exam score
    # log_reg = LogisticRegression(max_iter=10000, solver='liblinear')
    # log_reg.fit(x_train, y_train)
    # y_pred = log_reg.predict(x_test)
    # ACC = accuracy_score(y_pred, y_test)
    # print("Test Accuracy: {}".format(ACC))
    
