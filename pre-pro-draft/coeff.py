import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# df = pd.read_csv("data/kiva_loans.csv")
df = pd.read_csv("data/student.csv")

# Initialize the encoder
le = LabelEncoder()

# Encode all categorical variables
for column in df.select_dtypes(include=['object']).columns:
    df[f'{column}_encoded'] = le.fit_transform(df[column])

# Select numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# scale the data using MinMaxScaler
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Calculate correlations
correlations = df[numerical_cols].corr()

print("\nCorrelations higher than 0.1 and lower than 1:")
# print(correlations[(correlations > 0.1) & (correlations < 1)])
for i in range(len(correlations)):
    for j in range(i+1, len(correlations)):
        if abs(correlations.iloc[i, j]) > 0.1 and abs(correlations.iloc[i, j]) < 1:
            print(f"{correlations.columns[i]} and {correlations.columns[j]}: {correlations.iloc[i, j]}")



##########################################

# df = pd.read_csv("data/student.csv")

# # Initialize the encoder
# le = LabelEncoder()

# # Encode 'gender' and 'family income'
# df['gender_encoded'] = le.fit_transform(df['Gender'])
# df['family_income_encoded'] = le.fit_transform(df['Family_Income'])

# # Encode all categorical variables
# for column in df.select_dtypes(include=['object']).columns:
#     df[f'{column}_encoded'] = le.fit_transform(df[column])

# # Select numerical columns
# numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# # Calculate correlations
# correlations = df[numerical_cols].corr()

# # Extract correlations for 'gender_encoded' and 'family_income_encoded'
# gender_corr = correlations['gender_encoded'].drop('gender_encoded')
# family_income_corr = correlations['family_income_encoded'].drop('family_income_encoded')

# print(correlations)
# # save to csv
# correlations.to_csv('data/correlations.csv')
# # find corr higher than 0.1
# print("\nCorrelations higher than 0.1 and lower than 1:")
# # print(correlations[(correlations > 0.1) & (correlations < 1)])
# for i in range(len(correlations)):
#     for j in range(i+1, len(correlations)):
#         if correlations.iloc[i, j] > 0.1 and correlations.iloc[i, j] < 1:
#             print(f"{correlations.columns[i]} and {correlations.columns[j]}: {correlations.iloc[i, j]}")

# print("\nCorrelation with Gender:")
# print(gender_corr)
# print("\nCorrelation with Family Income:")
# print(family_income_corr)

# # correlations = df.corr()
# # print(correlations)

