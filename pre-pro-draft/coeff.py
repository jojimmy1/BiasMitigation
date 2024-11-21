import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/student.csv")

# Initialize the encoder
le = LabelEncoder()

# Encode 'gender' and 'family income'
df['gender_encoded'] = le.fit_transform(df['Gender'])
df['family_income_encoded'] = le.fit_transform(df['Family_Income'])

# Select numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Calculate correlations
correlations = df[numerical_cols].corr()

# Extract correlations for 'gender_encoded' and 'family_income_encoded'
gender_corr = correlations['gender_encoded'].drop('gender_encoded')
family_income_corr = correlations['family_income_encoded'].drop('family_income_encoded')

print(correlations)

print("\nCorrelation with Gender:")
print(gender_corr)
print("\nCorrelation with Family Income:")
print(family_income_corr)

