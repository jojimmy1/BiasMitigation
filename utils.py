import pandas as pd
from sklearn.preprocessing import LabelEncoder

class KivaDataProcessor:
    def __init__(self, file_path, output_path):
        self.file_path = file_path
        self.output_path = output_path
        self.data = None

    def label_kiva(self):
        # Load the data
        self.data = pd.read_csv(self.file_path)

        # Drop unnecessary columns
        self.data = self.data.drop(columns=[
            'id', 'activity', 'country', 'posted_time', 
            'disbursed_time', 'funded_time', 'tags', 'use', 
            'region', 'date'
        ])

        # Drop rows with missing values
        self.data = self.data.dropna()

        # Delete data points that represent multiple borrowers
        self.data = self.data[~self.data['borrower_genders'].str.contains(',', na=False)]
        
        # Initialize LabelEncoder
        le = LabelEncoder()

        # Encode categorical columns
        self.data['country_code'] = le.fit_transform(self.data['country_code'])
        self.data['currency'] = le.fit_transform(self.data['currency'])
        self.data['repayment_interval'] = le.fit_transform(self.data['repayment_interval'])
        self.data['sector'] = le.fit_transform(self.data['sector'])
        self.data['borrower_genders'] = le.fit_transform(self.data['borrower_genders'])

        # Save the cleaned data
        self.data.to_csv(self.output_path, index=False)
        print(f"Data saved to {self.output_path}")

    def onehot_kiva(self):
        self.data = pd.read_csv(self.file_path)
        self.data = self.data.drop(columns=['id',  'activity', 'country', 'posted_time', 
                          'disbursed_time', 'funded_time', 'tags', 'use', 
                          'region', 'date'])
        self.data = self.data.dropna()
        self.data = self.data[~self.data['borrower_genders'].str.contains(',', na=False)]
        self.data = pd.get_dummies(self.data, columns=['country_code', 'currency', 
                                                       'repayment_interval', 'sector', 'borrower_genders'])
        self.data.to_csv(self.output_path, index=False)
        print(f"Data saved to {self.output_path}")