import pandas as pd  # Importing the pandas library for data manipulation

# Function to load and preprocess the dataset based on the data type and sensitive feature

data_paths = {'credit': '/home/li3975/cs587/data/creditcardfraud/creditcard.csv',
             'kiva': '/home/li3975/cs587/data/kiva/kiva_loans.csv',
             'student': '/home/li3975/cs587/data/student-performance-factors/StudentPerformanceFactors.csv'}

# data_paths = {'credit': '../data/creditcardfraud/creditcard.csv',
#              'kiva': '../data/kiva/kiva_loans.csv',
#              'student': '../data/student-performance-factors/StudentPerformanceFactors.csv'}

def dataloader(data, Sensitive_Features):
    # Load the dataset from the specified path (assuming a CSV file format)
    data_path = data_paths[data]
    df = pd.read_csv(data_path)
    df = df.dropna()
    if data == 'credit':
    # Define numeric variables
        numvars = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        
        # Initialize the target variable with the 'Class' column
        target = df[['Class', Sensitive_Features]]

        # Map sensitive feature values if necessary (e.g., binary or categorical encoding)
        # Here, you can customize mappings if certain values in V5 and V16 need encoding as privileged/unprivileged

        # Drop sensitive features from the main dataset to avoid leakage
        df = df.drop(columns=Sensitive_Features)
        
        # Drop the target column 'Class' from the dataset
        dataset = df.drop("Class", axis=1)

        # Identify categorical columns (none are specified, so assuming all remaining are numeric)
        categorical = dataset.columns.difference(numvars)

        # Return the dataset (without sensitive features), target (Class, V5, V16), numeric and categorical features
        return dataset, target, numvars, categorical
    elif data == 'student':
        numvars = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores']

        # Initialize the target variable with the 'Exam_Score' column
        target = df[['Exam_Score', Sensitive_Features]]

        # Map sensitive feature values if necessary
        # if 'Gender' in df.columns:
        #     # Encode Gender: Male as 1, Female as 0
        #     target['Gender'].replace(['Male', 'Female'], [1, 0], inplace=True)

        if 'Family_Income' in df.columns:
            # Example encoding: High income as 2, Medium as 1, Low as 0
            target.replace(['High', 'Medium', 'Low'], [2, 1, 0], inplace=True)

        # # For Parental_Education_Level, you can apply custom mappings if needed
        # if 'Parental_Education_Level' in df.columns:
        #     # Example encoding: Postgraduate as 3, College as 2, High School as 1, None as 0
        #     target['Parental_Education_Level'].replace(['Postgraduate', 'College', 'High School', 'None'], [3, 2, 1, 0], inplace=True)

        # Drop sensitive features from the main dataset to avoid bias leakage
        df = df.drop(columns=Sensitive_Features)

        # Drop the target column 'Exam_Score' from the dataset
        dataset = df.drop("Exam_Score", axis=1)

        # Identify categorical columns by excluding numeric ones
        categorical = dataset.columns.difference(numvars)

        # Return the dataset (without sensitive features), target (Exam_Score and sensitive features), numeric and categorical features
        return dataset, target, numvars, categorical 
    
    elif data == 'kiva':
    # Define numeric variables (assuming these are the relevant numeric features)
        numvars = ['funded_amount', 'loan_amount', 'lender_count']

        # Initialize the target variable (including sensitive features)
        columns_to_drop = ['activity', 'country', 'posted_time', 'disbursed_time', 'funded_time', 'term_in_months']
        df = df.drop(columns=columns_to_drop)
        
        target = df['funded_amount', Sensitive_Features]

        # Map sensitive feature values if necessary
        if 'Borrower_genders' in df.columns:
            # Example encoding: female as 1, male as 0
            target.replace(['female', 'male'], [1, 0], inplace=True)

        # Keep only the relevant numeric features
        dataset = df.drop(columns=Sensitive_Features)
        dataset = df.drop("Borrower_genders", axis=1)
        # Identify categorical columns (these are the remaining columns in the original dataframe)
        categorical = ['sector', 'use', 'partner_id', 'tags', 'repayment_interval', 'country_code']

        # Add categorical data to the dataset
        dataset = pd.concat([dataset, df[categorical]], axis=1)

        # Return the dataset (with numeric and categorical features), target (Borrower_genders and country_code), numeric variables, and categorical features
        return dataset, target, numvars, categorical



def dataloader_old(data, sensitive_feature):
    
    # If the dataset is 'GERMAN', load the corresponding file
    if data.upper() == 'GERMAN':
        # Load the German dataset (assumed to be in TSV format)
        df = pd.read_csv("data/German.tsv", sep='\t')  # Reading the dataset with tab-separated values
        numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'existingcredits', 'peopleliable']
        # List of numeric variables in the dataset that won't be treated as categorical
        
        Sensitive_Features = ['gender', 'foreignworker']  # Defining the sensitive features to consider
        
        # If the sensitive feature is 'gender'
        if sensitive_feature.lower() == 'gender':
            # Select the classification column (target) and the 'gender' sensitive feature
            target = df[['classification', Sensitive_Features[0]]]  # ['classification', 'gender']
            
            # Map privileged and unprivileged groups for 'gender' (Male is privileged, Female is unprivileged)
            mappingPrivUnpriv = {'privilaged': 'M', 'unprivilaged': 'F'}
            
            # Replace the gender column values: 'M' -> 1 (privileged), 'F' -> 0 (unprivileged)
            target.replace(['M', 'F'], [1, 0], inplace=True)
            
            # Drop sensitive features (gender) from the dataset to avoid bias leakage
            df = df.drop(columns=Sensitive_Features)
            
            # Drop the 'classification' column (target variable) from the dataset
            dataset = df.drop("classification", axis=1)
            
            # Identify categorical columns by excluding numeric ones
            categorical = dataset.columns.difference(numvars)
            
            # Return the dataset (without sensitive features), target (classification and gender), numeric and categorical features
            return (dataset, target, numvars, categorical)
        
        # If the sensitive feature is 'foreign worker'
        elif sensitive_feature.lower() == 'foreignworker':
            # Select the classification column (target) and 'foreign worker' sensitive feature
            target = df[['classification', Sensitive_Features[1]]]  # ['classification', 'foreignworker']
            
            # Map privileged and unprivileged groups for 'foreign worker' (no -> privileged, yes -> unprivileged)
            mappingPrivUnpriv = {'privilaged': 'no', 'unprivilaged': 'yes'}
            
            # Replace 'foreign worker' column values: 'no' -> 1 (privileged), 'yes' -> 0 (unprivileged)
            target.replace(['no', 'yes'], [1, 0], inplace=True)
            
            # Drop sensitive features (foreign worker) from the dataset
            df = df.drop(columns=Sensitive_Features)
            
            # Drop the 'classification' column (target variable) from the dataset
            dataset = df.drop("classification", axis=1)
            
            # Identify categorical columns by excluding numeric ones
            categorical = dataset.columns.difference(numvars)
            
            # Return the dataset, target, numeric and categorical features
            return (dataset, target, numvars, categorical)
        
        # Raise an error if the provided sensitive feature doesn't exist in the dataset
        else:
            if sensitive_feature not in df.columns.to_list():
                raise "The required Sensitive Feature does not exist"  # Custom error message
            else:
                raise NotImplementedError  # If the feature exists but is not handled, raise a NotImplementedError

    # If the dataset is 'ADULT', load the corresponding file
    if data.upper() == 'ADULT':
        # Load the Adult dataset (assumed to be in TSV format)
        df = pd.read_csv("data/adult.tsv", sep='\t')
        
        # Drop any rows with missing values
        df = df.dropna()
        
        # List of numeric variables in the dataset
        numvars = ['education-num', 'capital gain', 'capital loss', 'hours per week', 'Age', 'fnlwgt']
        
        # Comment about bias and collinearity: some features should be dropped to avoid bias, but they are kept to match the paper's approach.
        
        # Define the sensitive features in the dataset
        Sensitive_Features = ['gender', 'marital-status']
        
        # If the sensitive feature is 'gender'
        if sensitive_feature.lower() == 'gender':
            # Select the income column (target) and 'gender' sensitive feature
            target = df[["income", "gender"]]
            
            # Replace gender values: 'Male' -> 1 (privileged), 'Female' -> 0 (unprivileged)
            target.replace([' Male', ' Female'], [1, 0], inplace=True)
            
            # Drop sensitive features (gender) from the dataset
            df = df.drop(columns=Sensitive_Features)
            
            # Drop the 'income' column (target variable) from the dataset
            dataset = df.drop("income", axis=1)
            
            # Identify categorical columns by excluding numeric ones
            categorical = dataset.columns.difference(numvars)
            
            # Return the dataset, target, numeric and categorical features
            return (dataset, target, numvars, categorical)
        
        # If the sensitive feature is 'marital-status'
        elif sensitive_feature == 'marital-status':
            # Replace marital status values with binary categories (married and not married)
            df.replace(to_replace=[' Divorced', ' Married-AF-spouse', ' Married-civ-spouse',
                                   ' Married-spouse-absent', ' Never-married', ' Separated', ' Widowed'], 
                       value=['not married', 'married', 'married', 'married', 'not married', 'not married', 'not married'], 
                       inplace=True)
            
            # Select the income column (target) and 'marital-status' sensitive feature
            target = df[["income", "marital-status"]]
            
            # Replace marital status values: 'married' -> 1 (privileged), 'not married' -> 0 (unprivileged)
            target.replace(['married', 'not married'], [1, 0], inplace=True)
            
            # Drop sensitive features (marital-status) from the dataset
            df = df.drop(columns=Sensitive_Features)
            
            # Drop the 'income' column (target variable) from the dataset
            dataset = df.drop("income", axis=1)
            
            # Identify categorical columns by excluding numeric ones
            categorical = dataset.columns.difference(numvars)
            
            # Return the dataset, target, numeric and categorical features
            return (dataset, target, numvars, categorical)
        
        # Raise an error if the provided sensitive feature doesn't exist in the dataset
        else:
            if sensitive_feature not in df.columns.to_list():
                raise "The required Sensitive Feature does not exist"  # Custom error message
            else:
                raise NotImplementedError  # If the feature exists but is not handled, raise a NotImplementedError
