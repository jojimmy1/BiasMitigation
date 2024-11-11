import pandas as pd


def dataloader(data, sensitive_feature):
    if data.upper() == 'KIVA':
        df = pd.read_csv("data/kiva_loans.csv", sep=',')
        df = df.dropna()
        # Remove rows where 'borrower_genders' is neither 'male' nor 'female'
        df = df[df['borrower_genders'].isin(['male', 'female'])]
        needed_feature = ['sector', 'partner_id', 'term_in_months', 'lender_count', 'repayment_interval']
        # drop columns not in needed_feature, not in sensitive_feature and not 'loan_amount'
        df = df.drop(columns=df.columns.difference(needed_feature + ['borrower_genders', 'country_code', 'loan_amount']))
        # label encode for sector, country_code, borrower_genders, repayment_interval
        df['sector'] = df['sector'].astype('category').cat.codes
        df['country_code'] = df['country_code'].astype('category').cat.codes
        df['borrower_genders'] = df['borrower_genders'].astype('category').cat.codes
        df['repayment_interval'] = df['repayment_interval'].astype('category').cat.codes
        # convert 'partner_id' and 'term_in_months' to int
        df['partner_id'] = pd.to_numeric(df['partner_id'], errors='coerce').fillna(0).astype(int)
        df['term_in_months'] = pd.to_numeric(df['term_in_months'], errors='coerce').fillna(0).astype(int)
        df['loan_amount'] = pd.to_numeric(df['loan_amount'], errors='coerce').fillna(0).astype(int)
        # Reset the index of the DataFrame after all filtering and modifications, and drop the old index
        df = df.reset_index(drop=True)
        numvars = ['term_in_months', 'lender_count']
        Sensitive_Features = ['borrower_genders', 'country_code']
        if sensitive_feature.lower() == 'borrower_genders':
            target = df[['loan_amount', Sensitive_Features[0]]]
            df = df.drop(columns=Sensitive_Features)
            dataset = df.drop("loan_amount", axis=1)
            categorical = dataset.columns.difference(numvars)
            return (dataset, target, numvars, categorical)
    if data.upper() =='GERMAN':
        df = pd.read_csv("data/German.tsv", sep='\t')
        numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'existingcredits', 'peopleliable']
        Sensitive_Features = ['gender', 'foreignworker']
        if sensitive_feature.lower() =='gender':
            target = df[['classification', Sensitive_Features[0]]]
            mappingPrivUnpriv = {'privilaged': 'M', 'unprivilaged': 'F'}
            target.replace(['M', 'F'], [1, 0], inplace=True)
            df = df.drop(columns=Sensitive_Features)
            dataset = df.drop("classification", axis=1)
            categorical = dataset.columns.difference(numvars)
            return (dataset,target,numvars,categorical)
        elif sensitive_feature.lower() == 'foreignworker':
            target = df[['classification', Sensitive_Features[1]]]
            mappingPrivUnpriv = {'privilaged': 'no', 'unprivilaged': 'yes'}
            target.replace(['no', 'yes'], [1, 0], inplace=True)
            df = df.drop(columns=Sensitive_Features)
            dataset = df.drop("classification", axis=1)
            categorical = dataset.columns.difference(numvars)
            return (dataset, target, numvars, categorical)
        else:
            if sensitive_feature not in df.columns.to_list():
                raise "The required Sensitive Feature does not exist"
            else:
                raise NotImplementedError


    if data.upper() == 'ADULT':
        df = pd.read_csv("data/adult.tsv", sep='\t')
        df = df.dropna()
        numvars = ['education-num', 'capital gain', 'capital loss', 'hours per week','Age','fnlwgt']
        #******************************FOLLOWING, MY CONSIDERATION*****************************************************
        # to be fair the next features should be dropped since source of bias and collinearity, however in
        # the paper the n° of features are 14 so the authors make use of them
        #df = df.drop(columns=['Age', 'race', 'relationship', 'fnlwgt', 'education', 'native-country'])
        Sensitive_Features = ['gender', 'marital-status']
        if sensitive_feature.lower() == 'gender':
            target = df[["income", "gender"]]  # 'marital-status'
            target.replace([' Male', ' Female'], [1, 0], inplace=True)
            df = df.drop(columns=Sensitive_Features)
            dataset = df.drop("income", axis=1)
            categorical = dataset.columns.difference(numvars)
            return (dataset, target, numvars, categorical)
        elif sensitive_feature == 'marital-status':
            df.replace(to_replace=[' Divorced', ' Married-AF-spouse',
                                   ' Married-civ-spouse', ' Married-spouse-absent',
                                   ' Never-married', ' Separated', ' Widowed'], value=
                       ['not married', 'married', 'married', 'married',
                        'not married', 'not married', 'not married'], inplace=True)
            target = df[["income", "marital-status"]]
            target.replace(['married', 'not married'], [1, 0], inplace=True)
            df = df.drop(columns=Sensitive_Features)
            dataset = df.drop("income", axis=1)
            categorical = dataset.columns.difference(numvars)
            return (dataset, target, numvars, categorical)
        else:
            if sensitive_feature not in df.columns.to_list():
                raise "The required Sensitive Feature does not exist"
            else:
                raise NotImplementedError