import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import joblib

# Global Mappings
GENDER_MAP = {'Male':0,'Female':1}
PAYMENT_HISTORY_MAP = {'On-Time':0, 'Delayed':1, 'Missed':2}
LEGAL_ACTION_MAP = {'No':0,'Yes':1}
LOAN_TYPE_MAP = {'Home': 0, 'Auto': 1, 'Personal': 2, 'Business': 3}
RECOVERY_STATUS_MAP = {'Fully Recovered':0, 'Partially Recovered':1, 'Written Off':2}


SEGMENT_NAME_MAP = {
    0: 'Stable Income, Manageable Loan, Low Risk',
    1: 'High Loan Burden, Critical Risk',
    2: 'High Missed Payments, Behavioral Risk',
    3: 'High Income, Large Loan, Managed Risk'
}


def save_scaler(scaler, path):
    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)

def save_kmeans_model(kmeans_model, path):
    joblib.dump(kmeans_model, path)

def load_kmeans_model(path):
    return joblib.load(path)

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)



def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    return df

def convert_to_numeric(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()

    if 'Gender' in df_copy.columns: df_copy['Gender'] = df_copy['Gender'].map(GENDER_MAP)
    if 'Payment_History' in df_copy.columns: df_copy['Payment_History'] = df_copy['Payment_History'].map(PAYMENT_HISTORY_MAP)
    if 'Legal_Action_Taken' in df_copy.columns: df_copy['Legal_Action_Taken'] = df_copy['Legal_Action_Taken'].map(LEGAL_ACTION_MAP)
    if 'Loan_Type' in df_copy.columns: df_copy['Loan_Type'] = df_copy['Loan_Type'].map(LOAN_TYPE_MAP)

    if 'Employment_Type' in df_copy.columns:
        df_copy = pd.get_dummies(df_copy, columns=['Employment_Type'], prefix='Employment_Type', drop_first=True)
    if 'Collection_Method' in df_copy.columns:
        df_copy = pd.get_dummies(df_copy, columns=['Collection_Method'], prefix='Collection_Method', drop_first=True)

    return df_copy

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    if 'Loan_Amount' in df_copy.columns and 'Monthly_Income' in df_copy.columns:
        df_copy['Loan_to_income_ratio'] = df_copy['Loan_Amount'] / (df_copy['Monthly_Income'] + 1e-6)
    if 'Monthly_EMI' in df_copy.columns and 'Monthly_Income' in df_copy.columns:
        df_copy['EMI_to_income_ratio'] = df_copy['Monthly_EMI'] / (df_copy['Monthly_Income'] + 1e-6)
    if 'Collateral_Value' in df_copy.columns:
        df_copy['Has_Collateral'] = (df_copy['Collateral_Value'] > 0).astype(int)

    return df_copy

def encode_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    if 'Recovery_Status' in df_copy.columns:
        df_copy['Recovery_Status'] = df_copy['Recovery_Status'].map(RECOVERY_STATUS_MAP)
    return df_copy

def drop_unwanted_features(df: pd.DataFrame, features_to_drop: list) -> pd.DataFrame:
    df_copy = df.copy()
    final_features_to_drop = [f for f in features_to_drop if f in df_copy.columns]
    df_copy.drop(columns=final_features_to_drop, inplace=True, errors='ignore')
    return df_copy

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_column_names(df)
    numerical_cols_to_convert_all = ['Monthly_Income', 'Loan_Amount', 'Outstanding_Loan_Amount', 'Monthly_EMI', 'Collateral_Value',
                                     'Age', 'Num_Dependents', 'Loan_Tenure', 'Interest_Rate', 'Num_Missed_Payments', 'Days_Past_Due', 'Collection_Attempts']
    df = convert_to_numeric(df, numerical_cols_to_convert_all)
    df.dropna(inplace=True)
    df = encode_categorical_features(df)
    df = create_derived_features(df)
    df = encode_target_variable(df)
    df = drop_unwanted_features(df, ['Loan_ID'])
    return df