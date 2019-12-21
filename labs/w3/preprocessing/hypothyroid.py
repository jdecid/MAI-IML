import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

CATEGORICAL_FEATURES = ['sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication', 'sick', 'pregnant',
                        'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium',
                        'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH_measured', 'T3_measured', 'TT4_measured',
                        'T4U_measured', 'FTI_measured', 'TBG_measured', 'referral_source']

NUMERICAL_FEATURES = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']


def preprocess(train_dataset, validation_dataset):
    train_data = train_dataset['data']
    val_data = validation_dataset['data']

    df_train = pd.DataFrame(train_data)
    df_val = pd.DataFrame(val_data)

    df_train = df_train.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    df_val = df_val.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    # Real Y labels
    le = LabelEncoder()
    y_train = df_train['Class'].copy()
    y_train = le.fit_transform(y_train)

    y_val = df_val['Class'].copy()
    y_val = le.transform(y_val)

    # Drop class and NaN columns
    df_train = df_train.drop(columns=['Class', 'TBG'])
    df_val = df_val.drop(columns=['Class'])

    # Encode categorical values into numerical with OHE
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    X_categorical_train = ohe.fit_transform(df_train[CATEGORICAL_FEATURES])
    X_categorical_val = ohe.transform(df_val[CATEGORICAL_FEATURES])

    columns = ohe.get_feature_names(input_features=CATEGORICAL_FEATURES)
    X_categorical_train = pd.DataFrame(data=X_categorical_train, columns=columns)
    X_categorical_val = pd.DataFrame(data=X_categorical_val, columns=columns)

    # Fill NA values
    mean_values = df_train[NUMERICAL_FEATURES].mean(axis=0, skipna=True)
    df_train = df_train[NUMERICAL_FEATURES].fillna(mean_values)
    df_val = df_val[NUMERICAL_FEATURES].fillna(mean_values)

    # Scale numerical values
    sc = MinMaxScaler()
    X_numerical_train = sc.fit_transform(df_train[NUMERICAL_FEATURES])
    X_numerical_val = sc.transform(df_val[NUMERICAL_FEATURES])

    X_numerical_train = pd.DataFrame(data=X_numerical_train, columns=NUMERICAL_FEATURES)
    X_numerical_val = pd.DataFrame(data=X_numerical_val, columns=NUMERICAL_FEATURES)

    # Numerical only data
    X_df_train = pd.concat((X_categorical_train, X_numerical_train), axis=1)
    X_df_val = pd.concat((X_categorical_val, X_numerical_val), axis=1)

    return (X_df_train.values, y_train), (X_df_val.values, y_val)
