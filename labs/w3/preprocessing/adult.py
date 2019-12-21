import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

CATEGORICAL_FEATURES = ['workclass', 'education', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'native-country']

NUMERICAL_FEATURES = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']


def preprocess(train_dataset, validation_dataset):
    train_data = train_dataset['data']
    val_data = validation_dataset['data']

    # TODO: Remove sample
    df_train = pd.DataFrame(train_data)

    df_val = pd.DataFrame(val_data)

    df_train = df_train.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    df_val = df_val.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    # Real Y labels
    le = LabelEncoder()
    y_train = df_train['class'].copy()
    y_train = le.fit_transform(y_train)

    y_val = df_val['class'].copy()
    y_val = le.transform(y_val)

    df_train = df_train.drop(columns=['class'])
    df_val = df_val.drop(columns=['class'])

    # Encode categorical values into numerical with OHE
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    X_categorical_train = ohe.fit_transform(df_train[CATEGORICAL_FEATURES])
    X_categorical_val = ohe.transform(df_val[CATEGORICAL_FEATURES])

    columns = ohe.get_feature_names(input_features=CATEGORICAL_FEATURES)
    X_categorical_train = pd.DataFrame(data=X_categorical_train, columns=columns)
    X_categorical_val = pd.DataFrame(data=X_categorical_val, columns=columns)

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
