import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from utils.dataset import read_dataset


def preprocess():
    dataset = read_dataset('adult')
    data = dataset['data']

    df = pd.DataFrame(data)
    df = df.sample(n=5000, replace=False, random_state=1)
    df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    # Real Y labels
    y = df['class'].copy()
    df = df.drop(columns=['class'])

    categorical_features = ['workclass',
                            'education',
                            'marital-status',
                            'occupation',
                            'relationship',
                            'race',
                            'sex',
                            'native-country']

    numerical_features = ['age',
                          'fnlwgt',
                          'education-num',
                          'capital-gain',
                          'capital-loss',
                          'hours-per-week']

    # Encode categorical values into numerical with OHE
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    X_categorical = ohe.fit_transform(df[categorical_features])

    columns = ohe.get_feature_names(input_features=categorical_features)
    X_categorical = pd.DataFrame(data=X_categorical, columns=columns)

    # Scale numerical values
    sc = MinMaxScaler()
    X_numerical = sc.fit_transform(df[numerical_features])
    X_numerical = pd.DataFrame(data=X_numerical, columns=numerical_features)

    # All to categorical
    X_numerical_as_categorical = X_numerical.copy()
    for feat in numerical_features:
        X_numerical_as_categorical[feat] = pd.qcut(x=X_numerical[feat], q=5, duplicates='drop')

    # Mix data
    X_df = pd.concat((df[categorical_features], X_numerical), axis=1)

    # Numerical only data
    X_df_num = pd.concat((X_categorical, X_numerical), axis=1)

    # Categorical only data
    X_df_cat = pd.concat((X_categorical, X_numerical_as_categorical), axis=1)

    # In[34]:

    X_df.to_csv(os.path.join('datasets', 'adult_clean.csv'), index=False)
    X_df_num.to_csv(os.path.join('datasets', 'adult_clean_num.csv'), index=False)
    X_df_cat.to_csv(os.path.join('datasets', 'adult_clean_cat.csv'), index=False)

    y.to_csv(os.path.join('datasets', 'adult_clean_y.csv'), index=False, header=False)

    return 'adult_clean_num.csv', 'adult_clean_cat.csv', 'adult_clean.csv', 'adult_clean_y.csv'
