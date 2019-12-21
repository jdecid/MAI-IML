import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def preprocess(train_dataset, validation_dataset):
    train_data = train_dataset['data']
    val_data = validation_dataset['data']

    # TODO: Remove sample
    df_train = pd.DataFrame(train_data)
    column_names = df_train.columns[:-1]
    df_train = df_train.sample(n=5000, replace=False, random_state=1).reset_index(drop=True)

    df_val = pd.DataFrame(val_data)

    df_train = df_train.applymap(lambda x: int(x.decode('utf-8')) if isinstance(x, bytes) else x)
    df_val = df_val.applymap(lambda x: int(x.decode('utf-8')) if isinstance(x, bytes) else x)


    # Real Y labels
    le = LabelEncoder()
    y_train = df_train['a17'].copy()
    y_train = le.fit_transform(y_train)

    y_val = df_val['a17'].copy()
    y_val = le.transform(y_val)

    df_train = df_train.drop(columns=['a17'])
    df_val = df_val.drop(columns=['a17'])

    # Scale numerical values
    sc = MinMaxScaler()
    X_train = sc.fit_transform(df_train)
    X_val = sc.transform(df_val)

    X_train = pd.DataFrame(data=X_train, columns=column_names)
    X_val = pd.DataFrame(data=X_val, columns=column_names)

    return (X_train.values, y_train), (X_val.values, y_val)

