import pickle
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def split_data(df, features, target):
    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


def lr_pipeline(X, y, numerical_features, categorical_features):
    preprocessor = ColumnTransformer([
        ('StandardScaler', StandardScaler(), numerical_features),
        ('OneHotEncoder', OneHotEncoder(
            drop='if_binary', sparse_output=False),
            categorical_features)
    ]).set_output(transform='pandas')

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])

    model_pipeline.fit(X, y)
    return model_pipeline


def train_model(df, numerical_features, categorical_features, target):
    '''
    Train LinearRegression model
    '''
    # Split data
    features = numerical_features + categorical_features
    X_train, X_test, y_train, y_test = split_data(df, features, target)

    # Train LR model
    lr_model = lr_pipeline(X_train, y_train, numerical_features, categorical_features)

    # Return fitted model
    return lr_model


def save_model(lr_model, lr_model_path):
    '''
    Serialize and save models
    '''
    # Save LR model
    with open(lr_model_path, 'wb') as file:
        pickle.dump(lr_model, file)

