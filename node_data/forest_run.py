import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def first_rand():
    features = pd.read_csv('dataset_9.csv', index_col=False)
    print('The shape of our features is:', features.shape)

    other_data = pd.read_csv('c_78_p_0.75_graph_WS_dataset_CIFAR10_seed_10_name_first_gen.csv',
                             index_col=0)
    other_data = other_data.drop('connection', axis=1)
    other_data = other_data.drop('seed', axis=1)
    other_data = other_data.drop('weight', axis=1)

    print(other_data)

    # Labels are the values we want to predict
    labels = np.array(features['weight'])  # Remove the labels from the features
    # axis 1 refers to the columns
    features = features.drop('weight', axis=1)  # Saving feature names for later use
    features = features.drop('seed', axis=1)
    features = features.drop('connection', axis=1)

    print(features)
    feature_list = list(features.columns)  # Convert to numpy array
    features = np.array(features)

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3,
                                                                                random_state=42)

    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    rf = RandomForestRegressor(n_estimators=1000, random_state=42)  # Train the model on training data
    rf.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)  # Calculate the absolute errors
    errors = abs(predictions - test_labels)  # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2))

    new_weights = rf.predict(other_data)

    correct_weights = pd.read_csv('c_78_p_0.75_graph_WS_dataset_CIFAR10_seed_10_name_first_gen.csv',
                                  index_col=0)

    correct_weights.weight = new_weights

    correct_weights.to_csv('new_first_gen_10.csv')

    # Calculate mean absolute percentage error (MAPE)
    mean_error = 100 * (errors / test_labels)  # Calculate and display accuracy
    accuracy = 100 - np.mean(mean_error)
    print('Accuracy:', round(accuracy, 2), '%.')
    print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, predictions))
    print('Mean Squared Error:', metrics.mean_squared_error(test_labels, predictions))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, predictions)))


def second_rand():
    dataset = pd.read_csv('dataset_9.csv')

    x = dataset.iloc[:, 3:].values
    y = dataset.iloc[:, 2].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    regression = RandomForestRegressor(n_estimators=1000, random_state=0)
    regression.fit(x_train, y_train)
    y_pred = regression.predict(x_test)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


def fifth_rand():
    features = pd.read_csv('dataset_9.csv', index_col=False)

    other_data = pd.read_csv('c_78_p_0.75_graph_WS_dataset_CIFAR10_seed_10_name_first_gen.csv',
                             index_col=0)
    other_data = other_data.drop('connection', axis=1)
    other_data = other_data.drop('seed', axis=1)
    other_data = other_data.drop('weight', axis=1)

    # Labels are the values we want to predict
    labels = np.array(features['weight'])  # Remove the labels from the features
    # axis 1 refers to the columns
    features = features.drop('weight', axis=1)  # Saving feature names for later use
    features = features.drop('seed', axis=1)
    features = features.drop('connection', axis=1)

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3,
                                                                                random_state=42)

    regr = RandomForestRegressor(n_estimators=1000, random_state=42)
    regr.fit(train_features, train_labels)

    predictions = regr.predict(test_features)

    print(train_features)
    # Define x axis
    x_axis = test_labels

    # Build scatterplot
    plt.scatter(x_axis, test_labels, c='b', alpha=0.5, marker='.', label='Real')
    plt.scatter(x_axis, predictions, c='r', alpha=0.5, marker='.', label='Predicted')
    plt.xlabel('level')
    plt.ylabel('weight')
    plt.grid(color='#D3D3D3', linestyle='solid')
    plt.legend(loc='lower right')
    plt.show()

    mae = metrics.mean_absolute_error(test_labels, predictions)

    # Mean squared error (MSE)
    mse = metrics.mean_squared_error(test_labels, predictions)

    # R-squared scores
    r2 = metrics.r2_score(test_labels, predictions)

    # Print metrics
    print('Mean Absolute Error:', round(mae, 2))
    print('Mean Squared Error:', round(mse, 2))
    print('R-squared scores:', round(r2, 2))

    parameters = {
        'max_depth': [70, 80, 90, 100],
        'n_estimators': [600, 1000, 1100]
    }

    gridforest = GridSearchCV(regr, parameters, cv=10, n_jobs=-1, verbose=1)
    gridforest.fit(train_features, train_labels)
    gridforest.best_params_

    characteristics = features.columns

    importances = list(regr.feature_importances_)
    characteristics_importances = [(characteristic, round(importance, 2)) for characteristic, importance in
                                   zip(characteristics, importances)]
    characteristics_importances = sorted(characteristics_importances, key=lambda x: x[1], reverse=True)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in characteristics_importances]

    plt.bar(characteristics, importances, orientation='vertical')
    plt.xticks(rotation='vertical')
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.grid(axis='y', color='#D3D3D3', linestyle='solid')
    plt.show()


def join_dataset():

    dataset = []
    for j in range(1, 11):
        data = pd.read_csv(f'c_78_p_0.75_graph_WS_dataset_CIFAR10_seed_{j}_name_first_gen.csv', index_col=0)
        dataset.append(data)

    df = pd.concat(dataset, ignore_index=True)

    df.to_csv('first_gen.csv')


def build_seed_dataset():

    dfs = []

    for i in range(1, 11):
        dataset = []

        for j in range(1, 11):
            if i != j:
                data = pd.read_csv(f'c_78_p_0.75_graph_WS_dataset_CIFAR10_seed_{j}_name_first_gen.csv', index_col=0)
                dataset.append(data)

        df = pd.concat(dataset, ignore_index=True)

        dfs.append(df)

    for i in range(1, 11):
        dfs[i-1].to_csv(f'dataset_{i}.csv', index=False)


def fifth_rand(seed):
    features = pd.read_csv(f'dataset_{seed}.csv', index_col=False)

    other_data = pd.read_csv(f'c_78_p_0.75_graph_WS_dataset_CIFAR10_seed_{seed}_name_first_gen.csv',
                             index_col=0)
    other_data = other_data.drop('connection', axis=1)
    other_data = other_data.drop('seed', axis=1)
    other_data = other_data.drop('weight', axis=1)

    # Labels are the values we want to predict
    labels = np.array(features['weight'])  # Remove the labels from the features
    features = features.drop('weight', axis=1)  # Saving feature names for later use
    features = features.drop('seed', axis=1)
    features = features.drop('connection', axis=1)

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3,
                                                                                random_state=42)

    regr = RandomForestRegressor(n_estimators=1000, random_state=42)
    regr.fit(train_features, train_labels)

    predictions = regr.predict(test_features)

    mae = metrics.mean_absolute_error(test_labels, predictions)
    mse = metrics.mean_squared_error(test_labels, predictions)
    r2 = metrics.r2_score(test_labels, predictions)

    # Print metrics
    print('Mean Absolute Error:', round(mae, 2))
    print('Mean Squared Error:', round(mse, 2))
    print('R-squared scores:', round(r2, 2))

    parameters = {
        'max_depth': [70, 80, 90, 100],
        'n_estimators': [600, 1000, 1100]
    }

    gridforest = GridSearchCV(regr, parameters, cv=10, n_jobs=-1, verbose=1)
    gridforest.fit(train_features, train_labels)
    gridforest.best_params_

    characteristics = features.columns

    importances = list(regr.feature_importances_)
    characteristics_importances = [(characteristic, round(importance, 2)) for characteristic, importance in
                                   zip(characteristics, importances)]
    characteristics_importances = sorted(characteristics_importances, key=lambda x: x[1], reverse=True)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in characteristics_importances]

    plt.bar(characteristics, importances, orientation='vertical')
    plt.xticks(rotation='vertical')
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.grid(axis='y', color='#D3D3D3', linestyle='solid')
    plt.savefig('')

# print('FIRST_RAND')
# first_rand()
# print('SECOND_RAND')
# second_rand()
# print('FOURTH_RAND')
# fourth_rand()
# print('FIFTH_RAND')
# fifth_rand()


def fourth_rand(seed):
    x = pd.read_csv(f'dataset_{seed}.csv', index_col=False)

    # Labels are the values we want to predict
    y = np.array(x['weight'])  # Remove the labels from the features
    x = x.drop('weight', axis=1)  # Saving feature names for later use
    x = x.drop('seed', axis=1)
    x = x.drop('connection', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    model = RandomForestRegressor(n_estimators=1000, random_state=42)
    model.fit(x_train, y_train)
    # evaluate the model
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
    n_scores = cross_val_score(model, x, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

    y_pred = model.predict(x_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    other_data = pd.read_csv(f'c_78_p_0.75_graph_WS_dataset_CIFAR10_seed_{seed}_name_first_gen.csv',
                             index_col=0)
    other_data = other_data.drop('connection', axis=1)
    other_data = other_data.drop('seed', axis=1)
    other_data = other_data.drop('weight', axis=1)

    new_weights = model.predict(other_data)

    print('new_weights:', len(new_weights))

    correct_weights = pd.read_csv(f'c_78_p_0.75_graph_WS_dataset_CIFAR10_seed_{seed}_name_first_gen.csv',
                                  index_col=0)

    print(correct_weights)

    correct_weights.weight = new_weights

    correct_weights.to_csv(f'./rand_forest/new_first_gen_{seed}.csv')

    characteristics = x.columns

    importances = list(model.feature_importances_)
    characteristics_importances = [(characteristic, round(importance, 2)) for characteristic, importance in
                                   zip(characteristics, importances)]
    characteristics_importances = sorted(characteristics_importances, key=lambda x: x[1], reverse=True)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in characteristics_importances]

    plt.bar(characteristics, importances, orientation='vertical')
    plt.xticks(rotation='vertical')
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.grid(axis='y', color='#D3D3D3', linestyle='solid')
    plt.savefig(f'./random_forest_importances/{seed}_importances.png')
    plt.clf()


def predict_weights(df):
    other_data = df
    other_data = other_data.drop('connection', axis=1)
    other_data = other_data.drop('seed', axis=1)
    other_data = other_data.drop('weight', axis=1)

    x = pd.read_csv(f'./node_data/first_gen.csv', index_col=0)

    # Labels are the values we want to predict
    y = np.array(x['weight'])  # Remove the labels from the features
    x = x.drop('weight', axis=1)  # Saving feature names for later use
    x = x.drop('seed', axis=1)
    x = x.drop('connection', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    model = RandomForestRegressor(n_estimators=1000, random_state=42)
    model.fit(x_train, y_train)
    # evaluate the model
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
    n_scores = cross_val_score(model, x, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

    y_pred = model.predict(x_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    new_weights = model.predict(other_data)

    df.weight = new_weights

    return df

# for i in range(1, 11):
#     fourth_rand(i)
