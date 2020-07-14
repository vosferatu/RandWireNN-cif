import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


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
        dfs[i - 1].to_csv(f'dataset_{i}.csv', index=False)


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


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f}.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, predictions))
    print('Mean Squared Error:', metrics.mean_squared_error(test_labels, predictions))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, predictions)))

    return accuracy


def fourth_rand(seed):
    x = pd.read_csv(f'first_gen.csv', index_col=False)

    train = x[x['seed'] != seed]
    test = x[x['seed'] == seed]

    validations = []
    while len(validations) < 2:
        np.random.seed()
        a = np.random.random_integers(1, 10)
        if a != seed:
            validations.append(a)

    val = x[x['seed'] in validations]

    # Labels are the values we want to predict
    y_train = train['weight']
    y_test = test['weight']
    y_val = test['weight']

    # Features
    # validation 2 randomly selected seeds
    # test 1 seed -> predicted seed
    x_train = train
    x_train.drop('weight', axis=1, inplace=True)
    x_train.drop('seed', axis=1, inplace=True)
    x_train.drop('connection', axis=1, inplace=True)

    x_test = test
    x_test.drop('weight', axis=1, inplace=True)
    x_test.drop('seed', axis=1, inplace=True)
    x_test.drop('connection', axis=1, inplace=True)

    x_val = val
    x_val.drop('weight', axis=1, inplace=True)
    x_val.drop('seed', axis=1, inplace=True)
    x_val.drop('connection', axis=1, inplace=True)

    base_model = RandomForestRegressor(n_estimators=1000, random_state=42)
    base_model.fit(x_train, y_train)

    from sklearn.model_selection import RandomizedSearchCV  # Number of trees in random forest
    n_estimators = 1000  # 1000
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(5, 30, num=5)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [5, 10]  # > 2
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [2, 4]  # > 2
    # Method of selecting samples for training each tree
    bootstrap = [True, False]  # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestRegressor()

    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=1000, cv=5, verbose=2,
                                   random_state=42, n_jobs=-1)

    rf_random.fit(x_val, y_val)  # validation

    print(rf_random.best_params_)

    best_random = rf_random.best_estimator_

    print('base_accuracy')
    base_accuracy = evaluate(base_model, x_test, y_test)
    print('random_accuracy')
    random_accuracy = evaluate(best_random, x_test, y_test)

    print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))

    # with open('path/to/file', 'wb') as f:
    #     pickle.dump(rf, f)


def plot_importances():
    with open('./forest_model', 'rb') as f:
        regr = pickle.load(f)

        x = pd.read_csv(f'./node_data/first_gen.csv', index_col=0)

        # Labels are the values we want to predict
        y = np.array(x['weight'])  # Remove the labels from the features
        x = x.drop('weight', axis=1)  # Saving feature names for later use
        x = x.drop('seed', axis=1)
        x = x.drop('connection', axis=1)

        characteristics = x.columns

        importances = list(regr.feature_importances_)
        characteristics_importances = [(characteristic, round(importance, 2)) for characteristic, importance in
                                       zip(characteristics, importances)]
        characteristics_importances = sorted(characteristics_importances, key=lambda x: x[1], reverse=True)
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in characteristics_importances]

        i = 0
        while i < len(importances):
            if importances[i] < 0.005:
                print('here')
                characteristics = characteristics.delete(i)
                importances.pop(i)
            else:
                i = i + 1

        print(characteristics)
        print(importances)

        fig = go.Figure(data=[
            go.Bar(name='WS', x=characteristics, y=importances, textposition='auto'),
        ])
        # Change the bar mode
        fig.update_layout(
            title=f'Random Forest feature importance',
            title_x=0.5,
            xaxis_title="Feature",
            yaxis_title="Importance",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#000000"
            ),
            barmode='group'
        )
        fig.write_image(f"plot/forest_importance.pdf")
        fig.show()


def predict_weights(df):
    with open('./forest_model', 'rb') as f:
        model = pickle.load(f)

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

        new_weights = model.predict(other_data)

        df.weight = new_weights

        evaluate(model, x_test, y_test)

        return df


def predict_new_model_weights(df):
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

    evaluate(model, x_test, y_test)

    # with open('./forest_model', 'wb') as f:
    #    pickle.dump(model, f)

    return df


def cluster_forest():
    x = pd.read_csv('cluster_frozen3.csv', index_col=0)

    y = np.array(x['weight1'])

    x.drop('node', axis=1, inplace=True)
    x.drop('channel2', axis=1, inplace=True)
    x.drop('weight1', axis=1, inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25,
                                                      random_state=1)  # 0.25 x 0.8 = 0.2

    base_model = RandomForestRegressor(n_estimators=1000, random_state=42)
    base_model.fit(x_train, y_train)

    from sklearn.model_selection import RandomizedSearchCV  # Number of trees in random forest
    n_estimators = [1500]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(5, 30, num=5)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 4, 6]  # > 2
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [2, 4]  # > 2
    # Method of selecting samples for training each tree
    bootstrap = [True, False]  # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestRegressor()

    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=1000, cv=5, verbose=2,
                                   random_state=42, n_jobs=-1)

    rf_random.fit(x_val, y_val)  # validation

    print(rf_random.best_params_)

    best_random = rf_random.best_estimator_

    print('base_accuracy')
    base_accuracy = evaluate(base_model, x_test, y_test)
    print('random_accuracy')
    random_accuracy = evaluate(best_random, x_test, y_test)

    print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))

    with open('./forest_node_model', 'wb') as f:
        pickle.dump(best_random, f)


def cluster_plot():
    with open('./forest_node_model', 'rb') as f:
        model = pickle.load(f)

        x = pd.read_csv('cluster_frozen3.csv', index_col=0)

        y = np.array(x['weight1'])

        x.drop('node', axis=1, inplace=True)
        x.drop('channel2', axis=1, inplace=True)
        x.drop('weight1', axis=1, inplace=True)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

        predictions = model.predict(x_test)

        fig, ax = plt.subplots()
        ax.scatter(y_test, predictions)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')

        plt.savefig('./weights.png')
        plt.show()


def kernel_forest():
    x = pd.read_csv('./node_data/first_gen_kernels.csv', index_col=0)

    y = np.array(x[['weight1', 'weight2', 'weight3', 'weight4', 'weight5', 'weight5', 'weight6', 'weight7', 'weight8',
                    'weight9']])

    x.drop('node', axis=1, inplace=True)
    x.drop('channel2', axis=1, inplace=True)
    x.drop('weight1', axis=1, inplace=True)
    x.drop('weight2', axis=1, inplace=True)
    x.drop('weight3', axis=1, inplace=True)
    x.drop('weight4', axis=1, inplace=True)
    x.drop('weight5', axis=1, inplace=True)
    x.drop('weight6', axis=1, inplace=True)
    x.drop('weight7', axis=1, inplace=True)
    x.drop('weight8', axis=1, inplace=True)
    x.drop('weight9', axis=1, inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25,
                                                      random_state=1)  # 0.25 x 0.8 = 0.2

    base_model = RandomForestRegressor(n_estimators=1000, random_state=42)
    base_model.fit(x_train, y_train)

    from sklearn.model_selection import RandomizedSearchCV  # Number of trees in random forest
    n_estimators = [1000]
    # Number of features to consider at every split
    max_features = ['auto']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(5, 30, num=5)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 4, 6]  # > 2
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [2, 4, 6]  # > 2
    # Method of selecting samples for training each tree
    bootstrap = [True]  # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestRegressor()

    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=200, cv=5, verbose=2,
                                   random_state=42, n_jobs=-1)

    rf_random.fit(x_val, y_val)  # validation

    print(rf_random.best_params_)

    best_random = rf_random.best_estimator_

    print('base_accuracy')
    base_accuracy = evaluate(base_model, x_test, y_test)
    print('random_accuracy')
    random_accuracy = evaluate(best_random, x_test, y_test)

    print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))

    predictions = best_random.predict(x_test)

    print('predictions: ', predictions)

    with open('./forest_node_model', 'wb') as f:
        pickle.dump(best_random, f)


def predict_new_kernels(df):
    with open('./forest_node_model', 'rb') as f:
        model = pickle.load(f)

        df.drop('node', axis=1, inplace=True)
        df.drop('channel2', axis=1, inplace=True)
        df.drop('weight1', axis=1, inplace=True)
        df.drop('weight2', axis=1, inplace=True)
        df.drop('weight3', axis=1, inplace=True)
        df.drop('weight4', axis=1, inplace=True)
        df.drop('weight5', axis=1, inplace=True)
        df.drop('weight6', axis=1, inplace=True)
        df.drop('weight7', axis=1, inplace=True)
        df.drop('weight8', axis=1, inplace=True)
        df.drop('weight9', axis=1, inplace=True)

        new_weights = model.predict(df)  # TODO

        print('new_weights: ', new_weights)

        return df


def main():
    plot_importances()


if __name__ == "__main__":
    main()
