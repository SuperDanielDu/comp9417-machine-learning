# python3
# COMP9417 Assignment
# Made by group happy
# group members:
# Sheng Du z5171466
# Chengze Du z5140893
# Mengxiao Shao z5204004

# import some useful tools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class DataReceiver(object):
    def __init__(self, path):
        self.data = pd.read_csv(path)


class DataProcessor(object):
    def __init__(self, data_set):
        self.data = data_set
        self.feature = None
        self.label = None

    # data processing
    def process(self):
        self.reduce_scope()
        self.time_transfer()
        self.filter_places()
        self.create_feature()
        self.create_label()
        return self.data, self.feature, self.label

    # reduce size
    def reduce_scope(self):
        self.data = self.data.query("x >= 5 & x < 5.5 & y >= 1.5 & y < 2")
        print(f'data shape: {self.data.shape}')
        # data_set = {}
        # sub_set = {}
        # for i in range(11):
        #     data_set[i] = self.data.query(f'x >= {i} & x < {i + 1}')
        #     for j in range(11):
        #         sub_set[j] = (data_set[i].query(f'y >= {j} & y < {j + 1}'))
        #         sub_set[j] = sub_set[j].sample(n=int(sub_set[j].shape[0] * 0.05))
        #         if not j == 0:
        #             sub_set[0] = sub_set[0].append(sub_set[j])
        #     data_set[i] = sub_set[0]
        #     if not i == 0:
        #         data_set[0] = data_set[0].append(data_set[i])
        # self.data = data_set[0]
        # print(f'data shape: {self.data.shape}')

    # change form of time stamp to year-month-day-time
    def time_transfer(self):
        time = pd.DatetimeIndex(pd.to_datetime(self.data["time"], unit="s"))
        # add "day","weekday", "hour" to data
        self.data["day"] = time.day
        self.data["weekday"] = time.weekday
        self.data["hour"] = time.hour

    # delete the place_id which less than 3 times
    def filter_places(self):
        place_count = self.data.groupby("place_id").count()["row_id"]
        self.data = self.data[self.data["place_id"].isin(place_count[place_count > 3].index.values)]

    def create_feature(self):
        self.feature = self.data[["x", "y", "accuracy", "day", "weekday", "hour"]]

    def create_label(self):
        self.label = self.data["place_id"]


def normalization(data_to_do):
    transfer = StandardScaler()
    return transfer.fit_transform(data_to_do)


def knn():
    print(f'KNN:')
    global data, features, label, x_test, x_test, y_train, y_test
    estimator = KNeighborsClassifier()
    param_dic = {"n_neighbors": [3, 4, 5, 6, 7, 8, 9, 10]}
    estimator = GridSearchCV(estimator, param_grid=param_dic, cv=3)
    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)
    print(f'y_predict:\n{y_predict}\n')
    score = estimator.score(x_test, y_test)
    print(f'Accuracy: {score}\n')

    print(f'Best K value: {estimator.best_params_}\n')
    print(f'Best result: {estimator.best_score_}\n')
    print(f'Best estimator: {estimator.best_estimator_}\n')
    print(f'cv result: {estimator.cv_results_}\n')

    return y_predict


def decision_tree():
    print(f'Decision Tree:')
    global data, features, label, x_test, x_test, y_train, y_test
    estimator = DecisionTreeClassifier()
    param_predict = {"max_depth": [7, 8, 9, 10, 11, 12, 13]}
    estimator = GridSearchCV(estimator, param_grid=param_predict, cv=3)
    estimator = estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)
    print(f'y_predict:\n{y_predict}\n')
    score = estimator.score(x_test, y_test)
    print(f'Accuracy: {score}\n')

    print(f'Best K value: {estimator.best_params_}\n')
    print(f'Best result: {estimator.best_score_}\n')
    print(f'Best estimator: {estimator.best_estimator_}\n')
    print(f'cv result: {estimator.cv_results_}\n')

    return y_predict
    # print(f'draw a tree:')
    # export_graphviz(estimator, out_file="tree.dot")
    # dot_data = export_graphviz(estimator, out_file=None)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.write_pdf('tree.pdf')


def random_forest():
    print(f'Random Forest:')
    global data, features, label, x_test, x_test, y_train, y_test
    estimator = RandomForestClassifier()
    param_predict = {"n_estimators": [35, 40, 45, 50, 55, 60, 65, 70]}
    # param_predict = {"n_estimators": [30, 40, 50], "max_depth": [4, 5, 6]}
    estimator = GridSearchCV(estimator, param_grid=param_predict, cv=3)
    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    score = estimator.score(x_test, y_test)
    print(f'Accuracy: {score}\n')

    print(f'Best K value: {estimator.best_params_}\n')
    print(f'Best result: {estimator.best_score_}\n')
    print(f'Best estimator: {estimator.best_estimator_}\n')
    print(f'cv result: {estimator.cv_results_}\n')

    return y_predict


if __name__ == "__main__":
    # get data
    data = DataReceiver("./train.csv").data
    # data processing
    data, features, label = DataProcessor(data).process()
    # data set partition
    x_train, x_test, y_train, y_test = train_test_split(features, label)
    # feature engineering - normalization
    x_train = normalization(x_train)
    x_test = normalization(x_test)

    predict_knn = knn()
    predict_decision_tree = decision_tree()
    predict_random_forest = random_forest()
