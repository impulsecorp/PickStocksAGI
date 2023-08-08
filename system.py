import datetime
import os

import warnings

import system

warnings.filterwarnings('ignore')
import random as rnd
import time as stime

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import date2num
from matplotlib.pyplot import gca
from matplotlib.pyplot import plot
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import scale
from tqdm.notebook import tqdm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier, BaggingClassifier, VotingRegressor, BaggingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
# from hyperopt import fmin, hp, rand, tpe
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler
from gplearn.genetic import SymbolicRegressor
# from imblearn.over_sampling import SMOTE
from copy import deepcopy
# from deap import base, creator, tools, algorithms
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
# from empyrical import sortino_ratio, omega_ratio, sharpe_ratio, calmar_ratio, stability_of_timeseries
import torch
import torch.nn as nn
import torch.nn.functional as F
import uuid

def uuname(): return str(uuid.uuid4()).replace('-','')[0:12]

def reseed():
    def seed_everything(s=0):
        rnd.seed(s)
        np.random.seed(s)
        os.environ['PYTHONHASHSEED'] = str(s)

    seed = 0
    while seed == 0:
        seed = int(stime.time() * 100000) % 1000000
    seed_everything(seed)
    return seed


def newseed():
    seed = 0
    while seed == 0:
        seed = int(stime.time() * 100000) % 1000000
    return seed


seed = reseed()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from datetime import datetime, time

# global parameters

train_set_end = 0.4  # percentage point specifying the training set end point (1.0 means all data is training set)
val_set_end = 0.7  # percentage point specifying the validation set end point (1.0 means no test set and all data after the previous point is validation )
max_tries = 0.2  # for optimization, percentage of the grid space to cover (1.0 = exchaustive search)
cv_folds = 5
balance_data = 1
scale_data = 1
multiclass = 0
multiclass_move_threshold = 1.0
regression = 0
regression_move_threshold = 1.0


# the objective function to maximize during optimization
def objective(s):
    return (0.05 * s['SQN'] +
            0.0 * s['Profit Factor'] +
            0.25 * s['Win Rate [%]'] / 100.0 +
            0.2 * s['Exposure Time [%]'] / 100.0 +
            1.0 * s['Return [%]']
            )


# keyword parameters nearly always the same
# btkw = dict(commission=.000, margin=1.0, trade_on_close=False, exclusive_orders=True)
# optkw = dict(method='grid', max_tries=max_tries, maximize=objective, return_heatmap=True)


def get_optdata(results, consts):
    return results[1][tuple([consts[y][0] for y in [x for x in consts.keys()]
                             if consts[y][0] in [x[0] for x in results[1].index.levels]])]


def plot_result(bt, results):
    try:
        bt.plot(plot_width=1200, plot_volume=False, plot_pl=1, resample=False)
    except Exception as ex:
        print(str(ex))
        plot(np.cumsum(results[0]['_trades']['PnL'].values));


def plot_optresult(rdata, feature_name):
    if rdata.index.to_numpy().shape[0] > 2:
        rdata.plot(kind='line', use_index=False);
        gca().set_xlabel(feature_name)
        gca().set_ylabel('objective')
    else:
        xs = rdata.index.values
        goodidx = np.where(~np.isnan(rdata.values))[0]
        xs = xs[goodidx]
        rda = rdata.values[goodidx]

        if not isinstance(xs[0], time):
            plt.plot(xs, rda)
            gca().set_xlabel(feature_name)
            gca().set_ylabel('objective')
            if xs.dtype.kind == 'f':
                try:
                    gca().set_xticks(np.linspace(np.min(xs), np.max(xs), 10), rotation=45)
                except:
                    gca().set_xticks(np.linspace(np.min(xs), np.max(xs), 10))
            else:
                gca().set_xticks(np.linspace(np.min(xs), np.max(xs), 10))
        else:
            # convert xs to a list of datetime.datetime objects with a fixed date
            fixed_date = datetime(2022, 1, 1)  # or any other date you prefer
            ixs = xs[:]
            xs = [datetime.combine(fixed_date, x) for x in xs]
            # convert xs to a list of floats using date2num
            xs = date2num(xs)

            # plot the data
            ax = gca()
            ax.plot(xs, rda)
            ax.set_xticks(xs)
            ax.set_xticklabels([x.strftime('%H:%M') for x in ixs], rotation=45)
            ax.set_xlabel(feature_name)
            ax.set_ylabel('objective')


def featformat(s):
    return 'X__' + '_'.join(s.lower().split(' '))


def featdeformat(s):
    return s[len('X__'):].replace('_', ' ').replace('-', ' ')


def filter_trades_by_feature(the_trades, data, feature, min_value=None, max_value=None, exact_value=None,
                             use_abs=False):
    # Create a copy of the trades DataFrame
    filtered_trades = the_trades.copy()

    # Get the relevant portion of the predictions indicator that corresponds to the trades
    relevant_predictions = data[feature].iloc[filtered_trades['entry_bar']]

    # Add the rescaled predictions as a new column to the trades DataFrame
    if use_abs:
        ft = abs(relevant_predictions.values)
    else:
        ft = relevant_predictions.values

    # Filter the trades by the prediction value
    if exact_value is not None:
        filtered_trades = filtered_trades.loc[ft == exact_value]
    else:
        # closed interval
        if (min_value is not None) and (max_value is not None):
            if min_value == max_value:
                filtered_trades = filtered_trades.loc[ft == min_value]
            else:
                min_value, max_value = np.min([min_value, max_value]), np.max([min_value, max_value])
                filtered_trades = filtered_trades.loc[(ft >= min_value) & (ft <= max_value)]
        else:
            # open intervals
            if (min_value is not None) and (max_value is None):
                filtered_trades = filtered_trades.loc[ft >= min_value]
            else:
                filtered_trades = filtered_trades.loc[ft <= max_value]

    return filtered_trades


def filter_trades_by_confidence(the_trades, min_conf=None, max_conf=None):
    trs = the_trades.copy()
    if (min_conf is None) and (max_conf is None):
        return trs
    elif (min_conf is not None) and (max_conf is None):
        return trs.loc[(np.abs(0.5 - trs['conf'].values) * 2.0) >= min_conf]
    elif (min_conf is None) and (max_conf is not None):
        return trs.loc[(np.abs(0.5 - trs['conf'].values) * 2.0) <= max_conf]
    else:
        return trs.loc[((np.abs(0.5 - trs['conf'].values) * 2.0) >= min_conf) & (
                (np.abs(0.5 - trs['conf'].values) * 2.0) <= max_conf)]


class Custom3DScaler():
    def fit(self, X):
        self.mean_ = X.mean(axis=(0, 1))
        self.std_ = X.std(axis=(0, 1))

    def transform(self, X):
        return (X - self.mean_) / (self.std_ + 1e-8)

    def inverse_transform(self, X):
        return X * self.std_ + self.mean_


class RnnNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.5):
        super(RnnNet, self).__init__()

        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 2)  # Change output dimension to 2
        self.softmax = nn.Softmax(dim=1)  # Replace sigmoid with softmax
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        rnn_out = rnn_out[:, -1]
        rnn_out = self.dropout(rnn_out)
        out = self.fc(rnn_out)
        out = self.softmax(out)  # Use softmax instead of sigmoid
        return out


class GruNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.5):
        super(GruNet, self).__init__()

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 2)  # Change output dimension to 2
        self.softmax = nn.Softmax(dim=1)  # Replace sigmoid with softmax
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out = gru_out[:, -1]
        gru_out = self.dropout(gru_out)
        out = self.fc(gru_out)
        out = self.softmax(out)  # Use softmax instead of sigmoid
        return out


class LstmNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.5):
        super(LstmNet, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 2)  # Change output dimension to 2
        self.softmax = nn.Softmax(dim=1)  # Replace sigmoid with softmax
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1]
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        out = self.softmax(out)  # Use softmax instead of sigmoid
        return out


class RecurrentNetWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, hidden_dim, window_size, quiet=0, num_layers=2, dropout=0.5,
                 batch_size=32, learning_rate=1e-3, n_epochs=50, type='lstm', device='cpu'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.num_layers=num_layers
        self.dropout = dropout
        self.type = type
        self.device = device
        self.quiet = quiet
        self.window_size = window_size
        self.scaler = Custom3DScaler()
        if type == 'lstm':
            self.model = LstmNet(input_dim, hidden_dim,
                                 num_layers=self.num_layers, dropout=self.dropout).to(device)
        elif type == 'gru':
            self.model = GruNet(input_dim, hidden_dim,
                                num_layers=self.num_layers, dropout=self.dropout).to(device)
        elif type == 'rnn':
            self.model = RnnNet(input_dim, hidden_dim,
                                num_layers=self.num_layers, dropout=self.dropout).to(device)
        else:
            # default is LSTM
            self.model = LstmNet(input_dim, hidden_dim,
                                 num_layers=self.num_layers, dropout=self.dropout).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(self, X, y):
        self.scaler.fit(X)
        X = self.scaler.transform(X)

        X_train_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        y_train_tensor = torch.tensor(y, dtype=torch.long).to(self.device)  # Change dtype to torch.long
        y_train_tensor = F.one_hot(y_train_tensor)  # Convert y to one-hot encoding

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        try:
            self.model.train()
            for epoch in range(self.n_epochs):
                epoch_loss = 0
                correct = 0
                total = 0
                for batch_x, batch_y in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y.to(torch.float))
                    epoch_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()

                    # Compute accuracy
                    predicted = torch.argmax(outputs, 1)
                    correct += (predicted == torch.argmax(batch_y, 1)).sum().item()
                    total += batch_y.size(0)

                epoch_acc = correct / total
                epoch_loss /= len(train_loader)

                if not self.quiet:
                    print(f'Epoch {epoch} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
        except KeyboardInterrupt:
            print('Interrupted.')
        return self

    def predict_proba(self, X):
        X = self.scaler.transform(X)
        X_test_tensor = torch.tensor(X, dtype=torch.float).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test_tensor).cpu().numpy().reshape(-1)
        return outputs


class RecurrentNetEnsemble:
    def __init__(self, n_classifiers, *args, **kwargs):
        self.classifiers = [RecurrentNetWrapper(*args, **kwargs) for _ in range(n_classifiers)]

    def fit(self, X, y):
        for i,classifier in enumerate(self.classifiers):
            print(f'Fitting net {i+1}...')
            classifier.fit(X, y)

    def predict_proba(self, X):
        probas = np.zeros((X.shape[0], 2)).reshape(-1)
        for classifier in self.classifiers:
            probas += classifier.predict_proba(X)
        probas /= len(self.classifiers)
        probas = np.exp(probas) / np.sum(np.exp(probas))
        return probas

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5), dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        return attn_output


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.5, num_layers=2):
        super(BinaryClassifier, self).__init__()

        self.layer_norm = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)

        self.self_attention = SelfAttention(hidden_dim)

        self.linear_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norm_layers.append(nn.BatchNorm1d(hidden_dim))
            self.relu_layers.append(nn.ReLU())
            self.dropout_layers.append(nn.Dropout(dropout_prob))

        self.linear_output = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.self_attention(x)

        for linear, batch_norm, relu, dropout in zip(self.linear_layers, self.batch_norm_layers, self.relu_layers,
                                                     self.dropout_layers):
            x = linear(x)
            x = batch_norm(x)
            x = relu(x)
            x = dropout(x)

        x = self.linear_output(x)
        x = self.sigmoid(x)
        return x


# Define the PyTorch wrapper to behave like an sklearn classifier
class NeuralNetClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, hidden_dim, quiet=0, dropout_prob=0.5, num_layers=2,
                 batch_size=32, learning_rate=1e-3, n_epochs=50, device='cpu'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.dropout_prob = dropout_prob
        self.num_layers = num_layers
        self.device = device
        self.scaler = StandardScaler()
        self.model = BinaryClassifier(input_dim, hidden_dim,
                                      dropout_prob=self.dropout_prob, num_layers=self.num_layers).to(device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.quiet = quiet

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        X_train_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        y_train_tensor = torch.tensor(y, dtype=torch.float).view(-1, 1).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        try:
            self.model.train()
            for epoch in range(self.n_epochs):
                epoch_loss = 0
                correct = 0
                total = 0
                for batch_x, batch_y in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    epoch_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()

                    # Compute accuracy
                    predicted = torch.round(outputs)
                    correct += (predicted == batch_y).sum().item()
                    total += batch_y.size(0)

                epoch_acc = correct / total
                epoch_loss /= len(train_loader)

                if not self.quiet:
                    print(f'Epoch {epoch} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
        except KeyboardInterrupt:
            print('Interrupted.')

        return self

    def predict_proba(self, X):
        X = self.scaler.transform(X)
        X_test_tensor = torch.tensor(X, dtype=torch.float).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test_tensor).cpu().numpy()
        return np.hstack((1 - outputs, outputs))

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


class SymbolicRegressionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            # population_size=1000,
            # generations=20,
            # tournament_size=20,
            # stopping_criteria=0.0,
            # const_range=(-1.0, 1.0),
            # init_depth=(2, 6),
            # init_method="half and half",
            # function_set=("add", "sub", "mul", "div"),
            # metric="mean absolute error",
            # parsimony_coefficient=0.001,
            # p_crossover=0.9,
            # p_subtree_mutation=0.01,
            # p_hoist_mutation=0.01,
            # p_point_mutation=0.01,
            # p_point_replace=0.05,
            # max_samples=1.0,
            # feature_names=None,
            # warm_start=False,
            # low_memory=False,
            # n_jobs=1,
            # verbose=0,
            # random_state=None
    ):
        self.scaler = StandardScaler()
        self.model = SymbolicRegressor(
            # population_size=population_size,
            # generations=generations,
            # tournament_size=tournament_size,
            # stopping_criteria=stopping_criteria,
            # const_range=const_range,
            # init_depth=init_depth,
            # init_method=init_method,
            # function_set=function_set,
            # metric=metric,
            # parsimony_coefficient=parsimony_coefficient,
            # p_crossover=p_crossover,
            # p_subtree_mutation=p_subtree_mutation,
            # p_hoist_mutation=p_hoist_mutation,
            # p_point_mutation=p_point_mutation,
            # p_point_replace=p_point_replace,
            # max_samples=max_samples,
            # feature_names=feature_names,
            # warm_start=warm_start,
            # low_memory=low_memory,
            # n_jobs=n_jobs,
            # verbose=verbose,
            # random_state=random_state
        )

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = self.scaler.transform(X)
        y_pred = self.model.predict(X)
        return (y_pred > 0.5).astype(int)


#####################
# STRATEGIES
#####################


bestsofar = None
bestsofar_score = None


def optimize_model(model_class, model_name, space, X_train, y_train, max_evals=120, test_size=0.2, **kwargs):
    global bestsofar, bestsofar_score
    defaults = model_class(**kwargs).get_params()

    # rstate = newseed()
    bestsofar = deepcopy(defaults)
    bestsofar_score = -99999.0

    def sanitize(p):  # because some classifiers are picky about the numeric types of parameters
        toints = ['n_estimators', 'num_leaves', 'max_depth', 'min_child_samples', 'num_iterations']
        for ti in toints:
            if ti in p:
                p[ti] = int(p[ti])

    def objective(params):
        global bestsofar, bestsofar_score
        try:
            model = model_class(**kwargs)
            X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
                X_train, y_train, test_size=test_size, random_state=newseed(),
            )
            sanitize(params)
            model.set_params(**params)
            model.fit(X_train_split, y_train_split)

            if regression:
                score = np.mean(model.score(X_test_split, y_test_split))
            else:
                score = model.score(X_test_split, y_test_split)

            if score > bestsofar_score:
                print('NEW RECORD:', score)
                bestsofar_score = score
                bestsofar = params

            return -score
        except Exception as ex:
            print(str(ex))
            return 9999999.0

    try:
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals)
    except KeyboardInterrupt:
        print('Interrupted. Best score so far:', bestsofar_score)
        best = bestsofar

    # if we can't instantiate and train the model, use the defaults
    try:
        model = model_class(**kwargs)
        sanitize(best)
        model.set_params(**best)
        model.fit(X_train[0:50], y_train[0:50])
    except Exception as ex:
        print(str(ex))
        print('No better parameters than the defaults were found.')
        best = defaults

    return best


def train_clf_ensemble(clf_classes, data, ensemble_size=1, time_window_size=1, n_jobs=-1, quiet=0, **kwargs):

    if not isinstance(clf_classes, list):
        clf_classes = [clf_classes]

    clfs = []
    if not quiet:
        print(f'Training ensemble: {ensemble_size} classifiers... ', end=' ')
    for enss in range(ensemble_size):
        for i, clf_class in enumerate(clf_classes):
            try:
                clf = clf_class(random_state=newseed(), **kwargs)
            except:
                clf = clf_class(**kwargs)
            clfs.append((f'clf_{uuname()}', clf))

    N_TRAIN = int(data.shape[0] * train_set_end)
    df = data.iloc[0:N_TRAIN]
    if time_window_size > 1:
        X, y = get_clean_Xy_3d(df, time_window_size)
    else:
        X, y = get_clean_Xy(df)
    scaler = None
    if scale_data and not (time_window_size > 1):
        scaler = StandardScaler()
        try:
            Xt = scaler.fit_transform(X)
        except:
            Xt = X
    else:
        Xt = X
    # if balance_data and not (time_window_size > 1):
    #     # Apply SMOTE oversampling to balance the training data
    #     sm = SMOTE(random_state=newseed())
    #     Xt, y = sm.fit_resample(Xt, y)

    # Create ensemble classifier
    ensemble = VotingClassifier(estimators=clfs, n_jobs=n_jobs, voting='soft')
    # Train ensemble on training data
    ensemble.fit(Xt, y)
    if not quiet:
        print(f'Done. Mean CV score: {np.mean(cross_val_score(ensemble, Xt, y, cv=cv_folds, scoring="accuracy")):.5f}')
    return ensemble, scaler


def train_reg_ensemble(reg_classes, data, ensemble_size=1, time_window_size=1, n_jobs=-1, quiet=0, **kwargs):

    if not isinstance(reg_classes, list):
        reg_classes = [reg_classes]

    regs = []
    if not quiet:
        print(f'Training ensemble: {ensemble_size} classifiers... ', end=' ')
    for enss in range(ensemble_size):
        for i, reg_class in enumerate(reg_classes):
            try:
                reg = reg_class(random_state=newseed(), **kwargs)
            except:
                reg = reg_class(**kwargs)
            regs.append((f'clf_{uuname()}', reg))

    N_TRAIN = int(data.shape[0] * train_set_end)
    df = data.iloc[0:N_TRAIN]
    if time_window_size > 1:
        X, y = get_clean_Xy_3d(df, time_window_size)
    else:
        X, y = get_clean_Xy(df)
    scaler = None
    if scale_data and not (time_window_size > 1):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X)
    else:
        Xt = X
    # Create ensemble regressor
    ensemble = VotingRegressor(estimators=regs, n_jobs=n_jobs)
    # Train ensemble on training data
    ensemble.fit(Xt, y)
    if not quiet:
        print(f'Done.')

    return ensemble, scaler


def train_classifier(clf_class, data, quiet=0, time_window_size=1, **kwargs):
    if not quiet:
        print('Training', clf_class.__name__.split('.')[-1], '...', end=' ')

    try:
        clf = clf_class(quiet=quiet, **kwargs)
    except Exception as ex:
        clf = clf_class(**kwargs)

    N_TRAIN = int(data.shape[0] * train_set_end)
    df = data.iloc[0:N_TRAIN]
    if time_window_size > 1:
        X, y = get_clean_Xy_3d(df, time_window_size)
    else:
        X, y = get_clean_Xy(df)
    scaler = None
    if scale_data and not (time_window_size > 1):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X)
    else:
        Xt = X
    if balance_data and not (time_window_size > 1):
        # Apply SMOTE oversampling to balance the training data
        sm = SMOTE(random_state=newseed())
        Xt, y = sm.fit_resample(Xt, y)

    if not quiet:
        print('Data collected.')
        print('Class 0 (up):', len(y[y == 0]))
        print('Class 1 (down):', len(y[y == 1]))
        print('Class 2 (none):', len(y[y == 2]))

    clf.fit(Xt, y)
    return clf, scaler


def train_regressor(reg_class, data, quiet=0, plot_dist=0, **kwargs):
    if not quiet:
        print('Training', reg_class.__name__.split('.')[-1], '...', end=' ')

    try:
        reg = reg_class(quiet=quiet, **kwargs)
    except:
        reg = reg_class(**kwargs)

    N_TRAIN = int(data.shape[0] * train_set_end)
    df = data.iloc[0:N_TRAIN]
    X, y = get_clean_Xy(df)
    scaler = StandardScaler()
    if scale_data:
        Xt = scaler.fit_transform(X)
    else:
        Xt = X

    if plot_dist:
        print('Data collected.')
        # Plot histogram of the target variable (y)
        plt.hist(y, bins='auto', alpha=0.7, color='blue', edgecolor='black')
        plt.title('Distribution of Target Variable (Price Move)')
        plt.xlabel('Price Move')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    reg.fit(Xt, y)

    return reg, scaler


def confidence_from_softmax(softmax_array):
    num_classes = len(softmax_array)
    max_value = np.max(softmax_array)
    if max_value == 1 / num_classes:
        return 0.0
    else:
        confidence = 1 - (max_value - 1 / num_classes) / (1 - 1 / num_classes)
        return confidence


class MLClassifierStrategy:
    def __init__(self, clf, feature_columns, scaler, min_confidence=0.0, window_size=1, reverse=False):
        self.clf = clf
        self.feature_columns = feature_columns
        self.min_confidence = min_confidence
        self.scaler = scaler
        self.reverse = reverse
        self.window_size = window_size

    def next(self, idx, data):
        if not hasattr(self, 'datafeats'):
            self.datafeats = data[self.feature_columns].values

        if idx < self.window_size:
            return 'none', 0

        if self.window_size > 1:
            start_idx = idx - self.window_size
            window_data = self.datafeats[start_idx:idx]
        else:
            window_data = self.datafeats[idx].reshape(1, -1)

        if self.scaler:
            features = self.scaler.transform(window_data)
        else:
            features = window_data

        try:
            if self.window_size > 1:
                features = features.reshape(1, self.window_size, -1)
                prediction_proba = self.clf.predict_proba(torch.tensor(features, dtype=torch.float32))
            else:
                try:
                    prediction_proba = self.clf.predict_proba(features).reshape(-1)
                except:
                    prediction_proba = self.clf.predict_proba(pd.DataFrame(features.reshape(1,-1))).values[0]#

            class_label = np.argmax(prediction_proba)
            conf = confidence_from_softmax(prediction_proba)
        except:
            class_label = self.clf.predict(features).reshape(-1)
            conf = 1.0


        if conf >= self.min_confidence:
            if not self.reverse:
                if class_label == 0:
                    return 'buy', conf
                elif class_label == 1:
                    return 'sell', conf
                elif class_label == 2:
                    return 'none', conf
            else:
                if class_label == 0:
                    return 'sell', conf
                elif class_label == 1:
                    return 'buy', conf
                elif class_label == 2:
                    return 'none', conf
        else:
            return 'none', conf


class MLRegressorStrategy:
    def __init__(self, reg, feature_columns, scaler, reverse=False):
        # the sklearn regressor is already fitted to the data, we just store it here
        self.reg = reg
        self.feature_columns = feature_columns
        self.scaler = scaler
        self.reverse = reverse

    def next(self, idx, data):
        if not hasattr(self, 'datafeats'):
            self.datafeats = data[self.feature_columns].values

        # the current row is data[idx]
        # extract features for the previous row
        if scale_data:
            features = self.scaler.transform(self.datafeats[idx].reshape(1, -1))
        else:
            features = (self.datafeats[idx].reshape(1, -1))

        # get the regressor prediction
        try:
            prediction = self.reg.predict(features)[0]
        except:
            # AutoGluon prediction fix
            prediction = self.reg.predict(pd.DataFrame(features)).values[0]

        # determine the action based on the predicted price move
        if abs(prediction) > regression_move_threshold:
            if not self.reverse:
                if prediction > 0:
                    return 'buy', prediction
                elif prediction <= 0:
                    return 'sell', -prediction
            else:
                if prediction > 0:
                    return 'sell', prediction
                elif prediction <= 0:
                    return 'buy', -prediction
        else:
            return 'none', 0


market_start_time = pd.Timestamp("09:30:00").time()
market_end_time = pd.Timestamp("16:00:00").time()
enter_on_close = True


def backtest_strategy_single(strategy, data, skip_train=1, skip_val=0, skip_test=1,
                             commission=0.0, slippage=0.0, position_value=100000, quiet=0):
    equity_curve = np.zeros(len(data))
    trades = []
    current_profit = 0

    if quiet:
        theiter = range(1, len(data))
    else:
        theiter = tqdm(range(1, len(data)))
    for idx in theiter:
        current_time = data.index[idx].time()
        if not data.daily:
            if (current_time < market_start_time) or (current_time > market_end_time):
                # Skip trading in pre/aftermarket hours
                equity_curve[idx] = current_profit
                continue
        if (idx <= int(train_set_end * len(data))) and skip_train:
            continue
        if (idx > int(train_set_end * len(data))) and (idx <= int(val_set_end * len(data))) and skip_val:
            continue
        if (idx > int(val_set_end * len(data))) and skip_test:
            continue

        action, confidence = strategy.next(idx, data)

        if system.enter_on_close:
            entry_price = data.iloc[idx-1]['Close']
        else:
            entry_price = data.iloc[idx]['Open']
        exit_price = data.iloc[idx]['Close']

        shares = int(position_value / entry_price)

        if action == 'buy':
            profit = (exit_price - entry_price - slippage) * shares - commission
        elif action == 'sell':
            profit = (entry_price - exit_price - slippage) * shares - commission
        elif action == 'none':
            profit = 0.0
        else:
            raise ValueError(f"Invalid action '{action}' at index {idx}")

        current_profit += profit
        equity_curve[idx] = current_profit
        if action != 'none':
            trades.append({
                'pos': action,
                'conf': confidence,
                'shares': shares,
                'entry_datetime': data.index[idx-1] if system.enter_on_close else data.index[idx],
                'exit_datetime': data.index[idx],
                'entry_bar': idx-1 if system.enter_on_close else idx,
                'exit_bar': idx,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit': profit
            })

    return equity_curve, *compute_stats(data, trades)


def backtest_strategy_multi(strategy, data_list, skip_train=1, skip_val=0, skip_test=1,
                             commission=0.0, slippage=0.0, position_value=100000, quiet=0):

    # data integrity check
    assert all([x.shape == data_list[0].shape for x in data_list])
    assert all([x.index[0] == data_list[0].index[0] for x in data_list])
    assert all([x.index[-1] == data_list[0].index[-1] for x in data_list])

    datalen = len(data_list[0])

    all_equity_curves = [np.zeros(datalen) for _ in range(len(data_list))]
    all_trades = [[] for _ in range(len(data_list))]
    all_current_profits = [0 for _ in range(len(data_list))]

    if quiet:
        theiter = range(1, datalen)
    else:
        theiter = tqdm(range(1, datalen))
    for idx in theiter:

        current_time = data_list[0].index[idx].time()
        if not data_list[0].daily:
            if (current_time < market_start_time) or (current_time > market_end_time):
                # Skip trading in pre/aftermarket hours
                for data_idx in range(datalen): all_equity_curves[data_idx][idx] = all_current_profits[data_idx]
                continue
        if (idx <= int(train_set_end * datalen)) and skip_train:
            continue
        if (idx > int(train_set_end * datalen)) and (idx <= int(val_set_end * datalen)) and skip_val:
            continue
        if (idx > int(val_set_end * datalen)) and skip_test:
            continue

        actions, confidences = strategy.next(idx, data_list)

        for data_idx, data in enumerate(data_list):
            entry_price = data.iloc[idx]['Open']
            exit_price = data.iloc[idx]['Close']

            shares = int(position_value / entry_price)

            if actions[data_idx] == 'buy':
                profit = (exit_price - entry_price - slippage) * shares - commission
            elif actions[data_idx] == 'sell':
                profit = (entry_price - exit_price - slippage) * shares - commission
            elif actions[data_idx] == 'none':
                profit = 0.0
            else:
                raise ValueError(f"Invalid action '{actions[data_idx]}' at index {idx}")

            all_current_profits[data_idx] += profit
            all_equity_curves[data_idx][idx] = all_current_profits[data_idx]
            if actions[data_idx] != 'none':
                all_trades[data_idx].append({
                    'pos': actions[data_idx],
                    'conf': confidences[data_idx],
                    'shares': shares,
                    'entry_datetime': data.index[idx],
                    'exit_datetime': data.index[idx],
                    'entry_bar': idx,
                    'exit_bar': idx,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit': profit
                })

    return all_equity_curves, [compute_stats(data, trades) for data,trades in zip(data_list, all_trades)]


def get_winner_pct(trades):
    if len(trades) > 0:
        winners = (len(trades.loc[trades['profit'].values >= 0.0]) / len(trades)) * 100.0
    else:
        winners = 0.0
    return winners


def get_profit_factor(trades):
    gross_profit = trades[trades['profit'] >= 0]['profit'].sum()
    gross_loss = np.abs(trades[trades['profit'] < 0]['profit'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0
    return profit_factor


def compute_stats(data, trades):
    if not isinstance(trades, pd.DataFrame):
        trades_df = pd.DataFrame(trades)
    else:
        trades_df = trades.copy()
    try:
        return get_profit_factor(trades_df), trades_df
    except:
        return 0, pd.DataFrame(columns=['pos', 'shares', 'entry_datetime', 'exit_datetime', 'entry_bar',
                                         'exit_bar', 'entry_price', 'exit_price', 'profit'])


def qbacktest(clf, scaler, data, quiet=0, reverse=False, window_size=1, min_confidence=0.0, **kwargs):
    s = MLClassifierStrategy(clf, list(data.filter(like='X')), scaler, min_confidence=min_confidence,
                             window_size=window_size, reverse=reverse)
    equity, pf, trades = backtest_strategy_single(s, data, quiet=quiet, **kwargs)
    if not quiet:
        plt.plot(equity)
        plt.xlabel('Bar #')
        plt.ylabel('Profit')
        print(f'Profit factor: {pf:.5f}, Winners: {get_winner_pct(trades):.2f}%, Trades: {len(trades)}')
    return equity, pf, trades


def rbacktest(reg, scaler, data, quiet=0, reverse=False, **kwargs):
    s = MLRegressorStrategy(reg, list(data.filter(like='X')), scaler, reverse=reverse)
    equity, pf, trades = backtest_strategy_single(s, data, quiet=quiet, **kwargs)
    if not quiet:
        plt.plot(equity)
        plt.xlabel('Bar #')
        plt.ylabel('Profit')
        print(f'Profit factor: {pf:.5f}, Winners: {get_winner_pct(trades):.2f}%, Trades: {len(trades)}')
    return equity, pf, trades


#####################
# DATA PROCEDURES
#####################

def fix_data(data1, data2):
    data1 = data1[::-1]
    data2 = data2[::-1]
    s1, s2 = data1.index[0], data2.index[0]
    if s1 < s2:
        # find the index of s2 in data1
        try:
            idx = data1.index.to_list().index(s2)
            data1 = data1[idx:]
        except:
            idx = data2.index.to_list().index(s2)
            data2 = data2[idx:]
    elif s1 > s2:
        # find the index of s1 in data2
        try:
            idx = data2.index.to_list().index(s1)
            data2 = data2[idx:]
        except:
            idx = data1.index.to_list().index(s1)
            data1 = data1[idx:]
    data1 = data1[::-1]
    data2 = data2[::-1]
    s1, s2 = data1.index[0], data2.index[0]
    if s1 < s2:
        # find the index of s1 in data2
        try:
            idx = data2.index.to_list().index(s1)
            data2 = data2[idx:]
        except:
            idx = data1.index.to_list().index(s1)
            data1 = data1[idx:]
    elif s1 > s2:
        # find the index of s2 in data1
        try:
            idx = data1.index.to_list().index(s2)
            data1 = data1[idx:]
        except:
            idx = data2.index.to_list().index(s2)
            data2 = data2[idx:]
    return data1, data2


def get_data(symbol, period='D', nrows=None, datadir='data'):
    print('Loading..', end=' ')
    if period == 'd': period = 'D'
    sfn = symbol + '_' + period
    if period != 'D':
        data = pd.read_csv(os.path.join(datadir, sfn + '.csv'), nrows=nrows, parse_dates=['time'], index_col=0)
        data.daily = 0
    else:
        data = pd.read_csv(os.path.join(datadir, sfn + '.csv'), nrows=nrows, parse_dates=['date'], index_col=0)
        data.daily = 1
    print('Done.')
    return data


def get_data_proc(symbol, period='D', nrows=None, datadir='data'):
    print('Loading..', end=' ')
    if period == 'd': period = 'D'
    sfn = symbol + '_' + period
    if period != 'D':
        data = pd.read_csv(os.path.join(datadir, sfn + '_proc.csv'), nrows=nrows, parse_dates=['time'], index_col=0)
        data.daily = 0
    else:
        data = pd.read_csv(os.path.join(datadir, sfn + '_proc.csv'), nrows=nrows, parse_dates=['date'], index_col=0)
        data.daily = 1
    print('Done.')
    return data


def get_data_pair(symbol1, symbol2, period='D'):
    s1, s1_f = get_data(symbol1, period=period)
    s2, s2_f = get_data(symbol2, period=period)

    s1, s2 = fix_data(s1, s2)
    s1_f, s2_f = fix_data(s1_f, s2_f)

    return (s1 / s2).dropna(), (s1_f / s2_f).dropna()


def get_data_features(data):
    return list(data.columns[0:5]) + [featdeformat(x) for x in data.columns[5:]]


def read_tradestation_data(filename, test_size=0.2, max_rows=None, nosplit=0):
    d = pd.read_csv(filename)
    d = d[::-1]
    dts = [x + ' ' + y for x, y in zip(d['<Date>'].values.tolist(), d[' <Time>'].values.tolist())]
    idx = pd.DatetimeIndex(data=dts)
    v = np.vstack([d[' <Open>'].values,
                   d[' <High>'].values,
                   d[' <Low>'].values,
                   d[' <Close>'].values,
                   d[' <Volume>'].values, ]).T
    # type: ignore
    data = pd.DataFrame(data=v, columns=['1. open', '2. high', '3. low', '4. close', '5. volume', ], index=idx)

    if max_rows is not None:
        data = data[0:max_rows]

    if not nosplit:
        cp = int(test_size * len(data))
        future_data = data[0:cp]
        data = data[cp:]

        return data, future_data
    else:
        return data


def make_synth_data(data_timeperiod='D', start='1990-1-1', end='2020-1-1', freq='1min', nosplit=0):
    idx = pd.date_range(start=start, end=end, freq=freq)
    start_price = 10000
    plist = []
    vlist = []
    p = start_price
    for i in range(len(idx)):
        plist.append(p)
        p += rnd.uniform(-0.1, 0.1)
    plist = np.array(plist)
    plot(plist);
    df = pd.DataFrame(data=plist, index=idx)
    data_timeperiod = 'D'
    rdf = df.resample(data_timeperiod).ohlc()
    ddf = pd.DataFrame(data=rdf.values, columns=['Open', 'High', 'Low', 'Close'], index=rdf.index)
    vlist = [rnd.randint(10000, 50000) for _ in range(len(ddf))]
    ddf['Volume'] = vlist
    data = ddf
    if not nosplit:
        test_size = 0.2
        cp = int(test_size * len(data))
        future_data = data[cp:]
        data = data[0:cp]
        return data, future_data
    else:
        return data


def get_X(data):
    """Return matrix X"""
    return data.filter(like='X').values


def get_y(data):
    """ Return dependent variable y """
    if regression:
        if system.enter_on_close:
            y = (data.Close - data.Close.shift(1)).astype(np.float32)
        else:
            y = (data.Close - data.Open).astype(np.float32)
        return y
    else:
        if not multiclass:
            if system.enter_on_close:
                y = ((data.Close - data.Close.shift(1)) < 0).astype(np.int32)  # False = 0, so class 0, True = 1, so class 1
            else:
                y = ((data.Close - data.Open) < 0).astype(
                    np.int32)  # False = 0, so class 0, True = 1, so class 1
            return y
        else:
            if system.enter_on_close:
                move = (data.Close - data.Close.shift(1)).astype(np.float32)
            else:
                move = (data.Close - data.Open).astype(np.float32)

            y = np.zeros_like(move, dtype=np.int32)

            y[move >= multiclass_move_threshold] = 0  # Class 0: 'buy'
            y[move <= -multiclass_move_threshold] = 1  # Class 1: 'sell'
            y[np.abs(move) < multiclass_move_threshold] = 2  # Class 2: 'do nothing'

            return y


def get_clean_Xy(df):
    """Return (X, y) cleaned of NaN values"""
    X = get_X(df)
    try:
        y = get_y(df).values
    except:
        y = get_y(df)
    isnan = np.isnan(y)
    X = X[~isnan]
    y = y[~isnan]

    return X, y


##########################
# sliding window version #
##########################


def get_X_3d(data, window_size):
    """Return matrix X with a sliding window of past data"""
    values = data.filter(like='X').values
    X = []
    for i in range(len(values) - window_size + 1):
        X.append(values[i:i + window_size])
    return np.array(X)


def get_y_3d(data, window_size):
    """ Return dependent variable y """
    if regression:
        y = (data.Close - data.Open).astype(np.float32)
        return y[window_size - 1:]
    else:
        if not multiclass:
            y = ((data.Close - data.Open) < 0).astype(np.int32)  # False = 0, so class 0, True = 1, so class 1
            return y[window_size - 1:]
        else:
            move = (data.Close - data.Open).astype(np.float32)

            y = np.zeros_like(move, dtype=np.int32)

            y[move >= multiclass_move_threshold] = 0  # Class 0: 'buy'
            y[move <= -multiclass_move_threshold] = 1  # Class 1: 'sell'
            y[np.abs(move) < multiclass_move_threshold] = 2  # Class 2: 'do nothing'

            return y[window_size - 1:]


def get_clean_Xy_3d(df, window_size):
    """Return (X, y) cleaned of NaN values"""
    X = get_X_3d(df, window_size)
    try:
        y = get_y_3d(df, window_size).values
    except:
        y = get_y_3d(df, window_size)

    # Remove rows with NaN values
    isnan = np.isnan(X).any(axis=(1, 2))
    X = X[~isnan]
    y = y[~isnan]

    return X, y


def make_dataset(input_source, to_predict,
                 winlen=1, sliding_window_jump=1, predict_time_ahead=1,
                 remove_outliers=0, outlier_bounds=None, scaling=0, predict_method="default"):
    # create training set
    c0 = []
    c1 = []

    for i in range(0, input_source.shape[1] - (winlen + predict_time_ahead + 1), sliding_window_jump):
        # form the input
        xs = input_source[:, i:i + winlen].T
        if scaling: xs = scale(xs, axis=0)
        xs = xs.reshape(-1)

        # form the output
        if predict_method == "default":
            before_idx = 0
            after_idx = 3
            if before_idx == after_idx:
                q = 1
            else:
                q = 0
            sp = to_predict[before_idx, i + winlen - q]
            st = to_predict[after_idx, i + winlen]
        elif predict_method == "alternate":
            q = 1
            sp = to_predict[i + winlen - q]
            st = to_predict[i + winlen]
        else:
            raise ValueError("predict_method must be either 'default' or 'alternate'")

        if remove_outliers and (isinstance(outlier_bounds, tuple) and len(outlier_bounds) == 2):
            if ((st - sp) < outlier_bounds[0]) or ((st - sp) > outlier_bounds[1]):
                # outlier - too big move, something isn't right, so skip it
                continue

        if st >= sp:  # up move
            c0.append((xs, np.array([0])))
        else:  # down move
            c1.append((xs, np.array([1])))

    return c0, c1


def shuffle_split(c0, c1, balance_data=1):
    # shuffle and shape data
    if balance_data:
        samplesize = min(len(c0), len(c1))
        s1 = rnd.sample(c0, samplesize)
        s2 = rnd.sample(c1, samplesize)
        a = s1 + s2
    else:
        a = c0 + c1
    rnd.shuffle(a)
    x = [x[0] for x in a]
    y = [x[1] for x in a]

    x = np.vstack(x)
    y = np.vstack(y)

    # use 80% as training set
    cutpoint = int(0.8 * x.shape[0])
    x_train = x[0:cutpoint]
    x_test = x[cutpoint:]
    y_train = y[0:cutpoint]
    y_test = y[cutpoint:]

    return x_train.astype(np.float32), x_test.astype(np.float32), y_train.astype(np.float32), y_test.astype(
        np.float32), x.astype(np.float32), y.astype(np.float32)


def do_RL_backtest(env, clfs, scaling=0, scalers=None,
                   envs=None, callback=None,
                   do_plot=1, keras=0, proba=0, force_action=None, remove_outliers=0, outlier_bounds=None):
    binary = env.binary
    observation = env.reset()
    if envs is not None: _ = [x.reset() for x in envs]
    done = False
    obs = [observation]
    acts = []
    preds = []
    rewards = [0]
    try:
        gener = tqdm(range(env.bars_per_episode)) if do_plot else range(env.bars_per_episode)
        for i in gener:
            if not keras:
                if (callback is not None) and callable(callback):
                    if scaling and (scalers is not None):
                        aa = callback(scalers[0].transform(observation.reshape(1, -1)), env, envs)
                    else:
                        aa = callback(observation.reshape(1, -1), env, envs)
                else:
                    if proba:
                        if scaling and (scalers is not None):
                            aa = [clf.predict_proba(sc.transform(observation.reshape(1, -1)))[0][1]
                                  for sc, clf in zip(scalers, clfs)]
                        else:
                            aa = [clf.predict_proba(observation.reshape(1, -1))[0][1] for clf in clfs]
                    else:
                        if scaling and (scalers is not None):
                            aa = [clf.predict(sc.transform(observation.reshape(1, -1)))[0]
                                  for sc, clf in zip(scalers, clfs)]
                        else:
                            aa = [clf.predict(observation.reshape(1, -1))[0] for clf in clfs]
            else:
                aa = []
                for clf in clfs:
                    o = observation
                    if scaling: o = scale(o, axis=0)
                    p = clf.predict(o.reshape(1, env.winlen, -1))[0]
                    if len(p) > 1:
                        aa += [p[1]]
                    else:
                        aa += [p]

            # get the average
            if np.mean(aa) > 0.5:
                a = 1
            else:
                a = 0

            if envs is None:
                if not binary:
                    if a == 0:  # up
                        action = 0  # buy
                    elif a == 1:  # mid
                        action = 3  # do nothing
                    elif a == 2:  # down
                        action = 1  # sell
                else:
                    if a == 0:  # up
                        action = 0  # buy
                    elif a == 1:  # down
                        action = 1  # sell

                if force_action is not None:
                    if isinstance(force_action, str) and (force_action == 'random'): action = rnd.choice([0, 1])
                    if isinstance(force_action, int): action = force_action

                observation, reward, done, info = env.step(action)
            else:
                if len(envs) == 2:  # pair trading

                    if not binary:
                        if (callback is not None) and callable(callback):
                            if scaling and (scalers is not None):
                                actions = callback(scalers[0].transform(observation.reshape(1, -1)), env, envs)
                            else:
                                actions = callback(observation.reshape(1, -1), env, envs)
                        else:

                            if a == 0:  # up
                                actions = (1, 0)  # sell/buy
                            elif a == 1:  # mid
                                actions = (3, 3)  # do nothing
                            elif a == 2:  # down
                                actions = (0, 1)  # buy/sell
                    else:
                        if a == 0:  # up
                            actions = (1, 0)  # sell/buy
                        elif a == 1:  # down
                            actions = (0, 1)  # buy/sell

                    if force_action is not None:
                        if isinstance(force_action, str) and (force_action == 'random'): actions = (rnd.choice([0, 1])
                                                                                                    for x in envs)
                        if isinstance(force_action, int): actions = (force_action for x in envs)
                        if isinstance(force_action, tuple): actions = force_action

                    observation, reward, done, info = env.step(0)
                    rs = [x.step(y) for x, y in zip(envs, actions)]
                    reward = np.sum([x[1] for x in rs])

                    action = actions

                else:

                    if not binary:
                        if (callback is not None) and callable(callback):
                            if scaling and (scalers is not None):
                                actions = callback(scalers[0].transform(observation.reshape(1, -1)), env, envs)
                            else:
                                actions = callback(observation.reshape(1, -1), env, envs)

                    observation, reward, done, info = env.step(0)
                    rs = [x.step(y) for x, y in zip(envs, actions)]
                    reward = np.sum([x[1] for x in rs])

                    action = actions

                # raise ValueError('Not implemented.')

            obs.append(observation)
            acts.append(action)
            rewards.append(reward)
            preds.append(np.mean(aa))

            if done: break
    except KeyboardInterrupt:
        pass

    obs = np.vstack([x.reshape(-1) for x in obs])
    acts = np.array(acts)
    rewards = np.array(rewards)
    preds = np.array(preds)

    if envs is None:
        navs = np.array(env.returns)
    else:
        allt = []
        for x in envs:
            allt += x.trades
        navs = sorted(allt, key=lambda k: k[-2])
        d = {}
        for n in navs:
            if n[-2] in d:
                d[n[-2]] += n[3]
            else:
                d[n[-2]] = n[3]
        kv = list([(k, v) for k, v in d.items()])
        kv = sorted(kv, key=lambda k: k[0])
        navs = [x[1] for x in kv]

    if remove_outliers and (isinstance(outlier_bounds, tuple) and len(outlier_bounds) == 2):
        idx = np.where((outlier_bounds[0] < rewards) & (rewards < outlier_bounds[1]))[0]

        obs = obs[idx]
        acts = acts[idx]
        rewards = rewards[idx]
        preds = preds[idx]
        navs = navs[idx]

    if np.sum(rewards) == 0:
        rewards = navs

    if do_plot:
        kl = []
        t = 0
        for n in navs:
            t = t + n
            kl.append(t)
        plt.plot(kl)
        plt.plot([0, len(navs)], [0.0, 0.0], color='g', alpha=0.5)  # the zero line
        plt.show()

    return obs, acts, rewards, preds


def kdplot(preds, rewards, *args, **kwargs):
    x = np.linspace(np.min(preds), np.max(preds), 100)
    y = np.linspace(np.min(rewards), np.max(rewards), 100)
    X, Y = np.meshgrid(x, y)

    n = np.vstack([np.array(preds),
                   np.array(rewards)]).T

    kde = KernelDensity(kernel='gaussian', bandwidth=0.8).fit(n)

    Z = np.zeros((Y.shape[0], X.shape[0]))
    Z.shape

    samples = []
    for ay in y:
        for ax in x:
            samples.append([ax, ay])

    samples = np.array(samples)
    mz = kde.score_samples(samples)
    nk = 0
    for ay in range(Z.shape[0]):
        for ax in range(Z.shape[1]):
            Z[ay, ax] = mz[nk]
            nk += 1

    plt.contourf(X, Y, Z, levels=80);
    plt.scatter(preds, rewards, color='r', alpha=0.15);
    plot([np.min(preds), np.max(preds)], [np.mean(rewards), np.mean(rewards)], color='g', alpha=0.5);
    plot([np.mean(preds), np.mean(preds)], [np.min(rewards), np.max(rewards)], color='g', alpha=0.5);


#########################
# GA code
#########################


def common_rows(dataframes):
    # check if the input is a list of at least two dataframes
    if not isinstance(dataframes, list) or len(dataframes) < 2:
        raise ValueError("Input must be a list of at least two dataframes")

    # merge dataframes one by one based on their common columns
    merged_df = dataframes[0]
    for df in dataframes[1:]:
        common_cols = list(set(merged_df.columns) & set(df.columns))
        merged_df = pd.merge(merged_df, df, on=common_cols, how='inner')

    # return the resulting dataframe
    return merged_df


def combined_trades(alltrades, combine_method='or'):
    if combine_method == 'or':
        return pd.concat(alltrades, axis=0).drop_duplicates().sort_index()
    else:
        return common_rows(alltrades).sort_index()


def compute_feature_matrix(data, base_trades, bins='doane', min_pf=0.1, min_trades=10, max_trades=10000, topn=None):
    feature_names = [featdeformat(x) for x in data.filter(like='X')]
    # feature_ranges = []
    feat_bins = []
    for fn in feature_names:
        d = data[featformat(fn)].values
        hd = np.histogram(d, bins=bins)[1]
        if len(hd) > len(np.unique(d)):
            feat_bins.append(np.linspace(np.min(d), np.max(d), len(np.unique(d)) + 1))
        else:
            feat_bins.append(hd)
    pf_matrix = []
    nt_matrix = []
    wn_matrix = []
    coords = []
    for row_idx, (fname, bins) in enumerate(zip(tqdm(feature_names), feat_bins)):
        for col_idx in range(1, len(bins)):
            if bins[col_idx - 1] > bins[col_idx]:
                bs = bins[col_idx], bins[col_idx - 1]
            else:
                bs = bins[col_idx - 1], bins[col_idx]
            pf, ntrades = compute_stats(data,
                                        filter_trades_by_feature(base_trades, data, featformat(fname), min_value=bs[0],
                                                                 max_value=bs[1]))
            if (pf != -1) and (len(ntrades) > 0):
                pf_matrix.append(pf)
                nt_matrix.append(len(ntrades))
                wn_matrix.append(get_winner_pct(ntrades))
                coords.append((row_idx, col_idx))
    zpd = sorted(list(zip(pf_matrix, nt_matrix, wn_matrix, coords)), key=lambda x: x[2], reverse=True)
    top_pfs = []
    top_nts = []
    top_wns = []
    all_coords = []
    fbins_lens = []
    fnames = []
    fbins = []
    for pf, nt, wn, coords in zpd:
        if (nt >= min_trades) and (nt <= max_trades) and (pf >= min_pf):
            top_pfs.append(pf)
            top_nts.append(nt)
            top_wns.append(wn)
            fbins.append(coords[1])
            fbins_lens.append(len(feat_bins[coords[0]]))
            fnames.append(feature_names[coords[0]])
            if topn is None:
                all_coords.append(coords)
            elif (topn > 0) and (len(all_coords) < topn):
                all_coords.append(coords)
    return (all_coords, feature_names, feat_bins,
            pd.DataFrame(data=list(zip(top_pfs, top_nts, top_wns, fnames, fbins, fbins_lens)),
                         columns=['PF', 'Trades', ' % Winners', 'feature name', 'bin', 'total bins']))


def compute_ranges(data):
    feature_names = [featdeformat(x) for x in data.filter(like='X')]
    cs = []
    for fn in feature_names:
        d = data[featformat(fn)].values
        cs.append((np.min(d), np.max(d)))
    return feature_names, cs


def get_genome_alltrades_nonbinned(data, genome, base_trades, feature_names, combine_method='and'):
    alltrades = []
    for i in range(len(genome)):
        try:
            r, c, d, a = genome[i]
            if d == 'above':
                _, mtrades = compute_stats(data,
                                           filter_trades_by_feature(base_trades, data,
                                                                    featformat(feature_names[r]),
                                                                    min_value=c,
                                                                    use_abs=a))
            elif d == 'below':
                _, mtrades = compute_stats(data,
                                           filter_trades_by_feature(base_trades, data,
                                                                    featformat(feature_names[r]),
                                                                    max_value=c,
                                                                    use_abs=a))
            else:
                _, mtrades = compute_stats(data,
                                           filter_trades_by_feature(base_trades, data,
                                                                    featformat(feature_names[r]),
                                                                    exact_value=c,
                                                                    use_abs=a))
            alltrades.append(mtrades)
        except Exception as ex:
            print(ex)
            print(i)
            print(genome)
    alltrades = combined_trades(alltrades, combine_method=combine_method)
    return alltrades


def get_genome_alltrades_binned(data, genome, base_trades, feature_names, feat_bins, combine_method='and'):
    alltrades = []
    for i in range(len(genome)):
        try:
            r, c = genome[i]
            _, mtrades = compute_stats(data,
                                       filter_trades_by_feature(base_trades, data,
                                                                featformat(feature_names[r]),
                                                                min_value=feat_bins[r][c - 1],
                                                                max_value=feat_bins[r][c]))
            alltrades.append(mtrades)
        except Exception as ex:
            print(ex)
            print(i)
            print(genome)
    alltrades = combined_trades(alltrades, combine_method=combine_method)
    return alltrades


def fitness_function(alltrades, objectives, eval_min_trades=10, worst_possible_fitness=-999999.0):
    if len(alltrades) >= eval_min_trades:
        xk = [x[0](alltrades) for x in objectives]
        xk = [(x if (not (np.isnan(x) | np.isinf(x))) else worst_possible_fitness) for x in xk]
        return tuple(xk)
    else:
        return tuple([worst_possible_fitness] * len(objectives))


def run_evolution(pop_size, toolbox, num_generations, survival_rate,
                  crossover_prob, mutation_prob, objectives, worst_possible_fitness,
                  target_score=None, parallel=True, quiet=0):
    weights = np.array([x[1] for x in objectives])
    # Create initial population
    pop = toolbox.population(n=pop_size)

    # Evaluate the initial population
    if not parallel:
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
    else:
        fitnesses = Parallel(n_jobs=-1)(delayed(toolbox.evaluate)(list(ind)) for ind in pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

    # Set up the statistics and logbook
    stats = tools.Statistics(lambda ind: np.dot(weights, np.array(ind.fitness.values)))
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "avg", "max"
    # Record initial population statistics
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    if not quiet:
        print(logbook.stream)
    # Run the genetic algorithm
    best_ever = worst_possible_fitness
    cbest = None
    try:
        ittt = range(1, num_generations + 1) if (target_score is None) else range(1, 100000000)
        for gen in ittt:

            selected = toolbox.select(pop, int(len(pop) * survival_rate))
            offspring = []

            while len(offspring) < len(pop):
                parents = rnd.sample(selected, 2)
                child1, child2 = toolbox.clone(parents[0]), toolbox.clone(parents[1])

                if rnd.random() < crossover_prob:
                    try:
                        toolbox.mate(child1, child2)  # single/twopoint
                    except:
                        toolbox.mate(child1, child2, 0.5)  # uniform
                    del child1.fitness.values
                    del child2.fitness.values

                offspring.append(child1)
                offspring.append(child2)

                if len(offspring) > len(pop):
                    offspring.pop()  # Remove extra individual if the population size is odd

            for mutant in offspring:
                if rnd.random() < mutation_prob:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate offspring
            if not parallel:
                fitnesses = list(map(toolbox.evaluate, offspring))
                for ind, fit in zip(offspring, fitnesses):
                    ind.fitness.values = fit
            else:
                fitnesses = Parallel(n_jobs=-1)(delayed(toolbox.evaluate)(list(ind)) for ind in offspring)
                for ind, fit in zip(offspring, fitnesses):
                    ind.fitness.values = fit

            # keep the best ever found
            ctop = tools.selBest(pop, 1)[0]
            ctf = np.dot(weights, np.array(ctop.fitness.values))
            if ctf > best_ever:
                print(f'[#{gen}] NEW RECORD: {ctf}')
                cbest = deepcopy(ctop)
                best_ever = ctf
                if (target_score is not None) and (ctf >= target_score):
                    print('Target reached.')
                    break
            # Replace the old population with the offspring and the best individuals
            pop[:] = offspring
            # Update the statistics and logbook
            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(pop), **record)
            if not quiet:
                print(logbook.stream)
    except KeyboardInterrupt:
        print('Interrupted.')

    # the best individual found
    best_ind = cbest
    try:
        print("\nBest score: {}".format(np.dot(weights, np.array(best_ind.fitness.values))))
        return best_ind, best_ever
    except:
        print('Evolution failed to find anything')
        raise ValueError


########################
# Test utilities

from scipy import stats


def get_corr_info(x, y, plot=1):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print(f'slope: {slope}, intercept: {intercept}, determ. coeff: {r_value ** 2}, p={p_value}')
    if plot:
        plt.plot(x, y, 'o', label='original data')
        plt.plot(x, intercept + slope * x, 'r', label='fitted line')
        plt.legend()
        plt.show()


def testloop(num_iters, runner):
    pf_datapoints = []
    pr_datapoints = []
    wn_datapoints = []
    bs_datapoints = []
    for iteration in tqdm(range(num_iters)):
        print('ITERATION:', iteration)
        valdata, testdata, bsdata = runner()
        pf_datapoints.append([valdata[0], testdata[0]])
        wn_datapoints.append([valdata[1], testdata[1]])
        pr_datapoints.append([valdata[3], testdata[3]])
        bs_datapoints.append([bsdata[0], bsdata[1]])

        bsval = np.array([x[0] for x in bs_datapoints])
        bstest = np.array([x[1] for x in bs_datapoints])
        pfsval = np.array([x[0] for x in pf_datapoints])
        pfstest = np.array([x[1] for x in pf_datapoints])
        prsval = np.array([x[0] for x in pr_datapoints])
        prstest = np.array([x[1] for x in pr_datapoints])

        print('bs val / bs test')
        get_corr_info(bsval, bstest, plot=0)
        print('PF val / PF test')
        get_corr_info(pfsval, pfstest, plot=0)
        print('profit val / profit test')
        get_corr_info(prsval, prstest, plot=0)
        print('bs val / PF test')
        get_corr_info(bsval, pfstest, plot=0)
        print('bs val / profit test')
        get_corr_info(bsval, prstest, plot=0)
    return bs_datapoints, pf_datapoints, pr_datapoints, wn_datapoints
