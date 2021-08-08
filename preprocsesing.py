import os
import pickle
import sys
import warnings
from abc import abstractmethod
from difflib import get_close_matches
from multiprocessing import Process
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from rich import print
from rich.console import Console
from scipy.signal import savgol_filter as sgf
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


class PreProcess:
    """
    prepossessing and training tool for dummies
    """
    _GLOBAL_MODEL_SETTING: list = ""
    _K_FOLDS = 5
    _MALTY_PROCESSES = os.cpu_count()-2

    def __init__(self, X, y):
        """
        :param X: the features of the data (type: DataFrame or Series)
        :param y: the target of the data (type: DataFrame or Series)
        """
        self.features, self.features_size, self.features_name = self._input_sequence(X)
        self.target, self.target_size, self.name = self._input_sequence(y)

    def __str__(self):
        return f"all my preprocessing data"

    def __len__(self):
        return self.features.shape[0]

    def preprocessing(self, preprocessing_list=None, derivative=2):
        """
        the preprocessing sequins for the training set
        :param preprocessing_list: list of the preprocessing methods (type: list)
        sgf: sgalov filter
        norm: normalize the data
        offset: offset the data
        """
        optinons_list = ["sgf", "norm", "offset", "drop_None", "opus_normalization"]
        for i in preprocessing_list:
            i = spelling_fixer(i, optinons_list)
            if i == "sgf":  # using sagalov filter to smote and divert data (it's c++ librerry os its not clean memory
                # so leave the try alone)
                try:
                    self.features = sgf(self.features, window_length=13, polyorder=3, deriv=derivative, mode="nearest")
                except Exception:
                    self.features = sgf(self.features, window_length=13, polyorder=3, deriv=derivative, mode="nearest")
            elif i == "norm":  # normalize the data
                self.features = MinMaxScaler(feature_range=(0, 1)).fit_transform(self.features)

            elif i == "offset":  # do offset to the data
                self.features = self.features - np.min(self.features)

            elif i == "drop_None":  # drop None from the data
                tamp = np.concatenate((self.features, self.target, self.group), axis=1)
                col = np.reshape(self.features_name, [self.features_name.shape[0], 1])
                col = np.append(col, ["target", "group"])
                tamp = pd.DataFrame(data=tamp, columns=col).dropna()
                self.target = tamp["target"].values
                del tamp["target"]
                self.group = tamp["group"].values
                del tamp["group"]
                self.features = tamp.values

            elif i == "opus_normalization":  # copy of the oups normalization technic
                tamp = np.mean(self.features, axis=1)
                squares = lambda x: x ** 2
                for num, j in enumerate(tamp):
                    self.features[num, :] -= j
                features = np.array(squares(self.features[:, :])).sum(axis=1)
                for num, j in enumerate(features):
                    self.features[num, :] /= j

    def expend_features(self, expend):
        """
        expend the features size and scaling them to same dynamic scale
        :param expend: the expansion (type: DataFrame or Series)
        :return: the expended x
        """
        expend, size, name = self._input_sequence(expend)
        self.features_size += size
        self.features_name = np.append(self.features_name, name)
        expend = MinMaxScaler(feature_range=(0, 1)).fit_transform(expend)
        element = np.concatenate((self.features, expend), axis=1)

        self.features = element.copy()

    @abstractmethod
    def feature_selection(self, teck, number_of_features=None):
        """
        :param teck: decide the scoring technique
        :param number_of_features: the number of features_size that you want to check (type: int)
        :return: features size and the targets that possessed
        """
        consol = Console(color_system="windows")
        consol.log("[green] feature selection started")
        test = SelectKBest(teck, k=number_of_features)  # do the fetcher selection
        self.features = test.fit_transform(self.features, self.target)
        self.features_name = self.features_name[test.get_support(True)]
        self.features_size = len(self.features_name)
        consol.log("[green] feature selection done")
        return self.features_name

    def grid_search(self, changes):
        """
        grid search in youer model
        :param changes: list of tuples that the first argument in ech tuple is the parameter and the second is the value
        :return:
        """

        consol = Console(color_system="windows")
        consol.log("[green] grid search hes started")

        tamp = {}
        parms = list(self.model.get_params().keys())
        for parmeter, value in changes:  # save the data as dictionary so its can be used
            parmeter = spelling_fixer(parmeter, parms)  # looking for spell mistakes

            if not isinstance(value, list):
                value = [value]
            tamp.update({parmeter: value})

        gss = GroupShuffleSplit(n_splits=1, train_size=0.9, random_state=42)
        gss.get_n_splits()
        for train_idx, test_idx in gss.split(self.features, self.target, self.group):  # this loop splitting dav out
            features = self.features[test_idx]
            target = self.target[test_idx]
            self.features = self.features[train_idx]
            self.target = self.target[train_idx]
            self.group = self.group[train_idx]

        self.model = GridSearchCV(estimator=self.model, param_grid=tamp, scoring='accuracy', cv=PreProcess._K_FOLDS,
                                  n_jobs=PreProcess._MALTY_PROCESSES).fit(features,
                                                                          target).best_estimator_

        consol.log("[green] grid search hes done")

    def model_modulation(self, changes):
        """
        set the model parameters no grid search
        :param changes: tuple of the changes (type: tuple of tuples wen index 0 is parameter name and index 1 is parameters value)
        :return: None
        """
        tamp = {}
        parms = list(self.model.get_params().keys())
        for parmeter, value in changes:  # save the data as dictionary so its can be used
            parmeter = spelling_fixer(parmeter, parms)  # looking for spell mistakes
            tamp.update({parmeter: value})
        consol = Console(color_system="windows")
        consol.log("[green] model hes been modulated")

        self.model = self.model.set_params(**tamp)
        self._modified=False

    @staticmethod
    def _print_out(data):
        """
        printing data frames (use to send to different trade)
        :param data: the data frame thet you want to print (type: data frame)
        :return:
        """
        data = data
        tamp_df = pd.DataFrame(data)
        tamp_df.T.plot()
        plt.show()

    def show_features(self):
        """
        print your features
        :return:
        """
        p = Process(target=PreProcess._print_out, args=[self.features.copy()])
        p.start()

    @staticmethod
    def add_to_report(text):
        """
        adding a little bit to all reports name
        :param text: the text you want to edd (type: string)
        :return:
        """
        text = " " + text
        PreProcess.ADD_TO_REPORT = text

    def model_print(self, ret=False):
        """
        print the model parameters
        :param ret: if you like to return the it for another use
        :return: the model parameters
        """
        if ret:
            return self.model.get_params()
        else:
            print(self.model.get_params())

    @staticmethod
    def _input_sequence(input_data: pd.DataFrame or pd.Series) -> [np.array, int, str]:
        """
        the input sequence to eny data
        :param input_data: the data (type: pandas data frame or pandas series)
        :return: the data himself, the number of his columns, the values of his columns (type: numpy array)
        """
        if isinstance(input_data, pd.DataFrame):  # for data frame
            data = input_data.to_numpy()
            name = input_data.columns.values
            size: int = data.shape[1]

        elif isinstance(input_data, pd.Series):  # for pandas serias
            data = input_data.to_numpy()
            data = np.reshape(data, (data.size, 1))
            size: int = input_data.shape
            name = input_data.name

        else:
            sys.exit("un know object kind")

        return data, size, name

    def upload_model(self, model_for_upload, new_model_name=None):
        """
        uploading existing model tool
        :param model_for_upload: the model you like to use (type: pickled model)
        :param new_model_name: the new model name (type: string)
        :return: None
        """
        model = pickle.load(open(model_for_upload, 'rb'))
        self.model = sklearn.base.clone(model)
        if new_model_name is not None:
            self.model_name = new_model_name

    @staticmethod
    def grid_search_k_folds(k_folds=5, multi_proses=100):
        """
        limit you're greed search K folds and there multi processing
        :param k_folds: the number of folds in the validation (type: int)
        :param multi_proses: number of trades in presses (type: int)
        :return: Non
        """
        PreProcess._K_FOLDS = k_folds
        cpu_coun = os.cpu_count()
        if cpu_coun <= multi_proses:
            PreProcess._MALTY_PROCESSES = cpu_coun - 2
        else:
            PreProcess._MALTY_PROCESSES = multi_proses


def lists_solver(input_list):
    """
    take a list of lists and brig it back as one long list
    :param input_list: the list of lists (type: list(of lists))
    :return: one long list (type: list)
    """
    list_out = []
    for i in input_list:
        if isinstance(i, list or np.array):
            beck = lists_solver(i)
            list_out.extend(beck)
        else:
            list_out.append(i)
    return list_out


def spelling_fixer(input_string, check_string):  # we had some dyslectic on the crow
    """
    fixing minor spelling mistakes
    :param input_string: the word you like to compere (type: string)
    :param check_string: list of strings to comparing whit (type: list of strings)
    :return: the closest word
    """
    if input_string not in check_string:
        chosen_word = get_close_matches(input_string, check_string, n=1)
        try:
            chosen_word = chosen_word[0]
        except:
            sys.exit(f"cant find close enough to the word {input_string} so system shout down")
        console = Console(color_system="windows")
        console.print(f"[blue]the word {input_string} is replaced by {chosen_word}[/blue]")
        return chosen_word
    else:
        return input_string


def dic_uniting(d1, d2):
    """
    unite two dictionaries whit the same parameters and tern the values to list of values
    :param d1: the first dictionary (type: dict)
    :param d2: the second dictionary (type: dict)
    :return:
    """
    dd = defaultdict(list)

    for d in (d1, d2):  # you can list as many input dicts as you want here
        for key, value in d.items():
            dd[key].append(value)
    return dd
