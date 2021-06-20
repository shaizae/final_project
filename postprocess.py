import math
from sklearn import metrics
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings

warnings.simplefilter("ignore")


class PostProcess:
    """
    my decision logic tools
    """

    def __init__(self, file):
        self.for_after_test_ = file
        self.fpt_, self.tpr_, self.threshold_ = None, None, None
        self.auc_ = None


    def _probabilities_crating(self):
        """
        crating the statistics block that use for the model
        :return: the statistics file that we using in the model
        """
        if self.for_after_test_ is None:
            file = pd.DataFrame(data=self.to_save_probability, columns=range(self.to_save_probability.shape[1]))
            if self._for_train is None:
                file["labels"] = self.target
                file["group"] = self.group
            else:
                file["labels"] = self._for_train[0]
                file["group"] = self._for_train[1]
            self.for_after_test_ = file
        else:
            file = self.for_after_test_.copy()
        return file

    def LLR_threshold(self, show_roc=True):

        """
        shows the ROC curve for LLR
        :return: the voting between groups inside sigmoid (type: for_after_test_)
        """
        self._probabilities_crating()
        df = self.for_after_test_.copy()
        groups = self.for_after_test_.loc[:, "group"].values
        groups = np.unique(groups)

        voting = []
        labels = []

        for group in groups:
            tamp = df.loc[df.loc[:, "group"] == group, :]
            normalize_size = tamp.shape[0]
            labels.append(tamp.iloc[0, 2].copy())
            tamp = tamp.loc[:, 0:1].values
            desigen_logic = np.log2(tamp)
            LLR_zero = np.sum(desigen_logic[:, 0]) / normalize_size
            LLR_ones = np.sum(desigen_logic[:, 1]) / normalize_size
            LLR = LLR_ones - LLR_zero
            LLR = _sigmoid(LLR)
            voting.append(LLR)

        df_save = pd.DataFrame()
        df_save["group"] = groups
        df_save["labels"] = labels
        df_save["voting"] = voting
        labels = np.array(labels).astype(float)
        voting = np.array(voting).astype(float)
        self.fpt_, self.tpr_, self.threshold_ = metrics.roc_curve(labels, voting)
        self.auc_ = metrics.auc(self.fpt_, self.tpr_)
        if show_roc:
            self.__show_roc("LLR_treshold")
        return df_save

    def LLR(self, threshold=0):
        """
        classing by log likelihood ratio vote
        :param threshold: the threshold for disease who is one (type: float)
        :return: the voting between groups (type: for_after_test_)
        """
        self._probabilities_crating()
        df = self.for_after_test_
        groups = df.loc[:, "group"].values
        groups = np.unique(groups)

        voting = []
        labels = []

        for group in groups:
            tamp = df.loc[df.loc[:, "group"] == group, :]
            normalize_size = tamp.shape[0]
            labels.append(tamp.iloc[0, 2].copy())
            tamp = tamp.iloc[:, :2].values
            designed_logic = np.log2(tamp)
            LLR_zero = np.sum(designed_logic[:, 0]) / normalize_size
            LLR_ones = np.sum(designed_logic[:, 1]) / normalize_size
            LLR = LLR_ones - LLR_zero
            LLR = _sigmoid(LLR)
            if LLR <= threshold:
                voting.append(0)
            else:
                voting.append(1)
        df_save = pd.DataFrame()
        df_save["group"] = groups
        df_save["labels"] = labels
        df_save["voting"] = voting
        self.voted = df_save
        labels = np.array(labels).astype(float)
        voting = np.array(voting).astype(float)
        self.fpt_, self.tpr_, self.threshold_ = metrics.roc_curve(labels, voting)
        self.auc_ = metrics.auc(self.fpt_, self.tpr_)

        return df_save

    def majority_vote(self, threshold=0.5):
        """
        classing by majority vote
        :param threshold: the threshold for disease who is one (type: float between 0 to 1)
        :return: the voting between groups (type: for_after_test_)
        """
        self._probabilities_crating()
        df = self.for_after_test_.copy()
        groups = df.loc[:, "group"].values
        groups = np.unique(groups)

        voting = []
        labels = []

        for group in groups:
            tamp = df.loc[df.loc[:, "group"] == group, :]
            labels.append(tamp.iloc[0, 2].copy())
            tamp = tamp.loc[:, 0:1].values
            tamp = tamp > threshold
            tamp = tamp + np.zeros([tamp.shape[0], tamp.shape[1]])
            onse = np.sum(tamp[:, 1])
            zerose = np.sum(tamp[:, 0])
            if onse > zerose:
                voting.append(1)
            else:
                voting.append(0)

        df_save = pd.DataFrame()
        df_save["group"] = groups
        df_save["labels"] = labels
        df_save["voting"] = voting
        self.voted = df_save
        labels = np.array(labels).astype(float)
        voting = np.array(voting).astype(float)
        self.fpt_, self.tpr_, self.threshold_ = metrics.roc_curve(labels, voting)
        self.auc_ = metrics.auc(self.fpt_, self.tpr_)
        self.__show_roc("majority vote threshold")
        return df_save

    def t_test(self, removing_threshold=0.05):
        """
        removing outlier data white t-test
        :param removing_threshold: the threshold presents (type: folate between 0 to 1)
        :return: the probebilitis file after voting
        """
        self._probabilities_crating()
        df = self.for_after_test_[1].values.copy()
        val, pdf_count = np.unique(df, return_counts=True)
        count = np.max(pdf_count) * removing_threshold
        count = np.where(pdf_count >= count)
        val = val[count]
        seen = []
        for i in val:
            tamp = np.where(df == i)[0].tolist()
            seen.extend(tamp)
        seen = np.array(seen)
        self.for_after_test_ = self.for_after_test_.iloc[seen, :]
        return self.for_after_test_

    def __show_roc(self, titel=None, legend_location=4, fig_size_x=7, fig_size_y=7):
        """
        show the roc of your classification
        :param legend_location: which corner you put the legend
        :param fig_size_x: the white of the figure
        :param fig_size_y:the height of the figure
        :return: None
        """

        plt.figure(figsize=(fig_size_x, fig_size_y))
        plt.title(titel)
        plt.rc("font", family="Times New Roman", size=16)
        plt.rc('axes', linewidth=2)
        plt.plot(self.fpt_, self.tpr_, label="(AUC = %0.2f)" % self.auc_)
        plt.plot([0, 1], [0, 1], "--r")
        plt.xlabel("1-Specificity", fontdict={"size": 21})
        plt.ylabel("Sensitivity", fontdict={"size": 21})
        plt.legend(loc=legend_location)
        plt.show()

    def sigmoid_confidence_interval(self):
        self._probabilities_crating()
        df = self.for_after_test_
        self.fpt_, self.tpr_, self.threshold_ = metrics.roc_curve(df.labels, df.loc[:, 1])
        best_acc_df = build_best_acc_df(df.labels, df.loc[:, 1])
        best_thres_prob = np.array(best_acc_df.iloc[0])[0]
        z_thresh = sigmoid_inverse(best_thres_prob)
        z_arr = sigmoid_inverse(df.loc[:, 1])
        return df


def _sigmoid(x, alfa=1, beta=0):
    return 1 / (1 + (alfa * math.exp(-x) + beta))


def sigmoid_inverse(p):
    z = np.log2(p) - np.log2(1 - p)
    return z


def build_best_acc_df(labels, probs):
    fpt, tpr, thresholds = metrics.roc_curve(labels, probs)
    accuracy_ls = []
    for thres in thresholds:
        y_pred = np.where(probs > thres, 1, 0)
        accuracy_ls.append(metrics.accuracy_score(labels, y_pred, normalize=True))

    accuracy_ls = pd.concat([pd.Series(thresholds), pd.Series(accuracy_ls), pd.Series(fpt), pd.Series(tpr)],
                            axis=1)
    accuracy_ls.columns = ['thresholds', 'accuracy', '1-SP', 'SE']
    accuracy_ls.sort_values(by='accuracy', ascending=False, inplace=True)
    # print(accuracy_ls.head(50))
    # print(np.array(accuracy_ls.iloc[0]))
    # accuracy_ls.head()
    return accuracy_ls
