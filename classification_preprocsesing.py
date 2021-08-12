from datetime import datetime
import xgboost as xgb
from sklearn import metrics
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import compute_sample_weight

from preprocsesing import *


class ClassificationPreprocessing(PreProcess):
    """
    all the preprocessing that unique to classification
    """
    ADD_TO_REPORT: str = ""
    SAVE = True

    def __init__(self, X, y):
        super().__init__(X, y)
        self.for_after_test_ = None
        self.to_save_probability = None
        self._for_train = None

    def feature_selection(self, number_of_features=None, tech=chi2):
        """
        :param tech: decide the scoring technique
        :param number_of_features: the number of features_size that you want to check (type: int)
        :param ret: if you like to return the features_size and the targets that possessed (type: bool)
        :return: features_size and the targets that possessed
        """

        features = super().feature_selection(number_of_features=number_of_features, teck=tech)
        return features

    def _class_mod_dis(self, model):
        """
        deists the learning classification algorithm
        :param model: the nam of thr model you kile to use
        """
        methods_list = ["xgboost", "gauss", "svm", "random", "logistic_regression_regulated", "logistic_regression",
                        "LDA"]
        model = spelling_fixer(model, methods_list)
        if model == "xgboost":
            self.model = xgb.XGBClassifier()
            self.model_name = "XGBoost"
        elif model == "gauss":
            self.model = GaussianNB()
            self.model_name = "Gaussian Naive Bayes"
        elif model == "svm":
            self.model = svm.SVC(probability=True)
            self.model_name = "support vector machine"
        elif model == "random":
            self.model = RandomForestClassifier()
            self.model_name = "random_forest"
        elif model == "logistic_regression_regulated":
            self.model = LogisticRegressionCV()
            self.model_name = "logistic regression"
        elif model == "logistic_regression":
            self.model = LogisticRegression()
            self.model_name = "logistic regression"
        elif model == "LDA":
            self.model = LinearDiscriminantAnalysis()
            self.model_name = "LDA"

    def save_classification_report(self, name):
        """
        name a report about your classification
        :param name: the file name
        :return: None
        """
        if ClassificationPreprocessing.SAVE:
            present_matrix = np.zeros((2, 2))  # crating percent base confusion matrix
            present_matrix[0, 0] = self.confusion_matrix_[0, 0] / (
                    self.confusion_matrix_[0, 0] + self.confusion_matrix_[0, 1]) * 100
            present_matrix[0, 1] = self.confusion_matrix_[0, 1] / (
                    self.confusion_matrix_[0, 0] + self.confusion_matrix_[0, 1]) * 100
            present_matrix[1, 0] = self.confusion_matrix_[1, 0] / (
                    self.confusion_matrix_[1, 0] + self.confusion_matrix_[1, 1]) * 100
            present_matrix[1, 1] = self.confusion_matrix_[1, 1] / (
                    self.confusion_matrix_[1, 0] + self.confusion_matrix_[1, 1]) * 100
            file_name = str(name + " " + ClassificationPreprocessing.ADD_TO_REPORT + ".txt")
            with open(file_name, "w+") as file:  # wraith all the parameters to the report
                file.write("classifier: ")
                file.write(self.model_name)
                file.write("\n")
                file.write("confusion matrix: ")
                file.write("\n")
                file.write(str(self.confusion_matrix_))
                file.write("\n")
                file.write("confusion matrix by presents")
                file.write("\n")
                file.write(str(present_matrix))
                file.write("\n")
                file.write("classification report")
                file.write("\n")
                file.write(str(self.classification_report_))
                file.write("number of features")
                file.write("\n")
                file.write(str(self.features_size))
                file.write("\n")
                file.write("aria under cave")
                file.write("\n")
                file.write(str(self.auc_roc_))
                file.write("\n")
                file.write(str(self.model_print(True)))
                file.write("\n")
                file.write("now:")
                file.write("\n")
                current_time = datetime.now()
                file.write(str(current_time))
                file.close()

        else:
            console = Console(color_system="windows")
            console.print(f"[red]save set off[/red]")

    def save_classification_probability(self, name):
        """
        save the probability of your classes
        :param name: the name that you want to give to the rapport (type: string)
        :return: None
        """
        if ClassificationPreprocessing.SAVE:
            name = str(name) + ClassificationPreprocessing.ADD_TO_REPORT + ".csv"
            file = pd.DataFrame(data=self.to_save_probability, columns=range(self.to_save_probability.shape[1]))
            if self.for_after_test_ is None:  # check if train test split happened
                if self._for_train is None:
                    file["labels"] = self.target
                    file["group"] = self.group
                else:
                    file["labels"] = self._for_train[0]
                    file["group"] = self._for_train[1]
                file.to_csv(name, index=False)
                self.for_after_test_ = file
            else:
                file = self.for_after_test_
                file.to_csv(name, index=False)
        else:
            console = Console(color_system="windows")
            console.print(f"[red]save set off[/red]")

    def show_roc(self, name=None, legend_location=4, fig_size_x=7, fig_size_y=7):
        """
        show the roc of your classification
        :param name:the name that you want to give to you're pic
        :param legend_location: which corner you put the legend
        :param fig_size_x: the white of the figure
        :param fig_size_y:the height of the figure
        :return: None
        """

        plt.figure(figsize=(fig_size_x, fig_size_y))  # set the plot parameters
        plt.rc("font", family="Times New Roman", size=16)
        plt.rc('axes', linewidth=2)
        plt.plot(self.fpr_, self.tpr_, label="%s (AUC = %0.2f)" % (self.model_name, self.auc_roc_))
        plt.plot([0, 1], [0, 1], "--r")  # add the Dashed red line
        if self.name is not None:
            plt.title("ROC: " + str(self.name) + " " + str(self.features_size))
        plt.xlabel("1-Specificity", fontdict={"size": 21})
        plt.ylabel("Sensitivity", fontdict={"size": 21})
        plt.legend(loc=legend_location)
        if name is None:
            plt.show()
        else:
            if ClassificationPreprocessing.SAVE:
                plt.savefig(str(name + " " + ClassificationPreprocessing.ADD_TO_REPORT + ".png"))
            else:
                console = Console(color_system="windows")
                console.print(f"[red]save set off[/red]")

    def show_det(self, name=None, legend_location=4, fig_size_x=7, fig_size_y=7):
        """
        show the roc of your classification
        :param name:the name that you want to give to you're pic
        :param legend_location: which corner you put the legend
        :param fig_size_x: the white of the figure
        :param fig_size_y:the height of the figure
        :return: None
        """

        plt.figure(figsize=(fig_size_x, fig_size_y))
        plt.rc("font", family="Times New Roman", size=16)
        plt.rc('axes', linewidth=2)
        plt.plot(self.fpr_det_, self.tpr_det_, label="%s (AUC = %0.2f)" % (self.model_name, self.auc_det_))
        if self.name is not None:
            plt.title("det: " + str(self.name) + " " + str(self.features_size))
        plt.plot([0, 1], [0, 1], "--r")  # add the Dashed red line
        plt.xlabel("FA", fontdict={"size": 21})
        plt.ylabel("Miss", fontdict={"size": 21})
        plt.legend(loc=legend_location)

        if name is None:
            plt.show()
        else:
            if ClassificationPreprocessing.SAVE:
                plt.savefig(str(name + " " + ClassificationPreprocessing.ADD_TO_REPORT + ".png"))
            else:
                console = Console(color_system="windows")
                console.print(f"[red]save set off[/red]")

    def _classification_local_report(self, labels, predictions, roc_labels):
        """
        analise the classification that currently happened
        :param labels: the labels of ech group
        :param predictions: the prediction of ech group
        :return: None
        """
        print(self.model_print())  # print the model hyper parameters
        self.confusion_matrix_ = metrics.confusion_matrix(labels, predictions)  # crate the confusion matrix
        print(self.confusion_matrix_)
        self.classification_report_ = metrics.classification_report(labels,
                                                                    predictions)  # crating the classification report
        print(self.classification_report_)
        self.accuracy_ = metrics.accuracy_score(labels, predictions)  # calculate the accuracy
        if self._for_train is None:
            self.fpr_, self.tpr_, self.threshold_ = metrics.roc_curve(roc_labels, self.to_save_probability[:, 1])
            self.fpr_det_, self.tpr_det_, self.threshold_det_ = metrics.det_curve(roc_labels,
                                                                                  self.to_save_probability[:, 1])
        else:
            self.fpr_, self.tpr_, self.threshold_ = metrics.roc_curve(self._for_train[0],
                                                                      self.to_save_probability[:, 1])
        self.auc_roc_ = metrics.auc(self.fpr_, self.tpr_)
        self.auc_det_ = metrics.auc(self.fpr_det_, self.tpr_det_)
        print(f"AUC: {self.auc_roc_}")

    def print_class_waits(self):
        """
        print the class weights and they counts
        :return: None
        """
        sample_weights = compute_sample_weight(class_weight='balanced',
                                               y=self.target)
        sample_weights = np.array(sample_weights)
        sample, count = np.unique(sample_weights, return_counts=True)
        for (i, j) in zip(sample, count):
            print(str(i) + ":" + str(j))

    def optimal_cut_point_on_roc_(self, delta_max=0.8, plot_point_on_ROC=False, save_report=None):
        """
        print the optimal cut on you're roc curve
        :param save_report: if you like to save local report (type: string)
        :param delta_max: the maximum delta between tpr and fpr (type: flute between 0 to 1)
        :param plot_point_on_ROC: is you like to show the roc curve now (type:bool)
        :return: report on you're optimal working point (type: dictionary)
        """
        tpr = self.fpr_
        fpr = self.tpr_
        n_p = self.target[self.target == 0].shape[0]
        n_n = self.target[self.target == 1].shape[0]
        sen = fpr[fpr > 0.55]
        spe = 1 - tpr[fpr > 0.55]

        delt = abs(sen - spe)
        ix_1 = np.argwhere(delt <= delta_max)

        acc = (n_p / (n_p + n_n)) * sen[ix_1] + (n_n / (n_p + n_n)) * spe[ix_1]
        best_point = (1 - spe[np.argmax(acc)], sen[np.argmax(acc)])
        auc = np.around(np.trapz(fpr, tpr), 2)

        recall_1 = sen[np.argmax(acc)]
        recall_2 = spe[np.argmax(acc)]
        precision_1 = (n_p * sen[np.argmax(acc)]) / (n_p * sen[np.argmax(acc)] + n_n * (1 - spe[np.argmax(acc)]))
        precision_2 = (n_n * spe[np.argmax(acc)]) / (n_n * spe[np.argmax(acc)] + n_p * (1 - sen[np.argmax(acc)]))

        report = {"auc": np.around(auc, 2), "acc": np.around(acc.max(), 2), "sen": np.around(recall_1, 2),
                  "spe": np.around(recall_2, 2), "npv": np.around(precision_1, 2),
                  "ppv": np.around(precision_2, 2)
                  }
        if plot_point_on_ROC:
            p = Process(target=ClassificationPreprocessing._plot_optimal_cut, args=(fpr, tpr, best_point,))
            p.start()

        if save_report is not None:
            with open(str(save_report) + ".txt", "w") as f:
                for key, valu in report.items():
                    f.write(str(key) + ": " + str(valu) + "\n")
        print(report)

        return report

    @staticmethod
    def _plot_optimal_cut(fpr, tpr, best_point):
        """
        print the optimal point
        *not for users*
        """
        plt.plot(tpr, fpr)
        plt.scatter(best_point[0], best_point[1], c="red")
        plt.xlabel("1-specificity")
        plt.ylabel("sensitivity")
        plt.plot([0, 1], [0, 1], "--r")
        plt.show()
