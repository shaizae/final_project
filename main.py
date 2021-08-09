from sklearn.datasets import make_classification
from classification import *

if __name__ == "__main__":
    Classification.SMOOTE = True  # setting bootstrap in the class
    X, y = make_classification(n_features=30, n_samples=200, weights=[.7, .3])
    X = pd.DataFrame(X)
    y = pd.Series(y)
    model = Classification(X, y, model="svm")  # crating the classification model
    model.preprocessing(["norm"])  # normalize the data
    model.feature_selection(15)  # use fetchers selection by chi 2 algorithm
    model.grid_search(("kernel", ["linear", "poly", "rbf"]), ("degree", [1, 2, 3, 4]),
                      ("probability", True))  # grid search dropping 10% as dave
    model.leave_one_out(method="most_likelihoods")  # leave one out validation can vote inside groups
    model.save_classification_report("test")  # save the report that print on the terminal
    model.save_classification_probability("test")  # save the classification probabilities
    model.show_roc("test")  # save the roc plot
