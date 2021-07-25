from multiprocessing import Pool

from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, StratifiedKFold
from sklearn.utils import resample

from classification_preprocsesing import *
from postprocess import *


class Classification(ClassificationPreprocessing, PostProcess):
    """
    my classification model
    binary models only
    """
    __CORES_LIMIT: int = None
    _GLOBAL_MODEL_SETTING = None
    BOOTSTRAP: bool = False
    SMOOTE: bool = False

    def __init__(self, features, target, model="random", group=None):
        """
        :param features: the features of the data (type: DataFrame or Series)
        :param target: the target of the data (type: DataFrame or Series)
        :param group: the groups of the data (type: DataFrame or Series)
        """
        super().__init__(features, target)
        if group is not None:
            self.group, _, _ = PreProcess._input_sequence(group)
            if Classification.SMOOTE:
                console = Console(color_system="windows")
                console.print(f"[blue]cant use smote on grouped data[/blue]")
                Classification.SMOOTE = False
        else:
            self.group = np.array(range(self.target.shape[0])).reshape([self.target.shape[0], 1])
        self.target = self.target.astype(int)
        # if len(np.unique(self.target)) < 2:
        #     sys.exit("the target have less then 2 labels")
        self._sacrifice_rate = None
        self._sacrifice_rate_flag = False
        self.best_treshold = None
        self.to_save_prediction = None
        self.to_save_probability = None
        super()._class_mod_dis(model=model)
        self.__modified = True
        self.roc_index=[]

    def __str__(self):
        return f"all my group classification tools"

    def model_modifying(self, *changes):
        """
        modifying your model
        :param changes: tuples when [0] is the parameter name and [1] is a list of parameter options
        :return:None
        """
        changes_list = []
        for i in changes:
            changes_list.append(i)
        if Classification._GLOBAL_MODEL_SETTING is not None:
            for i in Classification._GLOBAL_MODEL_SETTING:
                changes_list.append(i)
        super().model_modifying(changes_list)
        self.__modified = False

    def sacrifice_rate_setter(self, sacrifice_rate=0):
        self._sacrifice_rate = sacrifice_rate
        self._sacrifice_rate_flag = True

    @staticmethod
    def _bootstrap(feature, target, group):
        """
        bootstrap the classification data
        :param feature: the classification features
        :param target: the classification targets
        :param group: the classification groups
        :return: the bootstrapped data
        """
        local_df = np.concatenate((feature, target, group), axis=1)
        col = list(range(feature.shape[1]))
        col = np.append(col, ["target", "group"])
        local_df = pd.DataFrame(local_df, columns=col)
        for_bootstrap = local_df.loc[:, ["target", "group"]].copy()
        p_bootstrap = for_bootstrap.loc[for_bootstrap["target"] == 1, "group"].values
        p_bootstrap = np.unique(p_bootstrap)
        n_bootstrap = for_bootstrap.loc[for_bootstrap["target"] == 0, "group"].values
        n_bootstrap = np.unique(n_bootstrap)
        balance_vale = max([p_bootstrap.shape[0], n_bootstrap.shape[0]])

        p_bootstrap = resample(p_bootstrap, n_samples=balance_vale, random_state=42)
        n_bootstrap = resample(n_bootstrap, n_samples=balance_vale, random_state=42)
        for_bootstrap = np.append(p_bootstrap, n_bootstrap)
        bootstrapped = pd.DataFrame(columns=col)
        for num, i in enumerate(for_bootstrap):
            tamp = local_df.loc[local_df["group"] == i, :]
            tamp["group"] = num
            bootstrapped = pd.concat([bootstrapped, tamp], axis=0)
        features = bootstrapped.iloc[:, :-2]
        target = bootstrapped.iloc[:, -2]
        group = bootstrapped.iloc[:, -1]
        return features, target, group

    @staticmethod
    def __smoote(feature, target):
        """
        bootstrap the classification data
        :param feature: the classification features
        :param target: the classification targets
        :param group: the classification groups
        :return: the bootstrapped data
        """
        feature, target = SMOTE().fit_sample(feature.astype('float'), target.astype('float'))

        return feature, target

    def _voting_systems(self, method, out, fix_list=True):
        """
        voting system manager
        :param method: the method that use for voting
        :param out: tuple of (model, feature train, target train, feature test, target test, train group, test group)
        :return: None
        """
        voting_systems = ["majority_vote", "most_likelihoods", "no_vote"]

        method = spelling_fixer(method, voting_systems)

        if self.__modified and Classification._GLOBAL_MODEL_SETTING is not None:
            super().model_modifying(tuple(Classification._GLOBAL_MODEL_SETTING))
            self.__modified = False

        if isinstance(Classification.__CORES_LIMIT, int):
            pool = Pool(Classification.__CORES_LIMIT)
        else:
            pool = Pool()

        if method == "majority_vote":
            voted_data = pool.map(_majority_vote, out)
        elif method == "most_likelihoods":
            voted_data = pool.map(_most_likelihood, out)
        elif method == "no_vote":
            voted_data = pool.map(_no_vote, out)
        else:
            sys.exit("unknown's method")
        pool.close()
        pool.join()
        labels = []
        predictions = []
        self.to_save_probability = np.array([[0, 0], [0, 0]])
        for test in voted_data:
            labels.append(test[0])
            predictions.append(test[1])
            self.to_save_probability = np.concatenate((self.to_save_probability, test[2]), axis=0)
        self.to_save_probability = self.to_save_probability[2:, :]
        if fix_list:
            labels = lists_solver(labels)
            predictions = lists_solver(predictions)

        return labels, predictions

    def leave_one_out(self, method="no_vote"):
        """
        do classification of you're data
        :param method: disease the voting method of the data (type:string)
        :return: none
        """
        consol = Console(color_system="windows")
        consol.log("[green] training started")
        try:
            group = np.squeeze(self.group)
        except:
            group = self.group.copy()
        loo = LeaveOneGroupOut()
        loo.get_n_splits(X=self.features, y=self.target, groups=group)
        roc_labels = []

        out = []
        for train_index, test_index in loo.split(X=self.features, y=self.target, groups=group):
            feature_train, feature_test = self.features[train_index], self.features[test_index]
            target_train, target_test = self.target[train_index], self.target[test_index]
            train_group, test_group = self.group[train_index], self.group[test_index]

            out.append(
                [clone(self.model), feature_train, target_train, feature_test, target_test, train_group, test_group])
            roc_labels.extend(target_test)
            self.roc_index.extend(test_index)

        labels, predictions = self._voting_systems(method=method, out=out)

        if self._sacrifice_rate_flag:
            consol.log("[green] starts confides interval")
            self.confidence_threshold_auto()
            self.target = self.target[self.best_index_]
            self.features = self.features[self.best_index_, :]
            self.group = self.group[self.best_index_]
            self._sacrifice_rate_flag = False
            self.leave_one_out(method=method)
        else:
            consol.log("[green] training done")
            super()._classification_local_report(labels=labels, predictions=predictions, roc_labels=roc_labels)
            self.roc_groups= self.group[self.roc_index]
            self.roc_labels=self.target[self.roc_index]

    def K_folds(self, number_of_folds=5, method="no_vote"):
        """
         do classification of y from X white K folds
        :param number_of_folds: the number folds that you want to use (type: int)
        :param method: disease the voting method of the data (type:string)
        :return: the trained model
        """
        consol = Console(color_system="windows")
        consol.log("[green] training started")
        try:
            group = np.squeeze(self.group)
        except:
            group = self.group.copy()
        loo = GroupKFold(number_of_folds)
        loo.get_n_splits(X=self.features, y=self.target, groups=group)
        roc_labels = []
        out = []
        for train_index, test_index in loo.split(X=self.features, y=self.target, groups=self.group):
            feature_train, feature_test = self.features[train_index], self.features[test_index]
            target_train, target_test = self.target[train_index], self.target[test_index]
            train_group, test_group = self.group[train_index], self.group[test_index]

            out.append(
                [clone(self.model), feature_train, target_train, feature_test, target_test, train_group, test_group])
            roc_labels.extend(target_test)
            self.roc_index.extend(test_index)

        labels, predictions = self._voting_systems(method=method, out=out)

        if self._sacrifice_rate_flag:
            consol = Console(color_system="windows")
            consol.log("[green] starts confides interval")
            self.confidence_threshold_auto()
            self.target = self.target[self.best_index_]
            self.features = self.features[self.best_index_, :]
            self.group = self.group[self.best_index_]
            self._sacrifice_rate_flag = False
            self.K_folds(number_of_folds=number_of_folds, method=method)
        else:
            consol.log("[green] training done")
            super()._classification_local_report(labels=labels, predictions=predictions, roc_labels=roc_labels)
            self.roc_groups= self.group[self.roc_index]
            self.roc_labels=self.target[self.roc_index]


    def K_folds_stratified(self, number_of_folds=5, method="no_vote"):
        """
         do classification of y from X white K folds and keep stratified
        :param number_of_folds: the number folds that you want to use (type: int)
        :param method: disease the voting method of the data (type:string)
        :return: the trained model
        """

        df = pd.DataFrame(data=self.features.copy(), columns=self.features_name.copy())
        df["target"] = self.target.copy()
        df["group"] = self.group.copy()
        df = df.sort_values(by=["group"])
        df_comper = df.copy()
        df = df.groupby("group").mean()
        group = np.unique(self.group.copy())
        target = df["target"].values
        del df["target"]
        df = df.values

        consol = Console(color_system="windows")
        consol.log("[green] training started")
        loo = StratifiedKFold(number_of_folds)
        loo.get_n_splits(X=df, y=target, groups=group)
        out = []
        for train_index, test_index in loo.split(X=df, y=target, groups=group):
            train_group, test_group = group[train_index], group[test_index]

            df_train = df_comper.loc[df_comper["group"].isin(train_group)]
            target_train = df_train["target"].values
            del df_train["target"]
            del df_train["group"]
            feature_train = df_train.values

            df_test = df_comper.loc[df_comper["group"].isin(test_group)]
            target_test = df_test["target"].values
            del df_test["target"]
            del df_test["group"]
            feature_test = df_test.values

            out.append(
                [clone(self.model), feature_train, target_train, feature_test, target_test, train_group, test_group])

        labels, predictions = self._voting_systems(method=method, out=out)

        if self._sacrifice_rate_flag:
            consol = Console(color_system="windows")
            consol.log("[green] starts confides interval")
            self.confidence_threshold_auto()
            self.target = self.target[self.best_index_]
            self.features = self.features[self.best_index_, :]
            self.group = self.group[self.best_index_]
            self._sacrifice_rate_flag = False
            self.K_folds_stratified(number_of_folds=number_of_folds, method=method)
        else:
            consol.log("[green] training done")
            super()._classification_local_report(labels=labels, predictions=predictions)

    def train_test_split(self, test_size=0.3, method="no_vote"):
        """
         do classification of y from X white K folds
        :param number_of_folds: the number folds that you want to use (type: int)
        :param method: disease the voting method of the data (type:string)
        :return: the trained model
        """
        consol = Console(color_system="windows")
        consol.log("[green] training started")
        try:
            group = np.squeeze(self.group)
        except:
            group = self.group.copy()
        loo = GroupShuffleSplit(1, test_size)
        loo.get_n_splits(X=self.features, y=self.target, groups=group)
        out = []
        for train_index, test_index in loo.split(X=self.features, y=self.target, groups=self.group):
            feature_train, feature_test = self.features[train_index], self.features[test_index]
            target_train, target_test = self.target[train_index], self.target[test_index]
            train_group, test_group = self.group[train_index], self.group[test_index]

            out.append(
                [clone(self.model), feature_train, target_train, feature_test, target_test, train_group, test_group])
            self._for_train = [target_test, test_group]

        labels, predictions = self._voting_systems(method=method, out=out)

        if self._sacrifice_rate_flag:
            consol = Console(color_system="windows")
            consol.log("[green] starts confides interval")
            self.confidence_threshold_auto()
            self.target = self.target[self.best_index_]
            self.features = self.features[self.best_index_, :]
            self.group = self.group[self.best_index_]
            self._sacrifice_rate_flag = False
            self.train_test_split(method=method)
        else:
            consol.log("[green] training done")
            super()._classification_local_report(labels=labels, predictions=predictions)

    def K_folds_nested(self, N_features, number_of_folds=5, method="no_vote"):
        """
         do classification of y from X white K folds
        :param N_features: the number of features for greed serch (type: list of int)
        :param number_of_folds: the number folds that you want to use (type: int)
        :param method: disease the voting method of the data (type:string)
        :return: the trained model
        """
        voting_systems = ["majority_vote", "most_likelihoods", "no_vote"]
        method = spelling_fixer(method, voting_systems)
        consol = Console(color_system="windows")
        consol.log("[green] training started")
        try:
            group = np.squeeze(self.group)
        except:
            group = self.group.copy()
        self.__modified = False
        loo = GroupKFold(number_of_folds)
        loo.get_n_splits(X=self.features, y=self.target, groups=group)
        number_of_featchers_out = []
        params_out = []
        fold_num = 0

        for train_index, test_index in loo.split(X=self.features, y=self.target, groups=self.group):
            fold_num += 1
            feature_train, feature_test = self.features[train_index].copy(), self.features[test_index].copy()
            target_train, target_test = self.target[train_index].copy(), self.target[test_index].copy()
            train_group, test_group = self.group[train_index].copy(), self.group[test_index].copy()
            consol.log(f"[green] fold number {fold_num} started from {number_of_folds} folds")

            acc_test = 0
            number_of_feathers = []
            parameters = []
            for number in N_features:
                test = SelectKBest(chi2, k=number)
                features_train_after = test.fit_transform(feature_train, target_train)
                feature_test_after = feature_test[:, test.get_support(True)]

                changes = Classification._GLOBAL_MODEL_SETTING
                tamp = {}
                parms = list(self.model.get_params().keys())

                for parmeter, value in changes:
                    parmeter = spelling_fixer(parmeter, parms)

                    if not isinstance(value, list):
                        value = [value]
                    tamp.update({parmeter: value})
                model = GridSearchCV(estimator=self.model, param_grid=tamp, scoring='accuracy', cv=number_of_folds,
                                     n_jobs=PreProcess._MALTY_PROCESSES).fit(
                    features_train_after,
                    target_train).best_estimator_
                send_for_test = (
                    model, features_train_after, target_train, feature_test_after, target_test, train_group,
                    test_group)

                if method == "majority_vote":
                    voted_data = _majority_vote(send_for_test)
                elif method == "most_likelihoods":
                    voted_data = _most_likelihood(send_for_test)
                elif method == "no_vote":
                    voted_data = _no_vote(send_for_test)
                else:
                    sys.exit("un know method")

                label, prediction, probability = voted_data
                acc = accuracy_score(label, prediction)
                if acc > acc_test:
                    number_of_feathers = number
                    parameters = model.get_params()

            number_of_featchers_out.append(number_of_feathers)
            params_out.append(parameters)

        number_of_featchers = np.array(number_of_featchers_out)
        vals, count = np.unique(number_of_featchers, return_counts=True)
        cosen_number = vals[np.argmax(count)]
        tamp_params = params_out.pop(0)
        for i in params_out:
            tamp_params = dic_uniting(tamp_params, i)

        tamp_params["n_parameters"] = cosen_number

        consol.print("[blue]chosen number of fetchers:")
        print(cosen_number)
        consol.print("[blue]the greed search values:")
        print(tamp_params)
        consol.log("[green] greed search hes done")
        return number_of_featchers_out, tamp_params

    def save_model(self, model_name: str = "training_model"):
        """
        train a model whit all of you're data and save him for future use
        :param model_name: the name that you want to giv to the model (type: string)
        :return: the trained model
        """
        consol = Console(color_system="windows")
        consol.log("[green] training started")
        features = self.features.copy()
        target = self.target.copy()

        if Classification.BOOTSTRAP:
            features, target, _ = Classification._bootstrap(features, target,
                                                            self.group)
        if Classification.SMOOTE:
            features, target = Classification.__smoote(features, target)
        model = self.model.fit(features, target)

        with open(model_name, 'wb') as file:
            pickle.dump(model, file)
        consol.log("[green] training done")
        return model

    def confidence_threshold_auto(self, min_thr=0.001, steps_thr=0.001, max_thr=1.0):
        """
        find the threshold of the data
        :param min_thr: starting position of threshold loop (type: flute)
        :param steps_thr: step size of threshold loop (type: flute)
        :param max_thr: where to stop the threshold loop (type: flute)
        :return:
        """

        if self._sacrifice_rate == 0:
            self._sacrifice_rate = 1 * 10 ** -6

        y_prob = np.reshape(self.to_save_probability[:, 0], [self.to_save_probability.shape[0], 1])
        # for num, i in enumerate(y_prob):
        #     if i < 0.5:
        #         y_prob[num] = 1 - i
        thresholdes = np.arange(min_thr, max_thr, steps_thr)
        std = np.std(y_prob)

        rat = np.arange(0, 0.5, 0.01)
        samples_sacrifice_rate = []

        best_treshold = []
        y_hat_best_ = []
        y_test_missed_classified_index_ = []
        y_test_best_ = []
        y_hat_glob_ = []
        y_test_glob_ = []

        max_acc = []
        y_test_glob = []
        thres = []
        for r in rat:

            def_thresh = r

            acc = []
            threshold = []
            for thr in thresholdes:
                y_hat = np.zeros([y_prob.shape[0], y_prob.shape[1]])
                y_hat[y_prob < thr] = 1
                y_hat[y_prob > thr] = 0

                y_hat[((thr - def_thresh) < y_prob) & (
                        y_prob < (thr + def_thresh))] = 888

                y_test_ = self.target[y_hat != 888]
                y_hat_ = y_hat[y_hat != 888]

                acc.append(accuracy_score(y_test_, y_hat_))
                threshold.append(thr)

            acc = np.array(acc)
            best_accuracy = np.max(acc)
            best_treshold.append(threshold[np.argmax(acc)])

            y_hat_best = np.zeros([y_prob.shape[0], y_prob.shape[1]])
            y_hat_best[y_prob < threshold[np.argmax(acc)]] = 1
            y_hat_best[y_prob > threshold[np.argmax(acc)]] = 0
            y_hat_best[((threshold[np.argmax(acc)] - def_thresh) < y_prob) & (
                    y_prob < (threshold[np.argmax(acc)] + def_thresh))] = 888
            y_hat_glob = y_hat_best
            y_hat_glob_.append(y_hat_glob)

            y_test_missed_classified_index = np.argwhere(y_hat_best == 888)
            y_test_missed_classified_index_.append(y_test_missed_classified_index)

            y_hat_best = y_hat_best[y_hat_best != 888]
            y_hat_best_.extend(y_hat_best)

            y_test_glob = self.target
            y_test_glob_.append(y_test_glob)

            y_test_best = self.target[y_hat_glob != 888]
            y_test_best_.append(y_test_best)

            max_acc.append(best_accuracy)
            samples_sacrifice_rate.append((1 - (len(y_hat_best) / (len(y_prob)))))
            thres.append(r)
            if (1 - (len(y_hat_best) / (len(y_prob)))) >= self._sacrifice_rate:
                break
        arg = np.argwhere(np.array(samples_sacrifice_rate) < self._sacrifice_rate)
        best_accuracy_ = np.around(np.max((np.array(max_acc))[arg]), 2)
        arg_max = np.argmax((np.array(max_acc))[arg])

        best_rate = np.around(np.array(samples_sacrifice_rate)[arg_max], 5)
        y_hat_best = y_hat_best_[arg_max]
        y_test_missed_classified_index = y_test_missed_classified_index_[arg_max]
        y_test_best = y_test_best_[arg_max]
        y_hat_glob = y_hat_glob_[arg_max]
        y_test_glob = y_test_glob_[arg_max]
        best_treshold = (best_treshold[arg_max],
                         thres[arg_max])

        self.best_treshold = np.around(best_treshold, 3)
        self.best_index_ = np.where(y_hat_glob != 888)[0]

        return best_accuracy_, np.around(best_treshold,
                                         3), y_hat_best, best_rate, y_test_missed_classified_index, y_test_best, y_hat_glob, y_test_glob

    @staticmethod
    def limit_cores(max_cores):
        """
        limit the amount of cores that use in training set
        :param max_cores: maximum number of cors thet you like to use (type: int)
        :return: None
        """
        cores_avoidable = os.cpu_count()
        if max_cores >= cores_avoidable:
            console = Console(color_system="windows")
            console.print(f"[yellow]not enough cores on system use {cores_avoidable} instead[/yellow]")
        else:
            Classification.__CORES_LIMIT = max_cores

    @staticmethod
    def set_global_setting(*changes):
        Classification._GLOBAL_MODEL_SETTING = changes


def _no_vote(arguments_input):
    """
    classify the data whit no vote
    :param arguments_input: tuple of (model, feature train, target train, feature test, target test, train group, test group)
    :return:list of [original labels,labels after classification, the posterior probabilities]
    """
    model, feature_train, target_train, feature_test, target_test, train_group, test_group = arguments_input
    if Classification.BOOTSTRAP:
        feature_train, target_train, train_group = Classification._bootstrap(feature_train, target_train,
                                                                             train_group)
    if Classification.SMOOTE:
        feature_train, target_train = Classification.__smoote(feature_train, target_train)
    model.fit(feature_train, target_train)
    prediction = model.predict(feature_test)
    prediction = prediction.tolist()
    probability = model.predict_proba(feature_test)
    label = lists_solver(target_test.tolist())
    out_put = [label, prediction, probability]
    return out_put


def _most_likelihood(arguments_input):
    """
    classify the data most likelihood
    :param arguments_input: tuple of (model, feature train, target train, feature test, target test, train group, test group)
    :return:list of [original labels,labels after classification, the posterior probabilities]
    """
    model, feature_train, target_train, feature_test, target_test, train_group, test_group = arguments_input
    if Classification.BOOTSTRAP:
        feature_train, target_train, train_group = Classification._bootstrap(feature_train, target_train,
                                                                             train_group)
    if Classification.SMOOTE:
        feature_train, target_train = Classification.__smoote(feature_train, target_train)
    model.fit(feature_train, target_train)
    group_u = np.unique(test_group)
    try:
        local_df = np.concatenate((feature_test, target_test, test_group), axis=1)
    except:
        target_test = np.reshape(target_test, [target_test.shape[0], 1])
        test_group = np.reshape(test_group, [test_group.shape[0], 1])
        local_df = np.concatenate((feature_test, target_test, test_group), axis=1)
    col = list(range(feature_train.shape[1]))
    col = np.append(col, ["target", "group"])
    local_df = pd.DataFrame(local_df, columns=col)
    label = []
    probabilities = np.array([[0, 0], [0, 0]])
    predictions = []
    for group in group_u:
        tamp = local_df.loc[local_df["group"] == group, :]
        feature = tamp.iloc[:, :-2].values
        label.append(tamp.iloc[0, -2].tolist())

        probability = model.predict_proba(feature)
        if probability.shape[1] == 2:
            probability_to_one = sum(list(map(lambda x: np.log2(x), probability[:, 1]))) / probability.shape[0]
            probability_to_zero = sum(list(map(lambda x: np.log2(x), probability[:, 0]))) / probability.shape[0]
            if probability_to_one > probability_to_zero:
                prediction = 1
            else:
                prediction = 0
            predictions.append(prediction)
            probabilities = np.concatenate((probabilities, probability), axis=0)
        else:
            predictions.append(probability[0, 0].astype(int))
            probability_rebild = np.zeros([probability.shape[0], 2])
            probability_rebild[:, probability[0, 0].astype(int)] = 1
            probabilities = np.concatenate((probabilities, probability_rebild), axis=0)
    probabilities = probabilities[2:, :]

    out_put = [label, predictions, probabilities]
    return out_put


def _majority_vote(arguments_input):
    """
    classify the data majority vote
    :param arguments_input: tuple of (model, feature train, target train, feature test, target test, train group, test group)
    :return:list of [original labels,labels after classification, the posterior probabilities]
    """
    model, feature_train, target_train, feature_test, target_test, train_group, test_group = arguments_input
    if Classification.BOOTSTRAP:
        feature_train, target_train, train_group = Classification._bootstrap(feature_train, target_train,
                                                                             train_group)
    if Classification.SMOOTE:
        feature_train, target_train = Classification.__smoote(feature_train, target_train)
    model.fit(feature_train, target_train)
    group_u = np.unique(test_group)
    try:
        local_df = np.concatenate((feature_test, target_test, test_group), axis=1)
    except:
        target_test = np.reshape(target_test, [target_test.shape[0], 1])
        test_group = np.reshape(test_group, [test_group.shape[0], 1])
        local_df = np.concatenate((feature_test, target_test, test_group), axis=1)
    col = list(range(feature_train.shape[1]))
    col = np.append(col, ["target", "group"])
    local_df = pd.DataFrame(local_df, columns=col)
    label = []
    probabilities = np.array([[0, 0], [0, 0]])
    predictions = []
    for group in group_u:
        tamp = local_df.loc[local_df["group"] == group, :]
        feature = tamp.iloc[:, :-2].values
        label.append(tamp.iloc[0, -2].tolist())
        prediction = model.predict(feature)
        unique_elements, counts_elements = np.unique(prediction, return_counts=True)
        prediction = unique_elements[np.argmax(counts_elements)].astype(int)
        predictions.append(prediction)
        probability = model.predict_proba(feature)
        probabilities = np.concatenate((probabilities, probability), axis=0)
    probabilities = probabilities[2:, :]

    out_put = [label, predictions, probabilities]
    return out_put
