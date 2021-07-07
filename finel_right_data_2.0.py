from itertools import combinations

from classification import *

if __name__ == "__main__":
    FEATURES_NUM = 35
    df_ = pd.read_csv("no_eveg_labeld.csv")
    os.chdir(f"C:\\Users\\shaiza\\Desktop\\New folder")
    df_ = df_.replace(["Bacterial", "Viral"], [1, 0])
    df_ = df_[(df_["analysis"] == 1) | (df_["analysis"] == 0)].dropna()
    print(df_.iloc[:, -3:].corr(), "\n")
    targ = ["WBC", "ANC", "crp"]
    random = 42
    Classification.set_global_setting(("random_state", random))
    Classification.grid_search_k_folds()
    # ClassificationPreprocessing.ADD_TO_REPORT = " by LLR"
    # Classification.limit_cores(3)
    # ClassificationPreprocessing.SAVE = False
    Classification.BOOTSTRAP = True
    clas_tec = 'random'
    print("only spectra")
    model = Classification(df_.loc[:, "1801.2639":"898.70345"].copy(),
                           df_.loc[:, "analysis"].copy(),
                           group=df_.loc[:, "ix"].copy(),
                           model=clas_tec)
    model.preprocessing(["sgf", "norm"])
    feat = model.feature_selection(number_of_features=FEATURES_NUM, ret=True)
    #
    # model.show_features()
    model.model_modifying(("max_depth", [3, 4, 5, 6, 7, 8]), ("warm_start", [True, False]),
                          ("n_estimators", [100, 200, 300, 400, 500]))
    # model.print_class_waits()
    model.leave_one_out(method="most_likelihoods")
    model.show_roc("only spectra by")
    model.save_classification_report("only spectra by")
    model.save_classification_probability("only spectra by")
    model.optimal_cut_point_on_roc_(plot_point_on_ROC=True)

    for comb_len in range(1, 4, 1):
        for current_combination in combinations(targ, comb_len):
            print(str(current_combination) + "only original data whit no spectra")
            model = Classification(df_.loc[:, current_combination].copy(),
                                   df_.loc[:, "analysis"].copy(),
                                   group=df_.loc[:, "ix"].copy(),
                                   model=clas_tec)
            model.preprocessing(["sgf", "norm"])
            model.model_modifying(("max_depth", [3, 4, 5, 6, 7, 8]), ("warm_start", [True, False]),
                                  ("n_estimators", [100, 200, 300, 400, 500]))
            # model.print_class_waits()
            model.leave_one_out(method="most_likelihoods")
            model.show_roc(str(current_combination) + "only original data by")
            model.save_classification_report(str(current_combination) + "only original data by")
            model.save_classification_probability(str(current_combination) + "only original data by")

            print(str(current_combination) + "spectra+original data")
            model = Classification(df_.loc[:, feat].copy(),
                                   df_.loc[:, "analysis"].copy(),
                                   group=df_.loc[:, "ix"].copy(),
                                   model=clas_tec)
            model.expend_features(df_.loc[:, current_combination].copy())
            model.preprocessing(["sgf", "norm"])
            model.model_modifying(("max_depth", [3, 4, 5, 6, 7, 8]), ("warm_start", [True, False]),
                                  ("n_estimators", [100, 200, 300, 400, 500]))
            # model.print_class_waits()
            model.leave_one_out(method="most_likelihoods")
            model.show_roc(str(current_combination) + "spectra+original data by")
            model.save_classification_report(str(current_combination) + "spectra+original data by")
            model.save_classification_probability(str(current_combination) + "spectra+original data by")
