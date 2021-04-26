from classification import *

if __name__ == "__main__":
    FEATURES_NUM = 30
    df_ = pd.read_csv("no_eveg_labeld.csv")
    df_ = df_.replace(["Bacterial", "Viral"], [1, 0])
    df_ = df_[(df_["analysis"] == 1) | (df_["analysis"] == 0)]
    print(df_.iloc[:, -3:].corr(), "\n")
    targ = ["WBC", "ANC", "crp"]
    random = 42
    Classification.set_global_setting(("random_state", random))
    ClassificationPreprocessing.ADD_TO_REPORT = "random forest"
    # Classification.limit_cores(3)
    PreProcess.SAVE = False
    Classification.BOOTSTRAP = True
    clas_tec = 'logistic_regression_regulated'
    print("only spectra")
    model = Classification(df_.loc[:, "1801.2639":"898.70345"].copy(),
                           df_.loc[:, "analysis"].copy(),
                           group=df_.loc[:, "ix"].copy()
                           )
    feat = model.feature_selection(number_of_features=FEATURES_NUM, ret=True)
    # model.sacrifice_rate_setter(0.1)
    # model.show_features()
    model.model_modifying(("max_depth", [3, 4, 5, 6, 7]), ("warm_start", [True, False]),
                          ("n_estimators", [100, 200, 300]))
    # model.print_class_waits()
    model.leave_one_out(method="most_liklihood")
    model.show_roc("only spectra by")
    model.save_classification_report("only spectra by")
    model.save_classification_probability("only spectra by")
    model.optimal_cut_point_on_roc_(plot_point_on_ROC=True)
