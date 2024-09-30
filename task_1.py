import pandas as pd
from sklearn.linear_model import LogisticRegression

from PreprocessEngine import PreprocessEngine


# >>> This function was used to find the best model for the task 1
# def get_best_model_for_task_1():
#     from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, \
#         AdaBoostClassifier, BaggingClassifier
#     from sklearn.metrics import f1_score
#     from sklearn.neighbors import KNeighborsClassifier
#     from sklearn.tree import DecisionTreeClassifier
#
#     df = pd.read_csv(TRAIN_DATA_PATH)
#     # Separate the data into test (25%) and train (75%) sets
#     train_df = PreprocessEngine(df.sample(frac=0.75, random_state=0), "TrainSetData").preprocess()
#     test_df = PreprocessEngine(df.drop(train_df.index), "TestSetData").preprocess() \
#         .reindex(columns=train_df.columns, fill_value=0)
#
#     models_score = dict()
#     # iterate over different classifiers models and print their accuracy
#     print("-------------------------------------------------")
#     all_models = [LogisticRegression(), Perceptron(), SGDClassifier(),
#                   KNeighborsClassifier(), DecisionTreeClassifier(),
#                   RandomForestClassifier(), AdaBoostClassifier(),
#                   GradientBoostingClassifier(), BaggingClassifier(),
#                   ExtraTreesClassifier()]
#
#     for model in all_models:
#         features_to_drop = ["has_canceled", "delta_time"]
#
#         print("Training ", model, "...")
#         model.fit(train_df.drop(columns=features_to_drop), train_df["has_canceled"])
#         print("Training ", model, " completed!")
#
#         print("Getting F1 score of ", model, "...")
#         # predict the test set
#         predictions = model.predict(test_df.drop(columns=features_to_drop))
#         # calculate the f1 score
#         F1_score = f1_score(test_df["has_canceled"], predictions)
#         print("F1 score of ", model, " is: ", F1_score)
#         print("-------------------------------------------------")
#         models_score[model] = F1_score
#
#     # return the model with the highest score
#     best_model = max(models_score, key=models_score.get)
#     return best_model


def cancellation_prediction(train_set: pd.DataFrame, test_set_input: str):
    # >>> Set the best classifier model (as we found after comparing between several models)
    classifier_model = LogisticRegression()

    # >>> Train the model
    features_to_drop = ["has_canceled", "delta_time"]
    print("Training ", classifier_model.__class__.__name__, "...")
    classifier_model.fit(train_set.drop(columns=features_to_drop), train_set["has_canceled"])
    print("Training ", classifier_model.__class__.__name__, " completed!")

    # >>> Preprocess the test data
    test_set = pd.read_csv(test_set_input)
    h_booking_id = test_set["h_booking_id"]
    test_df = PreprocessEngine(test_set_input, task_index=1).preprocess() \
        .reindex(columns=train_set.columns, fill_value=0)

    # >>> Predict the response of test set and save it to csv file
    print("Predicting the cancellation of ", test_set_input, "...")
    predictions = classifier_model.predict(test_df.drop(columns=features_to_drop))
    print("Predicting the cancellation of  ", test_set_input, " completed!")

    # >>> Save the predictions to csv file
    pd.DataFrame({
        "h_booking_id": h_booking_id,
        "cancellation": list(predictions)
    }).to_csv("agoda_cancellation_prediction.csv", index=False)

    return classifier_model
