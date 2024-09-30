import pandas as pd
from sklearn.linear_model import *
from PreprocessEngine import PreprocessEngine
from task_1 import cancellation_prediction
from task_2 import cost_of_cancellation_prediction
from task_3 import feature_evaluation

TRAIN_DATA_PATH = "agoda_cancellation_train.csv"
TEST1_DATA_PATH = "Agoda_Test_1.csv"
TEST2_DATA_PATH = "Agoda_Test_2.csv"

TASK_1_OUTPUT_PATH = "agoda_cancellation_prediction.csv"
TASK_2_OUTPUT_PATH = "agoda_cost_of_cancellation.csv"

if __name__ == '__main__':
    # Preprocess the train data
    train_set = PreprocessEngine(TRAIN_DATA_PATH).preprocess()

    # ==========================
    # ========= Task 1 =========
    # ==========================

    # >>> Predict the cancellation of test set
    classifier_model = cancellation_prediction(train_set, TEST1_DATA_PATH)

    # ==========================
    # ========= Task 2 =========
    # ==========================
    # >>> Predict the cost of cancellation of test set 2

    cost_of_cancellation_prediction(classifier_model, train_set, TEST2_DATA_PATH)

    #if you want to run task 3 it might be worth putting task 2 in comment (faster)
    # ==========================
    # ========= Task 3 =========
    # ==========================
    feature_evaluation(train_set.drop(columns=["has_canceled", "delta_time"]), train_set["has_canceled"])