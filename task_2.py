import pandas as pd
from sklearn.linear_model import Ridge

from PreprocessEngine import PreprocessEngine


def calculate_cost_of_cancellation():
    pass


def original_selling_amount_prediction(train_set: pd.DataFrame, test_set_input: str):
    # >>> Set the best regression model (as we found after comparing between several models)
    regression_model = Ridge()

    # >>> Train the model
    features_to_drop = ["has_canceled", "delta_time", "original_selling_amount"]
    print("Training ", regression_model.__class__.__name__, "...")
    regression_model.fit(train_set.drop(columns=features_to_drop), train_set["original_selling_amount"])
    print("Training ", regression_model.__class__.__name__, " completed!")

    # >>> Preprocess the test data
    test_set = pd.read_csv(test_set_input)
    h_booking_id = test_set["h_booking_id"]
    cancellation_policy_code = test_set["cancellation_policy_code"]
    checkin_date = test_set["checkin_date"]
    checkout_date = test_set["checkout_date"]
    test_df = PreprocessEngine(test_set_input, task_index=2).preprocess() \
        .reindex(columns=train_set.columns, fill_value=0)

    # >>> Predict the response of test set
    print("Predicting the original selling amount of ", test_set_input, "...")
    predictions = regression_model.predict(test_df.drop(columns=features_to_drop))
    print("Predicting the original selling amount of  ", test_set_input, " completed!")

    test_df["original_selling_amount"] = predictions

    return cancellation_policy_code, checkin_date, checkout_date, h_booking_id, test_df


def add_cost_per_night_feature(df: pd.DataFrame):
    df[['checkin_date', 'checkout_date']] = df[
        ['checkin_date', 'checkout_date']].apply(pd.to_datetime)
    df["cost_per_night"] = df["original_selling_amount"] / (
            df["checkout_date"] - df["checkin_date"]).dt.days

def add_delta_time_feature(df: pd.DataFrame):
    df[['checkin_date', 'cancellation_datetime']] = df[
        ['checkin_date', 'cancellation_datetime']].apply(pd.to_datetime)
    df["delta_time"] = (df["checkin_date"] - df["cancellation_datetime"]).dt.days
    # set negative values in self.df["delta_time"] to -1
    df.loc[df["delta_time"] < 0, "delta_time"] = -1
    df["delta_time"] = df["delta_time"].fillna(-2)
    df = df.drop(columns=["cancellation_datetime"])


def get_predicted_selling_amount(row):
    original_selling_amount = float(row['original_selling_amount'])
    cost_per_night = float(row['cost_per_night'])
    cancellation_policy_code = row['cancellation_policy_code']
    delta_time = row['delta_time']

    if row['has_canceled'] == 1:
        return calc_cost_from_policy(cancellation_policy_code, delta_time, cost_per_night, original_selling_amount)
    elif row['has_canceled'] == 0:
        return -1
    else:
        return None


def calc_cost_from_policy(cancellation_policy_code, delta_time, cost_per_night, original_selling_amount):
    if delta_time == -2:  # unlikely to be cancelled
        return -1
    elif cancellation_policy_code == 'UNKNOWN': # unknown cancelation code
        return 0
    # Extract the cancellation policy components
    else:
        policy_components = cancellation_policy_code.split('_')

        # Initialize the cancellation cost
        cancellation_cost = 0

        # Process each policy component
        for component in policy_components:
            parts = component.split('D')
            days = 0
            for part in parts:
                if part.endswith('N'):
                    # Cancellation charge in prices per night
                    nights = int(part[:-1])  # Extract the number of nights
                    if (delta_time < days):
                        cancellation_cost = nights * cost_per_night
                elif part.endswith('P'):
                    # Cancellation charge in percentage of the entire deal
                    percentage = int(part[:-1]) / 100  # Extract the percentage
                    if (delta_time < days):
                        cancellation_cost = percentage * original_selling_amount
                else:
                    days = int(part)
        return cancellation_cost


def cost_of_cancellation_prediction(classifier_model, train_set: pd.DataFrame, test_set_input: str):
    # >>> Predict the original_selling_amount of test set
    cancellation_policy_code, checkin_date, checkout_date, h_booking_id, test_df = \
        original_selling_amount_prediction(train_set, test_set_input)

    # >>> Predict the response of test set and save it to csv file
    print("Predicting the cancellation of ", test_set_input, "...")
    test_df = test_df.drop(columns=["has_canceled", "delta_time"])
    test_df["has_canceled"] = classifier_model.predict(test_df)
    print("Predicting the cancellation of  ", test_set_input, " completed!")

    delta_time_prediction(train_set, test_df)

    test_df["checkin_date"] = checkin_date
    test_df["checkout_date"] = checkout_date
    add_cost_per_night_feature(test_df)

    test_df["cancellation_policy_code"] = cancellation_policy_code

    test_df["predicted_selling_amount"] = test_df.apply(get_predicted_selling_amount, axis=1)

    pd.DataFrame({
        "h_booking_id": h_booking_id,
        "predicted_selling_amount": test_df["predicted_selling_amount"]
    }).to_csv("agoda_cost_of_cancellation.csv", index=False)

def delta_time_prediction(train_set, test_df):
    regression_model = Ridge()
    features_to_drop = ["delta_time"]
    print("Training ", regression_model.__class__.__name__, "for time from cancellation to checkin...")
    regression_model.fit(train_set.drop(columns=features_to_drop), train_set["delta_time"])
    print("Training ", regression_model.__class__.__name__, " completed!")

    print("Predicting the time from cancellation to checkin of test set...")
    test_df["delta_time"] = regression_model.predict(test_df)
    print("Predicting the time from cancellation to checkin of test set completed!")