import pandas as pd

non_dummy_features = ["hotel_star_rating", "guest_is_not_the_customer", "no_of_adults", "no_of_children",
                      "no_of_extra_bed", "no_of_room", "original_selling_amount", "request_nonesmoke",
                      "request_latecheckin",
                      "request_highfloor", "request_largebed", "request_twinbeds", "request_airport",
                      "request_earlycheckin",
                      "has_canceled", "delta_time", "cost_per_night"]
features_to_drop = ["hotel_chain_code", "hotel_brand_code", "hotel_city_code", "hotel_live_date", "h_booking_id",
                    "h_customer_id", "hotel_country_code", "hotel_id"]
request_features = ["request_nonesmoke", "request_latecheckin",
                    "request_highfloor", "request_largebed", "request_twinbeds", "request_airport",
                    "request_earlycheckin"]
date_features = ["booking_datetime", "checkin_date", "checkout_date"]


class PreprocessEngine:
    def __init__(self, input_path: str, task_index: int = 0):
        self.df = pd.read_csv(input_path)
        self.name = input_path
        self.task_index = task_index

    def preprocess(self):
        print("Preprocessing ", self.name, "...")
        self.remove_unnecessary_features()
        if self.task_index == 0:  # we are on train mode
            self.add_has_canceled_feature()
            self.add_delta_time_feature()
        self.handle_date_features()
        self.handle_missing_values()
        self.create_dummies()
        print("Preprocessing ", self.name, " completed!")
        return self.df

    def remove_unnecessary_features(self):
        self.df = self.df.drop(columns=features_to_drop)

    def add_has_canceled_feature(self):
        if 'cancellation_datetime' in self.df.columns:
            self.df['has_canceled'] = self.df['cancellation_datetime'].notnull().astype(int)

    def add_delta_time_feature(self):
        self.df[['checkin_date', 'cancellation_datetime']] = self.df[
            ['checkin_date', 'cancellation_datetime']].apply(pd.to_datetime)
        self.df["delta_time"] = (self.df["checkin_date"] - self.df["cancellation_datetime"]).dt.days
        # set negative values in self.df["delta_time"] to -1
        self.df.loc[self.df["delta_time"] < 0, "delta_time"] = -1
        self.df["delta_time"] = self.df["delta_time"].fillna(-2)
        self.df = self.df.drop(columns=["cancellation_datetime"])

    def handle_request_classes(self):
        for feat in request_features:
            self.df[feat] = self.df[feat].fillna(-1)

    def handle_date_features(self):
        self.df[date_features] = self.df[date_features].apply(pd.to_datetime)
        for feat in date_features:
            year_feat = feat + "_year"
            month_feat = feat + "_month"
            day_feat = feat + "_day"
            self.df[year_feat] = self.df[feat].dt.year
            self.df[month_feat] = self.df[feat].dt.month
            self.df[day_feat] = self.df[feat].dt.day
            self.df = self.df.drop(columns=[feat])

    def handle_missing_values(self):
        self.handle_request_classes()

    def create_dummies(self):
        all_features = self.df.columns.tolist()
        dummy_features = [feature for feature in all_features if feature not in non_dummy_features]
        for feat in dummy_features:
            prefix = feat + "_"
            self.df = pd.get_dummies(self.df, prefix=prefix, columns=[feat])
