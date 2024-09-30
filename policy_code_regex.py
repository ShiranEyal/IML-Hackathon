"""
This function takes a policy code as input and splits it into 4 columns for each of its parts
using a regular expression.
"""
import re
import os

import pandas as pd

# global variables
path_to_dataset = os.path.join('dataset', 'agoda_cancellation_train.csv')
entries_in_policy_code = 6

def regex_split(policy_code):
    policy_codes = policy_code.split('_')
    policies = []
    code_entries = [-1] * entries_in_policy_code
    for code in policy_codes:
        match = re.search("^(\d+D)?(\d+(?:N|P))$", code)
        policies.append(match)

    j = 0
    for match in policies:
        if match:
            groups = match.groups()
            for i in range(len(groups)):
                code_entries[j] = groups[i]
                j += 1

    return code_entries


def add_policy_codes_columns():
    # get dataset table
    dataset = pd.read_csv(path_to_dataset)
    code_policy_split_df = dataset['cancellation_policy_code'].apply(regex_split).apply(pd.Series)
    code_policy_split_df.rename(columns={0 : 'dats until 1st penalty', 1 : '1st penalty', 2 : 'days until 2nd penalty',
                                         3 : '2nd penalty', 4 : 'remove this column', 5 : 'no show penalty'}, inplace=True)

    dataset.to_csv('expanded_code_policy_version.csv')


add_policy_codes_columns()
# dataset = pd.read_csv(pth_to_dataset)
# policy_codes_column = dataset['cancellation_policy_code'].apply(regex_split).apply(pd.Series)
# print("temp")
