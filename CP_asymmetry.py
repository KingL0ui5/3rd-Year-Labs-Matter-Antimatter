"""
Count the number of B mesons in the dataset, and seperate them into B+ and B- mesons.
20/01 - created
"""

import predictions_data


def load_data():
    signal = predictions_data.main()
    return signal


def count_B_mesons(data):
    """
    Count the total number of B mesons in the dataset, and seperate them into B+ and B- mesons.
    returns:
    total_B : int
        Total number of B mesons in the dataset
    total_B_plus : int
        Total number of B+ mesons in the dataset
    total_B_minus : int
        Total number of B- mesons in the dataset
    """
    total_B = data.shape[0]
    total_B_plus = data[data['B assumed particle type'] > 0].shape[0]
    total_B_minus = data[data['B assumed particle type'] < 0].shape[0]

    return total_B, total_B_plus, total_B_minus
