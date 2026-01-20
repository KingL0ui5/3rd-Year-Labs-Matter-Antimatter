"""
Count the number of B mesons in the dataset, and seperate them into B+ and B- mesons.
20/01 - created
"""

from copyreg import pickle
import predictions_data


def __load_signal_data():
    signal = predictions_data.main()
    return signal


def __load_JpsiK_data():
    with open('datasets/rapidsim_JpsiK.pkl', 'rb') as infile:
        JpsiK_data = pickle.load(infile)
        return JpsiK_data


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


def compute_asymmetry():
    data = __load_signal_data()
    total_B, total_B_plus, total_B_minus = count_B_mesons(data)
    print(f"Total number of B mesons: {total_B}")
    print(f"Total number of B+ mesons: {total_B_plus}")
    print(f"Total number of B- mesons: {total_B_minus}")
    print(
        f"CP Asymmetry: {(total_B_plus - total_B_minus) / (total_B_plus + total_B_minus):.4f}")


if __name__ == "__main__":
    data = __load_JpsiK_data()
    print('\n'.join(data.keys()))
