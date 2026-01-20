"""
Count the number of B mesons in the dataset, and seperate them into B+ and B- mesons.
20/01 - created
"""

import pickle
import predictions_data


def __load_signal_data():
    signal = predictions_data.main()
    return signal


def __load_simulation_data():
    """
    Load the simulation datasets for J/psi K and K mu mu.
    Returns:
    JpsiK_data : pd.DataFrame
        The dataset for the J/psi K simulation.
    Kmumu_data : pd.DataFrame
        The dataset for the K mu mu simulation."""
    with open('datasets/rapidsim_JpsiK.pkl', 'rb') as infile:
        JpsiK_data = pickle.load(infile)

    with open('datasets/rapidsim_Kmumu.pkl', 'rb') as infile:
        Kmumu_data = pickle.load(infile)

    return JpsiK_data, Kmumu_data


def __load_magnet_data():
    with open('datasets/dataset_2012_MagnetUp.pkl', 'rb') as infile:
        magnet_up_data = pickle.load(infile)

    with open('datasets/dataset_2012_MagnetDown.pkl', 'rb') as infile:
        magnet_down_data = pickle.load(infile)

    return magnet_up_data, magnet_down_data


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
    data = __load_simulation_data()
    print('\n'.join(data.keys()))
