"""
Count the number of B mesons in the dataset, and seperate them into B+ and B- mesons.
20/01 - created
"""

import pickle


def __load_signal_data():
    """
    Load the cleaned signal 2011 dataset after background fitting and weighting.
    Returns:
    pd.DataFrame
        The cleaned dataset with event weights applied.
    """
    with open('data/cleaned_data_2011.pkl', 'rb') as f:
        cleaned_data = pickle.load(f)
    return cleaned_data


def __load_cleaned_mag_data():
    """
    Load the cleaned magnetic 2012 dataset after background fitting and weighting.
    Returns:
    pd.DataFrame
        The cleaned dataset with event weights applied.
    """
    with open('data/cleaned_data_2012.pkl', 'rb') as f:
        cleaned_data = pickle.load(f)

    mag_up = cleaned_data[cleaned_data['polarity'] == 1]
    mag_down = cleaned_data[cleaned_data['polarity'] == 0]
    return mag_up, mag_down


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

# %%


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
    total_B_plus = data[data['Kaon assumed particle type'] > 0].shape[0]
    total_B_minus = data[data['Kaon assumed particle type'] < 0].shape[0]

    return total_B, total_B_plus, total_B_minus


def compute_b_asymmetry(data):
    total_B, total_B_plus, total_B_minus = count_B_mesons(data)
    print(f"Total number of B mesons: {total_B}")
    print(f"Total number of B+ mesons: {total_B_plus}")
    print(f"Total number of B- mesons: {total_B_minus}")
    print(
        f"CP Asymmetry: {(total_B_plus - total_B_minus) / (total_B_plus + total_B_minus):.4f}")


def compute_dimuon_asymmetry(data):
    # Ensure we have the weights from the previous background fit
    # Standard J/psi window
    jpsi_mask = data['dimuon-system invariant mass'].between(3000, 3150)
    pos_events = data[jpsi_mask & (data['B assumed particle type'] > 0)]
    neg_events = data[jpsi_mask & (data['B assumed particle type'] < 0)]

    yield_pos_jpsi = pos_events['event_weight'].sum()
    yield_neg_jpsi = neg_events['event_weight'].sum()

    # We do the same for the psi(2S) window, separate to the J/psi
    # Standard psi(2S) window
    psi2s_mask = data['dimuon-system invariant mass'].between(3600, 3750)
    pos_events_psi2s = data[psi2s_mask & (data['B assumed particle type'] > 0)]
    neg_events_psi2s = data[psi2s_mask & (data['B assumed particle type'] < 0)]
    yield_pos_psi2s = pos_events_psi2s['event_weight'].sum()
    yield_neg_psi2s = neg_events_psi2s['event_weight'].sum()

    if (yield_pos_jpsi + yield_neg_jpsi) == 0:
        return 0

    if (yield_pos_psi2s + yield_neg_psi2s) == 0:
        return 0

    cp_asymmetry_Jpsi = (yield_pos_jpsi - yield_neg_jpsi) / \
        (yield_pos_jpsi + yield_neg_jpsi)
    cp_asymmetry_psi2s = (yield_pos_psi2s - yield_neg_psi2s) / \
        (yield_pos_psi2s + yield_neg_psi2s)

    print(f'Signal Yield (B+ J/psi): {yield_pos_jpsi:.2f}')
    print(f'Signal Yield (B- J/psi): {yield_neg_jpsi:.2f}')
    print(f'Signal Yield (B+ psi(2S)): {yield_pos_psi2s:.2f}')
    print(f'Signal Yield (B- psi(2S)): {yield_neg_psi2s:.2f}')
    print(f'CP Asymmetry (J/psi): {cp_asymmetry_Jpsi:.4f}')
    print(f'CP Asymmetry (psi(2S)): {cp_asymmetry_psi2s:.4f}')

    return cp_asymmetry_Jpsi, cp_asymmetry_psi2s


def CP_asymmetry_mag():
    mag_data_up, mag_data_down = __load_cleaned_mag_data()
    print("Magnetic DOWN Data CP Asymmetry:")
    compute_b_asymmetry(mag_data_down)
    compute_dimuon_asymmetry(mag_data_down)
    print("Magnetic UP Data CP Asymmetry:")
    compute_b_asymmetry(mag_data_up)
    compute_dimuon_asymmetry(mag_data_up)


def CP_asymmetry_signal():
    print("Signal Data CP Asymmetry:")
    signal_data = __load_signal_data()
    compute_b_asymmetry(signal_data)
    compute_dimuon_asymmetry(signal_data)


if __name__ == "__main__":
    CP_asymmetry_mag()
    CP_asymmetry_signal()
