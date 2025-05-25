# simulation vectorized forecast functions
import numpy as np

def simulate_eligibles(annual_eligibles, n_months, n_simulations, uncertainty=0.025):
    """
    Simulates the number of eligible patients per month over time with uncertainty, across multiple simulations.

    Parameters:
        annual_eligibles (int): The expected annual number of eligible patients.
        n_months (int): Number of months to forecast.
        n_simulations (int): Number of simulation runs.
        uncertainty (float): Proportional uncertainty applied to monthly patient incidence (default: 2.5%).

    Returns:
        np.ndarray: A (n_months, n_simulations) array of simulated eligible patients per month.
    """
    
    avg_monthly = annual_eligibles / 12
    rate_factors = np.random.uniform(1 - uncertainty, 1 + uncertainty, size=(n_months, n_simulations))
    monthly_eligibles = np.random.poisson(lam=avg_monthly * rate_factors)

    return monthly_eligibles


def simulate_seen_by_users(monthly_eligibles, n_months, n_simulations,
                           initial_user_share=0.10, max_user_share=0.70,
                           share_uncertainty=0.05, months_to_peak=36):
    """
    Simulates how many eligible patients are seen by users of Drug X each month,
    based on a linear uptake of user share with uncertainty.

    Parameters:
        monthly_eligibles (np.ndarray): Array of shape (n_months, n_simulations) with eligible patients.
        n_months (int): Number of months to forecast.
        n_simulations (int): Number of simulation runs.
        initial_user_share (float): Starting proportion of users (default: 0.10).
        max_user_share (float): Maximum proportion of users reached over time (default: 0.70).
        share_uncertainty (float): Uncertainty applied to user share estimates (default: 0.05).
        months_to_peak (int): Number of months to reach maximum user share.

    Returns:
        np.ndarray: A (n_months, n_simulations) array of patients seen by users per month.
    """

    t = np.arange(n_months)
    # Linear ramp-up in user share over time
    expected_user_share = initial_user_share + (max_user_share - initial_user_share) * (t / months_to_peak)
    expected_user_share = np.clip(expected_user_share, 0, max_user_share).reshape(-1, 1)
    
    # Add uncertainty range
    p_min = np.clip(expected_user_share - share_uncertainty, 0, 1)
    p_max = np.clip(expected_user_share + share_uncertainty, 0, 1)
    
    # Sample individual probabilities for each time step and simulation
    realized_user_share = np.random.uniform(low=p_min, high=p_max, size=(n_months, n_simulations))
    # Simulate how many eligible patients are seen by users
    seen_by_users = np.random.binomial(monthly_eligibles, p=realized_user_share)

    return seen_by_users


def simulate_new_pts_treated(seen_by_users, n_months, n_simulations,
                             mean_treatment_prob=0.25, treat_uncertainty=0.05):
    """
    Simulates how many patients are treated with Drug X each month, based on the
    number of patients seen by users and the probability to prescribe.

    Parameters:
        seen_by_users (np.ndarray): Array of shape (n_months, n_simulations) with patients seen by users.
        n_months (int): Number of months to forecast.
        n_simulations (int): Number of simulation runs.
        mean_treatment_prob (float): Average probability that a user prescribes Drug X (default: 0.25).
        treat_uncertainty (float): Uncertainty applied to the treatment probability (default: 0.05).

    Returns:
        np.ndarray: A (n_months, n_simulations) array of new patients treated with Drug X per month.
    """

    p_min = np.clip(mean_treatment_prob - treat_uncertainty, 0, 1)
    p_max = np.clip(mean_treatment_prob + treat_uncertainty, 0, 1)

    # Simulate variable prescribing probability over time and simulations
    treatment_probs = np.random.uniform(low=p_min, high=p_max, size=(n_months, n_simulations))
    new_treated = np.random.binomial(n=seen_by_users, p=treatment_probs)

    return new_treated


def simulate_treatment_persistence(new_treated, median_duration, max_duration=None):
    """
    Simulates persistence over time using exponential decay applied to a matrix of new treated patients.
    
    Parameters:
        new_treated: (n_months, n_simulations) matrix of newly treated patients
        median_duration: median number of months a patient stays on treatment
        max_duration: maximum treatment duration (cutoff)
    
    Returns:
        (n_months, n_simulations) matrix of patients on treatment at each month
    """
    n_months, n_simulations = new_treated.shape
    lambda_ = np.log(2) / median_duration  # exponential decay parameter

    # Create a persistence decay kernel matrix: shape (n_months, n_months)
    # persistence_matrix[t, d] = retention from cohort at month d, d months after treatment
    t = np.arange(n_months)
    decay_matrix = np.exp(-lambda_ * t[:, None])  # shape (n_months, 1)

    # If needed, zero out values beyond max_duration
    if max_duration is not None:
        decay_matrix[t > max_duration, :] = 0

    # Build a 3D tensor where we apply the decay for each cohort over time
    on_treatment = np.zeros_like(new_treated, dtype=float)

    for start_month in range(n_months):
        duration = n_months - start_month
        decay = decay_matrix[:duration, 0]  # decay curve for that cohort
        treated_cohort = new_treated[start_month]  # shape: (n_simulations,)
        on_treatment[start_month:start_month+duration] += decay[:, None] * treated_cohort

    return np.round(on_treatment).astype(int)