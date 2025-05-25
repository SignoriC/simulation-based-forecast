# Functions for Forecasting Patient Uptake for a Drug X

import numpy as np  
import pandas as pd

# Simulate number of eligible patients per month
def simulate_monthly_eligibles(annual_eligibles, n_months, start_date=None, fluctuation_pct=0.025):
    """
    Simulates the number of monthly eligible patients over time using a Poisson process.

    Parameters:
    annual_eligibles (int): Number of eligible patients per year.
    n_months (int): Number of months to simulate.
    start_date (str, optional): Start date (YYYY-MM-DD). Defaults to first day of current month.
    fluctuation_pct (float, optional): Percent fluctuation in monthly incidence (default is 2.5%).

    Returns:
    pd.DataFrame: DataFrame indexed by month with simulated "Eligible_pts".
    """
    if start_date is None:
        start_date = pd.Timestamp.today().replace(day=1).strftime("%Y-%m-%d")
        
    avg_monthly = annual_eligibles / 12
    monthly_fluctuation = np.random.uniform(1 - fluctuation_pct, 1 + fluctuation_pct, n_months)
    monthly_counts = np.random.poisson(avg_monthly * monthly_fluctuation, n_months)
    date_index = pd.date_range(start=start_date, periods=n_months, freq="M")
    
    return pd.DataFrame({'Eligible_pts': monthly_counts}, index=date_index)


# Simulate Patients Seen by a Prescriber
def simulate_pts_seen_by_users(eligible_df, max_user_share=0.70, initial_user_share=0.10,
                                share_uncertainty=0.05, months_to_peak=36):
    """
    Adds a column to the DataFrame with the number of eligible patients seen by users of Drug X each month,
    assuming linear user adoption over time with uncertainty.

    Parameters:
    eligible_df (pd.DataFrame): DataFrame with 'Eligible_pts' column (monthly eligible patients).
    max_user_share (float): Maximum share of doctors using Drug X (e.g., 0.70).
    initial_user_share (float): Initial share of users at launch (e.g., 0.10).
    share_uncertainty (float): +/- uncertainty on user share (e.g., 0.05).
    months_to_peak (int): Number of months to reach max_user_share (default 36 months).

    Returns:
    pd.DataFrame: Modified DataFrame with new column 'Pts_seen_by_users'.
    """
    t = np.arange(len(eligible_df))

    # Linear ramp-up in user share over time
    base_share = initial_user_share + (max_user_share - initial_user_share) * (t / months_to_peak)
    base_share = np.clip(base_share, 0, max_user_share)

    # Add uncertainty range
    p_min = np.clip(base_share - share_uncertainty, 0, 1)
    p_max = np.clip(base_share + share_uncertainty, 0, 1)

    # Sample individual probabilities for each time step
    user_probs = np.random.uniform(p_min, p_max)

    # Simulate how many eligible patients are seen by users
    seen_by_users = np.random.binomial(n=eligible_df['Eligible_pts'].values, p=user_probs)

    eligible_df = eligible_df.copy()
    eligible_df['Pts_seen_by_users'] = seen_by_users

    return eligible_df


# Simulate Patients Prescribed Drug X
def simulate_new_pts_treated(user_seen_df, mean_treatment_prob=0.25, treatment_uncertainty=0.05):
    """
    Adds a column to the DataFrame with the number of patients actually treated with Drug X,
    among those seen by users, accounting for prescribing uncertainty.

    Parameters:
    user_seen_df (pd.DataFrame): DataFrame with column 'Pts_seen_by_users'.
    mean_treatment_prob (float): Mean probability a user prescribes Product X (default: 0.25).
    treatment_uncertainty (float): Range for uniform fluctuation around the mean (default: ±5%).

    Returns:
    pd.DataFrame: Modified DataFrame with new column 'New_pts_treated'.
    """
    n_months = len(user_seen_df)

    # Simulate variable prescribing probability over time
    treatment_probs = np.random.uniform(
        low=np.clip(mean_treatment_prob - treatment_uncertainty, 0, 1),
        high=np.clip(mean_treatment_prob + treatment_uncertainty, 0, 1),
        size=n_months
    )

    new_treated = np.random.binomial(n=user_seen_df['Pts_seen_by_users'].values, p=treatment_probs)

    user_seen_df = user_seen_df.copy()
    user_seen_df['New_pts_treated'] = new_treated

    return user_seen_df


# Simulate Duration on Treatment
def simulate_treatment_persistence(n_new_treated, median_duration_months=10, max_duration_months=None):
    """
    Simulates how many of the newly treated patients remain on treatment
    over time, optionally capped at a max duration.

    Parameters:
    n_new_treated (int): Number of new patients starting treatment.
    median_duration_months (float): Median treatment duration (default 10 months).
    max_duration_months (int or None): Maximum duration to consider (e.g., 12 for 12 months, 
        or None for full duration).

    Returns:
    List[int]: Number of patients still on treatment at each month.
    """
    scale = median_duration_months / np.log(2)
    durations = np.random.exponential(scale=scale, size=n_new_treated)

    if max_duration_months is None:
        max_month = int(np.ceil(durations.max()))
    else:
        max_month = max_duration_months

    patients_remaining = [np.sum(durations > month) for month in range(max_month)]

    return patients_remaining


# Simulate One Whole Forecast
def simulate_forecast(df, median_duration_months=10, max_duration_months=None):
    """
    Simulates the number of patients on Drug X each month based on treatment duration and uptake.

    Optimized for speed when scaling up to thousands of simulations.

    Parameters:
    - df (pd.DataFrame): Must contain 'New_pts_treated' column.
    - median_duration_months (float): Median duration patients stay on treatment.
    - max_duration_months (int or None): Max duration a patient can remain on treatment.

    Returns:
    - pd.DataFrame: Same DataFrame with 'Pts_on_treatment' column added.
    """
    n_months = len(df)
    treatment_window = max_duration_months if max_duration_months is not None else n_months
    persistence_matrix = np.zeros((n_months, treatment_window), dtype=int)

    # Simulate durations for all patients in a single pass (flat arrays, avoids Python loops)
    for month_idx, n_new in enumerate(df['New_pts_treated'].astype(int).values):
        if n_new == 0:
            continue
        # Simulate durations
        durations = np.random.exponential(scale=median_duration_months / np.log(2), size=n_new).astype(int)
        if max_duration_months:
            durations = np.clip(durations, 1, max_duration_months)
        else:
            durations = np.clip(durations, 1, n_months)

        # Count patients staying on treatment for at least t months
        for dur in durations:
            end = min(dur, treatment_window)
            persistence_matrix[month_idx, :end] += 1

    # Efficient diagonal summation to get calendar-month totals
    total_on_treatment = np.array([persistence_matrix[::-1, :].diagonal(i).sum()
                                   for i in range(-n_months + 1, treatment_window)])
    
    df['Pts_on_treatment'] = total_on_treatment[:n_months]
    return df
