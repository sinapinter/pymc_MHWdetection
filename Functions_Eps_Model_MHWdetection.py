import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from scipy.stats import norm
import scipy.stats as stats
import matplotlib.colors as mcolors
import seaborn as sns
import xarray as xr
from datetime import datetime
from scipy.signal import find_peaks
from scipy.ndimage import label
from fastdtw import fastdtw
from collections import defaultdict
import string
import pickle

# Add Graphviz binaries to PATH so PyMC can find 'dot.exe'
# Adjust this path if Graphviz is installed in a different location
os.environ["PATH"] += os.pathsep + r"C:\Users\24048369\OneDrive - The University of Western Australia\Graphviz-12.2.1-win64\bin"

sys.path.append('C:/Users/24048369/OneDrive - The University of Western Australia/Desktop/Sina/marineHeatWaves-python/')
import marineHeatWavesCopy as mhw_hobday



# ### Important Functions

# In[3]:


RANDOM_SEED = 8929
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

plt.rcParams.update(
    {
        "figure.figsize": (14, 4),
        "figure.dpi": 144,
        "figure.constrained_layout.use": True,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.bottom": False,
        "axes.spines.left": False,
        "axes.spines.right": False,
    }
)


def sampler_kwargs():
    return dict(
        nuts_sampler="nutpie",
        chains=4,
        draws=500,
        nuts_sampler_kwargs={"backend": "jax", "gradient_backend": "jax"},
    )




def comparing_to_Hobday(df_hobday, ClimatologyPeriod = None):
    """
    df_hobday = pandas dataframe or Series with time index and 
        temperture timeseries
    """
    
    full_date_range = pd.date_range(start=df_hobday.index.min(), end=df_hobday.index.max(), freq='D')

    # # Reindex DataFrame to include missing dates
    df_reindex = df_hobday.reindex(full_date_range)
    
    # # Fill missing values using interpolation
    df_reindex = df_reindex.interpolate()
    
    
    if ClimatologyPeriod is None:   
        ClimatologyPeriod = [df_reindex.index.min().year, df_reindex.index.max().year]
   
    
    sst  = np.array(df_reindex)
    
    # Create a regular daily time vector from 1 Jan 1992 to 31 Dec 2019
    times_dates_mhw = df_reindex.index
    t = np.array([times_dates_mhw[i].toordinal() for i in range(len(times_dates_mhw)) ])  # final daily time vector 
    
    
    mhws, clim = mhw_hobday.detect(t, sst, climatologyPeriod=ClimatologyPeriod,
                               smoothPercentileWidth=31)

    return mhws, clim, times_dates_mhw



# Compute the empirical CDF using the full distribution of p90
def empirical_cdf(value, samples):
    return np.mean(samples <= value)  # Fraction of p90 samples ≤ value


def calculate_prob_PTR(og_timeseries, PTR, variable="Temp"):
    """
    Calculate the probability of the observed values in `of_timeseries` exceeding the PTR range.

    Parameters:
    -----------
    of_timeseries : pd.DataFrame
        Observed time series with a datetime index and at least one variable column (default "Temp").
    PTR : np.ndarray
        Array of simulated values. Shape: (n_simulations, n_timepoints)
   
    variable : str
        Column name in `og_timeseries` to use (default: "Temp").

    Returns:
    --------
    prob_PTR : list
        List of probabilities (in percent) for each date in `time_array`/ of_timeseries.index.
    """
    prob_PTR = []
    time_array = og_timeseries.index
    for id_ptr, date_ptr in enumerate(time_array):
        # Find the index of the date in the observed time series
        id_d_index = np.where(og_timeseries.index == date_ptr)[0]
        
        if len(id_d_index) == 0:
            raise ValueError(f"Date {date_ptr} not found in observed time series index.")
        
        # Get observed value and apply empirical CDF against PTR
        obs_value = og_timeseries[variable].iloc[id_d_index].values
        prob = empirical_cdf(obs_value, PTR[:, id_ptr]) * 100
        prob_PTR.append(prob)

    return prob_PTR


def get_hemisphere(lat: float) -> str:
    """
    Return hemisphere based on latitude.
    
    Parameters
    ----------
    lat : float
        Latitude in degrees.
    
    Returns
    -------
    str
        "Northern Hemisphere" or "Southern Hemisphere"
    """
    if lat >= 0:
        return "Northern Hemisphere"
    else:
        return "Southern Hemisphere"



def generate_time_slices(df, start_date, end_date, window_size, step_years=1, hemis="Northern Hemisphere"):
    """
    Generate rolling time slices from a time series dataframe,
    aligned with hemisphere-specific 'climate years'.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a DateTimeIndex.
    start_date : pd.Timestamp
        Start of the full period.
    end_date : pd.Timestamp
        End of the full period.
    window_size : int
        Window size in years.
    step_years : int, optional
        Step size between windows in years (default 1).
    hemis : str, optional
        "Northern Hemisphere" (year = Jan–Dec)
        "Southern Hemisphere" (year = Jul–Jun)

    Yields
    ------
    (slice_start, slice_end, data_slice) : tuple
        Start date, end date, and corresponding dataframe slice.
    """
    # --- Adjust start_date to the first "climate year" boundary ---
    if hemis == "Southern Hemisphere":
        # If start_date is before July → push back to previous July
        if start_date.month < 7:
            start_date = pd.Timestamp(year=start_date.year - 1, month=7, day=1)
        else:
            start_date = pd.Timestamp(year=start_date.year, month=7, day=1)

        # End date → last June
        if end_date.month < 7:
            end_date = pd.Timestamp(year=end_date.year, month=6, day=30)
        else:
            end_date = pd.Timestamp(year=end_date.year + 1, month=6, day=30)

    else:  # Northern Hemisphere (Jan–Dec)
        start_date = pd.Timestamp(year=start_date.year, month=1, day=1)
        end_date = pd.Timestamp(year=end_date.year, month=12, day=31)

    # --- Calculate how many windows fit ---
    max_windows = ((end_date.year - start_date.year + 1) - window_size) // step_years + 1
    if max_windows < 1:
        return  # nothing fits
    max_windows = max_windows -1
    print(f'Total Number of Time Windows: {max_windows}')
    # --- Loop through windows ---
    for i in range(max_windows):
        window_start = start_date + pd.DateOffset(years=i * step_years)
        if hemis == "Southern Hemisphere":
            window_end = window_start + pd.DateOffset(years=window_size) - pd.DateOffset(days=1)
            window_end = pd.Timestamp(year=window_end.year, month=6, day=30)  # force June 30
        else:
            window_end = pd.Timestamp(year=window_start.year + window_size - 1, month=12, day=31)

        if window_end > end_date:
            window_end = end_date

        # Slice dataframe
        start_id = get_nearest_idx(df.index, window_start)
        end_id = get_nearest_idx(df.index, window_end)
        data_slice = df.iloc[start_id:end_id, :]

        yield window_start, window_end, data_slice




def get_custom_soft_rainbow(n):
    # Define base colors in hex
    base_colors = [
        '#FF666680',  # firebrick
        '#FF8C0080',  # darkorange, semi-transparent
        '#FFD70080',  # gold, semi-transparent
        '#228B2280',  # forestgreen, semi-transparent
        '#40C9C980',  # soft turquoise, semi-transparent now
        '#1E90FF80',  # dodgerblue, semi-transparent
        '#00008080',  # navy
        '#80008080',  # purple, semi-transparent
    ]


   
    
    # Convert to RGB array (without alpha)
    base_rgb = np.array([mcolors.to_rgb(c) for c in base_colors])
    
    # Create an interpolation function for RGB
    x = np.linspace(0, 1, len(base_colors))
    xi = np.linspace(0, 1, n)
    
    r = np.interp(xi, x, base_rgb[:,0])
    g = np.interp(xi, x, base_rgb[:,1])
    b = np.interp(xi, x, base_rgb[:,2])
    
    # Combine and add alpha
    colors = [mcolors.to_hex([r[i], g[i], b[i]]) for i in range(n)]
    return colors



def get_freq_seaslenght(time_array):
    freq = pd.infer_freq(time_array)
    # Get seasonal length based on frequency
    if freq is None:
        raise ValueError("Could not infer frequency. Please check the index.")
    
    freq_lower = freq.lower()
    # print(freq)
    if freq_lower.startswith('d'):
        seas_length = 365.25
    elif freq_lower.startswith('w'):
        seas_length = 52
    elif freq_lower.startswith('m'):
        seas_length = 12
    else:
        raise ValueError(f"Unsupported frequency: {freq_lower}")
    return seas_length, freq



def summarize_uncertainty_eps(idata,  time_array):
    
    # For seasonality: average std across all dates (time dependent)
    seasonality_samples = idata.posterior["seasonality"]  # (chains, draws, date)
    seasonality_std = seasonality_samples.std(dim=("chain", "draw")).mean().values
   
    # Beta_fourier (seasonality coefficients)
    beta_std = idata.posterior["beta_fourier"].std(dim=("chain", "draw")).mean(dim="fourier_features").values

    # For alpha (scalar)
    alpha_samples = idata.posterior["alpha"]
    alpha_std = alpha_samples.std(dim=("chain", "draw")).values
    
    # # For epsilon coefficient (scalar)
    #epsilon_samples = idata.posterior["epsilon"]
    #epsilon_std = epsilon_samples.std(dim=("chain", "draw")).values

    # For observation noise sigma(scalar)
    sigma_samples = idata.posterior["sigma"]
    sigma_std = sigma_samples.std(dim=("chain", "draw")).values

    # For mu (time dependent)
    mu_sample = idata.posterior["mu"].values
    mu_std = np.std(mu_sample, axis=0).mean()
    

    seas_length, freq = get_freq_seaslenght(time_array)
 

    ### PTR AND PCR (time dependent)
    test_like = idata.posterior_predictive.likelihood.values.reshape(-1, len(time_array))
    if seas_length > 300: 
        window = 5
    elif seas_length > 50 & seas_length < 60:
        window = 2   
    PCR, PTR = compute_seasonal_PCR_and_PTR_rolling(test_like, time_array, seas_length=seas_length, window=window)
    PTR_std =  np.std(PTR, axis = 0).mean()
    PCR_std =  np.std(PCR, axis = 0).mean()

    ## Overall likelihood (time dependent)
    likelihood_sample = idata.posterior_predictive.likelihood.values
    likelihood_std = np.std(likelihood_sample, axis=0).mean()
    
    
    return {
        "seasonality_std": seasonality_std,
        "beta_std" : beta_std,
        "alpha_std": alpha_std,
        "epsilon_std": epsilon_std,
        "sigma_std": sigma_std,
        "mu_std": mu_std,
        "PTR_std": PTR_std,
        "PCR_std": PCR_std,
        "likelihood_std": likelihood_std
         }






def get_nearest_idx(idx, target_date):
    pos = idx.searchsorted(target_date)
    if pos == 0:
        return 0
    elif pos == len(idx):
        return len(idx) - 1
    else:
        before = idx[pos - 1]
        after = idx[pos]
        if abs(target_date - before) <= abs(after - target_date):
            return pos - 1
        else:
            return pos

def get_doy_index(time_index, seas_length):
    """Returns 0-based day-of-year or week-of-year index."""
    if seas_length > 360:  # assume daily
        return time_index.dayofyear - 1
    elif seas_length == 52:  # assume weekly
        return time_index.isocalendar().week.values - 1



def compute_seasonal_PCR_and_PTR_rolling(test_like, time_index, seas_length=365, window=5):
    """
    Computes a seasonal rolling std.
    Output: seasonal_std of shape (seas_length,), repeating over time.
    """
    if seas_length > 360:
        seas_length = 366
        
    n_samples, n_time = test_like.shape
    season_idx = get_doy_index(time_index, seas_length)
    feb28 = 59
    feb29 = 60
    mar01 = 61


    
    # Step 1: Group time indices by seasonal index (0 to seas_length-1)
    seasonal_groups = {i: np.where(season_idx == i)[0] for i in range(seas_length)}

    # Step 2: Precompute rolling seasonal windows
    rolling_windows = {
        i: [(i + offset) % seas_length for offset in range(-window, window + 1)]
        for i in range(seas_length)
    }

    # Step 3: Allocate seasonal std array
    seasonal_PCR = np.empty((n_samples, seas_length))
    seasonal_PTR = np.empty((n_samples, seas_length))
    # Step 4: For each seasonal index, compute std across pooled samples
    for i in range(seas_length):
        if i == feb29:
            continue
       
        window_bins = rolling_windows[i]
        pooled_time_indices = np.concatenate([seasonal_groups[j] for j in window_bins if j in seasonal_groups])

        # Flatten over all samples and times in the seasonal window
        pooled_vals = test_like[:, pooled_time_indices]#.flatten()
    
        seasonal_PTR[:,i] = np.percentile(pooled_vals, 90, axis = 1)
        seasonal_PCR[:,i] = np.percentile(pooled_vals, 50, axis = 1)

    if seas_length> 300: ## ajdustment for 28th february only for dialy data 
        seasonal_PCR[:,feb29] = 0.5*seasonal_PCR[:,feb28] + 0.5*seasonal_PCR[:,mar01]
        seasonal_PTR[:,feb29] = 0.5*seasonal_PTR[:,feb28] + 0.5*seasonal_PTR[:,mar01]
   
    return seasonal_PCR[:,season_idx.astype(int)-1], seasonal_PTR[:,season_idx.astype(int)-1] # shape: (seas_length,)


# Function to calculate Dynamic Time Warping distance between two time series
def calculate_dtw(series_a, series_b):
    distance, _ = fastdtw(series_a, series_b)
    return distance

def climate_yearwise_dtw_similarity(data_slice,window_size, i ):
    data_slice = data_slice.copy()
    data_slice.index = pd.to_datetime(data_slice.index)
    
   # Similarity via Pearson correlation of daily curves
    daily_curves = []
    for j in range(window_size):
        start = f"{1994+i+j}-07-01"
        end = f"{1994+i+j+1}-06-30"
        mask = (data_slice.index >= start) & (data_slice.index <= end)
        df_sub = pd.DataFrame(data_slice.loc[mask].copy())
        df_sub['doy'] = df_sub.index.dayofyear
        daily_curve = df_sub.groupby('doy')['Temp'].mean()
        daily_curves.append(daily_curve)
    

    distances = []
    for a in range(len(daily_curves)):
        for b in range(a+1, len(daily_curves)):
            # print(daily_curves[a], daily_curves[b])
            d = calculate_dtw(daily_curves[a], daily_curves[b])
            distances.append(d)

    return np.mean(distances)





def detect_peaks_minima_plateaus(climatology, flat_threshold=0.08,var = 'Climatology'):
    """
    Detect peaks, minima, and plateaus in a seasonal climatology array.

    Parameters:
    -----------
    climatology : 1D numpy array
        Array of seasonal values (e.g., monthly averages).
    flat_threshold : float, optional
        Threshold for gradient magnitude to consider a region flat (plateau).

    Returns:
    --------
    peaks : numpy array
        Indices of detected peaks.
    minima : numpy array
        Indices of detected minima.
    plateau_indices : list of numpy arrays
        List where each element contains indices of a plateau region.
    """
    # Detect peaks
    peaks, _ = find_peaks(climatology)

    # Detect minima by inverting the data
    minima, _ = find_peaks(-climatology)

    # Calculate gradient
    gradient = np.gradient(climatology)

    # Detect flat regions where gradient magnitude is below threshold
    flat_regions = np.abs(gradient) < flat_threshold
    
    # # Label contiguous flat regions as plateaus
    labeled_plateaus, num_plateaus  = label(flat_regions)

    # Convert peaks and minima to a set for fast lookup
    peak_min_indices = set(peaks).union(set(minima))
    
    # Filter plateaus to exclude those overlapping peaks or minima
    plateau_indices = []
    for i in range(1, num_plateaus + 1):
        indices = np.where(labeled_plateaus == i)[0]
        
        # Check overlap with peaks or minima
        if not any(idx in peak_min_indices for idx in indices):
            plateau_indices.append(indices)

    x = np.arange(len(climatology))
    climatology = np.array(climatology)

    fig,axs = plt.subplots(1,2, figsize=(12,5))
    ax = axs[0]
    ax.plot(gradient, label = 'Gradient', color = 'black')
    ax.set_title('Gradient')
    ax.axhline(flat_threshold, label = 'flat_threshold')
    ax.axhline(-flat_threshold, label = '-flat_threshold')
    ax.legend()
    ax.set_xlabel('Time Index')
    
    ax = axs[1]
    ax.plot(x, climatology, label=var, color = 'black')

    # Plot peaks and minima
    ax.plot(x[peaks], climatology[peaks], 'ro', label='Peaks')
    ax.plot(x[minima], climatology[minima], 'bo', label='Minima')

   
    # Shade plateau regions
    for idx in plateau_indices:
        start = int(x[idx[0]] )
        end = int(x[idx[-1] + 1])
        ax.axvspan(start, end, color='orange', alpha=0.3, label='Plateau (Wide Peak)')
        

    ax.legend()
    ax.set_title(f'{var} with Peaks, Minima, and Plateaus')
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Value')
    plt.grid(True)
    
    plt.show()

    # Base extrema count
    n_extrema = len(peaks) + len(minima)
    base_n_order = int(np.ceil(n_extrema / 2))

     # Adjust for plateaus (can make this +1 or +len(plateaus) depending on preference)
    if len(plateau_indices) >= 1:
        adjusted_n_order = base_n_order + len(plateau_indices) + 1
    elif len(plateau_indices) == 0:
        adjusted_n_order = base_n_order + 1

    # Cap n_order to prevent overfitting
    final_n_order = min(adjusted_n_order, 12)
    print(f"Final suggested harmonics n_order: {base_n_order} to {final_n_order}")
    

    return final_n_order


def get_PCRs_and_PTR(idata, og_timeseries):
   
    # Time axis
    future_dates = og_timeseries.index 

    seas_length, freq =get_freq_seaslenght(future_dates)


    ### PCR and PTR
    test_like = idata.posterior_predictive.likelihood.values.reshape(-1,  len(future_dates))
    PCR, PTR = compute_seasonal_PCR_and_PTR_rolling(test_like, future_dates, seas_length=seas_length, window=1)
  
    return PCR, PTR, future_dates


def make_subset(df, center_date, years, mode="centered", verbose=True, plot= True):
    """
    Create subsets around a selected date with flexible positioning
    and automatic fallback if window exceeds timeseries bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with DateTimeIndex.
    center_date : str or pd.Timestamp
        The date of interest.
    years : int or float
        Window length in years.
    mode : str
        - 'centered':   center date in the middle of the window
        - 'beginning':  date_of_interest = first day
        - 'end':        date_of_interest = last day
        - 'early':      date_of_interest in first year of window
        - 'late':       date_of_interest in last year of window
    verbose : bool
        If True, prints warnings when fallback is applied.

    Returns
    -------
    subset : pd.DataFrame
    """

    center_date = pd.to_datetime(center_date)
    total_days = pd.Timedelta(days=365.25 * years)

    # Default calculation
    if mode == "centered":
        half = total_days / 2
        start, end = center_date - half, center_date + half
    elif mode == "beginning":
        start, end = center_date, center_date + total_days
    elif mode == "end":
        start, end = center_date - total_days, center_date
    elif mode == "early":
        start = center_date - pd.Timedelta(days=365.25)
        end   = start + total_days
    elif mode == "late":
        end   = center_date + pd.Timedelta(days=365.25)
        start = end - total_days
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # --- Check bounds ---
    series_start, series_end = df.index.min(), df.index.max()

    # --- Check bounds and fallback ---
    series_start, series_end = df.index.min(), df.index.max()

    if start < series_start:
        if verbose:
            print(f"[WARN] '{mode}' window before data start → shifting to series start")
        start = series_start
        end = start + total_days
    if end > series_end:
        if verbose:
            print(f"[WARN] '{mode}' window beyond data end → shifting to series end")
        end = series_end
        start = end - total_days

    if plot == True: 
        plt.figure()
        plt.plot(df.loc[start:end].index, df.loc[start:end], color = 'dodgerblue')
        plt.axvline(center_date, color = 'black')

    return df.loc[start:end].copy()

def plot_probs_range(PR, time_array, ax, color = 'red', label = 'PTR', mean_line = True):
     PR_p5, PR_p50, PR_p95 = np.percentile(PR, [1, 50, 99], axis=0)
     ax.fill_between(time_array, PR_p5, PR_p95, color=color, alpha=0.4, label = label)
     if mean_line == True: 
         ax.plot(time_array, PR_p50, color = color, label = f"Mean {label}")
         
    



def detect_mhws_from_hps(HPS, T_obs, PCR, PTR, time=None, detection_threshold=50,  ramp_threshold=0.2,  max_dip=10,  min_duration=1):
    """
    Detect marine heatwaves from HPS + observed SST + PCR posterior draws.

    Parameters
    ----------
    HPS : array-like or pd.Series
        Heat probability score (0..1) for each time step, aligned with T_obs.
    T_obs : pd.Series or array-like
        Observed SST (same length as HPS). If pd.Series, its index is used as dates.
    PCR : np.ndarray
        Probabilistic climatology range (n_samples x n_times)., shape (n_samples, n_times).
        Percentiles will be extracted across the sample axis.
    PTR : np.ndarray
        Probabilistic threshold range (n_samples x n_times).
    time : array-like or None
        If provided, used as index. If None and T_obs is a Series, uses its index.
    detection_threshold : float
        HPS threshold for a core MHW day (default 0.7).
    ramp_threshold : float
        Lower HPS threshold used to back-extend start to ramp-up (default 0.2).
    max_dip : int
        Max consecutive days below detection_threshold inside an event before splitting (default 2).
    min_duration : int
        Minimum event length (days) to keep.

    Returns
    -------
    mhw_dict : dict
        Hobday-style dict with lists:
          - date_start, date_end, duration
          - intensity_max_p50, intensity_mean_p50, intensity_cumulative_p50
          - intensity_max_p1,  intensity_mean_p1,  intensity_cumulative_p1
          - intensity_max_p99, intensity_mean_p99, intensity_cumulative_p99
          - n_events (int, total number of detected events)
    """
    # ---------- collapse PCR + PTR to percentiles ----------
    if PCR.ndim != 2 or PTR.ndim != 2:
        raise ValueError("PCR and PTR must both be (n_samples, n_times)")
        
    # ---------- collapse PCR samples to time-series percentiles ----------
    # Expect PCR shape (n_samples, n_times)
    if not isinstance(PCR, np.ndarray) or PCR.ndim != 2:
        raise ValueError("PCR must be a numpy array of shape (n_samples, n_times)")

    p1 = np.percentile(PCR, 1, axis=0)
    p50 = np.percentile(PCR, 50, axis=0)
    p99 = np.percentile(PCR, 99, axis=0)

    # threshold range (90th percentile distribution)
    ptr_p1  = np.percentile(PTR, 1, axis=0)
    ptr_p50 = np.percentile(PTR, 50, axis=0)
    ptr_p99 = np.percentile(PTR, 99, axis=0)
    

    # ---------- align inputs ----------
    if isinstance(T_obs, pd.Series):
        T = T_obs.copy().astype(float)
        idx = T.index
    else:
        T = pd.Series(np.asarray(T_obs).astype(float))
        idx = T.index

    H = pd.Series(np.asarray(HPS).astype(float), index=idx)

    if time is not None:
        time = pd.to_datetime(time)
        if len(time) != len(idx):
            raise ValueError("time length must match HPS/T_obs length")
        idx = pd.Index(time)
        T.index = idx
        H.index = idx

    p1 = pd.Series(p1, index=idx)
    p50 = pd.Series(p50, index=idx)
    p99 = pd.Series(p99, index=idx)

    ptr_p1  = pd.Series(ptr_p1, index=idx)
    ptr_p50 = pd.Series(ptr_p50, index=idx)
    ptr_p99 = pd.Series(ptr_p99, index=idx)
    

    # ---------- detect core days ----------
    core_mask = H >= detection_threshold
    n = len(H)
    values = core_mask.values.astype(int)

    events = []
    i = 0
    while i < n:
        if values[i] == 0:
            i += 1
            continue
        # start of a core block
        j = i
        while j + 1 < n and values[j + 1] == 1:
            j += 1
        # allow short dips
        k = j + 1
        while k < n:
            gap = 0
            while k < n and values[k] == 0 and gap <= max_dip:
                gap += 1
                k += 1
            if gap == 0 or gap > max_dip or k >= n:
                break
            if k < n and values[k] == 1:
                j = k
                k = j + 1
            else:
                break

        # back-extend with ramp_threshold
        s = i
        while s - 1 >= 0 and H.iloc[s - 1] >= ramp_threshold:
            s -= 1
        e = j
        while e + 1 < n and H.iloc[e + 1] >= ramp_threshold:
            e += 1

        events.append((s, e))
        i = j + 1

    # after detecting events [(s, e), ...]
    merge_gap = 14  # days allowed between events to merge
    
    merged_events = []
    if events:
        current_start, current_end = events[0]
        for (s, e) in events[1:]:
            gap = (s - current_end)  # in index units (days if daily index)
            if gap <= merge_gap:
                # merge: extend current event
                current_end = e
            else:
                # save old event, start new one
                merged_events.append((current_start, current_end))
                current_start, current_end = s, e
        merged_events.append((current_start, current_end))
    else:
        merged_events = []
    
    # use merged_events instead of events
    events = merged_events

    # ---------- compute stats ----------
    mhw_dict = {
        'date_start': [],
        'date_end': [],
        'duration': [],
        'intensity_max_p50': [],
        'intensity_mean_p50': [],
        'intensity_cumulative_p50': [],
        'intensity_max_p1': [],
        'intensity_mean_p1': [],
        'intensity_cumulative_p1': [],
        'intensity_max_p99': [],
        'intensity_mean_p99': [],
        'intensity_cumulative_p99': [],
        # categories (relative to PTR, baseline = PCR p50)
        'category_p50': [], 'category_p1': [], 'category_p99': [],
        
        'n_events': 0
    }

    def classify_event(intensity_series, thr_series, clim_series):
        """
        Classify MHW category for a given event based on the day of max intensity.
    
        intensity_series : 1D array or pd.Series of event intensities
        thr_series       : 1D array or pd.Series of thresholds
        """
    
        # convert to Series if needed (so we can use dropna/idxmax)
        if not isinstance(intensity_series, pd.Series):
            intensity_series = pd.Series(intensity_series)
        if not isinstance(thr_series, pd.Series):
            thr_series = pd.Series(thr_series)
        if not isinstance(clim_series, pd.Series):
            clim_series = pd.Series(clim_series)
        
        idx_max = np.where(intensity_series == np.nanmax(intensity_series))[0]

        I_max = float(intensity_series.iloc[idx_max].iloc[0])
        T_thr = float(thr_series.iloc[idx_max].iloc[0])
        T_cli = float(clim_series.iloc[idx_max].iloc[0])

        T_steps = T_thr - T_cli
      
    
        diff = I_max - T_steps
        if diff <= 0:
            # print(diff)
            category = "undefined"
        elif diff < 2 * T_steps:
            category = "moderate"
        elif diff < 3 * T_steps:
            category = "strong"
        elif diff < 4 * T_steps:
            category = "severe"
        else:
            category = "extreme"
    
        return category


    
    for (s, e) in events:
        idx_range = np.arange(s, e + 1)
        T_event = T.iloc[idx_range].dropna()
        if len(T_event) < min_duration:
            continue

        p1_event = p1.iloc[idx_range].reindex(T_event.index)
        p50_event = p50.iloc[idx_range].reindex(T_event.index)
        p99_event = p99.iloc[idx_range].reindex(T_event.index)

        ptr1  = ptr_p1.iloc[idx_range].reindex(T_event.index)
        ptr50 = ptr_p50.iloc[idx_range].reindex(T_event.index)
        ptr99 = ptr_p99.iloc[idx_range].reindex(T_event.index)

        
        # intensity relative to p50,p1 and p99
        diff_p50 = T_event.values - p50_event.values
        diff_p1  = T_event.values - p1_event.values
        diff_p99 = T_event.values - p99_event.values

        def summarize(diff):
            positive = np.where(diff > 0, diff, 0)
            return (
                float(np.nanmax(positive)),
                float(np.nanmean(positive)),
                float(np.nansum(positive))
            )

        max50, mean50, cum50 = summarize(diff_p50)
        max1, mean1, cum1    = summarize(diff_p1)
        max99, mean99, cum99 = summarize(diff_p99)

        # categories relative to PTR, baseline = pcr50
        cat_p50 =  classify_event(diff_p50, ptr50, p50_event)
        cat_p1  =  classify_event(diff_p50, ptr1, p50_event)
        cat_p99 =  classify_event(diff_p50, ptr99, p50_event)

        
        mhw_dict['date_start'].append(T.index[s])
        mhw_dict['date_end'].append(T.index[e])
        mhw_dict['duration'].append(len(T_event))

        mhw_dict['intensity_max_p50'].append(max50)
        mhw_dict['intensity_mean_p50'].append(mean50)
        mhw_dict['intensity_cumulative_p50'].append(cum50)

        mhw_dict['intensity_max_p1'].append(max1)
        mhw_dict['intensity_mean_p1'].append(mean1)
        mhw_dict['intensity_cumulative_p1'].append(cum1)

        mhw_dict['intensity_max_p99'].append(max99)
        mhw_dict['intensity_mean_p99'].append(mean99)
        mhw_dict['intensity_cumulative_p99'].append(cum99)

        mhw_dict['category_p50'].append(cat_p50)
        mhw_dict['category_p1'].append(cat_p1)
        mhw_dict['category_p99'].append(cat_p99)
        

        

    mhw_dict['n_events'] = len(mhw_dict['date_start'])
    return mhw_dict




def plot_mhw_phases(T_obs, HPS, mhw_dict, ax, detection_threshold=50, ramp_threshold=20):
    HPS = pd.Series(np.asarray(HPS).astype(float), index=T_obs.index)
   
    # plot observed SST
    ax.plot(T_obs.index, T_obs, label= 'Observered\nTemperature', color="black",lw = 2)

    # loop through events
    for i, (start, end) in enumerate(zip(mhw_dict["date_start"], mhw_dict["date_end"])):
        event_mask = (T_obs.index >= start) & (T_obs.index <= end)
        hps_event = HPS[event_mask]
     

       
        # development phase = from start until first day above detection_threshold
        if (hps_event >= detection_threshold).any():
            core_start = hps_event.index[hps_event >= detection_threshold][0]
            core_end   = hps_event.index[hps_event >= detection_threshold][-1]
        else:
            core_start, core_end = start, end  # fallback (shouldn’t happen)

        # only label the first occurrence
        dev_label = 'MHW Development Phase' if i == 0 else None
        core_label = 'MHW Core Phase' if i == 0 else None
        dec_label = 'MHW Decay Phase' if i == 0 else None

        ax.axvspan(start, core_start, color="gold", alpha=0.3, label=dev_label, zorder = 0)
        ax.axvspan(core_start, core_end, color="darkorange", alpha=0.3, label=core_label, zorder = 0)
        ax.axvspan(core_end, end, color="green", alpha=0.3, label=dec_label, zorder = 0)
        

