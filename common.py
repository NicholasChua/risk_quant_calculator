from scipy.stats import beta, poisson, gaussian_kde
import numpy as np


def get_beta_parameters_for_kurtosis(kurtosis: int) -> tuple[float, float]:
    """Helper function to estimate parameters 'a' and 'b' for the beta distribution to achieve a desired kurtosis.

    This is a simplified heuristic mapping of kurtosis values to beta distribution parameters since the actual implementation involves math way beyond my pay grade/education level, and the actual implementation also runs too slowly for interactive use.

    Args:
        kurtosis: The desired kurtosis value for the distribution.

    Returns:
        tuple[float, float]: The parameters 'a' and 'b' for the beta distribution.
    """
    # For demonstration purposes, we'll adjust 'a' and 'b' based on the desired kurtosis.
    if kurtosis <= 1.8:
        a = b = 0.5  # Higher kurtosis (leptokurtic)
    elif 1.8 < kurtosis <= 3:
        a = b = 1  # Standard uniform distribution
    elif 3 < kurtosis <= 9:
        a = b = 2  # Lower kurtosis (platykurtic)
    else:
        a = b = 5  # Even lower kurtosis

    return a, b


def calculate_sle(asset_value: float, exposure_factor: float) -> float:
    """Calculate the single loss expectancy (SLE) given the asset value and exposure factor.

    Args:
        asset_value: Asset value
        exposure_factor: Exposure factor

    Returns:
        float: Single loss expectancy
    """
    return asset_value * exposure_factor


def calculate_ale(
    asset_value: float, exposure_factor: float, annual_rate_of_occurrence: float
) -> float:
    """Calculate the annualized loss expectancy (ALE) given the asset value, exposure factor, and annual rate of occurrence.

    Args:
        asset_value: Asset value
        exposure_factor: Exposure factor
        annual_rate_of_occurrence: Annual rate of occurrence

    Returns:
        float: Annualized loss expectancy
    """
    return asset_value * exposure_factor * annual_rate_of_occurrence


def calculate_rosi(
    ale_before: float, ale_after: float, control_costs: list[float] | float
) -> float:
    """Calculate the return on security investment (ROSI) given the ALE before and after implementing controls and the control costs.

    Args:
        ale_before: Annualized loss expectancy before implementing controls
        ale_after: Annualized loss expectancy after implementing controls
        control_costs: Annualized cost of controls (either a single float or a list of floats)

    Returns:
        float: Return on security investment
    """
    if isinstance(control_costs, float):
        total_control_costs = control_costs
    else:
        total_control_costs = sum(control_costs)

    return (ale_before - ale_after - total_control_costs) / total_control_costs * 100


def calculate_mode(values: np.ndarray) -> tuple[float | None, float | None]:
    """Calculate mode using kernel density estimation and its percentage in the distribution.

    Args:
        values: Array of values to analyze

    Returns:
        tuple[float | None, float | None]: A tuple containing:
            - The mode value (or None if undefined)
            - The percentage of values within 1% of mode (or None if undefined)
    """
    if len(values) < 2:
        return (values[0], 100.0) if len(values) == 1 else (None, None)

    kde = gaussian_kde(values)
    grid = np.linspace(min(values), max(values), 100)
    mode_idx = np.argmax(kde(grid))
    mode = grid[mode_idx]

    # Calculate percentage of values within 1% of mode
    mode_range = 0.01 * mode
    count = np.sum((values >= mode - mode_range) & (values <= mode + mode_range))
    percentage = (count / len(values)) * 100

    return mode, percentage


def find_first_non_zero_percentile(
    data: np.ndarray, decimal_places: int = 2
) -> tuple[float, float]:
    """Helper function to find the first non-zero value and its percentile in a data array.
    Considers a value non-zero if it rounds to non-zero at specified decimal places.

    Args:
        data: Numpy array of numeric values to analyze.
        decimal_places: Number of decimal places to consider for zero comparison (default: 2).

    Returns:
        tuple[float, float]: A tuple containing:
            - The percentile (0-100) at which first non-zero value occurs
            - The first non-zero value found

    Raises:
        TypeError: If input is not a numpy array.
        ValueError: If array is empty or decimal_places is negative.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if len(data) == 0:
        raise ValueError("Input array cannot be empty")
    if decimal_places < 0:
        raise ValueError("Decimal places must be non-negative")

    # Round data to specified decimal places
    rounded_data = np.round(data, decimals=decimal_places)
    sorted_data = np.sort(data)  # Original values for return
    sorted_rounded = np.sort(rounded_data)  # Rounded for comparison

    # Find first non-zero in rounded data
    non_zero_idx = np.argmax(sorted_rounded > 0)
    if non_zero_idx == 0 and sorted_rounded[0] == 0:
        return 0.0, 0.0

    percentile = (non_zero_idx / len(data)) * 100
    return float(percentile), float(sorted_data[non_zero_idx])


def setup_sensitivity_problem(
    **kwargs: dict[str, list[float]]
) -> dict[str, int | list[str] | list[float]]:
    """Define the model inputs and their bounds for sensitivity analysis.

    Args:
        **kwargs: Dictionary of parameter names and their bounds in a list with two floats

    Returns:
        dict[str, int | list[str] | list[float]]: Problem dictionary for sensitivity analysis

    Raises:
        ValueError: If the parameter values are not lists of two floats
    """
    # Verify kwargs is a dictionary with the correct type
    if not isinstance(kwargs, dict):
        raise ValueError("Input parameters must be provided as a dictionary")

    # Verify input parameters contain only two floats
    for key, value in kwargs.items():
        if not isinstance(key, str):
            raise ValueError(f"Parameter name {key} must be a string")
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not all(isinstance(i, float) for i in value)
        ):
            raise ValueError(f"Parameter {key} must be a list of two floats")

    names = list(kwargs.keys())
    bounds = list(kwargs.values())

    problem = {
        "num_vars": len(names),
        "names": names,
        "bounds": bounds,
    }
    return problem


def simulate_exposure_factor_sobol(
    sobol_samples: np.ndarray, ef_range: list[float], kurtosis: int
) -> np.ndarray:
    """Simulate exposure factor using Sobol samples mapped to a Beta distribution with a specified kurtosis.

    Args:
        sobol_samples: Sobol samples
        ef_range: Exposure factor range
        kurtosis: Kurtosis of the distribution

    Returns:
        np.ndarray: Simulated exposure factors
    """
    # To avoid beta.ppf(0 or 1) = inf/NaN, clamp the samples:
    eps = 1e-9
    sobol_samples = np.clip(sobol_samples, eps, 1 - eps)

    # Calculate Beta parameters based on kurtosis
    alpha_param, beta_param = get_beta_parameters_for_kurtosis(kurtosis)

    # Map Sobol samples to Beta distribution using inverse CDF quantile function
    beta_samples = beta.ppf(sobol_samples, alpha_param, beta_param)

    # Scale Beta samples to the desired range
    exposure_factors = ef_range[0] + beta_samples * (ef_range[1] - ef_range[0])

    return exposure_factors


def simulate_annual_rate_of_occurrence_sobol(
    sobol_samples: np.ndarray, aro_range: list[float], decimal_places: int = 2
) -> np.ndarray:
    """Simulate annual rate of occurrence using Sobol samples mapped to a Poisson distribution with specified decimal precision. As Poisson distribution accepts only integer values, the Sobol samples are scaled up to the desired precision, mapped to the Poisson distribution, and then scaled back to the original scale, bypassing the decimal limitation.

    Args:
        sobol_samples: Sobol samples
        aro_range: Annual rate of occurrence range
        decimal_places: Number of decimal places for the generated values (default is 2)

    Returns:
        np.ndarray: Simulated annual rates of occurrence
    """
    # To avoid poisson.ppf(0 or 1) = inf/NaN, clamp the samples:
    eps = 1e-9
    sobol_samples = np.clip(sobol_samples, eps, 1 - eps)

    # Scale factor for desired decimal precision
    scale_factor = 10**decimal_places

    # Scale Sobol samples to integers
    scaled_sobol_samples = sobol_samples * scale_factor

    # Map Sobol samples to Poisson distribution using inverse CDF quantile function
    poisson_samples = poisson.ppf(
        scaled_sobol_samples / scale_factor, mu=np.mean(aro_range) * scale_factor
    )

    # Convert back to the original scale
    aro_samples = poisson_samples / scale_factor

    # Clip the values to the desired range
    aro_samples = np.clip(aro_samples, aro_range[0], aro_range[1])

    return aro_samples


def simulate_control_effectiveness_sobol(
    sobol_samples: np.ndarray, ce_range: list[float]
) -> np.ndarray:
    """Simulate control effectiveness using Sobol samples mapped to a Uniform distribution.

    Args:
        sobol_samples: Sobol samples.
        ce_range: Control effectiveness range.

    Returns:
        np.ndarray: Simulated control effectiveness values
    """
    # Scale Sobol samples to the desired range
    control_effectiveness = ce_range[0] + sobol_samples * (ce_range[1] - ce_range[0])

    return control_effectiveness


def randomize_sobol_samples(sobol_samples: np.ndarray) -> np.ndarray:
    """Randomize Sobol samples using a uniform random shift.

    Args:
        sobol_samples: Original Sobol samples

    Returns:
        np.ndarray: Randomized Sobol samples
    """
    random_shift = np.random.uniform(size=sobol_samples.shape)
    randomized_samples = (sobol_samples + random_shift) % 1
    return randomized_samples


def convert_to_serializable(obj: any) -> any:
    """Recursively converts numpy arrays, dictionaries, and lists into serializable formats.

    Args:
        obj: The object to be converted. It can be a numpy array, dictionary, list, or any other type.

    Returns:
        any: The converted object in a serializable format. Numpy arrays are converted to lists, dictionaries are recursively processed, and lists are recursively processed. Other types are returned as is.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj
