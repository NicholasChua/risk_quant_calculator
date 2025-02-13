import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, poisson, gaussian_kde
import csv
import json

from verification import verify_mcmc_implementation

MONTE_CARLO_SEED = 42
NUM_SIMULATIONS = 10000
KURTOSIS = 1.7  # Default value is 3
CURRENCY_SYMBOL = "\\$"


def load_csv_data(file_path: str) -> dict[str, list[float]]:
    """Helper function to load data from a CSV file containing input parameters for the risk calculation.

    The CSV file should have the following columns in order:
        - id
        - asset_value
        - exposure_factor
        - annual_rate_of_occurrence
        - percentage_reduction
        - cost_of_control

    Args:
        file_path: The path to the CSV file.

    Returns:
        dict[str, list[dict]]: A dictionary with 'data' key containing list of risk dictionaries.
    """
    result = {"data": []}

    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            risk = {
                "id": int(row["id"]),
                "asset_value": float(row["asset_value"]),
                "exposure_factor": float(row["exposure_factor"]),
                "annual_rate_of_occurrence": float(row["annual_rate_of_occurrence"]),
                "percentage_reduction": float(row["percentage_reduction"]),
                "cost_of_control": float(row["cost_of_control"]),
            }
            result["data"].append(risk)

    return result


def load_json_data(file_path: str) -> dict[str, list[float]]:
    """Helper function to load data from a JSON file containing input parameters for the risk calculation.

    Args:
        file_path: The path to the JSON file.

    Returns:
        dict: A dictionary containing the input parameters for the risk calculation.
    """
    float_fields = [
        "asset_value",
        "exposure_factor",
        "annual_rate_of_occurrence",
        "percentage_reduction",
        "cost_of_control",
    ]

    with open(file_path, "r") as file:
        data = json.load(file)

    for risk in data["data"]:
        for field in float_fields:
            risk[field] = float(risk[field])

    return data


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


def calculate_sle_ale(
    asset_value: float, exposure_factor: float, annual_rate_of_occurrence: float
) -> tuple[float, float]:
    """Helper function to calculate Single Loss Expectancy (SLE) and Annualized Loss Expectancy (ALE) for a given asset.

    Single Loss Expectancy (SLE) is the expected monetary loss every time a risk event occurs.

    Annualized Loss Expectancy (ALE) is the expected monetary loss that can be expected for a year.

    Args:
        asset_value: The value of the asset at risk
        exposure_factor: The percentage of asset value at risk
        annual_rate_of_occurrence: The frequency of the risk event

    Returns:
        tuple[float, float]: A tuple containing:
            - The Single Loss Expectancy (SLE)
            - The Annualized Loss Expectancy (ALE)
    """
    SLE = asset_value * exposure_factor
    ALE = SLE * annual_rate_of_occurrence
    return SLE, ALE


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


def _simulate_clamped_ef(input_ef: float, variability: int = 0, kurtosis: float = KURTOSIS, num_simulations: int = NUM_SIMULATIONS) -> np.ndarray:
    """Helper function to simulate clamped exposure factors using a Beta distribution.

    The clamped exposure factor is generated by sampling from a Beta distribution with parameters adjusted for the desired kurtosis. The sampled values are then clamped to be within ± variability of the input EF. This function simulates uncertainty in the provided EF (percentage of asset value at risk). If the variability is set to 0, the results will have no variability and will be fixed.

    Args:
        input_ef: The input exposure factor to clamp around.
        variability: The percentage variability around the input EF (default is 0).
        kurtosis: The desired kurtosis value for the Beta distribution (default is the constant KURTOSIS).
        num_simulations: The number of simulations to generate (default is the constant NUM_SIMULATIONS).

    Returns:
        np.ndarray: Array of simulated clamped exposure factors.
    """
    lower_bound = max(0.0, input_ef - input_ef * variability / 100)
    upper_bound = input_ef + input_ef * variability / 100
    
    # Sample from Beta, then rescale to [lower_bound, upper_bound]
    alpha, beta_param = get_beta_parameters_for_kurtosis(kurtosis)
    raw = beta(a=alpha, b=beta_param).rvs(num_simulations)
    return lower_bound + raw * (upper_bound - lower_bound)


def _simulate_clamped_aro(input_aro: float, variability: int = 0, num_simulations: int = NUM_SIMULATIONS) -> np.ndarray:
    """Helper function to simulate clamped Annual Rate of Occurrence (ARO) values.

    The clamped ARO is generated by sampling from a Poisson distribution with a mean adjusted for the desired variability around the input ARO. The sampled values are then clamped to be within ± variability of the input ARO. This function simulates uncertainty in the provided ARO (frequency of the risk event). If the variability is set to 0, the results will have no variability and will be fixed.

    Args:
        input_aro: The input annual rate of occurrence to clamp around.
        variability: The percentage variability around the input ARO (default is 0).
        num_simulations: The number of simulations to generate (default is the constant NUM_SIMULATIONS).

    Returns:
        np.ndarray: Array of simulated clamped ARO values.
    """
    lower_bound = max(0, input_aro - input_aro * variability / 100)
    upper_bound = input_aro + input_aro * variability / 100
    return np.clip(poisson(mu=input_aro).rvs(num_simulations), lower_bound, upper_bound)


def _validate_simulation_params(**kwargs) -> None:
    """Helper function to validate simulation parameters. Raises a ValueError if any parameter is invalid.

    Checks the following parameters:
        - asset_value: Must be positive
        - exposure_factor: Must be between 0 and 1
        - annual_rate_of_occurrence: Must be positive
        - reduction_percentage: Must be 99 or less
        - cost_of_controls: Must be positive
        - output_json_file: Must be .json or None
        - output_png_file: Must be .png or None
        - monte_carlo_seed: Must be positive
        - num_simulations: Must be positive
        - kurtosis: Must be positive
        - simulation_method: Must be 0 or 1

    Args:
        **kwargs: Arbitrary keyword arguments to be validated.

    Raises:
        ValueError: If any parameter is invalid.
    """
    param_rules = {
        "asset_value": (lambda x: x >= 0, "must be a positive value"),
        "exposure_factor": (lambda x: 0 <= x <= 1, "must be between 0 and 1"),
        "annual_rate_of_occurrence": (lambda x: x >= 0, "must be a positive value"),
        "reduction_percentage": (
            lambda x: x <= 99,
            "must be 99 or less",
        ),  # Allow negative values to model increase in risk, disallow 100 as it is risk avoidance and not risk reduction
        "cost_of_controls": (lambda x: x >= 0, "must be a positive value"),
        "output_json_file": (
            lambda x: x is None or x.endswith(".json"),
            "must be .json or None",
        ),
        "output_png_file": (
            lambda x: x is None or x.endswith(".png"),
            "must be .png or None",
        ),
        "monte_carlo_seed": (lambda x: x > 0, "must be a positive value"),
        "num_simulations": (lambda x: x > 0, "must be a positive value"),
        "kurtosis": (lambda x: x >= 0, "must be a positive value"),
        "simulation_method": (lambda x: x in [0, 1], "must be 0 or 1"),
    }

    for param, (rule, message) in param_rules.items():
        if param in kwargs and not rule(kwargs[param]):
            raise ValueError(f"{param} {message}")


def _simulate_losses_monte_carlo(
    asset_value: float,
    exposure_factor: float,
    annual_rate_of_occurrence: float,
    num_simulations: int,
    kurtosis: int,
) -> np.ndarray:
    """Helper function to simulate losses using Monte Carlo simulation.

    Args:
        asset_value: The value of the asset at risk
        exposure_factor: The percentage of asset value at risk
        annual_rate_of_occurrence: The frequency of the risk event
        num_simulations: Number of simulations to run
        kurtosis: Kurtosis value for beta distribution

    Returns:
        np.ndarray: Array of simulated losses
    """
    # Sample from beta distribution to capture kurtosis shape, then clamp around input EF
    alpha, beta_param = get_beta_parameters_for_kurtosis(kurtosis)
    # Sample a base EF from Beta, then clamp around the input EF
    base_EF_samples = beta(a=alpha, b=beta_param).rvs(num_simulations)
    simulated_EF = _simulate_clamped_ef(
        input_ef=exposure_factor,
        variability=0, # Remove variability in simple Monte Carlo calculations
        kurtosis=kurtosis,
        num_simulations=num_simulations
    ) * base_EF_samples / base_EF_samples.mean()

    # Sample a base ARO from Poisson clamped around the input ARO
    simulated_ARO = _simulate_clamped_aro(
        input_aro=annual_rate_of_occurrence,
        variability=0, # Remove variability in simple Monte Carlo calculations
        num_simulations=num_simulations
    )

    # Calculate losses
    losses = asset_value * simulated_EF * simulated_ARO

    return losses


def metropolis_hastings(
    num_samples: int, initial_control: float, alpha: float, beta_param: float
) -> np.ndarray:
    """Metropolis-Hastings algorithm to generate control effectiveness samples between 0 and 1.

    Args:
        num_samples: Number of MCMC samples to generate.
        initial_control: Starting control effectiveness value (decimal between 0 and 1).
        alpha: Beta distribution alpha parameter.
        beta_param: Beta distribution beta parameter.

    Returns:
        np.ndarray: Samples of control effectiveness values (decimals between 0 and 1).

    Raises:
        ValueError: If alpha or beta_param are non-positive.
    """
    # Validate parameters
    if alpha <= 0 or beta_param <= 0:
        raise ValueError("Alpha and beta parameters must be positive")

    samples = []
    current_state = initial_control
    step_size = 0.05  # Adjusted step size for better convergence

    for _ in range(num_samples):
        # Propose new control effectiveness with scaled noise
        proposal = current_state + step_size * np.random.normal(0, 1.0)
        proposal = np.clip(
            proposal, 0.0001, 0.9999
        )  # Avoid exactly 0 or 1 for Beta distribution

        # Calculate log probabilities
        try:
            current_log_prob = beta.logpdf(current_state, alpha, beta_param)
            proposal_log_prob = beta.logpdf(proposal, alpha, beta_param)

            # Calculate acceptance ratio in log space
            log_acceptance_ratio = proposal_log_prob - current_log_prob

            # Accept/reject the proposal
            if np.log(np.random.rand()) < log_acceptance_ratio:
                current_state = proposal
        except:
            continue  # Skip invalid proposals

        samples.append(current_state)

    return np.array(samples)


def _simulate_losses_with_mcmc(
    asset_value: float,
    exposure_factor: float,
    annual_rate_of_occurrence: float,
    num_simulations: int,
    kurtosis: int,
    reduction_percentage: float,
) -> np.ndarray:
    """Helper function to simulate losses using Markov Chain Monte Carlo to model control effectiveness. This method uses Metroplis-Hastings algorithm to generate control effectiveness samples and apply them to the exposure factor.

    Args:
        asset_value: The value of the asset at risk.
        exposure_factor: The percentage of asset value at risk.
        annual_rate_of_occurrence: The frequency of the risk event.
        num_simulations: Number of simulations to run.
        kurtosis: Kurtosis value for beta distribution.
        reduction_percentage: Initial control effectiveness percentage.

    Returns:
        np.ndarray: Array of simulated losses.
    """
    # Convert reduction percentage to decimal
    initial_control = reduction_percentage / 100.0

    # Get beta parameters
    alpha, beta_param = get_beta_parameters_for_kurtosis(kurtosis)

    # Generate base shapes from Beta
    base_EF_samples = beta(a=alpha, b=beta_param).rvs(num_simulations)
    base_ARO_samples = beta(a=alpha, b=beta_param).rvs(num_simulations)

    # Get clamped values
    clamped_ef = _simulate_clamped_ef(
        input_ef=exposure_factor,
        variability=0,
        kurtosis=kurtosis,
        num_simulations=num_simulations
    )
    clamped_aro = _simulate_clamped_aro(
        input_aro=annual_rate_of_occurrence,
        variability=0,
        num_simulations=num_simulations
    )

    # Scale while preserving shape
    simulated_ef = clamped_ef * (base_EF_samples / base_EF_samples.mean())
    simulated_aro = clamped_aro * (base_ARO_samples / base_ARO_samples.mean())

    # Run Metropolis-Hastings to generate control effectiveness samples
    control_samples = metropolis_hastings(
        num_samples=num_simulations,
        initial_control=initial_control,
        alpha=alpha,
        beta_param=beta_param,
    )
    control_effectiveness = control_samples.flatten()

    # verify_mcmc_implementation(control_effectiveness, num_simulations) # Debug code

    # Apply controls and calculate losses
    adjusted_ef = simulated_ef * (1 - control_effectiveness)
    adjusted_aro = simulated_aro * (1 - control_effectiveness)
    losses = asset_value * adjusted_ef * adjusted_aro

    return losses


def _calculate_mode_percentage(losses: np.ndarray) -> tuple[float, float]:
    """Helper function to calculate the mode and its percentage occurrence.

    Args:
        losses: Array of simulated losses.

    Returns:
        tuple[float, float]: The mode and its percentage occurrence.
    """
    hist, bins = np.histogram(losses, bins=100)
    mode_index = np.argmax(hist)
    mode = (bins[mode_index] + bins[mode_index + 1]) / 2
    mode_percentage = (hist[mode_index] / len(losses)) * 100
    return mode, mode_percentage


def _calculate_statistics(losses: np.ndarray) -> dict:
    """Helper function to calculate comprehensive statistics for loss distribution.

    Args:
        losses: Array of simulated losses.

    Returns:
        dict: Statistics including mean, median, mode, std dev, and percentiles.
    """
    mode, mode_percentage = _calculate_mode_percentage(losses)
    stats_dict = {
        "Mean": np.mean(losses),
        "Median": np.median(losses),
        "Mode": mode,
        "Mode Percentage": mode_percentage,
        "Std Dev": np.std(losses),
        "1st Percentile": np.percentile(losses, 1),
        "2.5th Percentile": np.percentile(losses, 2.5),
        "5th Percentile": np.percentile(losses, 5),
        "10th Percentile": np.percentile(losses, 10),
        "25th Percentile": np.percentile(losses, 25),
        "75th Percentile": np.percentile(losses, 75),
        "90th Percentile": np.percentile(losses, 90),
        "95th Percentile": np.percentile(losses, 95),
        "97.5th Percentile": np.percentile(losses, 97.5),
        "99th Percentile": np.percentile(losses, 99),
    }

    return stats_dict


def _plot_risk_distribution(
    ax: plt.Axes,
    losses: np.ndarray,
    adjusted_losses: np.ndarray | None = None,
    num_simulations: int = NUM_SIMULATIONS,
    currency_symbol: str = CURRENCY_SYMBOL,
) -> None:
    """Helper function to plot risk distribution histogram with KDE curves.

    Args:
        ax: Matplotlib axes to plot on.
        losses: Array of losses before controls.
        adjusted_losses: Optional array of losses after controls.
        num_simulations: Number of simulations run.
        currency_symbol: Currency symbol for formatting.
    """
    bins = np.linspace(
        0,
        (
            losses.max()
            if adjusted_losses is None
            else max(losses.max(), adjusted_losses.max())
        ),
        50,
    )

    # Plot base case
    ax.hist(
        losses,
        bins=bins,
        alpha=0.5,
        color="blue",
        label="Base Case" if adjusted_losses is None else "Before Controls",
    )
    kde_losses = gaussian_kde(losses)
    x_values = np.linspace(0, bins[-1], 1000)
    ax.plot(
        x_values,
        kde_losses(x_values) * num_simulations * np.diff(bins)[0],
        color="blue",
    )

    # Plot adjusted case if provided
    if adjusted_losses is not None:
        ax.hist(
            adjusted_losses, bins=bins, alpha=0.5, color="green", label="After Controls"
        )
        kde_adjusted = gaussian_kde(adjusted_losses)
        ax.plot(
            x_values,
            kde_adjusted(x_values) * num_simulations * np.diff(bins)[0],
            color="green",
        )

    # Format axes
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'{currency_symbol}{format(int(x), ",")}')
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ",")))

    ax.set_title(f"Risk Distribution over {num_simulations} Simulations")
    ax.set_xlabel("Total Expected Loss")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True)


def _calculate_exceedance_probabilities(losses: np.ndarray) -> np.ndarray:
    """Helper function to calculate exceedance probabilities for a given array of losses.

    Args:
        losses: Numpy array of losses.

    Returns:
        np.ndarray: Exceedance probabilities.
    """
    exceedance_prob = 100 * (1.0 - np.arange(1, len(losses) + 1) / len(losses))
    return exceedance_prob


def _plot_exceedance_curve(
    ax: plt.Axes,
    losses: np.ndarray,
    adjusted_losses: np.ndarray | None = None,
    currency_symbol: str = CURRENCY_SYMBOL,
) -> None:
    """Helper function to plot loss exceedance curve.

    Args:
        ax: Matplotlib axes to plot on.
        losses: Array of losses before controls.
        adjusted_losses: Optional array of losses after controls.
        currency_symbol: Currency symbol for formatting.
    """
    # Calculate exceedance probabilities for base case
    exceedance_prob = _calculate_exceedance_probabilities(losses)
    ax.plot(
        np.sort(losses),
        exceedance_prob,
        color="blue",
        label="Base Case" if adjusted_losses is None else "Before Controls",
    )

    # Calculate exceedance probabilities for adjusted case if provided
    if adjusted_losses is not None:
        nonzero_adjusted = adjusted_losses[adjusted_losses > 0]
        adjusted_prob = _calculate_exceedance_probabilities(nonzero_adjusted)
        ax.plot(
            np.sort(nonzero_adjusted),
            adjusted_prob,
            color="green",
            label="After Controls",
        )

    # Format axes
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'{currency_symbol}{format(int(x), ",")}')
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0f}%".format(y)))

    ax.set_title("Loss Exceedance Curve")
    ax.set_xlabel("Total Loss")
    ax.set_ylabel("Probability of Exceedance (%)")
    ax.legend()
    ax.grid(True)


def plot_risk_calculation_before_after(
    asset_value: float,
    exposure_factor: float,
    annual_rate_of_occurrence: float,
    reduction_percentage: float,
    cost_of_controls: float,
    output_json_file: str | None = None,
    output_png_file: str | None = None,
    monte_carlo_seed: int = MONTE_CARLO_SEED,
    num_simulations: int = NUM_SIMULATIONS,
    kurtosis: int = KURTOSIS,
    currency_symbol: str = CURRENCY_SYMBOL,
    simulation_method: int = 0,
) -> None:
    """Estimate the mean loss and the effectiveness of risk controls using Monte Carlo or Markov Chain Monte Carlo (MCMC) simulation.

    This function simulates the potential losses to an asset based on a given exposure factor (EF) and annual rate of occurrence (ARO) for a specific asset value (AV). It then simulates the impact of risk reduction controls that lower the ARO and calculates how these controls reduce potential losses. It provides a risk distribution and a loss exceedance curve both before and after controls are applied. The statistics and simulation results are either plotted or returned as a dictionary.

    The simulation uses:
        - **Beta Distribution**: To simulate the Exposure Factor (EF), representing the percentage of asset value that is at risk during a risk event. The Beta distribution allows for variability in the EF, with the distribution's shape controlled by parameters that introduce kurtosis (peakedness). The implementation allows for variability around the input EF to model uncertainty and possible outcomes around the input EF.
        - **Poisson Distribution**: To simulate the Annual Rate of Occurrence (ARO), representing the frequency of the risk event over a year. It models the randomness of how often the event will occur. After applying controls, the ARO is reduced by a specified percentage and recalculated using the Poisson distribution. The implementation allows for variability around the input ARO to model uncertainty and possible outcomes around the input ARO.

    The plot shows two main visualizations and four statistical tables:
        - **Risk Distribution**: A histogram showing the distribution of potential losses before and after controls. It also includes Kernel Density Estimation (KDE) curves to visualize the probability density of the losses.
        - **Loss Exceedance Curve**: A line plot showing the cumulative probability of exceeding a given loss amount before and after controls. It helps visualize the likelihood of different loss scenarios.
        - **Statistical Summary Table**: Displays key statistics such as Mean, Median, Mode, Mode Percentage, Standard Deviation, Confidence Interval, Interquartile Range, and 99th Percentile of losses before and after controls.
        - **Input Parameters Table**: Shows the user-provided input parameters such as Asset Value (AV), Exposure Factor (EF), Annual Rate of Occurrence (ARO), Single Loss Expectancy (SLE), and Annualized Loss Expectancy (ALE) before controls.
        - **Calculated Parameters Table**: Displays the Adjusted ARO, SLE, ALE, Expected Benefit, Cost of Controls, and Return on Security Investment (ROSI) after applying controls.

    The JSON output includes:
        - **Statistics**: All the statistical values in the visualizations, and percentile values used to calculate 90, 95, and 99 confidence intervals. The first non-zero percentile and value are also included.
        - **Input Parameters**: Asset Value (AV), Exposure Factor (EF), Annual Rate of Occurrence (ARO), Single Loss Expectancy (SLE), and Annualized Loss Expectancy (ALE).
        - **Calculated Parameters**: Adjusted ARO after controls, Adjusted ALE, and the expected reduction in loss due to controls.
        - **Simulation Results**: Lists of simulated losses and exceedance probabilities before and after controls.

    Args:
        asset_value: The value of the asset at risk, expressed in monetary units.
        exposure_factor: The percentage of the asset value that is at risk during a risk event, expressed as a decimal.
        annual_rate_of_occurrence: The frequency of the risk event over a year, expressed as a decimal.
        reduction_percentage: The percentage reduction in the ARO after applying risk controls, expressed as a percentage.
        cost_of_controls: The cost of implementing the risk controls, expressed in monetary units.
        output_json_file: The path to save the output file as a JSON file. If None, the output is not saved (default is None).
        output_png_file: The path to save the output plot as a PNG file. If None, the plot is not saved (default is None).
        monte_carlo_seed: The seed for the Monte Carlo simulation to ensure reproducibility (default is constant MONTE_CARLO_SEED).
        num_simulations: The number of simulations to run for the Monte Carlo analysis (default is constant NUM_SIMULATIONS).
        kurtosis: The kurtosis value to adjust the shape of the beta distribution for the EF (default is constant KURTOSIS).
        currency_symbol: The currency symbol to use in the plot displays. (default is constant CURRENCY_SYMBOL).
        simulation_method: The method to use for simulation. 0 for Monte Carlo, 1 for MCMC (default is 0).

    Returns:
        dict | None: If output_file is None, returns a dictionary containing the statistics and simulation results. If output_file is provided, returns None.
    """
    # Input validation
    _validate_simulation_params(
        asset_value=asset_value,
        exposure_factor=exposure_factor,
        annual_rate_of_occurrence=annual_rate_of_occurrence,
        reduction_percentage=reduction_percentage,
        cost_of_controls=cost_of_controls,
        output_file=output_json_file,
        monte_carlo_seed=monte_carlo_seed,
        num_simulations=num_simulations,
        kurtosis=kurtosis,
        # Don't need to validate currency_symbol
        simulation_method=simulation_method,
    )

    # Set seed for reproducibility
    np.random.seed(monte_carlo_seed)

    # Calculate SLE and ALE
    single_loss_expectancy, annualized_loss_expectancy = calculate_sle_ale(
        asset_value, exposure_factor, annual_rate_of_occurrence
    )

    if simulation_method == 0:
        # Simulate losses in the before scenario with Monte Carlo
        losses = _simulate_losses_monte_carlo(
            asset_value=asset_value,
            exposure_factor=exposure_factor,
            annual_rate_of_occurrence=annual_rate_of_occurrence,
            num_simulations=num_simulations,
            kurtosis=kurtosis,
        )
    elif simulation_method == 1:
        # Simulate losses in the before scenario with MCMC
        losses = _simulate_losses_with_mcmc(
            asset_value=asset_value,
            exposure_factor=exposure_factor,
            annual_rate_of_occurrence=annual_rate_of_occurrence,
            num_simulations=num_simulations,
            kurtosis=kurtosis,
            reduction_percentage=0,
        )

    # Adjusted ARO after controls
    adjusted_ARO = annual_rate_of_occurrence * (1 - reduction_percentage / 100)

    if simulation_method == 0:
        # Simulate losses in the after scenario with Monte Carlo
        adjusted_losses = _simulate_losses_monte_carlo(
            asset_value=asset_value,
            exposure_factor=exposure_factor,
            annual_rate_of_occurrence=adjusted_ARO,
            num_simulations=num_simulations,
            kurtosis=kurtosis,
        )
    elif simulation_method == 1:
        # Simulate losses in the after scenario with MCMC
        adjusted_losses = _simulate_losses_with_mcmc(
            asset_value=asset_value,
            exposure_factor=exposure_factor,
            annual_rate_of_occurrence=adjusted_ARO,
            num_simulations=num_simulations,
            kurtosis=kurtosis,
            reduction_percentage=reduction_percentage,
        )

    # Calculate statistics before and after controls
    calc_stats = _calculate_statistics(losses)
    calc_adjusted_stats = _calculate_statistics(adjusted_losses)

    # Calculate exceedance probabilities
    exceedance_prob = _calculate_exceedance_probabilities(losses)
    adjusted_exceedance_prob = _calculate_exceedance_probabilities(adjusted_losses)

    # Calculate expected reduction in loss
    expected_loss_before = np.mean(losses)
    expected_loss_after = np.mean(adjusted_losses)
    expected_reduction_in_loss = expected_loss_before - expected_loss_after

    # Calculate ROSI
    benefit = expected_reduction_in_loss
    rosi_percentage = (benefit - cost_of_controls) / cost_of_controls * 100

    # For after controls, 2.5% percentile is likely to be zero due to the reduction
    # Calculate the first non-zero percentile and value to show in the JSON output
    first_nonzero_pct, first_nonzero_val = find_first_non_zero_percentile(
        adjusted_losses
    )
    calc_adjusted_stats["First Non-Zero Percentile"] = first_nonzero_pct
    calc_adjusted_stats["First Non-Zero Value"] = first_nonzero_val

    nonzero_adjusted_losses = adjusted_losses[adjusted_losses > 0]

    # Output results as a dictionary
    return_data = {
        "before_controls_stats": {
            "mean": float(calc_stats["Mean"]),
            "median": float(calc_stats["Median"]),
            "mode": float(calc_stats["Mode"]),
            "mode_percentage": float(calc_stats["Mode Percentage"]),
            "std_dev": float(calc_stats["Std Dev"]),
            "percentile_1": float(calc_stats["1st Percentile"]),
            "percentile_2.5": float(calc_stats["2.5th Percentile"]),
            "percentile_5": float(calc_stats["5th Percentile"]),
            "percentile_10": float(calc_stats["10th Percentile"]),
            "percentile_25": float(calc_stats["25th Percentile"]),
            "percentile_75": float(calc_stats["75th Percentile"]),
            "percentile_90": float(calc_stats["90th Percentile"]),
            "percentile_95": float(calc_stats["95th Percentile"]),
            "percentile_97.5": float(calc_stats["97.5th Percentile"]),
            "percentile_99": float(calc_stats["99th Percentile"]),
        },
        "after_controls_stats": {
            "mean": float(calc_adjusted_stats["Mean"]),
            "median": float(calc_adjusted_stats["Median"]),
            "mode": float(calc_adjusted_stats["Mode"]),
            "mode_percentage": float(calc_adjusted_stats["Mode Percentage"]),
            "std_dev": float(calc_adjusted_stats["Std Dev"]),
            "percentile_1": float(calc_adjusted_stats["1st Percentile"]),
            "percentile_2.5": float(calc_adjusted_stats["2.5th Percentile"]),
            "percentile_5": float(calc_adjusted_stats["5th Percentile"]),
            "percentile_10": float(calc_adjusted_stats["10th Percentile"]),
            "percentile_25": float(calc_adjusted_stats["25th Percentile"]),
            "percentile_75": float(calc_adjusted_stats["75th Percentile"]),
            "percentile_90": float(calc_adjusted_stats["90th Percentile"]),
            "percentile_95": float(calc_adjusted_stats["95th Percentile"]),
            "percentile_97.5": float(calc_adjusted_stats["97.5th Percentile"]),
            "percentile_99": float(calc_adjusted_stats["99th Percentile"]),
            "first_nonzero_percentile": float(
                calc_adjusted_stats["First Non-Zero Percentile"]
            ),
            "first_nonzero_value": float(calc_adjusted_stats["First Non-Zero Value"]),
        },
        "input_parameters": {
            "asset_value": float(asset_value),
            "exposure_factor": float(exposure_factor),
            "annual_rate_of_occurrence": float(annual_rate_of_occurrence),
            "single_loss_expectancy": float(single_loss_expectancy),
            "annualized_loss_expectancy": float(annualized_loss_expectancy),
        },
        "calculated_parameters": {
            "adjusted_aro": round(adjusted_ARO, 4),
            "adjusted_ale": round(
                annualized_loss_expectancy * (1 - reduction_percentage / 100), 2
            ),
            "reduction_percentage": round(reduction_percentage, 2),
            "expected_benefit": round(benefit, 2),
            "cost_of_controls": round(cost_of_controls, 2),
            "rosi_percentage": round(rosi_percentage, 2),
        },
        "simulation_results": {
            "losses": losses.tolist(),
            "adjusted_losses": nonzero_adjusted_losses.tolist(),  # Filter out zero values
            "exceedance_probabilities": exceedance_prob.tolist(),
            "adjusted_exceedance_probabilities": adjusted_exceedance_prob.tolist(),
        },
    }

    # Save JSON output if specified
    if output_json_file:
        with open(output_json_file, "w") as file:
            json.dump(return_data, file, indent=4)

    if output_png_file:
        # Set default figure size as 2:1 aspect ratio with 120 DPI
        plt.rcParams["figure.figsize"] = [16, 8]
        plt.rcParams["figure.dpi"] = 120

        # Create a figure with two subplots
        _, (ax1, ax2) = plt.subplots(2, 1)

        # Plot Risk Distribution
        _plot_risk_distribution(
            ax1,
            losses,
            adjusted_losses,
            num_simulations=num_simulations,
            currency_symbol=currency_symbol,
        )

        # Plot Loss Exceedance Curves
        _plot_exceedance_curve(
            ax2, losses, adjusted_losses, currency_symbol=currency_symbol
        )

        # Add statistics tables
        before_controls_text = "\n".join(
            [
                "Statistical Summary (Before Controls)",
                "-------------------------------------",
                f'Mean: {currency_symbol}{calc_stats["Mean"]:,.2f}',
                f'Median: {currency_symbol}{calc_stats["Median"]:,.2f}',
                f'Mode: {currency_symbol}{calc_stats["Mode"]:,.2f} ({calc_stats["Mode Percentage"]:.2f}%)',
                f'Std Dev: {currency_symbol}{calc_stats["Std Dev"]:,.2f}',
                f'Interquartile Range: {currency_symbol}{calc_stats["25th Percentile"]:,.2f} - {currency_symbol}{calc_stats["75th Percentile"]:,.2f}',
                f'90% CI: {currency_symbol}{calc_stats["5th Percentile"]:,.2f} - {currency_symbol}{calc_stats["95th Percentile"]:,.2f}',
                f'99th Percentile: {currency_symbol}{calc_stats["99th Percentile"]:,.2f}',
            ]
        )

        after_controls_text = "\n".join(
            [
                "Statistical Summary (After Controls)",
                "------------------------------------",
                f'Mean: {currency_symbol}{calc_adjusted_stats["Mean"]:,.2f}',
                f'Median: {currency_symbol}{calc_adjusted_stats["Median"]:,.2f}',
                f'Mode: {currency_symbol}{calc_adjusted_stats["Mode"]:,.2f} ({calc_adjusted_stats["Mode Percentage"]:.2f}%)',
                f'Std Dev: {currency_symbol}{calc_adjusted_stats["Std Dev"]:,.2f}',
                f'Interquartile Range: {currency_symbol}{calc_adjusted_stats["25th Percentile"]:,.2f} - {currency_symbol}{calc_adjusted_stats["75th Percentile"]:,.2f}',
                f'90% CI: {currency_symbol}{calc_adjusted_stats["5th Percentile"]:,.2f} - {currency_symbol}{calc_adjusted_stats["95th Percentile"]:,.2f}',
                f'99th Percentile: {currency_symbol}{calc_adjusted_stats["99th Percentile"]:,.2f}',
            ]
        )

        # Display user input parameters before controls
        input_parameters = "\n".join(
            [
                "Input Parameters (Before Controls)",
                "--------------------------------",
                f"Asset Value (AV): {currency_symbol}{asset_value:,.2f}",
                f"Exposure Factor (EF): {exposure_factor:.2%}",
                f"Annualized Rate of Occurrence (ARO): {annual_rate_of_occurrence:.2f}",
                f"Single Loss Expectancy (SLE): {currency_symbol}{single_loss_expectancy:,.2f}",
                f"Annualized Loss Expectancy (ALE): {currency_symbol}{annualized_loss_expectancy:,.2f}",
            ]
        )

        # Display parameters after controls
        calculated_parameters = "\n".join(
            [
                "Calculated Parameters (After Controls)",
                "-------------------------------",
                f"ARO after {reduction_percentage:.0f}% reduction: {adjusted_ARO:.2f}",
                f"New SLE: {currency_symbol}{single_loss_expectancy * (1 - reduction_percentage/100):,.2f}",
                f"New ALE: {currency_symbol}{annualized_loss_expectancy * (1 - reduction_percentage/100):,.2f}",
                f"Expected Benefit: {currency_symbol}{benefit:,.2f}",
                f"Cost of Controls: {currency_symbol}{cost_of_controls:,.2f}",
                f"ROSI: {rosi_percentage:,.2f}%",
            ]
        )

        # Update the figtext positioning
        plt.figtext(
            0.05,  # Left side
            0.02,
            before_controls_text,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        )

        plt.figtext(
            0.30,  # Center-left
            0.02,
            after_controls_text,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        )

        plt.figtext(
            0.55,  # Center-right
            0.02,
            input_parameters,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        )

        plt.figtext(
            0.80,  # Right side
            0.02,
            calculated_parameters,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        )

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25, hspace=0.25)  # Make room for the tables

        # Save or display the plot
        if output_png_file:
            plt.savefig(output_png_file, bbox_inches="tight")
        else:
            plt.show()

    return None


def main():
    # Load data from JSON or CSV files
    # data = load_json_data("input_example.json")
    data = load_csv_data("input_example.csv")

    # Plot the risk calculation for a before-and-after scenario using each item in the data
    for item in data["data"]:
        plot_risk_calculation_before_after(
            item["asset_value"],
            item["exposure_factor"],
            item["annual_rate_of_occurrence"],
            item["percentage_reduction"],
            item["cost_of_control"],
            output_json_file=f"output_{item['id']}.json",
            output_png_file=f"output_{item['id']}.png",
            simulation_method=0,
        )


if __name__ == "__main__":
    main()
