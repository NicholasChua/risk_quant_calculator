import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import beta, poisson, gaussian_kde
import json

from mcmc_verification import verify_mcmc_implementation

MONTE_CARLO_SEED = 42
NUM_SIMULATIONS = 10000
KURTOSIS = 1.7  # Default value is 3
CURRENCY_SYMBOL = "\\$"


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
        data: numpy array of numeric values to analyze
        decimal_places: number of decimal places to consider for zero comparison (default: 2)

    Returns:
        tuple[float, float]: A tuple containing:
            - The percentile (0-100) at which first non-zero value occurs
            - The first non-zero value found

    Raises:
        TypeError: If input is not a numpy array
        ValueError: If array is empty or decimal_places is negative
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


def _validate_simulation_params(**kwargs) -> None:
    """Helper function to validate simulation parameters. Raises a ValueError if any parameter is invalid.

    Checks the following parameters:
        - exposure_factor: Must be between 0 and 1
        - reduction_percentage: Must be 99 or less
        - annual_rate_of_occurrence: Must be positive
        - asset_value: Must be positive
        - kurtosis: Must be positive
        - num_simulations: Must be positive
        - monte_carlo_seed: Must be positive
        - cost_of_controls: Must be positive
        - plot: Must be boolean

    Args:
        **kwargs: Arbitrary keyword arguments to be validated.

    Raises:
        ValueError: If any parameter is invalid.
    """
    param_rules = {
        "exposure_factor": (lambda x: 0 <= x <= 1, "must be between 0 and 1"),
        "reduction_percentage": (
            lambda x: x <= 99,
            "must be 99 or less",
        ),  # Allow negative values to model increase in risk, disallow 100 as it is risk avoidance and not risk reduction
        "annual_rate_of_occurrence": (lambda x: x >= 0, "must be a positive value"),
        "asset_value": (lambda x: x >= 0, "must be a positive value"),
        "kurtosis": (lambda x: x >= 0, "must be a positive value"),
        "num_simulations": (lambda x: x > 0, "must be a positive value"),
        "monte_carlo_seed": (lambda x: x > 0, "must be a positive value"),
        "cost_of_controls": (lambda x: x >= 0, "must be a positive value"),
        "plot": (lambda x: isinstance(x, bool), "must be a boolean value"),
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
) -> tuple[np.ndarray, np.ndarray]:
    """Helper function to simulate losses using Monte Carlo simulation.

    Args:
        asset_value: The value of the asset at risk
        exposure_factor: The percentage of asset value at risk
        annual_rate_of_occurrence: The frequency of the risk event
        num_simulations: Number of simulations to run
        kurtosis: Kurtosis value for beta distribution

    Returns:
        tuple containing:
        - simulated losses array
        - simulated EF array
    """
    # Get beta distribution parameters
    alpha, beta_param = get_beta_parameters_for_kurtosis(kurtosis)

    # Monte Carlo simulation
    simulated_EF = beta(a=alpha, b=beta_param).rvs(num_simulations)
    simulated_ARO = poisson(mu=annual_rate_of_occurrence).rvs(num_simulations)

    # Calculate losses
    losses = asset_value * simulated_EF * simulated_ARO

    return losses, simulated_EF


def metropolis_hastings(
    num_samples: int, initial_control: float, alpha: float, beta_param: float
) -> np.ndarray:
    """Metropolis-Hastings algorithm to generate control effectiveness samples between 0 and 1.

    Args:
        num_samples: Number of MCMC samples to generate
        initial_control: Starting control effectiveness value (decimal between 0 and 1)
        alpha: Beta distribution alpha parameter
        beta_param: Beta distribution beta parameter

    Returns:
        np.ndarray: Samples of control effectiveness values (decimals between 0 and 1)
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
        asset_value: The value of the asset at risk
        exposure_factor: The percentage of asset value at risk
        annual_rate_of_occurrence: The frequency of the risk event
        num_simulations: Number of simulations to run
        kurtosis: Kurtosis value for beta distribution
        reduction_percentage: Initial control effectiveness percentage

    Returns:
        np.ndarray: Array of simulated losses
    """
    # Convert reduction percentage to decimal
    initial_control = reduction_percentage / 100.0

    # Get beta distribution parameters for control effectiveness distribution
    alpha, beta_param = get_beta_parameters_for_kurtosis(kurtosis)

    # Generate base ARO samples
    aro_samples = poisson(mu=annual_rate_of_occurrence).rvs(num_simulations)

    # Run Metropolis-Hastings to generate control effectiveness samples
    control_samples = metropolis_hastings(
        num_simulations,
        initial_control,  # Single control with initial effectiveness
        alpha,
        beta_param,
    )

    # Extract control effectiveness values
    control_effectiveness = control_samples.flatten()

    # verify_mcmc_implementation(control_effectiveness, num_simulations) # Debug code

    # Generate exposure factor samples
    simulated_ef = beta(a=alpha, b=beta_param).rvs(num_simulations)

    # Apply controls to both EF and ARO
    adjusted_ef = simulated_ef * (1 - control_effectiveness)
    adjusted_aro = aro_samples * (1 - control_effectiveness)

    # Calculate final losses
    losses = asset_value * adjusted_ef * adjusted_aro

    return losses


def _calculate_statistics(losses: np.ndarray) -> dict:
    """Helper function to calculate comprehensive statistics for loss distribution.

    Args:
        losses: Array of simulated losses

    Returns:
        dict: Statistics including mean, median, mode, std dev, and percentiles
    """
    stats_dict = {
        "Mean": np.mean(losses),
        "Median": np.median(losses),
        "Mode": float(stats.mode(np.round(losses))[0]),
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
        ax: Matplotlib axes to plot on
        losses: Array of losses before controls
        adjusted_losses: Optional array of losses after controls
        num_simulations: Number of simulations run
        currency_symbol: Currency symbol for formatting
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
        losses: Numpy array of losses

    Returns:
        np.ndarray: Exceedance probabilities
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
        ax: Matplotlib axes to plot on
        losses: Array of losses before controls
        adjusted_losses: Optional array of losses after controls
        currency_symbol: Currency symbol for formatting
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
    plot: bool = True,
    monte_carlo_seed: int = MONTE_CARLO_SEED,
    num_simulations: int = NUM_SIMULATIONS,
    kurtosis: int = KURTOSIS,
    currency_symbol: str = CURRENCY_SYMBOL,
) -> None | dict:
    """Estimate the mean loss and the effectiveness of risk controls using Markov Chain Monte Carlo simulations for a before-and-after scenario.

    This function simulates the potential losses to an asset based on a given exposure factor (EF) and annual rate of occurrence (ARO) for a specific asset value (AV). It then simulates the impact of risk reduction controls that lower the ARO and calculates how these controls reduce potential losses. It provides a risk distribution and a loss exceedance curve both before and after controls are applied. The statistics and simulation results are either plotted or returned as a dictionary.

    The simulation uses:
        - **Beta Distribution**: To simulate the Exposure Factor (EF), representing the percentage of asset value that is at risk during a risk event. The Beta distribution allows for variability in the EF, with the distribution's shape controlled by parameters that introduce kurtosis (peakedness).
        - **Poisson Distribution**: To simulate the Annual Rate of Occurrence (ARO), representing the frequency of the risk event over a year. It models the randomness of how often the event will occur. After applying controls, the ARO is reduced by a specified percentage and recalculated using the Poisson distribution.

    The plot shows two main visualizations and four statistical tables:
        - **Risk Distribution**: A histogram showing the distribution of potential losses before and after controls. It also includes Kernel Density Estimation (KDE) curves to visualize the probability density of the losses.
        - **Loss Exceedance Curve**: A line plot showing the cumulative probability of exceeding a given loss amount before and after controls. It helps visualize the likelihood of different loss scenarios.
        - **Statistical Summary Table**: Displays key statistics such as Mean, Median, Mode, Standard Deviation, Confidence Interval, and Percentiles (1%, 99%) of losses before and after controls. For the after control table, it dynamically uses *the first non-zero percentile and value* if the 2.5th percentile in the 95% CI is zero.
        - **Input Parameters Table**: Shows the user-provided input parameters such as Asset Value (AV), Exposure Factor (EF), Annual Rate of Occurrence (ARO), Single Loss Expectancy (SLE), and Annualized Loss Expectancy (ALE) before controls.
        - **Calculated Parameters Table**: Displays the Adjusted ARO after controls, Adjusted ALE, and the maximum acceptable cost of implementing controls based on the expected reduction in loss.

    The JSON output includes:
        - **Statistics**: Mean, Standard Deviation, 1st Percentile, 95% Confidence Interval (CI), and 99th Percentile of losses before and after controls. It also includes the first non-zero percentile and value after controls.
        - **Input Parameters**: Asset Value (AV), Exposure Factor (EF), Annual Rate of Occurrence (ARO), Single Loss Expectancy (SLE), and Annualized Loss Expectancy (ALE).
        - **Calculated Parameters**: Adjusted ARO after controls, Adjusted ALE, and the expected reduction in loss due to controls.
        - **Simulation Results**: Lists of simulated losses and exceedance probabilities before and after controls.

    Args:
        asset_value: The value of the asset at risk, expressed in monetary units.
        exposure_factor: The percentage of the asset value that is at risk during a risk event, expressed as a decimal.
        annual_rate_of_occurrence: The frequency of the risk event over a year, expressed as a decimal.
        reduction_percentage: The percentage reduction in the ARO after applying risk controls, expressed as a percentage.
        cost_of_controls: The cost of implementing the risk controls, expressed in monetary units.
        plot: A boolean indicating whether to plot the results (default is True).
        monte_carlo_seed: The seed for the Monte Carlo simulation to ensure reproducibility (default is constant MONTE_CARLO_SEED).
        num_simulations: The number of simulations to run for the Monte Carlo analysis (default is constant NUM_SIMULATIONS).
        kurtosis: The kurtosis value to adjust the shape of the beta distribution for the EF (default is constant KURTOSIS).
        currency_symbol: The currency symbol to use in the plot displays. (default is constant CURRENCY_SYMBOL).

    Returns:
        dict | None: A dictionary containing the statistics, input parameters, calculated parameters, and simulation results if plot is False. If plot is True, the function displays the visualizations and tables without returning a dictionary.
    """
    # Input validation
    _validate_simulation_params(
        exposure_factor=exposure_factor,
        reduction_percentage=reduction_percentage,
        annual_rate_of_occurrence=annual_rate_of_occurrence,
        asset_value=asset_value,
        cost_of_controls=cost_of_controls,
        kurtosis=kurtosis,
        num_simulations=num_simulations,
        monte_carlo_seed=monte_carlo_seed,
        plot=plot,
    )

    # Set seed for reproducibility
    np.random.seed(monte_carlo_seed)

    # Calculate SLE and ALE
    single_loss_expectancy, annualized_loss_expectancy = calculate_sle_ale(
        asset_value, exposure_factor, annual_rate_of_occurrence
    )

    # Simulate loss in the before scenario with MCMC
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

    # Simulate losses in the after scenario with MCMC
    adjusted_losses = _simulate_losses_with_mcmc(
        asset_value=asset_value,
        exposure_factor=exposure_factor,
        annual_rate_of_occurrence=annual_rate_of_occurrence,
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
    if cost_of_controls > 0:
        rosi_percentage = (benefit - cost_of_controls) / cost_of_controls * 100
    else:
        rosi_percentage = float("inf")

    # For after controls, 2.5% percentile is likely to be zero due to the reduction
    # Calculate the first non-zero percentile and value to show in the table instead
    first_nonzero_pct, first_nonzero_val = find_first_non_zero_percentile(
        adjusted_losses
    )
    calc_adjusted_stats["First Non-Zero Percentile"] = first_nonzero_pct
    calc_adjusted_stats["First Non-Zero Value"] = first_nonzero_val

    nonzero_adjusted_losses = adjusted_losses[adjusted_losses > 0]

    if plot:
        # Create a figure with two subplots
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

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
                f'Mode: {currency_symbol}{calc_stats["Mode"]:,.2f}',
                f'Std Dev: {currency_symbol}{calc_stats["Std Dev"]:,.2f}',
                f'1st Percentile: {currency_symbol}{calc_stats["1st Percentile"]:,.2f}',
                f'95% CI: {currency_symbol}{calc_stats["2.5th Percentile"]:,.2f} - {currency_symbol}{calc_stats["97.5th Percentile"]:,.2f}',
                f'99th Percentile: {currency_symbol}{calc_stats["99th Percentile"]:,.2f}',
            ]
        )

        after_controls_lines = [
            "Statistical Summary (After Controls)",
            "------------------------------------",
            f'Mean: {currency_symbol}{calc_adjusted_stats["Mean"]:,.2f}',
            f'Median: {currency_symbol}{calc_adjusted_stats["Median"]:,.2f}',
            f'Mode: {currency_symbol}{calc_adjusted_stats["Mode"]:,.2f}',
            f'Std Dev: {currency_symbol}{calc_adjusted_stats["Std Dev"]:,.2f}',
        ]

        # Conditionally add the 1st percentile if it is non-zero (otherwise use the first non-zero value)
        if round(calc_adjusted_stats["1st Percentile"], 2) > 0:
            after_controls_lines.append(
                f'1st Percentile: {currency_symbol}{calc_adjusted_stats["1st Percentile"]:,.2f}'
            )
            after_controls_lines.append(
                f'95% CI: {currency_symbol}{calc_adjusted_stats["2.5th Percentile"]:,.2f} - {currency_symbol}{calc_adjusted_stats["97.5th Percentile"]:,.2f}'
            )
        else:
            after_controls_lines.append(
                f'{calc_adjusted_stats["First Non-Zero Percentile"]:.1f}th Percentile: {currency_symbol}{calc_adjusted_stats["First Non-Zero Value"]:,.2f}'
            )
            after_controls_lines.append(
                f'CI {calc_adjusted_stats["First Non-Zero Percentile"]:.1f}%-95%: {currency_symbol}{calc_adjusted_stats["First Non-Zero Value"]:,.2f} - {currency_symbol}{calc_adjusted_stats["97.5th Percentile"]:,.2f}'
            )

        after_controls_lines.extend(
            [
                f'99th Percentile: {currency_symbol}{calc_adjusted_stats["99th Percentile"]:,.2f}',
            ]
        )

        after_controls_text = "\n".join(after_controls_lines)

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
                f"Annualized Loss Expectancy (ALE): {currency_symbol}{annualized_loss_expectancy * (1 - reduction_percentage/100):,.2f}",
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
        plt.show()

    # If plot is False, return the statistics
    else:
        return {
            "before_controls_stats": {
                "mean": float(calc_stats["Mean"]),
                "median": float(calc_stats["Median"]),
                "mode": float(calc_stats["Mode"]),
                "std_dev": float(calc_stats["Std Dev"]),
                "percentile_1": float(calc_stats["1st Percentile"]),
                "percentile_2.5": float(calc_stats["2.5th Percentile"]),
                "percentile_5": float(calc_stats["5th Percentile"]),
                "percentile_10": float(calc_stats["10th Percentile"]),
                "percentile_25": float(calc_stats["25th Percentile"]),
                "percentile_75": float(calc_stats["75th Percentile"]),
                "percentile_90": float(calc_stats["90th Percentile"]),
                "percentile_97.5": float(calc_stats["97.5th Percentile"]),
                "percentile_99": float(calc_stats["99th Percentile"]),
            },
            "after_controls_stats": {
                "mean": float(calc_adjusted_stats["Mean"]),
                "median": float(calc_adjusted_stats["Median"]),
                "mode": float(calc_adjusted_stats["Mode"]),
                "std_dev": float(calc_adjusted_stats["Std Dev"]),
                "percentile_1": float(calc_adjusted_stats["1st Percentile"]),
                "percentile_2.5": float(calc_adjusted_stats["2.5th Percentile"]),
                "percentile_5": float(calc_adjusted_stats["5th Percentile"]),
                "percentile_10": float(calc_adjusted_stats["10th Percentile"]),
                "percentile_25": float(calc_adjusted_stats["25th Percentile"]),
                "percentile_75": float(calc_adjusted_stats["75th Percentile"]),
                "percentile_90": float(calc_adjusted_stats["90th Percentile"]),
                "percentile_97.5": float(calc_adjusted_stats["97.5th Percentile"]),
                "percentile_99": float(calc_adjusted_stats["99th Percentile"]),
                "first_nonzero_percentile": float(
                    calc_adjusted_stats["First Non-Zero Percentile"]
                ),
                "first_nonzero_value": float(
                    calc_adjusted_stats["First Non-Zero Value"]
                ),
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


def main():
    # Get user inputs
    AV = float(input("Enter the Asset Value (AV): "))
    EF = float(input("Enter the Exposure Factor (EF) between 0 and 1: "))
    ARO = float(input("Enter the Annual Rate of Occurrence (ARO): "))
    reduction_percentage = float(
        input("Enter the Percentage reduction after controls (%): ")
    )
    control_cost = float(input("Enter the cost of implementing controls: "))

    # Hardcoded inputs for testing
    # AV, EF, ARO, reduction_percentage, control_cost = 100000, 0.5, 2, 50, 10000

    # Plot the risk calculation for a before-and-after scenario
    test = plot_risk_calculation_before_after(
        AV, EF, ARO, reduction_percentage, control_cost, plot=True
    )
    # with open("risk_simulation_results_mcmc.json", "w") as f:
    #     json.dump(test, f, indent=4)


if __name__ == "__main__":
    main()
