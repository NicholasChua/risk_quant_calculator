import numpy as np
from scipy.stats import beta, poisson
from itertools import permutations
import matplotlib.pyplot as plt
import json
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
import csv

from risk_simulator import (
    get_beta_parameters_for_kurtosis,
)


RANDOM_SEED = 42
NUM_SAMPLES = 16384  # Number of Sobol samples 2^14
KURTOSIS = 1.7  # Results in alpha and beta of 0.5


def load_csv_data(
    file_path: str,
) -> dict[str, list[dict[str, int | float | list[float]]]]:
    """Helper function to load data from a CSV file containing input parameters for the risk calculation.

    The CSV file should have the following columns in order:
        - id
        - asset_value
        - exposure_factor_min/max
        - annual_rate_of_occurrence_min/max
        - cost_adjustment_min/max
        - control_reduction_i (alternating columns)
        - control_cost_i (alternating columns)

    Args:
        file_path: The path to the CSV file.

    Returns:
        dict[str, list[dict[str, int | float | list[float]]]]: A dictionary with 'data' key containing list of risk dictionaries.

    Raises:
        FileNotFoundError: If CSV file not found
        ValueError: If control columns missing or invalid
    """
    result = {"data": []}

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            if not reader.fieldnames:
                raise ValueError("CSV file is empty or has no headers")

            headers = reader.fieldnames

            # Extract and validate control numbers
            control_numbers = sorted(
                [
                    int(col.split("_")[-1])
                    for col in headers
                    if col.startswith("control_reduction_")
                ]
            )

            if not control_numbers:
                raise ValueError("No control columns found")

            # Process each row
            for row in reader:
                control_reductions = []
                control_costs = []

                # Process controls
                for num in control_numbers:
                    red_key = f"control_reduction_{num}"
                    cost_key = f"control_cost_{num}"

                    if red_key not in row or cost_key not in row:
                        raise ValueError(f"Missing control {num} data")

                    control_reductions.append(float(row[red_key]))
                    control_costs.append(float(row[cost_key]))

                # Build risk dictionary
                risk = {
                    "id": int(row["id"]),
                    "asset_value": float(row["asset_value"]),
                    "ef_range": [
                        float(row["exposure_factor_min"]),
                        float(row["exposure_factor_max"]),
                    ],
                    "aro_range": [
                        float(row["annual_rate_of_occurrence_min"]),
                        float(row["annual_rate_of_occurrence_max"]),
                    ],
                    "cost_adjustment_range": [
                        float(row["cost_adjustment_min"]),
                        float(row["cost_adjustment_max"]),
                    ],
                    "control_reductions": control_reductions,
                    "control_costs": control_costs,
                    "num_years": len(control_costs),
                }
                result["data"].append(risk)

    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    except csv.Error as e:
        raise ValueError(f"CSV parsing error: {str(e)}")

    return result


# TODO: Move this to a common library file
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


def calculate_compounding_costs(
    control_costs: list[float],
    cost_adjustment_range: list[float],
    years: int,
    num_samples: int = NUM_SAMPLES,
) -> dict[str, dict[str, dict[str, np.float64]]]:
    """Given a list of control costs, a cost adjustment range, and a number of years, calculate the cost for each control for each year.

    Args:
        control_costs: Base annualized cost of controls
        cost_adjustment_range: Cost adjustment per year range
        years: Number of years to simulate
        num_samples: Number of samples to simulate. Default is NUM_SAMPLES

    Returns:
        dict[str, dict[str, dict[str, np.float64]]]: Dictionary of costs and adjustments for each control, for each year
    """
    # Initialize costs dictionary
    costs = {}

    # Define the problem for cost adjustments using setup_sensitivity_problem
    problem = setup_sensitivity_problem(
        **{
            f"cost_adj_{i}": cost_adjustment_range
            for i in range(len(control_costs) * years)
        }
    )

    # Generate Sobol samples for cost adjustments. Number of sobol samples is equal to samples * dimensions (years * controls)
    sobol_samples = sobol_sample.sample(
        problem, num_samples, calc_second_order=False, seed=RANDOM_SEED
    )

    # Randomize Sobol samples to avoid correlation with EF and ARO
    sobol_samples = randomize_sobol_samples(sobol_samples)

    # Year 0 is the base cost and has no adjustment
    costs["year_0"] = {
        f"control_{index + 1}": {
            "cost": np.float64(control_cost),
            "adjustment": np.float64(0.0),
        }
        for index, control_cost in enumerate(control_costs)
    }

    # Dynamically compute how many samples we can average per control-year
    num_dimensions = len(control_costs) * years
    # Determine the number of samples to average for each control-year with a minimum of 1
    n_avg = max(1, num_samples // num_dimensions)

    # Calculate costs for each year on a compounding basis for each control
    for year in range(1, years + 1):
        costs[f"year_{year}"] = {}
        for index, _ in enumerate(control_costs):
            # Use mean of multiple Sobol samples for adjustment
            sample_start = ((year - 1) * len(control_costs) + index) * n_avg
            sample_end = sample_start + n_avg
            adjustment_values = sobol_samples[
                sample_start:sample_end, sample_start // n_avg
            ]
            adjustment_mean = adjustment_values.mean()
            adjustment = np.float64(
                adjustment_mean * (cost_adjustment_range[1] - cost_adjustment_range[0])
                + cost_adjustment_range[0]
            )

            previous_year_cost = costs[f"year_{year - 1}"][f"control_{index + 1}"][
                "cost"
            ]
            costs[f"year_{year}"][f"control_{index + 1}"] = {
                "cost": np.float64(
                    previous_year_cost + (previous_year_cost * adjustment)
                ),
                "adjustment": adjustment,
            }
    return costs


def calculate_statistics_for_permutation_per_year(
    asset_value: float,
    costs: dict[str, dict[str, dict[str, np.float64]]],
    ef_samples: np.ndarray,
    aro_samples: np.ndarray,
    permutation: tuple[int],
    control_reductions: list,
    num_of_simulations: int = NUM_SAMPLES,
) -> dict[str, float]:
    """Given a permutation of controls, calculate the ROSI for each year and the total ROSI for the permutation.

    Args:
        asset_value: The value of the asset at risk, expressed in monetary units
        costs: Dictionary of costs and adjustments for each control, for each year
        ef_samples: Simulated exposure factors
        aro_samples: Simulated annual rates of occurrence
        permutation: A single permutation of controls represented as a tuple of integers
        control_reductions: List of control reduction percentages
        num_of_simulations: Number of samples to simulate

    Returns:
        dict[str, float]: Dictionary of results for the permutation
    """
    # Initialize ROSI
    rosi_per_simulation = []

    for i in range(num_of_simulations):
        rosi_per_year = []
        results = {
            "permutation": permutation,
            "total_rosi": 0.0,
        }

        for year in range(len(permutation)):
            # Retrieve the appropriate control and its cost for the year
            control = permutation[year]
            control_cost = costs[f"year_{year + 1}"][f"control_{control}"]["cost"]

            # Calculate the total cost by summing the costs of all controls up to the current year, inclusive
            total_cost = sum(
                [
                    costs[f"year_{year + 1}"][f"control_{control}"]["cost"]
                    for control in permutation[: year + 1]
                ]
            )

            # Calculate the new ARO after applying the control
            aro_after = aro_samples[i][year] * (1 - control_reductions[control - 1])

            # Calculate the new ALE after applying the control
            ale_after = calculate_ale(asset_value, ef_samples[i][year], aro_after)

            # Calculate the ALE before applying the control
            if year == 0:
                ale_before = calculate_ale(
                    asset_value, ef_samples[i][year], aro_samples[i][year]
                )
            else:
                ale_before = results[f"year_{year}"]["ale_after"]

            # Calculate the ROSI after applying the control and save it
            rosi = calculate_rosi(ale_before, ale_after, total_cost)
            rosi_per_year.append(rosi)

            # Add year-by-year information to the results
            results[f"year_{year + 1}"] = {
                "ale_before": ale_before,
                "ale_after": ale_after,
                "control_cost": control_cost,
                "total_cost": total_cost,
                "rosi": rosi,
            }

        # Calculate the total ROSI for the permutation
        results["total_rosi"] = np.sum(rosi_per_year)
        rosi_per_simulation.append(results["total_rosi"])

    # Average the ROSI over all simulations
    average_rosi = np.mean(rosi_per_simulation)
    results["total_rosi"] = average_rosi

    return results


def calculate_statistics_for_permutation_aggregate(
    asset_value: float,
    costs: dict[str, dict[str, dict[str, np.float64]]],
    ef_samples: np.ndarray,
    aro_samples: np.ndarray,
    permutation: tuple[int],
    control_reductions: list,
    num_of_simulations: int = NUM_SAMPLES,
) -> dict[str, float]:
    """Given a permutation of controls, calculate the ROSI for the entire period. Currently not being used in the main simulation, as we are interested in year-by-year results.

    Args:
        asset_value: The value of the asset at risk, expressed in monetary units
        costs: Dictionary of costs and adjustments for each control, for each year
        ef_samples: Simulated exposure factors
        aro_samples: Simulated annual rates of occurrence
        permutation: A single permutation of controls represented as tuple of integers
        control_reductions: List of control reduction percentages
        num_of_simulations: Number of samples to simulate. Default is NUM_SAMPLES

    Returns:
        dict[str, float]: Dictionary with permutation and total ROSI
    """
    rosi_per_simulation = []

    for i in range(num_of_simulations):
        results = {
            "permutation": permutation,
            "total_rosi": 0.0,
        }

        # Calculate initial ALE and yearly data
        for year in range(len(permutation)):
            control = permutation[year]
            control_cost = costs[f"year_{year + 1}"][f"control_{control}"]["cost"]

            # Calculate total cost up to this year
            total_cost = sum(
                costs[f"year_{y + 1}"][f"control_{permutation[y]}"]["cost"]
                for y in range(year + 1)
            )

            # Calculate ALE before and after for this year
            ale_before = calculate_ale(
                asset_value, ef_samples[i][year], aro_samples[i][year]
            )
            reduction = 1 - control_reductions[control - 1]
            ale_after = calculate_ale(
                asset_value, ef_samples[i][year], aro_samples[i][year] * reduction
            )

            # Store year data
            results[f"year_{year + 1}"] = {
                "ale_before": ale_before,
                "ale_after": ale_after,
                "control_cost": control_cost,
                "total_cost": total_cost,
            }

        # Calculate aggregate ROSI
        total_ale_before = sum(
            results[f"year_{y + 1}"]["ale_before"] for y in range(len(permutation))
        )
        total_ale_after = sum(
            results[f"year_{y + 1}"]["ale_after"] for y in range(len(permutation))
        )
        total_costs = results[f"year_{len(permutation)}"]["total_cost"]

        rosi = calculate_rosi(total_ale_before, total_ale_after, total_costs)
        rosi_per_simulation.append(rosi)

    # Set final ROSI value
    results["total_rosi"] = float(np.mean(rosi_per_simulation))

    return results


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


def setup_sensitivity_problem(
    **kwargs: dict[str, list[float]]
) -> dict[str, list[float]]:
    """Define the model inputs and their bounds for sensitivity analysis.

    Args:
        **kwargs: Dictionary of parameter names and their bounds in a list with two floats

    Returns:
        dict[str, list[float]]: Problem dictionary for sensitivity analysis

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


def evaluate_model(
    asset_value: float,
    control_costs: list[float],
    control_reductions: list[float],
    X: np.ndarray,
    problem: dict,
) -> np.array:
    """Evaluate model for sensitivity analysis samples. The model calculates the ROSI for each sample. Each row in X corresponds to one set of parameter values, in the same order as problem["names"].

    For now, this function supports the following parameters:
    - EF_variance: Exposure factor variance
    - ARO_variance: Annual rate of occurrence variance
    - cost_variance: Cost adjustment variance

    Args:
        asset_value: The value of the asset at risk, expressed in monetary units
        control_costs: List of control costs, expressed in monetary units
        control_reductions: List of control reduction percentages, expressed as decimals
        X: Samples generated for sensitivity analysis

    Returns:
        np.array: ROSI values for each sample
    """
    # Initialize output array
    Y = []

    # Retrieve parameter names from the problem dictionary
    param_names = problem["names"]

    for row in X:
        # Map each parameter name to its value
        param_values = dict(zip(param_names, row))

        # Retrieve parameters by name
        ef = param_values.get("EF_variance", 0.5)
        aro = param_values.get("ARO_variance", 2.0)
        cost_adj = param_values.get("cost_variance", 0.0)

        # Calculate base ALE
        ale = calculate_ale(asset_value, ef, aro)

        # Adjust costs
        adjusted_costs = [cost * (1 + cost_adj) for cost in control_costs]

        # Calculate ALE after applying first control
        ale_after = ale * (1 - control_reductions[0])

        # Calculate ROSI for demonstration
        rosi = calculate_rosi(ale, ale_after, adjusted_costs[0])
        Y.append(rosi)

    return np.array(Y)


def perform_sensitivity_analysis(
    asset_value: float,
    ef_range: list[float],
    aro_range: list[float],
    cost_adjustment_range: list[float],
    control_costs: list[float],
    control_reductions: list[float],
    num_samples: int = NUM_SAMPLES,
) -> dict:
    """Perform Sobol sensitivity analysis on the model using the specified number of samples.

    Args:
        asset_value: The value of the asset at risk, expressed in monetary units
        ef_range: Exposure factor range
        aro_range: Annual rate of occurrence range
        cost_adjustment_range: Cost adjustment per year range
        control_costs: List of control costs, expressed in monetary units
        control_reductions: List of control reduction percentages, expressed as decimals
        num_samples: Number of samples to generate for sensitivity analysis. Default is NUM_SAMPLES

    Returns:
        dict: Sensitivity analysis results
    """
    problem = setup_sensitivity_problem(
        EF_variance=ef_range,
        ARO_variance=aro_range,
        cost_variance=cost_adjustment_range,
    )

    # Generate samples using sobol sampler
    param_values = sobol_sample.sample(
        problem, num_samples, calc_second_order=False, seed=RANDOM_SEED
    )

    # Run model evaluations
    Y = evaluate_model(
        asset_value, control_costs, control_reductions, param_values, problem
    )

    # Create a copy of the problem dictionary to convert to numpy arrays
    problem_array = problem.copy()
    for key in problem_array.keys():
        if key != "num_vars":
            problem_array[key] = np.array(problem_array[key])

    # Calculate sensitivity indices using sobol analyzer
    Si = sobol_analyze.analyze(
        problem_array, Y, calc_second_order=False, seed=RANDOM_SEED
    )

    return Si


def plot_sensitivity_analysis(Si: dict, problem: dict, output_file: str) -> None:
    """Plot the results of the sensitivity analysis.

    Args:
        Si: Sensitivity analysis results
        problem: Problem definition used in the sensitivity analysis
        output_file: Output file to save the sensitivity analysis plot

    Returns:
        None
    """
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.bar(
        problem["names"],
        Si["S1"],
        yerr=Si["S1_conf"],
        label="First Order",
        color="blue",
    )
    plt.bar(
        problem["names"],
        Si["ST"],
        yerr=Si["ST_conf"],
        alpha=0.5,
        label="Total Order",
        color="orange",
    )
    plt.xlabel("Parameters")
    plt.ylabel("Sensitivity Index")
    plt.legend()
    plt.title("Parameter Sensitivity Analysis")
    plt.tight_layout()
    plt.savefig(output_file)


def plot_permutations_by_weighted_risk_reduction(
    sorted_permutations: list[dict[str, float]],
    control_cost_values: dict[str, dict[str, dict[str, np.float64]]],
    control_reductions: list[float],
    output_file: str,
    show_all_annotations: bool = False,
) -> None:
    """Plot permutations by weighted risk reduction and mean ROSI in a scatter plot, highlighting the best-performing permutation. Weighted risk reduction is calculated as the sum of control costs multiplied by the control reduction percentages. The best-performing permutation is the one with the highest mean ROSI.

    Args:
        sorted_permutations: List of sorted permutations with total ROSI values
        control_cost_values: Dictionary of costs and adjustments for each control, for each year
        control_reductions: List of control reduction percentages
        output_file: Output file to save the plot
        show_all_annotations: Boolean to show annotations for all points. Default is False

    Returns:
        None
    """
    weighted_reductions = []
    mean_rosi_values = []
    permutations = []

    for permutation_data in sorted_permutations:
        permutation = permutation_data["permutation"]
        mean_rosi = permutation_data["total_rosi"]

        # Calculate total weighted risk reduction for the permutation
        total_weighted_reduction = 0
        for year, control in enumerate(permutation, start=1):
            control_cost = control_cost_values[f"year_{year}"][f"control_{control}"][
                "cost"
            ]
            control_reduction = control_reductions[control - 1]
            total_weighted_reduction += control_cost * control_reduction

        weighted_reductions.append(total_weighted_reduction)
        mean_rosi_values.append(mean_rosi)
        permutations.append(permutation)

    # Plot scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(weighted_reductions, mean_rosi_values, alpha=0.7, label="Permutations")

    # Highlight best-performing permutation
    best_index = mean_rosi_values.index(max(mean_rosi_values))
    plt.scatter(
        weighted_reductions[best_index],
        mean_rosi_values[best_index],
        color="red",
        label="Best Permutation",
        zorder=5,
    )

    # Decide which annotations to show
    if show_all_annotations:
        for i, permutation in enumerate(permutations):
            plt.text(
                weighted_reductions[i],
                mean_rosi_values[i],
                str(permutation),
                fontsize=8,
                ha="right",
            )
    else:
        plt.text(
            weighted_reductions[best_index],
            mean_rosi_values[best_index],
            str(permutations[best_index]),
            fontsize=8,
            ha="right",
        )

    # Add labels and title
    plt.xlabel("Total Weighted Risk Reduction")
    plt.ylabel("Mean ROSI (%)")
    plt.title("Permutations by Mean ROSI vs. Weighted Risk Reduction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)


def plot_ale_progression_by_year(
    best_permutation_data: dict[str, float], output_file: str
) -> None:
    """Plot the year-by-year ALE progression before and after implementing controls for the best-performing permutation.

    Args:
        best_permutation_data: Data for the best-performing permutation
        output_file: Output file to save the plot

    Returns:
        None
    """
    years = []
    ale_values = []

    # Extract ALE data year-by-year
    for year in range(1, len(best_permutation_data["permutation"]) + 1):
        years.append(year)
        ale_values.append(best_permutation_data[f"year_{year}"]["ale_after"])

    # Plot ALE before and after controls
    plt.figure(figsize=(10, 6))
    plt.plot(
        years,
        ale_values,
        label="ALE After Controls",
        marker="o",
        linestyle="-",
        color="green",
    )

    # Add ALE values as annotations with smart positioning to avoid overlap with the line
    padding = 0.02 * (max(ale_values) - min(ale_values))
    for i, ale in enumerate(ale_values):
        # Check if the current ALE is lower than the previous and next ALE values
        if (i > 0 and ale_values[i - 1] < ale) and (
            i < len(ale_values) - 1 and ale_values[i + 1] < ale
        ):
            # Place annotation on top
            plt.text(
                years[i],
                ale + (padding / 2),
                f"{ale:.2f}",
                fontsize=8,
                ha="center",
                va="bottom",
            )
        else:
            # Place annotation below
            plt.text(
                years[i], ale - padding, f"{ale:.2f}", fontsize=8, ha="center", va="top"
            )

    # Add labels and title
    plt.xlabel("Year")
    plt.ylabel("Annualized Loss Expectancy (ALE)")
    plt.title("Year-by-Year ALE Progression")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Set x-axis to show only discrete years
    plt.xticks(years)

    # Show the plot
    plt.savefig(output_file)


def simulate_control_sequence_optimization(
    asset_value: float,
    ef_range: list[float],
    aro_range: list[float],
    control_costs: list[float],
    cost_adjustment_range: list[float],
    control_reductions: list[float],
    num_years: int,
    kurtosis: float = KURTOSIS,
    num_samples: int = NUM_SAMPLES,
    output_json_file: str = None,
    output_png_file: str = None,
) -> None:
    """Given a set of security controls to be implemented, with one control per year, determine the optimal control implementation sequence to maximize the Return on Security Investment (ROSI) over a specified number of years, using Randomized Quasi-Monte Carlo (RQMC) and Sobol sensitivity analysis.

    The simulation uses:
        - Sobol samples for the exposure factor, annual rate of occurrence, and cost adjustments
        - RQMC for adding randomness to the Sobol samples and averaging multiple samples per control-year
        - Beta distribution for introducing kurtosis to the exposure factor
        - Poisson distribution for simulating the annual rate of occurrence, with a specified number of decimal places to bypass the integer limitation of the distribution
        - Sobol sensitivity analysis as a global sensitivity analysis method to determine the impact of varying the exposure factor, annual rate of occurrence, and control costs on the ROSI
            - First order sensitivity index (S1) measures the impact of each parameter on the output
            - Total order sensitivity index (ST) measures the impact of each parameter, including interactions with other parameters

    The implementation performs a multivariate time series simulation to evaluate the ROSI for each year, for each permutation of controls. The simulation takes into account variability in the exposure factor, annual rate of occurrence, and control costs. The total ROSI is derived from a summation of the ROSI for all years. The simulation also performs a sensitivity analysis to determine the impact of varying the exposure factor, annual rate of occurrence, and control costs on the ROSI.

    An example scenario is as follows:
        - You have an asset value of X
        - You have 4 security controls to implement (referred to as 1, 2, 3, 4), with a base annualized cost of A, B, C, D
        - You predict the exposure factor to be between X_ef and Y_ef
        - You predict the annual rate of occurrence to be between X_aro and Y_aro
        - You predict the cost adjustment for each control to be between X_adj and Y_adj per year
        - You know the reduction in risk for each control to be applied to the annual rate of occurrence
        - You want to know the optimal sequence of controls to implement to maximize the ROSI over those years
        - You want to know the impact of varying the exposure factor, annual rate of occurrence, and control costs on the ROSI

    Args:
        asset_value: The value of the asset at risk, expressed in monetary units.
        ef_range: The range of the exposure factor, representing the percentage of the asset value that is at risk during a risk event, expressed as decimals.
        aro_range: The range of the annual rate of occurrence, representing the frequency of the risk event over a year, expressed as decimals.
        control_costs: The base annualized cost of implementing each security control, expressed in monetary units.
        cost_adjustment_range: The range of the cost adjustment for each control per year, expressed as decimals.
        control_reductions: The percentage reduction in risk for each control, expressed as decimals.
        num_years: The number of years to simulate.
        kurtosis: The kurtosis of the distribution for the exposure factor. Default is constant KURTOSIS
        num_samples: The number of samples to generate for the simulation. Default is constant NUM_SAMPLES
        json_output_file: The output JSON file to save the simulation results. Default is None
        png_output_file: The output PNG file to save the simulation results. Default is None

    Returns:
        None
    """
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)

    # Define EF and ARO problems for sensitivity analysis
    ef_params = {f"EF_{i+1}": ef_range for i in range(num_years)}
    aro_params = {f"ARO_{i+1}": aro_range for i in range(num_years)}

    problem_ef = setup_sensitivity_problem(**ef_params)
    problem_aro = setup_sensitivity_problem(**aro_params)

    # Generate Sobol samples for EF and ARO
    sobol_samples_ef = sobol_sample.sample(
        problem_ef, num_samples, calc_second_order=False, seed=RANDOM_SEED
    )
    sobol_samples_aro = sobol_sample.sample(
        problem_aro, num_samples, calc_second_order=False, seed=RANDOM_SEED
    )

    # Randomize Sobol samples to avoid correlation with cost adjustments
    sobol_samples_ef = randomize_sobol_samples(sobol_samples_ef)
    sobol_samples_aro = randomize_sobol_samples(sobol_samples_aro)

    # Simulate EF and ARO using the randomized Sobol samples
    ef_samples = simulate_exposure_factor_sobol(sobol_samples_ef, ef_range, kurtosis)
    aro_samples = simulate_annual_rate_of_occurrence_sobol(sobol_samples_aro, aro_range)

    # Calculate compounding costs for each control
    control_cost_values = calculate_compounding_costs(
        control_costs, cost_adjustment_range, num_years
    )

    # Determine permutations of control orderings (starting from 1 instead of 0)
    all_permutations = list(permutations(range(1, num_years + 1)))

    # List to store total ROSI values for each permutation after calculating all samples
    simulate_all_permutations = [
        calculate_statistics_for_permutation_per_year(  # You can use either calculate_statistics_for_permutation_aggregate or calculate_statistics_for_permutation_per_year
            asset_value,
            control_cost_values,
            ef_samples,
            aro_samples,
            permutation,
            control_reductions,
            num_samples,
        )
        for permutation in all_permutations
    ]

    # Sort the permutations by total ROSI descending
    sorted_permutations = sorted(
        simulate_all_permutations, key=lambda x: x["total_rosi"], reverse=True
    )
    best_permutation = sorted_permutations[0]["permutation"]
    best_rosi = sorted_permutations[0]["total_rosi"]

    # Run sensitivity analysis
    sensitivity_results = perform_sensitivity_analysis(
        asset_value,
        ef_range,
        aro_range,
        cost_adjustment_range,
        control_costs,
        control_reductions,
    )

    # Initialize a results dictionary
    results = {
        "simulation_parameters": {
            "asset_value": asset_value,
            "ef_range": ef_range,
            "aro_range": aro_range,
            "control_reductions": control_reductions,
            "control_costs": control_costs,
            "cost_adjustment_range": cost_adjustment_range,
            "num_samples": num_samples,
            "num_years": num_years,
            "kurtosis": kurtosis,
        },
        "results": {
            "best_permutation": best_permutation,
            "best_rosi": best_rosi,
            "control_cost_values": control_cost_values,
        },
        "ranked_permutations": sorted_permutations,
        "sensitivity_results": sensitivity_results,
    }

    serializable_results = convert_to_serializable(results)

    with open(output_json_file, "w") as f:
        json.dump(serializable_results, f, indent=4)

    # TODO: Combine plotting functions into a single plot output

    # Plot sensitivity analysis
    # plot_sensitivity_analysis(sensitivity_results, problem_ef, output_png_file)

    # Plot permutations by weighted risk reduction and mean ROSI
    # plot_permutations_by_weighted_risk_reduction(
    #     sorted_permutations, control_cost_values, control_reductions
    # )

    # Plot ALE progression by year for the best-performing permutation
    # plot_ale_progression_by_year(sorted_permutations[0])


def main():
    test = load_csv_data("rqmc_sobol_example.csv")
    for data in test["data"]:
        simulate_control_sequence_optimization(
            data["asset_value"],
            data["ef_range"],
            data["aro_range"],
            data["control_costs"],
            data["cost_adjustment_range"],
            data["control_reductions"],
            data["num_years"],
            output_json_file="test.json",
            output_png_file="sensitivity_analysis.png",
        )


if __name__ == "__main__":
    main()
