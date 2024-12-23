import numpy as np
from scipy.stats import beta, poisson, qmc
from itertools import permutations
import matplotlib.pyplot as plt
import json

# TODO: Fix cost calculations
# Fix cost adjustments per year

# Constants
RANDOM_SEED = 42

ASSET_VALUE = 500000
EF_RANGE = (0.4, 0.6)  # Exposure factor range
ARO_RANGE = (1.5, 2.5)  # Annual rate of occurrence range
CONTROL_REDUCTIONS = [0.2, 0.35, 0.3, 0.45]  # Reduction percentages for controls
CONTROL_COSTS = [10000, 30000, 20000, 35000]  # Base annualized cost of controls
COST_ADJUSTMENT_RANGE = (0.0, 0.2)  # Cost adjustment per year range
NUM_SAMPLES = 10000  # Number of Sobol samples
NUM_YEARS = 4


# Helper functions
def generate_sobol_samples(dimensions, num_samples):
    """Generates Sobol samples for the given dimensions."""
    sobol_engine = qmc.Sobol(d=dimensions, scramble=True)
    samples = sobol_engine.random_base2(m=int(np.log2(num_samples)))
    return samples


def scale_sobol_samples(samples, min_val, max_val):
    """Scales Sobol samples to the desired range."""
    return min_val + samples * (max_val - min_val)


def simulate_ef(sobol_samples, kurtosis=2):
    """Simulates EF using Sobol samples mapped to a Beta distribution."""
    alpha, beta_param = kurtosis, kurtosis
    beta_samples = beta.ppf(sobol_samples, a=alpha, b=beta_param)
    return scale_sobol_samples(beta_samples, *EF_RANGE)


def simulate_aro(sobol_samples):
    """Simulates ARO using Sobol samples mapped to a Poisson distribution."""
    poisson_samples = poisson.ppf(sobol_samples, mu=np.mean(ARO_RANGE))
    return np.clip(poisson_samples, *ARO_RANGE)


def calculate_ale(asset_value, ef, aro):
    """Calculates the Annualized Loss Expectancy (ALE)."""
    return asset_value * ef * aro


def calculate_rosi(ale_before, ale_after, cost):
    """Calculates the Return on Security Investment (ROSI)."""
    return (ale_before - ale_after - cost) / cost


def calculate_cumulative_cost(permutation, base_costs, cost_adjustments):
    """
    Calculates the cumulative cost for a given permutation of controls over the years.

    Args:
        permutation (list): Order of controls (indices) to be implemented.
        base_costs (list): Base annualized costs of each control.
        cost_adjustments (np.ndarray): Cost adjustment factors (NUM_SAMPLES x NUM_YEARS).

    Returns:
        cumulative_costs (list): Cumulative total cost for each year.
    """
    cumulative_costs = []
    total_cost = 0

    for year, control in enumerate(permutation):
        # Initialize the cost for the current year
        current_cost = base_costs[control]

        # Adjust costs for previously implemented controls
        for prev_year, prev_control in enumerate(permutation[:year]):
            adjustment_factors = cost_adjustments[:, prev_year:year].mean(axis=1)
            current_cost += base_costs[prev_control] * np.prod(1.0 + adjustment_factors)

        # Add the current control's base cost (no adjustment for itself in the year it's implemented)
        total_cost += current_cost
        cumulative_costs.append(total_cost)

    return cumulative_costs


# Generate Sobol samples
sobol_samples = generate_sobol_samples(
    dimensions=3 * NUM_YEARS, num_samples=NUM_SAMPLES
)
ef_samples = simulate_ef(sobol_samples[:, 0:NUM_YEARS])  # EF samples
aro_samples = simulate_aro(sobol_samples[:, NUM_YEARS : 2 * NUM_YEARS])  # ARO samples
cost_adjustments = scale_sobol_samples(
    sobol_samples[:, 2 * NUM_YEARS :], *COST_ADJUSTMENT_RANGE
)

# Initialize variables
all_permutations = list(permutations(range(4)))  # All permutations of control indices
best_permutation = None
best_mean_rosi = -np.inf
cumulative_cost_details = None

# Evaluate each permutation
for permutation in all_permutations:
    ale_before = calculate_ale(ASSET_VALUE, ef_samples[:, 0], aro_samples[:, 0])
    ale_previous = ale_before.copy()
    rosi_per_year = []

    for year, control in enumerate(permutation):
        # Apply the control
        reduction = CONTROL_REDUCTIONS[control]
        ale_after = ale_previous * (1 - reduction)

        # Adjust cost if not year 1
        cost = CONTROL_COSTS[control]
        if year > 0:
            cost *= 1 + cost_adjustments[:, year - 1].mean()

        # Calculate ROSI for the year
        rosi = calculate_rosi(ale_previous, ale_after, cost)
        rosi_per_year.append(rosi)

        # Update ALE for next year
        ale_previous = ale_after

    # Aggregate ROSI for this permutation
    mean_rosi = np.mean(np.hstack(rosi_per_year), axis=0)

    # Calculate cumulative cost
    cumulative_costs = calculate_cumulative_cost(
        permutation, CONTROL_COSTS, cost_adjustments
    )

    # Update best permutation if this one is better
    if mean_rosi > best_mean_rosi:
        best_mean_rosi = mean_rosi
        best_permutation = permutation
        cumulative_cost_details = cumulative_costs

# Print results
print(f"Best Permutation: {best_permutation}")
print(f"Highest Mean ROSI: {best_mean_rosi:.2f}")
print(f"Cumulative Costs by Year: {cumulative_cost_details}")


# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

# Generate Sobol samples
sobol_samples = generate_sobol_samples(
    dimensions=3 * NUM_YEARS, num_samples=NUM_SAMPLES
)
ef_samples = simulate_ef(sobol_samples[:, 0:NUM_YEARS])  # EF samples
aro_samples = simulate_aro(sobol_samples[:, NUM_YEARS : 2 * NUM_YEARS])  # ARO samples
cost_adjustments = scale_sobol_samples(
    sobol_samples[:, 2 * NUM_YEARS :], *COST_ADJUSTMENT_RANGE
)

# Initialize variables
all_permutations = list(permutations(range(4)))  # All permutations of control indices
best_permutation = None
best_mean_rosi = -np.inf

# Evaluate each permutation
for permutation in all_permutations:
    ale_before = calculate_ale(ASSET_VALUE, ef_samples[:, 0], aro_samples[:, 0])
    ale_previous = ale_before.copy()
    rosi_per_year = []

    for year, control in enumerate(permutation):
        # Apply the control
        reduction = CONTROL_REDUCTIONS[control]
        ale_after = ale_previous * (1 - reduction)

        # Adjust cost if not year 1
        cost = CONTROL_COSTS[control]
        if year > 0:
            cost *= 1 + cost_adjustments[:, year - 1]

        # Calculate ROSI for the year
        rosi = calculate_rosi(ale_previous, ale_after, cost)
        rosi_per_year.append(rosi)

        # Update ALE for next year
        ale_previous = ale_after

    # Aggregate ROSI for this permutation
    mean_rosi = np.mean(np.hstack(rosi_per_year), axis=0)

    # Update best permutation if this one is better
    if mean_rosi > best_mean_rosi:
        best_mean_rosi = mean_rosi
        best_permutation = permutation

# Calculate complete ALE progression first
ale_before = calculate_ale(ASSET_VALUE, ef_samples[:, 0], aro_samples[:, 0])
ale_previous = ale_before.copy()
ale_values = [np.mean(ale_before)]

for control in best_permutation:
    reduction = CONTROL_REDUCTIONS[control]
    ale_after = ale_previous * (1 - reduction)
    ale_values.append(np.mean(ale_after))
    ale_previous = ale_after

# Collect data for JSON export
export_data = {
    "best_permutation": best_permutation,
    "best_mean_rosi": best_mean_rosi,
    "ale_values": ale_values,
    "controls": [],
}

for idx, (control, reduction, cost) in enumerate(
    zip(best_permutation, CONTROL_REDUCTIONS, CONTROL_COSTS)
):
    control_data = {
        "control": control,
        "reduction": reduction,
        "base_cost": cost,
        "cumulative_total_cost": np.sum(
            cost * (1 + cost_adjustments[:, :idx].sum(axis=1))
        ),
        "ef": ef_samples[:, 0][control],
        "aro": aro_samples[:, 0][control],
        "cost_adjustment": cost_adjustments[:, idx].mean(),
        "ale_before": ale_values[idx],
        "ale_after": ale_values[idx + 1],
        "rosi": np.mean(rosi_per_year[idx]),
    }
    export_data["controls"].append(control_data)

# Add summary data
export_data["summary"] = {
    "input_asset_value": ASSET_VALUE,
    "input_ef_range": EF_RANGE,
    "input_aro_range": ARO_RANGE,
    "input_control_reductions": CONTROL_REDUCTIONS,
    "input_control_costs": CONTROL_COSTS,
    "input_cost_adjustment_range": COST_ADJUSTMENT_RANGE,
    "total_samples": NUM_SAMPLES,
    "initial_ale": ale_values[0],
    "final_ale": ale_values[-1],
    "total_reduction": 1 - ale_values[-1] / ale_values[0],
}

# Export to JSON file
with open("rqmc_sobol_test_results.json", "w") as json_file:
    json.dump(export_data, json_file, indent=4)

# Setup figure and grid
plt.figure(figsize=(20, 10))
gs = plt.GridSpec(2, 4)

# ALE progression plot (left half of top row)
ax1 = plt.subplot(gs[0, 0:2])
ax1.plot(range(NUM_YEARS + 1), ale_values, marker="o")
ax1.set_title(f"ALE Progression for Best Permutation: {best_permutation}")
ax1.set_xlabel("Year")
ax1.set_ylabel("Mean ALE ($)")
ax1.grid(True)
ax1.set_xticks(range(NUM_YEARS + 1))

# Create 4 control tables (2 in top right, 2 in bottom left)
table_positions = [(0, 2), (0, 3), (1, 0), (1, 1)]
for idx, (control, reduction, cost, pos) in enumerate(
    zip(best_permutation, CONTROL_REDUCTIONS, CONTROL_COSTS, table_positions)
):
    ax = plt.subplot(gs[pos])
    ax.axis("off")

    control_stats = [
        f"Control {control}",
        f"-------------",
        f"Reduction: {reduction:.0%}",
        f"Base Cost: ${cost:,.0f}",
        f"Cumulative Total Cost: ${np.sum(cost * (1 + cost_adjustments[:, :idx].sum(axis=1))):,.0f}",
        f"EF: {ef_samples[:, 0][control]:.2f}",
        f"ARO: {aro_samples[:, 0][control]:.2f}",
        f"Cost Adjustment: {cost_adjustments[:, idx].mean():.2f}",
        f"ALE Before: ${ale_values[idx]:,.0f}",
        f"ALE After: ${ale_values[idx+1]:,.0f}",
        f"ROSI: {np.mean(rosi_per_year[idx]):.1f}%",
    ]

    ax.text(
        0.1,
        0.5,
        "\n".join(control_stats),
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        fontsize=10,
        family="monospace",
        transform=ax.transAxes,
        va="center",
    )

# Create summary table (bottom row, centered)
ax = plt.subplot(gs[1, 2:4])
ax.axis("off")

summary_stats = [
    "Best Configuration Summary",
    "------------------------",
    f"Input Asset Value: ${ASSET_VALUE:,.0f}",
    f"Input EF Range: {EF_RANGE}",
    f"Input ARO Range: {ARO_RANGE}",
    f"Input Control Reductions: {CONTROL_REDUCTIONS}",
    f"Input Control Costs: {CONTROL_COSTS}",
    f"Input Cost Adjustment Range: {COST_ADJUSTMENT_RANGE}",
    f"Total Samples: {NUM_SAMPLES:,}",
    f"Optimal Control Sequence: {best_permutation}",
    f"Mean ROSI Across Years: {best_mean_rosi:.1f}%",
    f"Initial ALE: ${ale_values[0]:,.0f}",
    f"Final ALE: ${ale_values[-1]:,.0f}",
    f"Total Reduction: {(1 - ale_values[-1]/ale_values[0]):.0%}",
]

ax.text(
    0.5,
    0.5,
    "\n".join(summary_stats),
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    fontsize=10,
    family="monospace",
    transform=ax.transAxes,
    va="center",
    ha="center",
)

plt.tight_layout()
plt.savefig("rqmc_sobol_test.png")
