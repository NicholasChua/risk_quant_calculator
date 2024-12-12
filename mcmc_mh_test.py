import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

# Inputs
AV = 1000000  # Asset Value in dollars
EF = 0.2      # Exposure Factor
ARO = 1.0     # Annual Rate of Occurrence
controls = [0.1, 0.2, 0.3, 0.5, 0.1]  # Initial percentage effectiveness of controls
alpha, beta_params = 0.5, 0.5  # Beta distribution params for priors
num_years = len(controls)
num_samples = 1000  # Number of MCMC iterations

# Define cumulative effectiveness
def cumulative_effectiveness(control_effectiveness):
    """Calculate cascading effect of controls.
    Each control reduces the remaining risk from previous controls."""
    remaining_risk = 1.0
    for control in control_effectiveness:
        remaining_risk *= (1.0 - control)
    return 1.0 - remaining_risk

def yearly_loss(control_effectiveness):
    """Calculate yearly losses with sequential control application."""
    initial_loss = AV * EF * ARO
    remaining_risk = 1.0
    losses = []
    
    for control in control_effectiveness:
        remaining_risk *= (1.0 - control)
        losses.append(initial_loss * remaining_risk)
    return losses

def test_control_cascade():
    """Verify cascading control calculations."""
    test_controls = [0.1, 0.2, 0.3, 0.5, 0.1]
    remaining = 1.0
    print(f"Initial Risk: 100%")
    
    for i, control in enumerate(test_controls, 1):
        remaining *= (1.0 - control)
        print(f"Year {i}: {remaining*100:.1f}% risk remaining")

def plot_risk_reduction(years, losses):
    initial_loss = AV * EF * ARO
    risk_remaining = [loss/initial_loss * 100 for loss in losses]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Loss plot
    ax1.plot(years, losses, marker='o', color='blue')
    ax1.set_title("Yearly Loss After Controls")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Loss ($)")
    ax1.grid(True)
    
    # Risk reduction plot
    ax2.plot(years, risk_remaining, marker='o', color='red')
    ax2.set_title("Remaining Risk Percentage")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Risk Remaining (%)")
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def metropolis_hastings(num_samples: int, initial_controls: list, alpha: float, beta_params: float) -> np.ndarray:
    """Metropolis-Hastings algorithm with improved numerical stability.
    
    Args:
        num_samples: Number of MCMC samples to generate
        initial_controls: Starting control effectiveness values
        alpha: Beta distribution alpha parameter 
        beta_params: Beta distribution beta parameter
    
    Returns:
        np.ndarray: Samples of control effectiveness values
    """
    # Validate parameters
    if alpha <= 0 or beta_params <= 0:
        raise ValueError("Alpha and beta parameters must be positive")
        
    samples = []
    current_state = np.array(initial_controls)

    for _ in range(num_samples):
        # Propose new controls with small noise
        proposal = current_state + np.random.normal(0, 0.01, size=len(initial_controls))
        proposal = np.clip(proposal, 1e-10, 1-1e-10)  # Avoid boundary values
        
        # Calculate log probabilities
        try:
            current_log_prior = np.sum([beta.logpdf(e, alpha, beta_params) for e in current_state])
            proposal_log_prior = np.sum([beta.logpdf(e, alpha, beta_params) for e in proposal])
            
            # Calculate acceptance ratio in log space
            log_acceptance_ratio = proposal_log_prior - current_log_prior
            
            # Accept/reject in log space
            if np.log(np.random.rand()) < log_acceptance_ratio:
                current_state = proposal
        except:
            continue  # Skip invalid proposals
            
        samples.append(current_state.copy())
        
    return np.array(samples)

# Run Metropolis-Hastings to sample control effectiveness
control_samples = metropolis_hastings(num_samples, controls, alpha, beta_params)

# Compute average yearly losses for each year
average_yearly_losses = []
for year in range(num_years):
    yearly_loss_samples = [
        AV * EF * ARO * (1 - cumulative_effectiveness(sample[:year+1]))
        for sample in control_samples
    ]
    average_yearly_losses.append(np.mean(yearly_loss_samples))

# Plot the results
years = list(range(1, num_years + 1))
test_control_cascade()
plot_risk_reduction(years, average_yearly_losses)
plt.show()
