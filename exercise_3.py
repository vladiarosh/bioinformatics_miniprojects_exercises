import pymc as pm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew, kurtosis

# 1: Defining two models
if __name__ == "__main__":
    alpha_term = 2
    beta_term = 3
    # For questions 1 - 3
    with pm.Model() as model1:
        # Defining our chain of distributions
        theta = pm.Gamma('theta', alpha=alpha_term, beta=beta_term)
        beta = pm.Normal('beta', mu=theta, sigma=1)
        y = pm.Normal('y', mu=beta, sigma=2)

        # MC sampling from the distribution (based on all possible theta)
        trace1 = pm.sample(10000, return_inferencedata=False)

    # # For question 4 (assuming only thetas that guarantee y=20 in y-distribution)
    with pm.Model() as model2:
        # Defining our chain of distributions
        theta = pm.Gamma('theta', alpha=alpha_term, beta=beta_term)
        beta = pm.Normal('beta', mu=theta, sigma=1)
        y = pm.Normal('y', mu=beta, sigma=2, observed=20)

        # MC sampling from the posterior theta distribution (only for y=20)
        trace2 = pm.sample(10000, return_inferencedata=False)

# 2: Calculate the expected E[y]
    mean_value_of_theta = alpha_term / beta_term
    expected_value_y = mean_value_of_theta
    print(f"The expected value E[y]: {expected_value_y:.3f}")

# 3: Extract values for plots and for descriptive statistics
    y_samples = trace1['y']
    theta_all_samples = trace1['theta']
    theta_sub_samples = trace2['theta']

# 4: Estimate the normality of y-distribution based on skewness and excess kurtosis
    skewness = skew(y_samples)
    kurt = kurtosis(y_samples)
    print(f"Skewness: {skewness:.3f}")
    print(f"Excess Kurtosis: {kurt:.3f}")
    print('If skewness and excess kurtosis are close to 0, it suggests the distribution is very close to normal')

# 5: Calculate 95% confidence interval for E[y]
    confidence_interval = [np.mean(y_samples) - 1.96 * np.std(y_samples) / np.sqrt(len(y_samples)),
                           np.mean(y_samples) + 1.96 * np.std(y_samples) / np.sqrt(len(y_samples))]
    print(f"95% confidence interval for E[y]: {confidence_interval[0]:.3f} - {confidence_interval[1]:.3f} ")

# 6: Plot y-distribution, theta-full distribution and theta-sub distribution (for y=20)
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    sns.histplot(y_samples, kde=True, color='skyblue', stat='density', linewidth=0, ax=axes[0])
    axes[0].set_title("Posterior Distribution of y")
    axes[0].set_xlabel("y")
    axes[0].set_ylabel("Density")

    sns.histplot(theta_all_samples, kde=True, color='green', stat='density', linewidth=0, ax=axes[1])
    axes[1].set_title("Distribution of All θ Values")
    axes[1].set_xlabel("θ")
    axes[1].set_ylabel("Density")

    sns.histplot(theta_sub_samples, kde=True, color='orange', stat='density', linewidth=0, ax=axes[2])
    axes[2].set_title("Posterior Distribution of θ for y = 20")
    axes[2].set_xlabel("θ")
    axes[2].set_ylabel("Density")
    plt.tight_layout()
    plt.show()

