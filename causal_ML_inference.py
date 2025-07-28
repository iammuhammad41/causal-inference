import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dowhy
from dowhy import CausalModel
import dowhy.datasets
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor

# Generate a synthetic dataset
np.random.seed(42)

# Define sample size
n_samples = 1000

# Create a synthetic dataset where 'treatment' affects 'outcome'
X = np.random.rand(n_samples, 3)  # Features
w0 = X[:, 0]  # First feature (confounder)
w1 = X[:, 1]  # Second feature (confounder)
treatment = 2 * w0 + 3 * w1 + np.random.normal(0, 0.1, n_samples)  # Treatment variable

# Outcome depends on the treatment with some noise
outcome = 5 * treatment + 2 * w0 + 3 * w1 + np.random.normal(0, 1, n_samples)  # Outcome variable

# Combine into a DataFrame
df = pd.DataFrame(data=np.column_stack([treatment, outcome, w0, w1]), columns=['treatment', 'outcome', 'w0', 'w1'])
print("Synthetic Data:")
print(df.head())

# Define the causal graph
model = CausalModel(
    data=df,
    treatment='treatment',
    outcome='outcome',
    common_causes=['w0', 'w1']
)

# Visualize the causal graph
model.view_model(layout="dot")

# Identification of causal effect
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print("\nIdentified Estimand:")
print(identified_estimand)

# Estimation of the causal effect
# We'll use a simple backdoor method to estimate the effect via linear regression
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression"
)
print("\nCausal Estimate:")
print(f"Mean Estimate: {estimate.value}")

# Plotting the treatment and outcome relationship
dowhy.plotter.plot_causal_effect(estimate, df['treatment'], df['outcome'])

# Refutation tests for robustness
# Test with placebo treatment (random treatment)
refute_results = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter")
print("\nRefutation Results (Placebo Treatment):")
print(refute_results)

# Using EconML for advanced causal effect estimation (Double Machine Learning)
from econml.dml import DMLCateEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

# Prepare model for Double Machine Learning (DML)
estimator = DMLCateEstimator(model_y=RandomForestRegressor(),
                             model_t=RandomForestRegressor(),
                             model_final=LassoCV(),
                             featurizer=None)

# Fit the model
X_features = df[['w0', 'w1']].values  # Feature columns
y = df['outcome'].values  # Outcome
treatment = df['treatment'].values  # Treatment

estimator.fit(X_features, y, treatment)

# Estimate the causal effect
dml_estimate = estimator.effect(X_features)
print("\nDML Causal Estimate:")
print(dml_estimate)

# Plot the effect
plt.plot(dml_estimate)
plt.title("Causal Effect (DML Estimate)")
plt.xlabel("Samples")
plt.ylabel("Estimated Effect")
plt.show()

# Refute using placebo treatment in DML model
dml_refute = estimator.refute(X_features, y, treatment)
print("\nDML Refutation Result:")
print(dml_refute)
