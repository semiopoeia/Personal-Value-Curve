# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 19:06:14 2025

@author: PWS5
"""

# -*- coding: utf-8 -*-
"""
Doing Utility Functions adjusting for sub-optimal decisioning
@author: PWS5
"""
!pip install pandas scikit-learn numpy pymc
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from scipy.optimize import minimize
import random
import pandas as pd
import pymc as pm
import numpy as np
import numpy as np
import arviz as az

#Step 1. Collect Responses to Lottery Tasks to determine a persons loss aversion and diminishing sensitivities to loss/gain
"choice tasks to determine a person's loss aversion with diminishing returns"
#define a set of choice tasks reflecting gains and losses
tasks = [
    #gain tasks
    {"sure": 50,  "lottery": [(100, 0.5), (0, 0.5)]},
    {"sure": 30,  "lottery": [(80, 0.4), (0, 0.6)]},
    {"sure": 70,  "lottery": [(150, 0.6), (0, 0.4)]},
    {"sure": 20,  "lottery": [(60, 0.3), (0, 0.7)]},
    {"sure": 90,  "lottery": [(200, 0.8), (0, 0.2)]},
    {"sure": 10,  "lottery": [(40, 0.25), (0, 0.75)]},
    {"sure": 120, "lottery": [(250, 0.9), (0, 0.1)]},
    {"sure": 40,  "lottery": [(100, 0.35), (0, 0.65)]},
    {"sure": 60,  "lottery": [(140, 0.55), (0, 0.45)]},
    {"sure": 80,  "lottery": [(180, 0.7), (0, 0.3)]},
    #loss tasks
    {"sure": -50,  "lottery": [(-100, 0.5), (0, 0.5)]},
    {"sure": -30,  "lottery": [(-80, 0.4), (0, 0.6)]},
    {"sure": -70,  "lottery": [(-150, 0.6), (0, 0.4)]},
    {"sure": -20,  "lottery": [(-60, 0.3), (0, 0.7)]},
    {"sure": -90,  "lottery": [(-200, 0.8), (0, 0.2)]},
    {"sure": -10,  "lottery": [(-40, 0.25), (0, 0.75)]},
    {"sure": -120, "lottery": [(-250, 0.9), (0, 0.1)]},
    {"sure": -40,  "lottery": [(-100, 0.35), (0, 0.65)]},
    {"sure": -60,  "lottery": [(-140, 0.55), (0, 0.45)]},
    {"sure": -80,  "lottery": [(-180, 0.7), (0, 0.3)]},
]

responses = []
#administering tasks
print("Please choose between the sure amount and the lottery for each task:\n")
for i, task in enumerate(tasks, start=1):
    sure = task["sure"]
    lottery_str = " or ".join([f"{p*100:.0f}% chance of {amt}" for amt, p in task["lottery"]])
    print(f"Task {i}: Sure amount = {sure}, Lottery = {lottery_str}")
    
    choice = input("Enter 'S' for sure amount or 'L' for lottery: ").strip().upper()
    responses.append({"task": i, "sure": sure, "lottery": task["lottery"], "choice": choice})
#store responses
ir_df = pd.DataFrame(responses)

#Step 2: Updating alpha, beta, lambda values based on individual's responses to tasks 
#reflecting their diminishing sensitivity to loss/gains and loss aversion level
#empircially 1992 Kahneman/Tversky found lamb=2.25, alpha=beta=.88
#we can use those as the prior parameters to update

#Prospect Theory value function
#set response data to be usable
choices = np.array([1 if r["choice"] == "L" else 0 for r in responses])
sure_values = [r["sure"] for r in responses]
lottery_values = [r["lottery"] for r in responses]

def prospect_value(x, alpha, beta, lamb):
    return x**alpha if x >= 0 else -lamb * (abs(x)**beta)

def lottery_value(lottery, alpha, beta, lamb):
    return sum(p * prospect_value(outcome, alpha, beta, lamb) for outcome, p in lottery)

with pm.Model() as model:
    #priors
    alpha = pm.TruncatedNormal("alpha", mu=0.88, sigma=0.2, lower=0.01, upper=1.5)
    beta = pm.TruncatedNormal("beta", mu=0.88, sigma=0.2, lower=0.01, upper=1.5)
    lamb = pm.TruncatedNormal("lambda", mu=2.25, sigma=0.5, lower=0.5, upper=5)
    gamma = pm.HalfNormal("gamma", sigma=1.0)

    #compute utilities
    V_sure = [prospect_value(s, alpha, beta, lamb) for s in sure_values]
    V_lottery = [lottery_value(l, alpha, beta, lamb) for l in lottery_values]

    #logistic choice model
    p = pm.Deterministic("p", pm.math.sigmoid(gamma * (pm.math.stack(V_lottery) - pm.math.stack(V_sure))))

    #likelihood
    pm.Bernoulli("obs", p=p, observed=choices)

    #sampling
    trace = pm.sample(2000, tune=1000, target_accept=0.9)

#posterior summary
print(az.summary(trace, var_names=["alpha", "beta", "lambda", "gamma"]))

#Step 3: Generate Propect Theory Value function with individual's updated alpha, beta, lambda values
"prospect theory (Kahneman & Tversky 1992)"
#Prospect Theory value function with safe handling for negative inputs
# x gives the payout/outcome being considered (x<0 loss, x>=0 gains)
#alpha controls curvature of gains, 
#alpha<1 diminished senstivity to gain (typical, concave), alpha=1 linear, 
#beta controls curvature of losses
#beta applies to losses, beta<1 diminsihed sensitivity to loss, beta=1 linear utility,
#with both gain and loss there's diminished sensitivity at scale 
#losing or gaining 50 to 100$ is more sensitive that gaining 1050 over 1000$
#lambda is the loss aversion coefficient, lamb>1 losses felt more strongly than gains
#extract posterior means
alpha_mean = trace.posterior["alpha"].mean().item()
beta_mean = trace.posterior["beta"].mean().item()
lambda_mean = trace.posterior["lambda"].mean().item()

print(f"Posterior means: alpha={alpha_mean:.3f}, beta={beta_mean:.3f}, lambda={lambda_mean:.3f}")

#define prospect value function using posterior means
def prospect_value(x, alpha=alpha_mean, beta=beta_mean, lamb=lambda_mean):
    x = np.array(x)
    v = np.empty_like(x, dtype=float)
    v[x >= 0] = x[x >= 0] ** alpha
    v[x < 0] = -lamb * (np.abs(x[x < 0]) ** beta)
    return v

#generate range and plot
x = np.linspace(-10, 10, 500)
v = prospect_value(x)

plt.figure(figsize=(8, 5))
plt.plot(x, v, label=f"α={alpha_mean:.2f}, β={beta_mean:.2f}, λ={lambda_mean:.2f}")
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.title("Prospect Theory Value Function (Posterior Means)")
plt.xlabel("Outcome (x)")
plt.ylabel("Value v(x)")
plt.legend()
plt.grid(True)
plt.show()


#with credibe interval bands
#extract posterior samples
alpha_samples = trace.posterior["alpha"].values.flatten()
beta_samples = trace.posterior["beta"].values.flatten()
lambda_samples = trace.posterior["lambda"].values.flatten()

#sampled posterior draws
n_samples = 1000
idx = np.random.choice(len(alpha_samples), size=n_samples, replace=False)

value_matrix = np.zeros((n_samples, len(x)))
for i, j in enumerate(idx):
    a = alpha_samples[j]
    b = beta_samples[j]
    l = lambda_samples[j]
    v = np.empty_like(x, dtype=float)
    v[x >= 0] = x[x >= 0] ** a
    v[x < 0] = -l * (np.abs(x[x < 0]) ** b)
    value_matrix[i, :] = v

#compute mean and 95% credible intervals
mean_curve = value_matrix.mean(axis=0)
lower_bound = np.percentile(value_matrix, 2.5, axis=0)
upper_bound = np.percentile(value_matrix, 97.5, axis=0)

#plot with uncertainty bands (credible intervals)
plt.figure(figsize=(8, 5))
plt.plot(x, mean_curve, color='blue', label='Posterior Mean')
plt.fill_between(x, lower_bound, upper_bound, color='blue', alpha=0.2, label='95% Credible Interval')
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.title("Prospect Theory Value Function with 95% Credible Interval")
plt.xlabel("Outcome (x)")
plt.ylabel("Value v(x)")
plt.legend()
plt.grid(True)
plt.show()

