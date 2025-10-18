
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
def prospect_value(x, alpha, beta, lamb):
    if x >= 0:
        return x ** alpha
    else:
        return -lamb * (abs(x) ** beta)

#compute subjective value of lottery
def lottery_value(lottery, alpha, beta, lamb):
    return sum(p * prospect_value(outcome, alpha, beta, lamb) for outcome, p in lottery)

#logistic choice probability
def choice_prob(sure, lottery, alpha, beta, lamb, gamma=1.0):
    V_sure = prospect_value(sure, alpha, beta, lamb)
    V_lottery = lottery_value(lottery, alpha, beta, lamb)
    return 1 / (1 + np.exp(-gamma * (V_lottery - V_sure)))

#negative log-likelihood
def neg_log_likelihood(params, responses):
    alpha, beta, lamb = params
    ll = 0
    for r in responses:
        p = choice_prob(r["sure"], r["lottery"], alpha, beta, lamb)
        if r["choice"] == "L":
            ll += np.log(p+1e-9)
        else:
            ll += np.log(1 - p+1e-9)
    return -ll

#initial guesses and bounds
init_params = [0.88, 0.88, 2.25]
bounds = [(0.01, 1.5), (0.01, 1.5), (0.5, 5)]

#optimize
result = minimize(neg_log_likelihood, init_params, args=(responses,), bounds=bounds)
alpha_upd, beta_upd, lambda_upd = result.x
print(f"Updated parameters: alpha={alpha_upd:.3f}, beta={beta_upd:.3f}, lambda={lambda_upd:.3f}")

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
def prospect_value(x, alpha=alpha_upd, beta=beta_upd, lamb=lambda_upd):
    x = np.array(x)
    v = np.empty_like(x, dtype=float)
    # Apply value function for gains
    v[x >= 0] = x[x >= 0] ** alpha
    # Apply value function for losses using absolute values
    v[x < 0] = -lamb * (np.abs(x[x < 0]) ** beta)
    return v

#generate a range of outcomes
x = np.linspace(-10, 10, 500)
v = prospect_value(x)

#plot the Prospect Theory value function
plt.plot(x, v)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.title("Value Function with Loss Aversion")
plt.xlabel("Outcome (x)")
plt.ylabel("Value v(x)")
plt.grid(True)
plt.show()


