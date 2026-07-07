import numpy as np
import pandas as pd
from scipy.optimize import linprog

# -----------------------------
# 1. Data
# -----------------------------

divisions = ["Thanthirimale", "Pulmude", "Rambewa", "Tirappane", "Rajanganaya"]
crops = ["Black gram", "Sesame", "Big onion"]

land = np.array([2000, 2300, 600, 1100, 500], dtype=float)

division_water_limit = np.array([3200, 3400, 800, 500, 600], dtype=float)

# Water needed per acre
water_per_acre = np.array([1.6, 2.9, 3.5], dtype=float)

# Profit per acre
# Black gram: 50 bushels/acre * Rs.2000 = Rs.100000
# Sesame: 1.5 tons/acre * Rs.40000 = Rs.60000
# Big onion: 2.2 tons/acre * Rs.50000 = Rs.110000
profit_per_acre = np.array([100000, 60000, 110000], dtype=float)

# Maximum acres according to sales limits
# Black gram: 110000 bushels / 50 = 2200 acres
# Sesame: 1800 tons / 1.5 = 1200 acres
# Big onion: 2200 tons / 2.2 = 1000 acres
max_crop_acres = np.array([2200, 1200, 1000], dtype=float)

# Minimum sesame acres
# At least 800 tons sesame, 1 acre gives 1.5 tons
min_sesame_acres = 800 / 1.5


# -----------------------------
# 2. Function to solve LP
# -----------------------------

def solve_farming_lp(total_water_available):
    n_vars = 15   # 5 divisions * 3 crops

    # scipy solves minimization.
    # We need maximization, so use negative profit.
    c = -np.tile(profit_per_acre, 5)

    A = []
    b = []

    # Land constraints for each division
    for i in range(5):
        row = np.zeros(n_vars)
        row[3*i : 3*i+3] = 1
        A.append(row)
        b.append(land[i])

    # Water constraints for each division
    for i in range(5):
        row = np.zeros(n_vars)
        row[3*i : 3*i+3] = water_per_acre
        A.append(row)
        b.append(division_water_limit[i])

    # Total water constraint
    row = np.tile(water_per_acre, 5)
    A.append(row)
    b.append(total_water_available)

    # Crop sales constraints
    for crop_index in range(3):
        row = np.zeros(n_vars)
        for i in range(5):
            row[3*i + crop_index] = 1
        A.append(row)
        b.append(max_crop_acres[crop_index])

    # Minimum sesame constraint:
    # sum sesame >= 533.33
    # linprog accepts <= only, so write:
    # -sum sesame <= -533.33
    row = np.zeros(n_vars)
    for i in range(5):
        row[3*i + 1] = -1
    A.append(row)
    b.append(-min_sesame_acres)

    # Non-negativity
    bounds = [(0, None)] * n_vars

    result = linprog(
        c,
        A_ub=np.array(A),
        b_ub=np.array(b),
        bounds=bounds,
        method="highs"
    )

    return result


# -----------------------------
# 3. Solve original problem
# -----------------------------

original_result = solve_farming_lp(7400)

if original_result.success:
    original_profit = -original_result.fun
    original_plan = original_result.x.reshape(5, 3)

    print("Original maximum profit:", original_profit)

    original_df = pd.DataFrame(
        original_plan,
        index=divisions,
        columns=crops
    )

    print("\nOriginal cultivation plan:")
    print(original_df)

else:
    print("Original LP did not solve:", original_result.message)


# -----------------------------
# 4. Solve with additional water
# -----------------------------

extra_result = solve_farming_lp(8000)  # 7400 + 600

if extra_result.success:
    extra_profit_before_fee = -extra_result.fun
    extra_plan = extra_result.x.reshape(5, 3)

    print("\nProfit with extra water before fee:", extra_profit_before_fee)

    extra_df = pd.DataFrame(
        extra_plan,
        index=divisions,
        columns=crops
    )

    print("\nCultivation plan with extra water:")
    print(extra_df)

    # Extra water fee
    water_fee = 6000000

    extra_gross_profit = extra_profit_before_fee - original_profit
    net_extra_profit = extra_gross_profit - water_fee

    print("\nExtra gross profit:", extra_gross_profit)
    print("Additional water fee:", water_fee)
    print("Net extra profit:", net_extra_profit)

    if net_extra_profit > 0:
        print("\nRecommendation: Buy the additional water.")
    else:
        print("\nRecommendation: Do not buy the additional water.")

else:
    print("Extra-water LP did not solve:", extra_result.message)