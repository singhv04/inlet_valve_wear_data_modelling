# generate_inlet_valve_data.py (Final Uniform Wear Distribution Fix)

import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# ------------------------------
# Constants and Lookup Tables
# ------------------------------

VALVE_MATERIALS = ['EMS200', 'X45CrSi9-3', '21-4N']
INSERT_MATERIALS = ['Ni-resist', 'Stellite', 'Inconel']
COATINGS = {
    'None': 1.0,
    'Chromium Nitride': 0.9,
    'Titanium Nitride': 0.85,
    'Thermal Spray': 0.92
}

HARDNESS_LOOKUP = {
    ('EMS200', 'Chromium Nitride'): 45,
    ('X45CrSi9-3', 'Titanium Nitride'): 48,
    ('21-4N', 'Thermal Spray'): 50,
    ('EMS200', 'None'): 35,
    ('21-4N', 'None'): 38
}

MATERIAL_MODIFIERS = {
    'EMS200': 1.0,
    'X45CrSi9-3': 0.95,
    '21-4N': 1.05
}

# ------------------------------
# Utility Functions
# ------------------------------

def derive_temperature(rpm, pressure):
    base_temp = 350 + 0.015 * rpm + 0.1 * pressure
    return base_temp + np.random.normal(0, 2)

def derive_velocity(rpm, diameter):
    stroke = 0.1 * diameter
    cam_ratio = 0.6
    velocity = stroke * rpm * cam_ratio / 1000
    return velocity + np.random.normal(0, 0.02 * velocity)

def compute_mismatch(seat_angle, insert_angle):
    return abs(seat_angle - insert_angle)

def compute_coeff_modifier(coating, lubrication_index, mismatch):
    base = COATINGS.get(coating, 1.0)
    modifier = base * (1 + 0.2 * (1 - lubrication_index)) * (1 + mismatch * 0.03)
    return modifier

def derive_hardness(valve_material, coating):
    base = HARDNESS_LOOKUP.get((valve_material, coating), 40)
    return base + np.random.normal(0, 1)

def compute_wear(pressure, velocity, time, temp_mult, material_mod, mismatch, hardness, width, coeff_mod):
    k1 = 0.5
    numerator = pressure * velocity * time * temp_mult * material_mod * (1 + mismatch * k1)
    denominator = hardness * width * coeff_mod
    wear = numerator / denominator
    return wear

# ------------------------------
# Main Data Generator Function
# ------------------------------

def generate_synthetic_data(n_samples=100_000, wear_bins=10):
    """
    Generates realistic synthetic inlet valve data with a uniform distribution of target `wear` values.

    Parameters:
        n_samples (int): Total number of samples to generate
        wear_bins (int): Number of bins (uniform wear levels)

    Returns:
        DataFrame: Synthetic dataset with engineered features and target
    """
    data = []
    samples_per_bin = n_samples // wear_bins
    wear_targets = np.linspace(0.3, 2.9, wear_bins)

    for wear_target in tqdm(wear_targets, desc="Generating bins"):
        for _ in range(samples_per_bin):

            # -----------------------
            # Param Sampling (mapped to wear range)
            # -----------------------
            scale = wear_target / 3.0  # scale factor from 0.1 to 1.0

            pressure = np.random.normal(150 + 100 * scale, 5)
            rpm = np.random.normal(800 + 1600 * scale, 30)
            seat_angle = np.random.uniform(25, 45)
            insert_angle = seat_angle + np.random.uniform(-1, 1)
            diameter = np.random.uniform(35, 55)
            duration = np.random.normal(200 + 2800 * scale, 30)
            lubrication_index = np.random.normal(1.0 - scale * 0.5, 0.02)
            face_width = np.random.normal(3.5 - scale * 2.0, 0.05)

            valve_material = random.choice(VALVE_MATERIALS)
            insert_material = random.choice(INSERT_MATERIALS)
            coating = random.choice(list(COATINGS.keys()))

            # Derived features
            temperature = derive_temperature(rpm, pressure)
            velocity = derive_velocity(rpm, diameter)
            mismatch = compute_mismatch(seat_angle, insert_angle)
            coeff_mod = compute_coeff_modifier(coating, lubrication_index, mismatch)
            hardness = derive_hardness(valve_material, coating)
            material_mod = MATERIAL_MODIFIERS.get(valve_material, 1.0)
            temp_mult = 1 + (temperature - 350) / 1000

            # Raw wear value
            raw_wear = compute_wear(
                pressure, velocity, duration, temp_mult,
                material_mod, mismatch, hardness,
                face_width, coeff_mod
            )

            # Normalize to target
            scaling = wear_target / (raw_wear + 1e-6)
            wear = raw_wear * scaling + np.random.normal(0, 0.015)

            wear = max(0.1, min(wear, 3.0))  # Clamp to bounds

            data.append({
                'pressure': pressure,
                'rpm': rpm,
                'temperature': temperature,
                'seat_angle': seat_angle,
                'insert_angle': insert_angle,
                'mismatch': mismatch,
                'valve_material': valve_material,
                'insert_material': insert_material,
                'coating': coating,
                'hardness': hardness,
                'diameter': diameter,
                'velocity': velocity,
                'duration': duration,
                'face_width': face_width,
                'lubrication_index': lubrication_index,
                'coeff_mod': coeff_mod,
                'wear': wear
            })

    return pd.DataFrame(data)

# ------------------------------
# Execution Block
# ------------------------------

if __name__ == '__main__':
    df = generate_synthetic_data(n_samples=100000)
    df.to_csv("data/synthetic_inlet_valve_data.csv", index=False)

    print("✅ Synthetic inlet valve data generated and saved.")

    # Summary
    print("\nWear Distribution Summary:")
    print(df['wear'].describe())

    print("\nVariable Summary:")
    print(df.describe().T[['min', 'max', 'std']])

    # Plot histogram
    plt.figure(figsize=(10, 4))
    df['wear'].hist(bins=60)
    plt.title("Wear Value Distribution")
    plt.xlabel("Wear")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/inlet_wear_distribution_histogram.png")
    plt.show()
    print("📊 Histogram saved as inlet_wear_distribution_histogram.png")
