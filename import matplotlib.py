import matplotlib.pyplot as plt
import pandas as pd

# Sample data for creating tables and plots based on the description
data_crossing = {
    "Obstacle Height (%)": [10, 20, 30],
    "Crossing Speed (MCI)": [1.1, 1.0, 0.9],
    "Crossing Speed (Control)": [1.3, 1.2, 1.1],
    "Trailing Toe-Obstacle Clearance (MCI)": [0.25, 0.28, 0.30],
    "Trailing Toe-Obstacle Clearance (Control)": [0.27, 0.30, 0.33],
    "Contralateral SLS Time (MCI)": [0.6, 0.65, 0.7],
    "Contralateral SLS Time (Control)": [0.55, 0.6, 0.65]
}

df_crossing = pd.DataFrame(data_crossing)

# Plot 1: Crossing Speed vs Obstacle Height
plt.figure(figsize=(8, 6))
plt.plot(df_crossing["Obstacle Height (%)"], df_crossing["Crossing Speed (MCI)"], label="MCI", marker='o')
plt.plot(df_crossing["Obstacle Height (%)"], df_crossing["Crossing Speed (Control)"], label="Control", marker='s')
plt.xlabel("Obstacle Height (%)")
plt.ylabel("Crossing Speed (m/s)")
plt.title("Crossing Speed vs Obstacle Height")
plt.legend()
plt.grid()
plt.show()

# Plot 2: Trailing Toe-Obstacle Clearance vs Obstacle Height
plt.figure(figsize=(8, 6))
plt.plot(df_crossing["Obstacle Height (%)"], df_crossing["Trailing Toe-Obstacle Clearance (MCI)"], label="MCI", marker='o')
plt.plot(df_crossing["Obstacle Height (%)"], df_crossing["Trailing Toe-Obstacle Clearance (Control)"], label="Control", marker='s')
plt.xlabel("Obstacle Height (%)")
plt.ylabel("Trailing Toe-Obstacle Clearance (m)")
plt.title("Trailing Toe-Obstacle Clearance vs Obstacle Height")
plt.legend()
plt.grid()
plt.show()

# Plot 3: Contralateral SLS Time vs Obstacle Height
plt.figure(figsize=(8, 6))
plt.plot(df_crossing["Obstacle Height (%)"], df_crossing["Contralateral SLS Time (MCI)"], label="MCI", marker='o')
plt.plot(df_crossing["Obstacle Height (%)"], df_crossing["Contralateral SLS Time (Control)"], label="Control", marker='s')
plt.xlabel("Obstacle Height (%)")
plt.ylabel("Contralateral SLS Time (s)")
plt.title("Contralateral SLS Time vs Obstacle Height")
plt.legend()
plt.grid()
plt.show()

# Display table to user
print("\nCrossing Parameters vs Obstacle Height:")
print(df_crossing.to_string(index=False))
