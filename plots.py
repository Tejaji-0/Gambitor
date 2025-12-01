import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Section 1: Energy System Analysis
# -------------------------------
# Mars solar irradiance and power generation

mars_solar_irradiance = 590  # W/m^2 (43% of Earth's ~1361 W/m^2)
solar_panel_efficiency = 0.30  # 30% efficiency
solar_farm_area = 5000  # m^2

solar_power_generated = mars_solar_irradiance * solar_panel_efficiency * solar_farm_area  # Watts
nuclear_backup = 5000000  # 5 MW backup

total_power = solar_power_generated + nuclear_backup  # Total available power

print(f"Solar Power Generated: {solar_power_generated/1000:.2f} kW")
print(f"Total Power (including nuclear backup): {total_power/1000:.2f} kW")

# Plot Energy Contribution
labels = ['Solar', 'Nuclear Backup']
power_values = [solar_power_generated, nuclear_backup]

plt.figure(figsize=(6,6))
plt.pie(power_values, labels=labels, autopct='%1.1f%%', startangle=140, colors=['orange','gray'])
plt.title('Energy Contribution in Mars Colony (Watts)')
plt.show()

# -------------------------------
# Section 2: Hydroponic Farming Yield
# -------------------------------
# Parameters
greenhouse_area = 1000  # m^2
yield_per_m2 = 18  # kg per m^2 per year
water_saving = 0.9  # 90% less water than soil

total_yield = greenhouse_area * yield_per_m2  # kg per year

print(f"Hydroponic Yield: {total_yield} kg/year")
print(f"Water Saving Compared to Soil Farming: {water_saving*100}%")

# Plot Hydroponic Yield
plt.figure(figsize=(8,5))
plt.bar(['Hydroponic'], [total_yield], color='green')
plt.ylabel('Vegetable Yield (kg/year)')
plt.title('Annual Hydroponic Yield for Mars Colony')
plt.show()

# -------------------------------
# Section 3: Break-Even Analysis
# -------------------------------
# Parameters
chip_price_per_unit = 5000  # USD per photonic chip
chips_produced_per_year = 2000
operational_cost_per_year = 7000000  # USD

total_revenue = chip_price_per_unit * chips_produced_per_year
break_even = total_revenue - operational_cost_per_year

print(f"Total Revenue: ${total_revenue}")
print(f"Operational Cost: ${operational_cost_per_year}")
print(f"Break-Even Margin: ${break_even}")

# Plot Revenue vs Operational Cost
labels = ['Revenue', 'Operational Cost']
values = [total_revenue, operational_cost_per_year]

plt.figure(figsize=(8,5))
plt.bar(labels, values, color=['blue','red'])
plt.ylabel('USD')
plt.title('Revenue vs Operational Cost for Photonic Chips')
plt.show()

days = np.arange(0, 30, 1)  # 30-day month
dust_storm_probability = 0.3  # 30% chance of dust storm each day
solar_output = solar_power_generated * (1 - dust_storm_probability * np.random.rand(len(days)))

plt.figure(figsize=(10,5))
plt.plot(days, solar_output/1000, marker='o', color='orange')
plt.xlabel('Day')
plt.ylabel('Solar Output (kW)')
plt.title('Daily Solar Output During Dust Storms Simulation')
plt.grid(True)
plt.show()

# -------------------------------
# Section 5: Hydroponic Water Use Comparison
# -------------------------------
soil_water_use = greenhouse_area * yield_per_m2 * 10  # assume 10 liters/kg in soil
hydro_water_use = soil_water_use * (1 - water_saving)

plt.figure(figsize=(8,5))
plt.bar(['Soil Farming', 'Hydroponics'], [soil_water_use, hydro_water_use], color=['brown','green'])
plt.ylabel('Water Used (liters/year)')
plt.title('Water Usage: Soil vs Hydroponic Farming')
plt.show()
