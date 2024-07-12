import numpy as np
import matplotlib.pyplot as plt
import energyflow

# Enable LaTeX rendering
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

# Load data
X, y = energyflow.qg_jets.load(num_data=2000000, pad=False, ncol=4, generator='pythia', with_bc=False, cache_dir='energyflow')

# Count the number of particles in each jet
particle_counts = [len(jet) for jet in X]

print(particle_counts)

# Calculate percentiles and IQR for identifying outliers
Q1 = np.percentile(particle_counts, 25)
Q3 = np.percentile(particle_counts, 75)
IQR = Q3 - Q1

# Calculate max and min number of particles
max_particles = max(particle_counts)
min_particles = min(particle_counts)

# Calculate outlier thresholds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Print statistics
print(f"Q1 (25th percentile): {Q1}")
print(f"Q3 (75th percentile): {Q3}")
print(f"IQR (Interquartile Range): {IQR}")
print(f"Maximum number of particles: {max_particles}")
print(f"Minimum number of particles: {min_particles}")
print(f"Lower bound for outliers: {lower_bound}")
print(f"Upper bound for outliers: {upper_bound}")

print(f"Original dataset size: {len(X)}")

# Plotting the original and filtered distributions
#plt.figure(figsize=(12, 6))
#num_different_particles = len(set(particle_counts))
#bins = np.arange(0.5, num_different_particles+0.5,1)
#plt.hist(particle_counts, bins=bins, color='blue', alpha=0.7)
#plt.xlabel("Number of Particles per jet")
#plt.ylabel("Frequency")
bins = range(min_particles, max_particles+2)

plt.hist(particle_counts, bins=bins,align='left')
plt.xlabel('Number of Particles')
plt.ylabel('Frequency')


plt.tight_layout()
plt.show()