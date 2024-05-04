import energyflow
import numpy as np
import matplotlib.pyplot as plt
import os


def load_data(num_samples=10000, part_dist = True, filter_outliers=False, subsampled_data_path = 'subsampled_data'):
    assert num_samples <= 2000000
 
    # Ensure the directory exists
    if not os.path.exists(subsampled_data_path):
        os.makedirs(subsampled_data_path, exist_ok=True)

    X_path = os.path.join(subsampled_data_path, 'X_' + str(num_samples))
    y_path = os.path.join(subsampled_data_path, 'y_' + str(num_samples))

    if part_dist:
        X_path += "_part_dist"
        y_path += "_part_dist"
    if filter_outliers:
        X_path += "_filtered"
        y_path += "_filtered"
    
    X_path += ".npy"
    y_path += ".npy"

    if os.path.exists(X_path) and os.path.exists(y_path):
        sampled_X = np.load(X_path, allow_pickle=True)
        sampled_y = np.load(y_path, allow_pickle=True)
        return sampled_X, sampled_y

    X_all, y_all = energyflow.qg_jets.load(num_data=2000000, pad=False, ncol=4, generator='pythia', with_bc=False, cache_dir='energyflow')
    
    if filter_outliers:
        
        particles_per_jet = np.array([len(jet) for jet in X_all])
        # Calculate percentiles and IQR for identifying outliers
        Q1 = np.percentile(particles_per_jet, 25)
        Q3 = np.percentile(particles_per_jet, 75)
        IQR = Q3 - Q1

        # Calculate outlier thresholds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out jets that are outliers
        filtered_indices = [i for i, count in enumerate(particles_per_jet) if lower_bound <= count <= upper_bound]
        X_all = X_all[filtered_indices]
        y_all = y_all[filtered_indices]


    if part_dist:
        sampled_X_all = []
        sampled_y_all = []
        jet_types = np.unique(y_all)
        num_jet_types = len(jet_types)
        for jet_type in jet_types:
            X = X_all[y_all==jet_type]
    
            particles_per_jet = np.array([len(jet) for jet in X])
            # Step 2: Find unique numbers of particles and the number of jets with that many particles
            unique_particles, counts = np.unique(particles_per_jet, return_counts=True)

            #print(unique_particles)
            #print(counts)

            # Step 3: Calculate the proportion of each group
            total_jets = len(X)
            proportions = counts / total_jets

            # Step 4: Sample jets from each group proportionally
            sampled_indices = []
            for num_particles, proportion in zip(unique_particles, proportions):
                
                # Number of jets to sample from this group
                num_to_sample = int(proportion * num_samples//num_jet_types) + 1
                
                # Get indices of all jets with 'num_particles' particles
                indices = np.where(particles_per_jet == num_particles)[0]
                
                # Randomly sample 'num_samples' indices from these indices
                sampled_indices.extend(np.random.choice(indices, num_to_sample, replace=False))
    
            # Since proportional sampling might not add up exactly to 10000 due to rounding, adjust if necessary
            sampled_indices = np.random.choice(sampled_indices, num_samples//num_jet_types, replace=False)

            # Get the sampled jets and their labels
            #sampled_X = [X[i] for i in sampled_indices]
            sampled_X = X[sampled_indices]
            sampled_y = np.full(num_samples//num_jet_types, jet_type)

            sampled_X_all.append(sampled_X)
            sampled_y_all.append(sampled_y)
        
        sampled_X_all = np.concatenate(sampled_X_all)
        sampled_y_all = np.concatenate(sampled_y_all)


    
    else:
        sampled_X_all = np.array()
        sampled_y_all = np.array()
        jet_types = np.unique(y_all)
        num_jet_types = len(jet_types)
        for jet_type in jet_types:
            X = X_all[y_all==jet_type]
            indices = np.random.choice(X.shape[0], num_samples//num_jet_types, replace=False)
            sampled_X = X[indices]
            sampled_y = np.full(num_samples//num_jet_types, jet_type)

            sampled_X_all.append(sampled_X)
            sampled_y_all.append(sampled_y)
        
        sampled_X_all = np.concatenate(sampled_X_all)
        sampled_y_all = np.concatenate(sampled_y_all)


    np.save(X_path, sampled_X_all)
    np.save(y_path, sampled_y_all)
    

    return sampled_X_all, sampled_y_all

def plot_hist(X):
    particle_counts = [len(jet) for jet in X]
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    num_different_particles = len(set(particle_counts))
    bins = np.arange(0.5, num_different_particles+0.5,1)    
    plt.hist(particle_counts, bins=bins, color='blue', alpha=0.7)
    plt.title("Original Distribution of Particles per Jet")
    plt.xlabel("Number of Particles")
    plt.ylabel("Frequency")
    plt.show()