import energyflow
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import math
import awkward as ak
import vector
from particle import Particle

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split

from terminal_colors import tcols

parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=105000, help="Total number of data samples to use")
parser.add_argument('--part_dist', type=bool, default=True, help="Sample according to the distribution of number of particles in the jets")
parser.add_argument('--filter_outliers', type=bool, default=False, help="Whether or not to filter out the outliers in terms of number of particles")
parser.add_argument('--subsampled_data_path', type=str, default='aux_data/subsampled_data', help="Path in which the subsampled data is saved")
parser.add_argument('--augmented_data_path', type=str, default='aux_data/augmented_data', help="Path in which the augmented data is saved")
parser.add_argument('--outdir', type=str, default='../data', help="Path to save the graph data for the autoencoder")
parser.add_argument('--train_samples', type=float, default=50000, help="Number of data samples to use for training")
parser.add_argument('--valid_samples', type=float, default=5000, help="Number of data samples to use for validation")
parser.add_argument('--test_samples', type=float, default=50000, help="Number of data samples to use for testing")
parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
parser.add_argument('--normalized_data_path', type=str, default='aux_data/normalized_data', help="Path in which the normalized data is saved")
parser.add_argument('--norm_name', type=str, default='std', help="Choose which way to normalise the data. Available options: vanilla (no norm), minmax," 
                    "maxabs, std, robust, power, quantile.")




def main():

    X, y = load_and_subsample_data(args.num_samples, args.part_dist, args.filter_outliers, args.subsampled_data_path)

    X, y = augment_features(X,y,args.num_samples, args.part_dist, args.filter_outliers, args.augmented_data_path)

    train_X, train_y, valid_X, valid_y, test_X, test_y = split_data(X, y, args.num_samples, args.train_samples, args.valid_samples, args.test_samples, args.seed)

    scaler = choose_norm(args.norm_name)
    train_X, valid_X, test_X = normalize_features(scaler, train_X, valid_X, test_X)

    #transform to graph format
    full_graph_data_path = args.outdir + "/graphdata_" + str(args.num_samples) + "_train" + str(args.train_samples) + "_valid" + str(args.valid_samples) + "_test" + str(args.test_samples)
    if args.part_dist:
        full_graph_data_path += "_part_dist"
    if args.filter_outliers:
        full_graph_data_path += "_filtered"
    
    full_graph_data_path += "_" + args.norm_name

    print(full_graph_data_path)

    format_to_graph_and_save(train_X, train_y, valid_X, valid_y, test_X, test_y, full_graph_data_path)


def load_and_subsample_data(num_samples, part_dist = True, filter_outliers=False, subsampled_data_path = 'aux_data/subsampled_data'):
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
        print("Data loaded and subsampled from cache successfully!")
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
    
    print("Data loaded and subsampled successfully!")

    return sampled_X_all, sampled_y_all

def get_features(X, y):

    # Step 1: Find the maximum number of rows
    max_rows = max(arr.shape[0] for arr in X)

    # Step 2: Pad each array
    padded_arrays = []
    for arr in X:
        # Calculate the padding needed
        padding_needed = max_rows - arr.shape[0]
        # Pad with zeros on the bottom of the array
        padded_array = np.pad(arr, ((0, padding_needed), (0, 0)), mode='constant', constant_values=0)
        padded_arrays.append(padded_array)

    # Step 3: Stack into a 3D array
    X = np.stack(padded_arrays).astype(np.float32)
    y = y.astype(int)
    origPT = X[:, :, 0]
    indices = np.argsort(-origPT, axis=1)

    _pt = np.take_along_axis(X[:, :, 0], indices, axis=1)
    _eta = np.take_along_axis(X[:, :, 1], indices, axis=1)
    _phi = np.take_along_axis(X[:, :, 2], indices, axis=1)
    _pid = np.take_along_axis(X[:, :, 3], indices, axis=1)

    mask = _pt > 0
    n_particles = np.sum(mask, axis=1)


    pt = ak.Array(_pt[mask])
    eta = ak.Array(_eta[mask])
    phi = ak.Array(_phi[mask])

   
    PID = ak.Array(_pid[mask])
    # To create an array of zeros with the same structure as pt
    #mass = ak.zeros_like(pt)
    mass = ak.Array([Particle.from_pdgid(pid).mass / 1000 if Particle.from_pdgid(pid) else 0 for pid in PID])
    

    pt = ak.unflatten(pt, n_particles)
    eta = ak.unflatten(eta, n_particles)
    phi = ak.unflatten(phi, n_particles)
    mass = ak.unflatten(mass, n_particles)
    PID = ak.unflatten(PID, n_particles)

    #p4 = TLorentzVectorArray.from_ptetaphim(pt, eta, phi, mass)
    p4 = vector.awk(ak.zip({
        "pt": pt,
        "eta": eta,
        "phi": phi,
        "mass": mass
    }, with_name="Momentum4D"))
    px = p4.x
    py = p4.y
    pz = p4.z
    energy = p4.energy


    jet_p4 = ak.sum(p4, axis=1)

    # outputs
    v = {}
    v['label'] = y

    v['jet_pt'] = jet_p4.pt
    v['jet_eta'] = jet_p4.eta
    v['jet_phi'] = jet_p4.phi
    v['jet_energy'] = jet_p4.energy
    v['jet_mass'] = jet_p4.mass
    v['jet_nparticles'] = n_particles

    v['part_px'] = px
    v['part_py'] = py
    v['part_pz'] = pz
    v['part_energy'] = energy

    _jet_etasign = np.sign(v['jet_eta'])
    #_jet_etasign[_jet_etasign == 0] = 1
    _jet_etasign = np.where(_jet_etasign == 0, 1, _jet_etasign)

    v['part_deta'] = (p4.eta - v['jet_eta']) * _jet_etasign
    v['part_dphi'] = p4.deltaphi(jet_p4)

    v['part_isCHPlus'] = ak.values_astype((PID == 211) + (PID == 321) * 0.5 + (PID == 2212) * 0.2, np.float32)
    v['part_isCHMinus'] = ak.values_astype((PID == -211) + (PID == -321) * 0.5 + (PID == -2212) * 0.2, np.float32)
    v['part_isNeutralHadron'] = ak.values_astype((PID == 130) + (PID == 2112) * 0.2 + (PID == -2112) * 0.2, np.float32)
    v['part_isPhoton'] = ak.values_astype(PID == 22, np.float32)
    v['part_isEPlus'] = ak.values_astype(PID == -11, np.float32)
    v['part_isEMinus'] = ak.values_astype(PID == 11, np.float32)
    v['part_isMuPlus'] = ak.values_astype(PID == -13, np.float32)
    v['part_isMuMinus'] = ak.values_astype(PID == 13, np.float32)


    v['part_isChargedHadron'] = v['part_isCHPlus'] + v['part_isCHMinus']
    v['part_isElectron'] = v['part_isEPlus'] + v['part_isEMinus']
    v['part_isMuon'] = v['part_isMuPlus'] + v['part_isMuMinus']

    v['part_charge'] = (v['part_isCHPlus'] + v['part_isEPlus'] + v['part_isMuPlus']
                        ) - (v['part_isCHMinus'] + v['part_isEMinus'] + v['part_isMuMinus'])
    
    v['part_mask'] = ak.ones_like(v['part_deta'])
    v['part_pt'] = np.hypot(v['part_px'], v['part_py'])
    v['part_pt_log'] = np.log(v['part_pt'])
    v['part_e_log'] = np.log(v['part_energy'])
    v['part_logptrel']  = np.log(v['part_pt']/v['jet_pt'])
    v['part_logerel'] = np.log(v['part_energy']/v['jet_energy'])
    v['part_deltaR'] = np.hypot(v['part_deta'], v['part_dphi'])

    
    v['part_eta'] = eta
    v['part_phi'] = phi

    
    for k in list(v.keys()):
        if k.endswith('Plus') or k.endswith('Minus'):
            del v[k]

    return v


def augment_features(X,y,num_samples=10000, part_dist = True, filter_outliers=False, augmented_data_path = "aux_data/augmented_data"):

    # Ensure the directory exists
    if not os.path.exists(augmented_data_path):
        os.makedirs(augmented_data_path, exist_ok=True)

    X_path = os.path.join(augmented_data_path, 'X_' + str(num_samples))
    y_path = os.path.join(augmented_data_path, 'y_' + str(num_samples))

    if part_dist:
        X_path += "_part_dist"
        y_path += "_part_dist"
    if filter_outliers:
        X_path += "_filtered"
        y_path += "_filtered"
    
    X_path += ".npy"
    y_path += ".npy"

    if os.path.exists(X_path) and os.path.exists(y_path):
        jets_array = np.load(X_path, allow_pickle=True)
        label_array = np.load(y_path, allow_pickle=True)
        print("Data augmented from cache successfully!")
        return jets_array, label_array

    data_dict = get_features(X,y)
    # List to hold structured jets information as 2D arrays
    jets_list = []
    # Number of jets - Assuming 'jet_pt' exists and is a good proxy for counting jets
    num_jets = len(data_dict['jet_pt'])
    label_array = data_dict['label']

    relevant_features = [
                'part_deta', 'part_dphi', 'part_pt_log', 'part_e_log', 'part_logptrel', 
                'part_logerel', 'part_deltaR', 'part_charge', 'part_isElectron', 
                'part_isMuon', 'part_isPhoton', 'part_isChargedHadron', 'part_isNeutralHadron'
            ]
    # Loop through each jet
    for i in range(num_jets):
        # Initialize a list to store particles' data temporarily
        particles_data = []
        # Determine the number of particles in the current jet for the first particle-specific feature
        # Assuming 'part_deta' exists and all particle arrays are aligned in length
        particles_data = np.array([data_dict[key][i] for key in relevant_features]).T

        # Convert list of particle features to a 2D NumPy array and append to the jets list
        jets_list.append(particles_data)
    
    jets_array = np.array(jets_list, dtype=object)

    np.save(X_path, jets_array)
    np.save(y_path, label_array)

    print("Data augmented successfully!")
    
    return jets_array, label_array 

def split_data(X, y, num_samples, train_samples, valid_samples, test_samples, random_seed=42):

    assert train_samples+valid_samples+test_samples==num_samples

    X_train_val, X_test, y_train_val, y_test = train_test_split(X,y, test_size=test_samples, random_state=random_seed, stratify=y)

    #X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, test_size=valid_size+test_size, random_state=random_seed, stratify=y)
    #new_test_size = round(test_size/(1-train_size),2)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_val,y_train_val, test_size=valid_samples, random_state=random_seed, stratify=y_train_val)

    #X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, test_size=new_test_size, random_state=random_seed, stratify=y_valid_test)

    print("Data split successfully!")

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def choose_norm(norm_name):
    """
    Normalise the mlready data.
    @norm_name :: Name of the normalisation applied, chosen by user.
    """
    print(tcols.OKBLUE)
    if norm_name == "vanilla":
        print("\nNo normalization...")
        return None
    elif norm_name == "minmax":
        print("\nApplying minmax normalization...")
        return MinMaxScaler()
    elif norm_name == "maxabs":
        print("\nApplying maxabs normalization...")
        return MaxAbsScaler()
    elif norm_name == "std":
        print("\nApplying standard normalization...")
        return StandardScaler()
    elif norm_name == "robust":
        print("\nApplying robust normalization...")
        return RobustScaler()
    elif norm_name == "power":
        print("\nApplying power normalization...")
        return PowerTransformer()
    elif norm_name == "quantile":
        return QuantileTransformer()
    else:
        print(tcols.ENDC)
        raise TypeError("Specified normalisation type does not exist!!!")

    print(tcols.ENDC)

def normalize_features(scaler, X_train, X_valid, X_test):

    if scaler is None:
        return X_train, X_valid, X_test

    X_train_particles = np.concatenate(X_train, axis=0)

    #print(X_train_particles)
    subset_features = X_train_particles[:, 2:7]

    scaler.fit(subset_features)

    for jet in X_train:
        jet[:, 2:7] = scaler.transform(jet[:,2:7])

    for jet in X_valid:
        jet[:,2:7] = scaler.transform(jet[:,2:7])

    for jet in X_test:
        jet[:,2:7] = scaler.transform(jet[:,2:7])

    
    print("Data normalized successfully!")


    return X_train, X_valid, X_test

def format_to_graph_and_save(train_X, train_y, valid_X, valid_y, test_X, test_y, data_path):
    def calculate_edge_feature(particle1, particle2):
        """Calculate edge feature based on rapidity and azimuthal angle."""
        delta_rapidity = abs(particle1[1] - particle2[1])
        delta_azimuthal = abs(particle1[2] - particle2[2])
        return math.sqrt(delta_rapidity**2 + delta_azimuthal**2)
    
    def get_graph_data(X,y, data_path, prefix):
        full_data_path = data_path + "/" + prefix + "/" + prefix + "/raw/"  #format for graphAE
        # Ensure the directory exists
        if not os.path.exists(full_data_path):
            os.makedirs(full_data_path, exist_ok=True)
        else:
            return
        
        DS_A = []
        DS_graph_indicator = []
        DS_graph_labels = []
        DS_node_attributes = []
        DS_edge_attributes = []

        global_node_index = 1
        global_graph_index = 1

        # Second pass to process jets and scale attributes
        for i, jet in enumerate(X):  # Again, change the slice as needed
            graph_label = int(y[i])

            for j, particle in enumerate(jet):
                features = particle
                DS_node_attributes.append(','.join(map(str, features)))
                DS_graph_indicator.append(global_graph_index)

                for k in range(j + 1, len(jet)):
                    edge_feature = calculate_edge_feature(particle, jet[k])
                    DS_A.append((global_node_index + j, global_node_index + k))
                    DS_A.append((global_node_index + k, global_node_index + j))
                    DS_edge_attributes.append(edge_feature)
                    DS_edge_attributes.append(edge_feature)

            global_node_index += len(jet)
            DS_graph_labels.append(graph_label)
            global_graph_index += 1

        print(len(DS_A))
        # Save files

        with open(full_data_path + prefix + '_A.txt', 'w') as f:
            for entry in DS_A:
            #for i in range(len(DS_A)):
                f.write(f'{entry[0]},{entry[1]}\n')
                #f.write(f'{DS_A[i][0]},{DS_A[i][1]}\n')
                #DS_A[i] = None  # Overwrite entry to free memory
            #while DS_A:
            #    entry = DS_A.pop(0)
            #    f.write(f'{entry[0]},{entry[1]}\n')
        del DS_A
        with open(full_data_path + prefix + '_graph_indicator.txt', 'w') as f:
            for entry in DS_graph_indicator:
                f.write(f'{entry}\n')
        del DS_graph_indicator
        with open(full_data_path + prefix + '_graph_labels.txt', 'w') as f:
            for label in DS_graph_labels:
                f.write(f'{label}\n')
        del DS_graph_labels
        with open(full_data_path + prefix + '_node_attributes.txt', 'w') as f:
            for attributes in DS_node_attributes:
                f.write(f'{attributes}\n')
        del DS_node_attributes
        with open(full_data_path + prefix + '_edge_attributes.txt', 'w') as f:
            for attribute in DS_edge_attributes:
                f.write(f'{attribute}\n')
        del DS_edge_attributes

        print("Saved " + prefix + " graph data successfully!")
    
    get_graph_data(train_X, train_y, data_path, "train")
    del train_X, train_y
    get_graph_data(valid_X, valid_y, data_path, "valid")
    del valid_X, valid_y
    get_graph_data(test_X, test_y, data_path, "test")
    del test_X, test_y

    print("Data formatted to graph format and saved successfully!")


    return

if __name__ == "__main__":
    args = parser.parse_args()
    main()