import numpy as np
import awkward as ak
import vector
from particle import Particle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math

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


def augment_features(X,y,num_samples=10000, part_dist = True, filter_outliers=False, augmented_data_path = "augmented_data"):

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

    return jets_array, label_array 


def split_data(X, y, train_size=0.8, valid_size=0.1, test_size=0.1, random_seed=42):
    assert train_size+valid_size+test_size==1.0

    X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, test_size=valid_size+test_size, random_state=random_seed, stratify=y)

    new_test_size = round(test_size/(1-train_size),2)

    X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, test_size=new_test_size, random_state=random_seed, stratify=y_valid_test)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def standardize_features(X_train, X_valid, X_test):

    X_train_particles = np.concatenate(X_train, axis=0)

    #print(X_train_particles)
    subset_features = X_train_particles[:, 2:7]

    scaler = StandardScaler()
    scaler.fit(subset_features)

    for jet in X_train:
        jet[:, 2:7] = scaler.transform(jet[:,2:7])

    for jet in X_valid:
        jet[:,2:7] = scaler.transform(jet[:,2:7])

    for jet in X_test:
        jet[:,2:7] = scaler.transform(jet[:,2:7])


    return X_train, X_valid, X_test


def format_to_graph(train_X, train_y, valid_X, valid_y, test_X, test_y, data_path):
    def calculate_edge_feature(particle1, particle2):
        """Calculate edge feature based on rapidity and azimuthal angle."""
        delta_rapidity = abs(particle1[1] - particle2[1])
        delta_azimuthal = abs(particle1[2] - particle2[2])
        return math.sqrt(delta_rapidity**2 + delta_azimuthal**2)
    
    def get_graph_data(X,y, data_path, prefix):
        full_data_path = data_path + "/" + prefix + "/"
        # Ensure the directory exists
        if not os.path.exists(full_data_path):
            os.makedirs(full_data_path, exist_ok=True)
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

        # Save files
        with open(full_data_path + prefix + '_A.txt', 'w') as f:
            for entry in DS_A:
                f.write(f'{entry[0]},{entry[1]}\n')
        with open(full_data_path + prefix + '_graph_indicator.txt', 'w') as f:
            for entry in DS_graph_indicator:
                f.write(f'{entry}\n')
        with open(full_data_path + prefix + '_graph_labels.txt', 'w') as f:
            for label in DS_graph_labels:
                f.write(f'{label}\n')
        with open(full_data_path + prefix + '_node_attributes.txt', 'w') as f:
            for attributes in DS_node_attributes:
                f.write(f'{attributes}\n')
        with open(full_data_path + prefix + '_edge_attributes.txt', 'w') as f:
            for attribute in DS_edge_attributes:
                f.write(f'{attribute}\n')
    
    get_graph_data(train_X, train_y, data_path, "train")
    get_graph_data(valid_X, valid_y, data_path, "valid")
    get_graph_data(test_X, test_y, data_path, "test")


    return
    

    