
import os
from autoencoders import util as ae_util
from classical import util as classical_util
from autoencoders import data as ae_data
from torch_geometric.loader import DataLoader
import pennylane as qml
from .vqc import VQC

from .data import QuantumGraphsLoader
from time import perf_counter

def main(args):
    ae_util.set_seeds(args["seed"])

    if args["device"] == "gpu": 
        device = ae_util.define_torch_device()
    else:
        device = "cpu"

    outdir = "./trained_vqcs/" + args["outdir"] + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # Load the data
    classical_util.save_compressed_data(device, args)

    train_graphs = ae_data.SelectGraph(args['compressed_data_path']+"/train")
    valid_graphs = ae_data.SelectGraph(args['compressed_data_path']+"/valid")

    if args["train_dataloader_type"] == "fixed_sampling":
        sampler = ae_data.BalancedFixedSubsetSampler(train_graphs, args["num_samples_train"])
        train_loader = DataLoader(train_graphs, batch_size=args["batch"], sampler=sampler)
    elif args["train_dataloader_type"] == "random_sampling":
        sampler = ae_data.BalancedRandomSubsetSampler(train_graphs, args["num_samples_train"])
        train_loader = DataLoader(train_graphs, batch_size=args["batch"], sampler=sampler)
    elif args["train_dataloader_type"] == "fixed_full":
        train_loader = DataLoader(train_graphs, batch_size=args["batch"], shuffle=True)
    else:
        raise TypeError("Specified train dataloader type not recognized")

    valid_loader = DataLoader(valid_graphs, batch_size=len(valid_graphs), shuffle=False)

    quantum_train_loader = QuantumGraphsLoader(train_loader)
    quantum_valid_loader = QuantumGraphsLoader(valid_loader)

    dev = qml.device(args["ideal_dev"], wires=args["n_qubits"])

    model = VQC(dev,args)
    model.export_hyperparameters(outdir)
    model.export_architecture(outdir)

    time_the_training(
        model.train_model, quantum_train_loader, quantum_valid_loader, args["epochs"], args["early_stopping"], outdir
    )


def time_the_training(train, *args):
    """Times the training of the VQC.

    Args:
        train (callable): The training method of the VQC.
        *args: Arguments for the train_model callable of the VQC.
    """
    train_time_start = perf_counter()
    train(*args)
    train_time_end = perf_counter()
    print(
        f"Training completed in: {train_time_end-train_time_start:.2e} s or "
        f"{(train_time_end-train_time_start)/3600:.2e} h."
    )