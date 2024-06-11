
import os
from autoencoders import util as ae_util
from classical import util as classical_util
from autoencoders import data as ae_data
from torch_geometric.loader import DataLoader
import pennylane as qml

from .util import choose_ae_vqc_model, transform_data_for_classifier

from time import perf_counter

def main(args):
    ae_util.set_seeds(args["seed"])

    if args["device"] == "gpu": 
        device = ae_util.define_torch_device()
    else:
        device = "cpu"

    outdir = "./trained_vqcs_guided/" + args["outdir"] + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)


    train_graphs = ae_data.SelectGraph(args['data_folder']+"/train")
    valid_graphs = ae_data.SelectGraph(args['data_folder']+"/valid")

    train_loader = DataLoader(train_graphs, batch_size=args["batch"], shuffle=True)
    valid_loader = DataLoader(valid_graphs, batch_size=len(valid_graphs), shuffle=False)


    dev = qml.device(args["ideal_dev"], wires=args["n_qubits"])
    #Autoencoder model definition
    model = choose_ae_vqc_model(args["ae_vqc_type"], dev, device, args)

    model.export_hyperparameters(outdir)
    model.export_architecture(outdir)

    time_the_training(
        model.train_autoencoder, train_loader, valid_loader, args["epochs"], outdir
    )
    model.loss_plot(outdir)

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