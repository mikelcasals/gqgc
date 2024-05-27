import sys
import os

sys.append('..')

from autoencoders import util as aeutil

class compressed_data:
    

    def __init__(self, data_folder, norm_name, num_samples, model_path, seed=42):
        
        device = "cpu"
        model_folder = os.path.dirname(model_path)
        hp_file = os.path.join(model_folder, "hyperparameters.json")
        hp=aeutil.import_hyperparams(hp_file)

        #Load data
        

    
        
        self.model = aeutil.choose_ae_model(hp["aetype"], device, hp)
        self.model.load_model(model_path)



        pass

    def get_latent_data(self, )