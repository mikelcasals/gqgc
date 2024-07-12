
from gae_models.MIAGAE import MIAGAE
from gae_models.SAG_model import SAG_model
from classifier_models.classical.classical_GNN import ClassicalGNN
from classifier_models.quantum.QGNN1 import QGNN1
from classifier_models.quantum.QGNN2 import QGNN2
from classifier_models.quantum.QGNN3 import QGNN3
from classifier_models.quantum.QGNN4 import QGNN4
from classifier_models.quantum.QGNN5 import QGNN5

from base_models.guided_classifier_base_model import GuidedClassifier


def choose_guided_classifier_model(gae_type, classifier_type, device, hyperparams):
    gae_classes = {
        "MIAGAE": MIAGAE,   
        "SAG_model": SAG_model
    }

    classifier_classes = {
        "GNN": ClassicalGNN,
        "QGNN1": QGNN1,
        "QGNN2": QGNN2,
        "QGNN3": QGNN3,
        "QGNN4": QGNN4,
        "QGNN5": QGNN5
    }

    gae_class = gae_classes.get(gae_type)
    classifier_class = classifier_classes.get(classifier_type)

    
    if not gae_class or not classifier_class:
        raise ValueError(f"Invalid ae_type or classifier_type: {gae_type}, {classifier_type}")
    
    class GuidedClassifierInstance(GuidedClassifier, gae_class, classifier_class):
        def __init__(self, device="cpu", hpars = {}):
            GuidedClassifier.__init__(self, device, hpars)
            gae_class.__init__(self, device, hpars)
            classifier_class.__init__(self, device, hpars)
            
            
        
        def classifier(self, *args, **kwargs):
            return classifier_class.classifier(self, *args, **kwargs)

        def classifier_network_summary(self):
            return classifier_class.classifier_network_summary(self)

        def encoder_decoder(self, *args, **kwargs):  # Ensure the method accepts the 'data' argument
            return gae_class.encoder_decoder(self, *args, **kwargs)

        def gae_network_summary(self):
            return gae_class.gae_network_summary(self)

        def instantiate_decoder(self):
            return gae_class.instantiate_decoder(self)

        def instantiate_encoder(self):
            return gae_class.instantiate_encoder(self)



    return GuidedClassifierInstance(device, hyperparams).to(device)
