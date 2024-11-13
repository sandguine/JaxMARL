import jaxmarl
from jaxmarl.networks.successor_features import SuccessorFeatureNetwork

class SRWrapper:
    """Wrapper to add SR capabilities to any algorithm"""
    def __init__(self, config):
        self.sr_module = SRModule(
            feature_dim=config["FEATURE_DIM"],
            encoder_type=config.get("ENCODER_TYPE", "cnn"),
            activation=config["ACTIVATION"]
        )
        self.gamma = config["GAMMA"]
        self.sf_coef = config["SF_COEF"]
    
    def process_observation(self, obs, training_state):
        """Process observation through SR networks"""
        encoded_state = self.sr_module.feature_encoder.apply(
            training_state.feature_state.params,
            obs
        )
        sf_features = self.sr_module.sf_network.apply(
            training_state.sf_state.params,
            encoded_state
        )
        return encoded_state, sf_features
    
    def compute_loss(self, encoded_state, next_encoded_state, done):
        """Compute SR loss component"""
        return self.sr_module.compute_sf_loss(
            encoded_state,
            next_encoded_state,
            done,
            self.gamma
        ) 