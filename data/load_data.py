from .develop_data import DevelopData
from .babycry import BabyCry


def load_dataloader(config):

    # This was just for testing purposes
    if config.data.develop:
        return DevelopData(batch_size = 4, sample_rate = 16000)
    
    return BabyCry(config.data.batch_size, config.data.root_dir, config.data.shuffle)