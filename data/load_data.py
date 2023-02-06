from .develop_data import DevelopData


def load_dataloader(config):
    if config.data.develop:
        return DevelopData(batch_size = 4, sample_rate = 16000)