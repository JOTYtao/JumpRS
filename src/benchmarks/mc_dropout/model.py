from src.models.multistep_baselines import predict_deterministic, train_deterministic


def train(splits, train_cfg):
    return train_deterministic("MC Dropout", splits, train_cfg, mc_dropout=True)


def predict(model, split, device, mc_samples=30):
    return predict_deterministic(model, split, device, mc_samples=mc_samples)

