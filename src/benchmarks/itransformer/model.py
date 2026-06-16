from src.models.multistep_baselines import predict_deterministic, train_sota_deterministic


def train(splits, train_cfg):
    return train_sota_deterministic("iTransformer", splits, train_cfg)


def predict(model, split, device):
    return predict_deterministic(model, split, device)

