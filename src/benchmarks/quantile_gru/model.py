from src.models.multistep_baselines import predict_quantile, train_quantile


def train(splits, train_cfg):
    return train_quantile(splits, train_cfg)


def predict(model, split, device):
    return predict_quantile(model, split, device)

