from src.models.multistep_baselines import predict_diffusion, train_diffusion


def train(splits, train_cfg):
    return train_diffusion("NsDiff-style", splits, train_cfg)


def predict(model, split, device, n_samples=100):
    return predict_diffusion(model, split, device, n_samples=n_samples)
