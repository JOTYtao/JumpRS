from src.run_multisite_baselines import baseline_predictions


def predict(split, thresholds, capacity_kw_by_site):
    return baseline_predictions(split, thresholds, capacity_kw_by_site, "Persistence")

