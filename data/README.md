# Data

This project uses real measured OEDI PVDAQ AC power only. Synthetic PV data must
not be used for training, validation, testing, figures, tables, or paper claims.

Expected local layout after downloading and preprocessing:

```text
data/
  raw/
    multisite/
      pvdaq_system_4/
      pvdaq_system_10/
      pvdaq_system_34/
    nsrdb/
      pvdaq_system_4/
      pvdaq_system_10/
      pvdaq_system_34/
  processed/
    multisite/
      pvdaq_system_4/processed_power.csv
      pvdaq_system_10/processed_power.csv
      pvdaq_system_34/processed_power.csv
  source_metadata/
```

Raw and processed data are ignored by normal Git. Recreate them with:

```bash
python scripts/download_pvdaq_multisite.py
export NSRDB_API_KEY="<your NREL developer API key>"
export NSRDB_EMAIL="<your NSRDB account email>"
python scripts/download_nsrdb_weather.py
bash scripts/prepare_multisite.sh
```

