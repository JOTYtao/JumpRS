import pandas as pd
import pytest

from src.data.nsrdb import align_nsrdb_weather, parse_nsrdb_psm3_csv


NSRDB_SAMPLE = """Source,Location ID,City,State,Country,Latitude,Longitude,Time Zone,Elevation
NSRDB,1,-,-,-,39.74,-105.18,0,1793
Year,Month,Day,Hour,Minute,Temperature,Wind Speed
2011,1,1,0,0,5.0,2.0
2011,1,1,0,30,7.0,4.0
"""


def test_parse_nsrdb_psm3_csv_uses_utc_and_site_timezone():
    weather = parse_nsrdb_psm3_csv(NSRDB_SAMPLE, "America/Denver")

    assert list(weather.columns) == ["air_temperature", "wind_speed"]
    assert str(weather.index.tz) == "America/Denver"
    assert weather.iloc[0].to_dict() == {"air_temperature": 5.0, "wind_speed": 2.0}


def test_align_nsrdb_weather_interpolates_to_target_timestamps():
    weather = parse_nsrdb_psm3_csv(NSRDB_SAMPLE, "America/Denver")
    target = pd.date_range(weather.index[0], periods=3, freq="15min")

    aligned = align_nsrdb_weather(weather, target)

    assert aligned["air_temperature"].tolist() == [5.0, 6.0, 7.0]
    assert aligned["wind_speed"].tolist() == [2.0, 3.0, 4.0]


def test_align_nsrdb_weather_leaves_out_of_coverage_boundaries_missing():
    weather = parse_nsrdb_psm3_csv(NSRDB_SAMPLE, "America/Denver")
    target = weather.index.insert(0, weather.index[0] - pd.Timedelta(minutes=15))

    aligned = align_nsrdb_weather(weather, target)

    assert aligned.iloc[0].isna().all()


def test_align_nsrdb_weather_supports_duplicate_target_timestamps():
    weather = parse_nsrdb_psm3_csv(NSRDB_SAMPLE, "America/Denver")
    target = weather.index.insert(1, weather.index[0])

    aligned = align_nsrdb_weather(weather, target)

    assert len(aligned) == 3
    assert aligned.iloc[0].equals(aligned.iloc[1])


def test_parse_nsrdb_rejects_missing_weather_columns():
    invalid = NSRDB_SAMPLE.replace(",Wind Speed", "")
    with pytest.raises(ValueError, match="weather columns"):
        parse_nsrdb_psm3_csv(invalid, "America/Denver")
