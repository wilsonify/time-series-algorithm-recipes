# airpassengers_analysis.py

from c01_getting_started import path_to_data
from c01_getting_started.r011a_reading_time_series_objects import load_airpassenger_data, plot_passenger_data


def test_r011a_reading_time_series_objects():
    csv_path = f"{path_to_data}/input/airpassengers.csv"
    data = load_airpassenger_data(csv_path)
    plot_passenger_data(data, show=False)
