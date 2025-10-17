import pandas as pd

def load_sensor_data(file_path):
    """
    Load sensor data from a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file containing sensor data.

    Returns:
    pd.DataFrame: A DataFrame containing the sensor data.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

accelerometer_data = load_sensor_data('iot/hw1/2025-10-12_16-18-03/Accelerometer.csv')
gyroscope_data = load_sensor_data('iot/hw1/2025-10-12_16-18-03/Gyroscope.csv')
gravity_data = load_sensor_data('iot/hw1/2025-10-12_16-18-03/Gravity.csv')