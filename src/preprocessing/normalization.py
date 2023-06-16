import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def min_max(data):
    # Create a MinMaxScaler object to perform min-max normalization
    min_max_scaler = MinMaxScaler()

    # Fit the scaler to the data and transform the data
    data_min_max = pd.DataFrame(min_max_scaler.fit_transform(data), columns=data.columns, index=data.index)

    # Save the normalized data to new CSV files
    data_min_max.to_csv('../../data/preprocessed/data_min_max.csv')


def z_score(data):
    # Create a StandardScaler object to perform z-score normalization
    standard_scaler = StandardScaler()

    # Fit the scaler to the data and transform the data
    data_z_score = pd.DataFrame(standard_scaler.fit_transform(data), columns=data.columns, index=data.index)

    # Save the normalized data to new CSV files
    data_z_score.to_csv('../../data/preprocessed/data_z_score.csv')


def test():
    # Load the cleaned and resampled data from a CSV file
    data = pd.read_csv('../../data/preprocessed/cleaned_data.csv', index_col=0, parse_dates=True)

    min_max(data)
    z_score(data)


if __name__ == "__main__":
    test()
