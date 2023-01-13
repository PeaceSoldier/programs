import numpy as np
from sklearn import preprocessing

input_data = np.array([[4.1, -5.9, -3.5],
                       [-1.9, 4.6, 3.9],
                       [-4.2, 6.8, 6.3],
                       [3.9, 3.4, 1.2]])
data_binarized = preprocessing.Binarizer(threshold=3.2).transform(input_data)
print(f"\nBinarized data:\n{data_binarized}")

print("\nBEFORE: ")
print(f"Mean = {input_data.mean(axis=0)}")
print(f"Std deviation = {input_data.std(axis=0)}")

data_scaled = preprocessing.scale(input_data)
print("\nAFTER: ")
print(f"Mean = {data_scaled.mean(axis=0)}")
print(f"Std deviation = {data_scaled.std(axis=0)}")

data_scaled_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled_minmax = data_scaled_minmax.fit_transform(input_data)
print(f"\nMin max scaled data:\n{data_scaled_minmax}")

data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')

print(f"\nL1 normalized data:\n{data_normalized_l1}")
print(f"\nL2 normalized data:\n{data_normalized_l2}")
