import pickle

file_path = "/home/TS2Vec/single_forecast_result/data_2.pkl"

with open(file_path, "rb") as file:
    data = pickle.load(file)

print(data)
