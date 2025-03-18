import pickle

with open('data/processed/text_features_batch_84.pkl', 'rb') as file:
    data = pickle.load(file)

print(data)
