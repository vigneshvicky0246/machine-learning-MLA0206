import pandas as pd
import numpy as np
from collections import Counter
# Load the dataset
data = pd.read_csv("id3.csv")
# Define the ID3 algorithm
def id3(data, target_col):
# If all target values are the same, return the target value
if len(np.unique(data[target_col])) == 1:
return np.unique(data[target_col])[0]
# If there are no more features to split on, return the majority target 
value
if len(data.columns) == 1:
return Counter(data[target_col]).most_common(1)[0][0]
# Calculate the information gain for each feature
info_gains = []
for feature in data.columns[:-1]: # Exclude the target column
info_gains.append(calc_info_gain(data, feature, target_col))
# Choose the feature with the highest information gain
best_feature_index = np.argmax(info_gains)
best_feature = data.columns[best_feature_index]
# Create a sub-tree
tree = {best_feature: {}}
unique_values = np.unique(data[best_feature])
for value in unique_values:
subtree_data = data[data[best_feature] == value].drop(best_feature, 
axis=1)
subtree = id3(subtree_data, target_col)
tree[best_feature][value] = subtree
return tree
# Calculate information gain for a feature
def calc_info_gain(data, feature, target_col):
entropy_before_split = calc_entropy(data[target_col])
unique_values = np.unique(data[feature])
entropy_after_split = 0
for value in unique_values:
subset = data[data[feature] == value]
subset_weight = len(subset) / len(data)
entropy_after_split += subset_weight *
calc_entropy(subset[target_col])
return entropy_before_split - entropy_after_split
# Calculate entropy
def calc_entropy(data):
counts = Counter(data)
probs = [count / len(data) for count in counts.values()]
return -sum(p * np.log2(p) for p in probs)
# Build the ID3 decision tree
target_col = "Answer"
tree = id3(data, target_col)
# Print the tree
print(tree)
# Define a function to classify a new sample
def classify_sample(sample, tree):
feature = list(tree.keys())[0]
value = sample[feature]
subtree = tree[feature][value]
if isinstance(subtree, dict):
return classify_sample(sample, subtree)
else:
return subtree
# Classify a new sample
new_sample = {"Outlook": "sunny", "Temp": "cool", "Humidity": "normal", 
"Wind": "True"}
classification = classify_sample(new_sample, tree)
print("Classification:", classification)
