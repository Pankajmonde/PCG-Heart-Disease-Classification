import pandas as pd
from sklearn.feature_selection import f_classif, SelectKBest

# Load your dataset
# df = pd.read_csv('path_to_your_dataset.csv')

# Assuming the last column is the target variable
X = df.iloc[:, :-1]  # feature set
Y = df.iloc[:, -1]   # target variable

# Applying ANOVA F-statistic
selector = SelectKBest(f_classif, k=30)
X_selected = selector.fit_transform(X, Y)

# Get the selected feature indices
selected_features = selector.get_support(indices=True)

# Display the top 30 features
print('Top 30 features based on ANOVA F-statistic:')
print(X.columns[selected_features])