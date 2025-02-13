import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, classification_report
import joblib
import time

# Step 1: Load the dataset
file_path = "/home/besong/project/project_data/turbine_1/turb_1_with_target.csv"  # Replace with your dataset path
data = pd.read_csv(file_path)

# Step 2: Define features (X) and target (y)
sensor_columns = ['Sensor_0', 'Sensor_1', 'Sensor_2', 'Sensor_3', 
                  'Sensor_4', 'Sensor_5', 'Sensor_6', 'Sensor_7']
X = data[sensor_columns]  # Features
y = data['target']        # Target

# Step 3: Normalize the features using MinMaxScaler
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Step 4: Handle missing values (Imputation)
imputer = SimpleImputer(strategy='mean')  # Replace NaN with the column mean
X_imputed = imputer.fit_transform(X_normalized)  # Impute missing values

# Step 5: Discretize features into 50 bins to simulate `max_bins=50`
binner = KBinsDiscretizer(n_bins=50, encode='ordinal', strategy='uniform')  # 50 bins, uniform binning
X_binned = binner.fit_transform(X_imputed)  # Apply binning to normalized features

# Step 6: Split the data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X_binned, y, test_size=0.5, random_state=42, stratify=y)

# Step 7: Hyperparameter Tuning using GridSearchCV
# Define the hyperparameter grid
param_grid = {
    'max_depth': [5, 10, 15],  # Depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'criterion': ['gini', 'entropy'],  # Splitting criteria
    'class_weight': ['balanced']  # Equal weight for classes
}

# Initialize the Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)

# Perform Grid Search
grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
start_time = time.time()  # Start timing
grid_search.fit(X_train, y_train)
end_time = time.time()  # End timing
training_time = end_time - start_time

# Best parameters from GridSearch
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

# Step 8: Train the model using the best parameters
best_model = grid_search.best_estimator_

# Save the trained model
model_file = "/home/besong/project/project_data/turbine_1/decision_tree_model_normalized_with_hyperparameters.joblib"
joblib.dump(best_model, model_file)
print(f"Model saved to {model_file}")

# Step 9: Load the saved model
loaded_model = joblib.load(model_file)
print("Model loaded successfully.")

# Step 10: Validate the loaded model on the test set
y_pred_loaded = loaded_model.predict(X_test)

# Step 11: Evaluate the loaded model
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_loaded)
tn, fp, fn, tp = conf_matrix.ravel()

# Precision
precision = precision_score(y_test, y_pred_loaded)

# Sensitivity (Recall)
sensitivity = recall_score(y_test, y_pred_loaded)

# Specificity
specificity = tn / (tn + fp)

# Accuracy
accuracy = accuracy_score(y_test, y_pred_loaded)

# F1-Score
f1 = f1_score(y_test, y_pred_loaded)

# Print Evaluation Metrics
print("\nValidation Results Using Loaded Model:")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Precision: {precision:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Training Time: {training_time:.4f} seconds")

# Classification Report
classification_rep = classification_report(y_test, y_pred_loaded)
print("\nClassification Report:")
print(classification_rep)

# Step 12: Feature Importance Analysis
feature_importances = loaded_model.feature_importances_
print("\nFeature Importance Analysis:")
for sensor, importance in zip(sensor_columns, feature_importances):
    print(f"{sensor}: {importance:.4f}")

# Step 13: Visualize Decision Tree (Optional, requires matplotlib)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(loaded_model, feature_names=sensor_columns, class_names=['Normal', 'Anomaly'], filled=True, rounded=True, fontsize=10)
plt.title(f"Decision Tree Visualization (Max Depth = {best_params['max_depth']})", fontsize=12)
plt.show()
