import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split  # Corrected import
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
file_path = 'Dataset of Diabetes .csv'
data = pd.read_csv(file_path)

# Encode categorical data

# Label Encoding for 'Gender'
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# One-Hot Encoding for 'CLASS'
one_hot_encoder = OneHotEncoder()
class_encoded = one_hot_encoder.fit_transform(data[['CLASS']]).toarray()
class_encoded_df = pd.DataFrame(class_encoded, columns=one_hot_encoder.get_feature_names_out(['CLASS']))
data = pd.concat([data, class_encoded_df], axis=1)
data.drop('CLASS', axis=1, inplace=True)

# Remove unnecessary 'ID' and 'No_Pation' columns
data.drop(['ID', 'No_Pation'], axis=1, inplace=True)

# Split the data into features and target
X = data.iloc[:, :-3]  # All columns except the one-hot encoded class columns
y = data.iloc[:, -3:]  # Only the one-hot encoded class columns

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
scaler_filename = "scaler.save"
joblib.dump(scaler, scaler_filename)

# At this point, you can print some information to check everything is as expected
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

# Building the neural network
model = Sequential([
    Dense(12, input_dim=X_train.shape[1], activation='relu'),  # Input layer with 'relu' activation
    Dense(8, activation='relu'),  # Hidden layer
    Dense(3, activation='softmax')  # Output layer with 'softmax' activation for multi-class classification
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=150, batch_size=10, validation_split=0.1)

# Save the model
model.save('diabetes_prediction_model.h5')

# Optional: Print out the history to see the accuracy and loss over epochs
print(history.history['accuracy'])
print(history.history['loss'])

# Optional: Save the history for later analysis
with open('training_history.pkl', 'wb') as file_pi:
    joblib.dump(history.history, file_pi)


# Make predictions on test data
y_pred = model.predict(X_test)

# Convert predictions to class labels
y_pred_classes = tf.argmax(y_pred, axis=1)
y_true_classes = tf.argmax(y_test.values, axis=1)

# Generate the confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report

conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
class_report = classification_report(y_true_classes, y_pred_classes)

# Print the confusion matrix and classification report
print(conf_matrix)
print(class_report)

# Optional: Save the evaluation metrics for later analysis
with open('evaluation_metrics.txt', 'w') as f:
    f.write(f"Confusion Matrix:\n{conf_matrix}\n")
    f.write(f"Classification Report:\n{class_report}\n")