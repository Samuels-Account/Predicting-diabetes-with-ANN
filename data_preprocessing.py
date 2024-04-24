import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
file_path = 'C:\\Users\\sledu\\OneDrive\\Documents\\University Work\\Year 2\\Artificial Intelligence\\Dataset of Diabetes .csv'
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
