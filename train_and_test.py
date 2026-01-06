import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import joblib

# Load the training dataset
train_data = pd.read_csv('data/emotion_dataset_training.csv')

# Split the training data into features and target
X_train = train_data['Text']
y_train = train_data['Emotion']

# Replace NaN values with empty strings
X_train = X_train.fillna('')

# Vectorize the training text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Train the model with increased max_iter
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Save the model and vectorizer
joblib.dump(model, 'models/emotion_classifier_pipe_lr2.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

# Load the testing dataset
test_data = pd.read_csv('data/emotion_dataset_testing.csv')

# Split the testing data into features and target
X_test = test_data['Text']
y_test = test_data['Emotion']

# Replace NaN values with empty strings
X_test = X_test.fillna('')

# Vectorize the testing text data
X_test_vec = vectorizer.transform(X_test)

# Test the model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
