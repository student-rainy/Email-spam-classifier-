#!/usr/bin/env python
# coding: utf-8

# In[3]:


# 📧 Email Spam Classifier Project using Python and Machine Learning
# ---------------------------------------------------------------
# Author: Your Name
# Description: Detects whether an email/message is spam or ham using TF-IDF and Naive Bayes.

# 1️⃣ Import Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# 2️⃣ Load Dataset
# Make sure you have spam.csv file in the same folder as this script
data = pd.read_csv("mail_data.csv", encoding='latin-1')

# Keep only necessary columns
print(data.columns)
# Display dataset info
print("✅ Dataset Loaded Successfully!")
print("Total Messages:", data.shape[0])
print(data.head())

# 3️⃣ Data Cleaning / Preprocessing
data['Category'] = data['Category'].map({'spam': 0, 'ham': 1})
data.dropna(inplace=True)

print("\n✅ Data Cleaned Successfully!")
print(data['Category'].value_counts())

# 4️⃣ Split Data into Training and Testing Sets
X = data['Message']
Y = data['Category']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
print("\n✅ Data Split Completed!")
print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# 5️⃣ Feature Extraction using TF-IDF
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Convert Y to int (safety)
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

print("\n✅ Feature Extraction Done!")
print("Total Features:", X_train_features.shape[1])

# 6️⃣ Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_features, Y_train)

print("\n✅ Model Training Completed!")

# 7️⃣ Evaluate the Model
training_prediction = model.predict(X_train_features)
training_accuracy = accuracy_score(Y_train, training_prediction)

test_prediction = model.predict(X_test_features)
test_accuracy = accuracy_score(Y_test, test_prediction)

print("\n📊 Model Evaluation Results:")
print("Training Accuracy:", round(training_accuracy * 100, 2), "%")
print("Testing Accuracy:", round(test_accuracy * 100, 2), "%")
print("\nClassification Report:\n", classification_report(Y_test, test_prediction))

# Confusion Matrix
cm = confusion_matrix(Y_test, test_prediction)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Spam', 'Ham'], yticklabels=['Spam', 'Ham'])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()

# 8️⃣ Test with a Custom Input
print("\n✉️ Test with a Custom Message:")
input_mail = ["Congratulations! You’ve won a $1000 Walmart gift card! Click here to claim now."]

# Convert input to feature vector
input_data_features = feature_extraction.transform(input_mail)

# Predict
prediction = model.predict(input_data_features)
if prediction[0] == 0:
    print("🔴 The mail is **SPAM**.")
else:
    print("🟢 The mail is **HAM (Not Spam)**.")

# 9️⃣ Save Model and Vectorizer for Future Use
joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(feature_extraction, "tfidf_vectorizer.pkl")
print("\n💾 Model and Vectorizer Saved Successfully!")


# In[ ]:




