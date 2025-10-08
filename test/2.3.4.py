import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
# Bước 1: Load dữ liệu
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
columns = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 
           'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 
           'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 
           'stalk-surface-below-ring', 'stalk-color-above-ring', 
           'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 
           'ring-type', 'spore-print-color', 'population', 'habitat']
df = pd.read_csv(url, header=None, names=columns)
# Bước 2: Encode categorical
label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
# Tách X, y (class: 0=edible, 1=poisonous)
X = df.drop('class', axis=1)
y = df['class']
# Bước 3: Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Bước 4: Train MultinomialNB
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
# Bước 5: Dự đoán
y_pred = nb_model.predict(X_test)
# Bước 6: Đánh giá
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', conf_matrix)
class_report = classification_report(y_test, y_pred, target_names=['Edible', 'Poisonous'])
print('Classification Report:\n', class_report)
# Vẽ Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Edible', 'Poisonous'], yticklabels=['Edible', 'Poisonous'])
plt.title('Confusion Matrix - Naive Bayes on Mushroom Dataset')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
