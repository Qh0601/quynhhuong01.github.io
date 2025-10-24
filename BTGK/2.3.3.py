import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Bước 1: Load dữ liệu từ URL raw (không cần file local)
df = pd.read_csv('https://raw.githubusercontent.com/sharmaroshan/Online-Shoppers-Purchasing-Intention/master/online_shoppers_intention.csv')


# Bước 2: Encode categorical
le_month = LabelEncoder()
df['Month'] = le_month.fit_transform(df['Month'])
le_vtype = LabelEncoder()
df['VisitorType'] = le_vtype.fit_transform(df['VisitorType'])
df['Weekend'] = df['Weekend'].astype(int)  # Đã boolean, chuyển int


# Tách X, y (Revenue là target binary)
X = df.drop('Revenue', axis=1)
y = df['Revenue'].astype(int)


# Bước 3: Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Bước 4: Train GaussianNB
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)


# Bước 5: Dự đoán
y_pred = nb_model.predict(X_test)


# Bước 6: Đánh giá
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', conf_matrix)
class_report = classification_report(y_test, y_pred)
print('Classification Report:\n', class_report)


# Vẽ Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Naive Bayes on Online Shoppers')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()