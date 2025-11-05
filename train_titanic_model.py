import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import requests
from io import StringIO

# AI-ассистент: использован DeepSeek для генерации кода

# Загрузка и предобработка данных
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
response = requests.get(url)
data = pd.read_csv(StringIO(response.text))


def preprocess_data(df):
    # Обработка пропущенных значений
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # Кодирование категориальных переменных
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()

    df['Sex_encoded'] = le_sex.fit_transform(df['Sex'])
    df['Embarked_encoded'] = le_embarked.fit_transform(df['Embarked'])

    features = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_encoded']
    X = df[features]
    y = df['Survived']

    return X, y


# Обучение модели
X, y = preprocess_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Сохранение модели
with open('titanic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Модель обучена и сохранена!")
print(f"📊 Accuracy: {model.score(X_test, y_test):.3f}")