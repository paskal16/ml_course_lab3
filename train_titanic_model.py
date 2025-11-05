import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # Исправлено: ensemble, не mosable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import requests
from io import StringIO  # Исправлено: io, не in; StringIO, не String10

# AI-ассистент: использован DeepSeek для генерации кода

# Загрузка датасета Titanic
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
response = requests.get(url)
data = pd.read_csv(StringIO(response.text))


# Предобработка данных
def preprocess_data(df):
    # Заполнение пропущенных значений
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # Кодирование категориальных переменных
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()

    df['Sex_encoded'] = le_sex.fit_transform(df['Sex'])
    df['Embarked_encoded'] = le_embarked.fit_transform(df['Embarked'])

    # Выбор фичей для модели
    features = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_encoded']
    X = df[features]
    y = df['Survived']

    return X, y


# Предобработка и разделение данных
X, y = preprocess_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Сохранение модели
with open('titanic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Оценка модели
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("✅ Модель Titanic обучена и сохранена!")
print(f"📊 Accuracy на тренировочных данных: {train_score:.3f}")
print(f"📊 Accuracy на тестовых данных: {test_score:.3f}")
print("🎯 Используемые фичи: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked")