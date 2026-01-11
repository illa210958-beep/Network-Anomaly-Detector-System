# random_forest_model.py (упрощенная версия)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


class RandomForestAnomalyDetector:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def prepare_features(self, df):
        """Подготовка признаков"""
        # Удаляем нечисловые колонки
        feature_columns = []
        for col in df.columns:
            if col not in ['ip', 'timestamp', 'label', 'prediction']:
                # Пробуем преобразовать в число
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    feature_columns.append(col)
                except:
                    pass

        if not feature_columns:
            return None

        X = df[feature_columns].fillna(0)
        return X

    def train(self, df, target_column='label', verbose=True):
        """Обучение модели"""
        try:
            X = self.prepare_features(df)
            if X is None or len(X) < 2:
                if verbose:
                    print("⚠️  Недостаточно данных для обучения")
                return self

            y = df[target_column]

            # Кодируем метки
            y_encoded = self.label_encoder.fit_transform(y)

            # Масштабируем признаки
            X_scaled = self.scaler.fit_transform(X)

            # Обучаем модель
            self.model.fit(X_scaled, y_encoded)

            if verbose:
                print("✅ Модель обучена")

            return self

        except Exception as e:
            if verbose:
                print(f"❌ Ошибка при обучении: {e}")
            return self

    def predict(self, df, verbose=True):
        """Предсказание аномалий"""
        try:
            X = self.prepare_features(df)
            if X is None:
                if verbose:
                    print("⚠️  Нет данных для предсказания")
                df['prediction'] = 'normal'
                return df

            X_scaled = self.scaler.transform(X)

            # Предсказания
            predictions = self.model.predict(X_scaled)
            df['prediction'] = self.label_encoder.inverse_transform(predictions)

            if verbose:
                print("✅ Предсказания выполнены")

            return df

        except Exception as e:
            if verbose:
                print(f"❌ Ошибка при предсказании: {e}")
            df['prediction'] = 'normal'
            return df