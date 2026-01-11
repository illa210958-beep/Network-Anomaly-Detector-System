# isolation_forest.py (упрощенная версия)
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np


class IsolationForestAnomalyDetector:
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def prepare_features(self, df):
        """Подготовка признаков"""
        feature_columns = []
        for col in df.columns:
            if col not in ['ip', 'timestamp', 'label', 'is_anomaly']:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    feature_columns.append(col)
                except:
                    pass

        if not feature_columns:
            return None

        X = df[feature_columns].fillna(0)
        return X

    def train(self, df, verbose=True):
        """Обучение модели"""
        try:
            X = self.prepare_features(df)
            if X is None or len(X) < 2:
                if verbose:
                    print("⚠️  Недостаточно данных для обучения")
                return self

            self.model.fit(X)

            if verbose:
                print("✅ Модель обучена")

            return self

        except Exception as e:
            if verbose:
                print(f"❌ Ошибка при обучении: {e}")
            return self

    def predict(self, df, verbose=True):
        """Обнаружение аномалий"""
        try:
            X = self.prepare_features(df)
            if X is None:
                if verbose:
                    print("⚠️  Нет данных для анализа")
                df['is_anomaly'] = False
                return df

            predictions = self.model.predict(X)
            df['is_anomaly'] = predictions == -1

            if verbose:
                print("✅ Анализ завершен")

            return df

        except Exception as e:
            if verbose:
                print(f"❌ Ошибка при анализе: {e}")
            df['is_anomaly'] = False
            return df