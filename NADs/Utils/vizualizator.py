# utils/visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import numpy as np


class TrafficVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.fig_size = (12, 8)

    def plot_traffic_overview(self, df, anomalies_df=None):
        """Обзор трафика с аномалиями"""
        fig, axes = plt.subplots(2, 2, figsize=self.fig_size)

        # График количества пакетов по времени
        if 'timestamp' in df.columns:
            time_series = df.groupby('timestamp').agg({
                'packet_count': 'sum',
                'total_bytes': 'sum'
            }).reset_index()

            axes[0, 0].plot(time_series['timestamp'], time_series['packet_count'])
            axes[0, 0].set_title('Количество пакетов по времени')
            axes[0, 0].set_xlabel('Время')
            axes[0, 0].set_ylabel('Пакеты')

            # Отметка аномалий
            if anomalies_df is not None:
                anomaly_times = anomalies_df['timestamp']
                for time in anomaly_times:
                    axes[0, 0].axvline(x=time, color='red', alpha=0.3)

        # Распределение размеров пакетов
        axes[0, 1].hist(df['avg_packet_size'].dropna(), bins=50, alpha=0.7)
        axes[0, 1].set_title('Распределение размеров пакетов')
        axes[0, 1].set_xlabel('Размер пакета (байты)')
        axes[0, 1].set_ylabel('Частота')

        # Количество уникальных портов
        axes[1, 0].scatter(df['packet_count'], df['unique_dst_ports'], alpha=0.6)
        axes[1, 0].set_title('Пакеты vs Уникальные порты')
        axes[1, 0].set_xlabel('Количество пакетов')
        axes[1, 0].set_ylabel('Уникальные порты')

        # SYN ratio
        axes[1, 1].hist(df['syn_ack_ratio'].replace([np.inf, -np.inf], np.nan).dropna(), bins=50, alpha=0.7)
        axes[1, 1].set_title('Распределение SYN/ACK ratio')
        axes[1, 1].set_xlabel('SYN/ACK Ratio')
        axes[1, 1].set_ylabel('Частота')

        plt.tight_layout()
        return fig

    def plot_anomalies(self, df, anomaly_column='is_anomaly'):
        """Визуализация аномалий"""
        if anomaly_column not in df.columns:
            print(f"Колонка {anomaly_column} не найдена в данных")
            return

        fig, axes = plt.subplots(1, 2, figsize=self.fig_size)

        # Аномалии в 2D пространстве признаков
        normal = df[~df[anomaly_column]]
        anomalies = df[df[anomaly_column]]

        axes[0].scatter(normal['packet_count'], normal['unique_dst_ports'],
                        alpha=0.5, label='Нормальный', color='green')
        axes[0].scatter(anomalies['packet_count'], anomalies['unique_dst_ports'],
                        alpha=0.8, label='Аномалии', color='red')
        axes[0].set_xlabel('Количество пакетов')
        axes[0].set_ylabel('Уникальные порты')
        axes[0].set_title('Обнаружение аномалий')
        axes[0].legend()

        # Временная шкала аномалий
        if 'timestamp' in df.columns:
            axes[1].plot(df['timestamp'], df['packet_count'], alpha=0.7, label='Трафик')
            anomaly_times = df[df[anomaly_column]]['timestamp']
            anomaly_values = df[df[anomaly_column]]['packet_count']
            axes[1].scatter(anomaly_times, anomaly_values, color='red',
                            label='Аномалии', zorder=5)
            axes[1].set_xlabel('Время')
            axes[1].set_ylabel('Пакеты')
            axes[1].set_title('Аномалии по времени')
            axes[1].legend()

        plt.tight_layout()
        return fig