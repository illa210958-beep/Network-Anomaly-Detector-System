# main.py - исправленная версия
import pandas as pd
import numpy as np
from feature_extractor import NetworkFeatureExtractor
from Models.random_forest_model import RandomForestAnomalyDetector
from Models.isolation_forest import IsolationForestAnomalyDetector
import argparse
import os
from datetime import datetime
import warnings

# Подавляем предупреждения pandas
warnings.filterwarnings('ignore')


class NetworkAnomalyDetector:
    def simple_plot_anomalies(self, results_df):
        """Простая визуализация аномалий"""
        try:
            import matplotlib.pyplot as plt

            print("\n🎨 Создание графика анализа...")

            # Проверяем, есть ли данные для визуализации
            if results_df.empty or len(results_df) < 2:
                print("⚠️  Недостаточно данных для визуализации")
                return

            # Проверяем наличие необходимых столбцов
            if 'packet_count' not in results_df.columns or 'total_bytes' not in results_df.columns:
                print("⚠️  Отсутствуют данные для визуализации")
                return

            # Создаем график
            plt.figure(figsize=(12, 8))

            # Определяем цвета для разных типов аномалий
            colors = []
            labels = []

            if 'prediction' in results_df.columns:
                # Random Forest - разные цвета для разных типов
                color_map = {
                    'normal': 'green',
                    'port_scan': 'orange',
                    'ddos': 'red',
                    'syn_flood': 'purple',
                    'brute_force': 'brown',
                    'suspicious': 'yellow'
                }

                for _, row in results_df.iterrows():
                    colors.append(color_map.get(row['prediction'], 'blue'))

                # Создаем легенду
                for label, color in color_map.items():
                    if label in results_df['prediction'].values:
                        labels.append(label)

            else:
                # Isolation Forest - только норма/аномалия
                colors = ['green' if not row['is_anomaly'] else 'red' for _, row in results_df.iterrows()]
                labels = ['Нормальный', 'Аномалия']

            # Рисуем точки
            scatter = plt.scatter(results_df['packet_count'], results_df['total_bytes'],
                                  c=colors, alpha=0.7, s=100)

            plt.xlabel('Количество пакетов (за временной интервал)', fontsize=12)
            plt.ylabel('Общий объем данных (байты)', fontsize=12)
            plt.title('Визуализация сетевых аномалий', fontsize=14, fontweight='bold')

            # Добавляем сетку
            plt.grid(True, alpha=0.3, linestyle='--')

            # Добавляем легенду
            from matplotlib.patches import Patch
            legend_elements = []
            for label in set(labels):
                if label == 'normal':
                    legend_elements.append(Patch(facecolor='green', label='Нормальный трафик', alpha=0.7))
                elif label == 'port_scan':
                    legend_elements.append(Patch(facecolor='orange', label='Сканирование портов', alpha=0.7))
                elif label == 'ddos':
                    legend_elements.append(Patch(facecolor='red', label='DDoS атака', alpha=0.7))
                elif label == 'syn_flood':
                    legend_elements.append(Patch(facecolor='purple', label='SYN Flood', alpha=0.7))
                elif label == 'brute_force':
                    legend_elements.append(Patch(facecolor='brown', label='Brute Force', alpha=0.7))
                elif label == 'suspicious':
                    legend_elements.append(Patch(facecolor='yellow', label='Подозрительный', alpha=0.7))
                elif label == 'Аномалия':
                    legend_elements.append(Patch(facecolor='red', label='Аномалия (Isolation Forest)', alpha=0.7))

            plt.legend(handles=legend_elements, loc='upper left')

            # Добавляем информационную надпись
            total_points = len(results_df)
            if 'prediction' in results_df.columns:
                anomalies = results_df[results_df['prediction'] != 'normal']
            else:
                anomalies = results_df[results_df['is_anomaly']]

            anomaly_percent = (len(anomalies) / total_points * 100) if total_points > 0 else 0

            info_text = f"Всего интервалов: {total_points}\nАномалий: {len(anomalies)} ({anomaly_percent:.1f}%)"
            plt.figtext(0.02, 0.02, info_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

            plt.tight_layout()
            plt.show()
            print("✅ График создан успешно")

        except Exception as e:
            print(f"⚠️  Не удалось создать график: {e}")

    def __init__(self, mode='file'):
        self.mode = mode
        self.feature_extractor = NetworkFeatureExtractor()
        print("✅ Система NetworkAnomalyDetector инициализирована")

    def analyze_pcap(self, pcap_file, model_type='random_forest'):
        """Анализ pcap файла"""
        print(f"\n{'=' * 70}")
        print(f"🔍 АНАЛИЗ СЕТЕВОГО ТРАФИКА")
        print(f"📁 Файл: {os.path.basename(pcap_file)}")
        print(f"{'=' * 70}")

        # Проверка существования файла
        if not os.path.exists(pcap_file):
            print(f"❌ ФАЙЛ НЕ НАЙДЕН: '{pcap_file}'")
            print(f"📁 Текущая директория: {os.getcwd()}")
            available_files = [f for f in os.listdir('.') if f.lower().endswith(('.pcap', '.pcapng'))]
            if available_files:
                print(f"📂 Доступные PCAP файлы: {', '.join(available_files)}")
            return None

        # Проверка размера файла
        file_size = os.path.getsize(pcap_file)
        print(f"📊 Размер файла: {file_size / 1024:.1f} КБ")

        if file_size == 0:
            print(f"❌ Файл пустой!")
            return None

        try:
            # Извлечение признаков
            print("\n📊 ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ИЗ ТРАФИКА...")
            features_df = self.feature_extractor.process_pcap_file(pcap_file)

            if features_df is None or features_df.empty or len(features_df) < 2:
                print(f"❌ Извлечено только {len(features_df) if features_df is not None else 0} записей")
                print("⚠️  Возможно, файл содержит слишком мало пакетов или неподдерживаемый формат")
                return None

            print(f"✅ Извлечено записей: {len(features_df)}")
            print(f"📈 Общий объем трафика: {features_df['total_bytes'].sum() / (1024 * 1024):.2f} МБ")

            # Обучение или загрузка модели
            if model_type == 'random_forest':
                print("\n🤖 ИСПОЛЬЗУЕТСЯ МОДЕЛЬ: Random Forest (классификация)")
                features_df = self._create_detailed_labels(features_df)
                model = RandomForestAnomalyDetector()
                model.train(features_df, verbose=False)
                results_df = model.predict(features_df, verbose=False)
                anomaly_column = 'prediction'

            else:  # isolation_forest
                print("\n🤖 ИСПОЛЬЗУЕТСЯ МОДЕЛЬ: Isolation Forest (обнаружение аномалий)")
                model = IsolationForestAnomalyDetector()
                model.train(features_df, verbose=False)
                results_df = model.predict(features_df, verbose=False)
                anomaly_column = 'is_anomaly'

            # Вывод подробных результатов
            print(f"\n{'=' * 70}")
            print("📊 РЕЗУЛЬТАТЫ АНАЛИЗА")
            print(f"{'=' * 70}")

            self._print_detailed_results(results_df, model_type)

            # Визуализация
            self.simple_plot_anomalies(results_df)

            # Сохранение результатов
            output_file = self.save_results(results_df, pcap_file, model_type)

            # Рекомендации
            self._print_recommendations(results_df, output_file, model_type)

            print(f"\n{'=' * 70}")
            print("✅ АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
            print(f"{'=' * 70}")
            return results_df

        except Exception as e:
            print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_detailed_labels(self, df):
        """Создание детализированных меток для обучения"""
        try:
            df = df.copy()
            df['label'] = 'normal'

            # Более сложные правила для реалистичных меток
            if len(df) > 3:
                # 1. Сканирование портов (много уникальных портов)
                port_threshold = df['unique_dst_ports'].quantile(0.85)
                if port_threshold > 20:
                    port_scan_mask = df['unique_dst_ports'] > port_threshold
                    df.loc[port_scan_mask, 'label'] = 'port_scan'

                # 2. DDoS (очень много пакетов)
                pps_threshold = df['packet_count'].quantile(0.9)
                if pps_threshold > 500:
                    ddos_mask = df['packet_count'] > pps_threshold
                    df.loc[ddos_mask, 'label'] = 'ddos'

                # 3. SYN Flood (низкое соотношение SYN/ACK)
                syn_ratio_threshold = df['syn_ack_ratio'].quantile(0.15)
                syn_count_threshold = df['syn_count'].quantile(0.85)
                if syn_ratio_threshold < 0.2 and syn_count_threshold > 50:
                    syn_mask = (df['syn_ack_ratio'] < syn_ratio_threshold) & (df['syn_count'] > syn_count_threshold)
                    df.loc[syn_mask, 'label'] = 'syn_flood'

                # 4. Brute Force (много маленьких пакетов)
                if 'packets_per_second' in df.columns:
                    # Преобразуем в числовой формат
                    df['packets_per_second'] = pd.to_numeric(df['packets_per_second'], errors='coerce').fillna(0)
                    pps_high = df['packets_per_second'].quantile(0.9)
                    bytes_low = df['avg_packet_size'].quantile(0.1)
                    if pps_high > 100 and bytes_low < 100:
                        brute_mask = (df['packets_per_second'] > pps_high) & (df['avg_packet_size'] < bytes_low)
                        df.loc[brute_mask, 'label'] = 'brute_force'

            # Считаем распределение меток
            label_counts = df['label'].value_counts()
            if len(label_counts) > 1:
                print(f"📋 Созданы метки: {', '.join([f'{label}: {count}' for label, count in label_counts.items()])}")

            return df

        except Exception as e:
            print(f"⚠️  Ошибка при создании меток: {e}")
            df['label'] = 'normal'
            return df

    def _print_detailed_results(self, results_df, model_type):
        """Вывод детализированных результатов анализа"""
        try:
            total_intervals = len(results_df)

            if model_type == 'random_forest' and 'prediction' in results_df.columns:
                # Анализ для Random Forest
                prediction_counts = results_df['prediction'].value_counts()

                print(f"\n📈 РАСПРЕДЕЛЕНИЕ ТИПОВ ТРАФИКА:")
                for label, count in prediction_counts.items():
                    percentage = (count / total_intervals * 100)
                    label_display = label.upper() if label != 'normal' else 'НОРМАЛЬНЫЙ'
                    print(f"   • {label_display:<15} {count:>4} интервалов ({percentage:>5.1f}%)")

                # Аномалии (все кроме normal)
                anomalies_df = results_df[
                    results_df['prediction'] != 'normal'].copy()  # Используем .copy() чтобы избежать предупреждения
                anomaly_count = len(anomalies_df)
                anomaly_percent = (anomaly_count / total_intervals * 100) if total_intervals > 0 else 0

                print(f"\n⚠️  ОБНАРУЖЕННЫЕ АНОМАЛИИ:")
                if anomaly_count > 0:
                    anomaly_types = anomalies_df['prediction'].value_counts()
                    for anomaly_type, count in anomaly_types.items():
                        anomaly_name = anomaly_type.replace('_', ' ').title()
                        print(f"   • {anomaly_name:<20} {count:>4} интервалов")
                else:
                    print("   • Аномалии не обнаружены")

                # Временной анализ
                if 'timestamp' in results_df.columns and anomaly_count > 0:
                    print(f"\n🕒 ВРЕМЕННОЙ АНАЛИЗ АНОМАЛИЙ:")
                    # Создаем копию для преобразования времени
                    anomalies_df_timestamp = anomalies_df.copy()
                    anomalies_df_timestamp['timestamp'] = pd.to_datetime(anomalies_df_timestamp['timestamp'])
                    time_range = anomalies_df_timestamp['timestamp'].max() - anomalies_df_timestamp['timestamp'].min()

                    print(f"   • Первая аномалия: {anomalies_df_timestamp['timestamp'].min()}")
                    print(f"   • Последняя аномалия: {anomalies_df_timestamp['timestamp'].max()}")
                    print(f"   • Длительность аномального периода: {time_range}")

                    # Анализ по часам
                    if len(anomalies_df_timestamp) > 1:
                        hours = anomalies_df_timestamp['timestamp'].dt.hour
                        peak_hour = hours.mode()
                        if not peak_hour.empty:
                            print(f"   • Пиковый час аномалий: {peak_hour.iloc[0]}:00")

            elif model_type == 'isolation_forest' and 'is_anomaly' in results_df.columns:
                # Анализ для Isolation Forest
                normal_count = len(results_df[~results_df['is_anomaly']])
                anomaly_count = len(results_df[results_df['is_anomaly']])

                print(f"\n📈 РЕЗУЛЬТАТЫ ОБНАРУЖЕНИЯ АНОМАЛИЙ:")
                print(
                    f"   • НОРМАЛЬНЫЙ ТРАФИК:    {normal_count:>4} интервалов ({(normal_count / total_intervals * 100):>5.1f}%)")
                print(
                    f"   • АНОМАЛЬНЫЙ ТРАФИК:    {anomaly_count:>4} интервалов ({(anomaly_count / total_intervals * 100):>5.1f}%)")

            print(f"\n{'=' * 40}")
            print("📊 ИТОГОВАЯ СТАТИСТИКА")
            print(f"{'=' * 40}")
            print(f"   • Всего временных интервалов: {total_intervals}")
            print(f"   • Процент подозрительного трафика: {(anomaly_count / total_intervals * 100):.1f}%")

            # Дополнительная статистика
            if 'total_bytes' in results_df.columns:
                total_traffic_mb = results_df['total_bytes'].sum() / (1024 * 1024)
                avg_traffic_kb = results_df['total_bytes'].mean() / 1024
                print(f"   • Общий объем трафика: {total_traffic_mb:.2f} МБ")
                print(f"   • Средний объем на интервал: {avg_traffic_kb:.1f} КБ")

            if 'packet_count' in results_df.columns:
                total_packets = results_df['packet_count'].sum()
                avg_packets = results_df['packet_count'].mean()
                print(f"   • Всего пакетов: {total_packets:,}")
                print(f"   • Среднее пакетов на интервал: {avg_packets:.1f}")

        except Exception as e:
            print(f"⚠️  Ошибка при выводе результатов: {e}")

    def _print_recommendations(self, results_df, output_file, model_type):
        """Вывод рекомендаций по результатам анализа"""
        print(f"\n{'=' * 70}")
        print("💡 РЕКОМЕНДАЦИИ И ДЕЙСТВИЯ")
        print(f"{'=' * 70}")

        try:
            total_intervals = len(results_df)

            if model_type == 'random_forest' and 'prediction' in results_df.columns:
                anomalies_df = results_df[results_df['prediction'] != 'normal'].copy()
                anomaly_count = len(anomalies_df)

                if anomaly_count > 0:
                    # Определяем типы атак
                    attack_types = anomalies_df['prediction'].unique()
                    attack_types = [at for at in attack_types if at != 'normal']

                    print(f"\n📋 ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ:")
                    print(f"   В сетевом трафике обнаружены {len(attack_types)} типа атак:")

                    for attack_type in attack_types:
                        count = len(anomalies_df[anomalies_df['prediction'] == attack_type])
                        attack_name = attack_type.replace('_', ' ').title()
                        print(f"   • {attack_name}: {count} временных интервалов")

                    anomaly_percent = (anomaly_count / total_intervals * 100)
                    print(f"   • {anomaly_percent:.1f}% трафика является подозрительным")

                else:
                    print(f"\n✅ Трафик чист. Аномалии не обнаружены.")

            elif model_type == 'isolation_forest' and 'is_anomaly' in results_df.columns:
                anomaly_count = len(results_df[results_df['is_anomaly']])

                if anomaly_count > 0:
                    print(f"\n📋 ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ:")
                    print(f"   • Обнаружено {anomaly_count} аномальных временных интервалов")
                    print(f"   • {(anomaly_count / total_intervals * 100):.1f}% трафика является подозрительным")
                else:
                    print(f"\n✅ Трафик чист. Аномалии не обнаружены.")

            # Общие рекомендации
            if anomaly_count > 0:
                print(f"\n🚀 РЕКОМЕНДУЕМЫЕ ДЕЙСТВИЯ:")
                print(f"   1. 📄 ОТКРЫТЬ CSV ФАЙЛ с результатами: {os.path.basename(output_file)}")
                print(f"   2. 🔍 НАЙТИ IP-АДРЕСА, участвующие в атаках (столбец 'ip')")
                print(f"   3. 🕒 ПРОВЕРИТЬ ВРЕМЕННЫЕ МЕТКИ аномалий (столбец 'timestamp')")
                print(f"   4. 🛡️  ПРИНЯТЬ МЕРЫ по блокировке подозрительных IP в фаерволе")

                if 'total_bytes' in results_df.columns:
                    total_traffic = results_df['total_bytes'].sum() / (1024 * 1024)  # в МБ
                    print(f"   5. 📊 ПРОАНАЛИЗИРОВАТЬ ОБЪЕМ трафика: {total_traffic:.2f} МБ")

                print(f"\n📌 Таким образом, программа предоставляет понятную визуализацию")
                print(f"   и детальный текстовый отчет о сетевых аномалиях,")
                print(f"   помогая специалистам по безопасности быстро идентифицировать угрозы.")
            else:
                print(f"\n✅ Все в порядке. Рекомендуется продолжить регулярный мониторинг.")

        except Exception as e:
            print(f"⚠️  Ошибка при выводе рекомендаций: {e}")

    def save_results(self, results_df, pcap_file, model_type):
        """Сохранение результатов анализа"""
        try:
            os.makedirs('results', exist_ok=True)

            base_name = os.path.splitext(os.path.basename(pcap_file))[0]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"results/{base_name}_{model_type}_{timestamp}.csv"

            # Сохраняем данные
            results_df.to_csv(output_file, index=False, encoding='utf-8')

            print(f"\n💾 РЕЗУЛЬТАТЫ СОХРАНЕНЫ:")
            print(f"   • Файл: {output_file}")
            print(f"   • Размер: {os.path.getsize(output_file) / 1024:.1f} КБ")
            print(f"   • Записей: {len(results_df)}")

            # Создаем краткий отчет
            report_file = f"results/{base_name}_{model_type}_{timestamp}_report.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("ОТЧЕТ ПО АНАЛИЗУ СЕТЕВОГО ТРАФИКА\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Исходный файл: {pcap_file}\n")
                f.write(f"Модель анализа: {model_type}\n")
                f.write(f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Всего интервалов: {len(results_df)}\n\n")

                if model_type == 'random_forest' and 'prediction' in results_df.columns:
                    anomalies = results_df[results_df['prediction'] != 'normal']
                    f.write("ОБНАРУЖЕННЫЕ АНОМАЛИИ:\n")
                    for attack_type in anomalies['prediction'].unique():
                        count = len(anomalies[anomalies['prediction'] == attack_type])
                        f.write(f"  • {attack_type.replace('_', ' ').title()}: {count}\n")

            return output_file

        except Exception as e:
            print(f"⚠️  Не удалось сохранить результаты: {e}")
            return None


def get_pcap_file():
    """Простой запрос имени файла"""
    print(f"\n📁 Текущая директория: {os.getcwd()}")

    # Показать доступные файлы
    pcap_files = [f for f in os.listdir('.') if f.lower().endswith(('.pcap', '.pcapng'))]

    if pcap_files:
        print(f"📂 Доступные PCAP файлы:")
        for i, filename in enumerate(pcap_files, 1):
            size_kb = os.path.getsize(filename) / 1024
            print(f"   {i}. {filename} ({size_kb:.1f} КБ)")
        print()

    filename = input("📁 Введите имя PCAP файла: ").strip()

    if not filename:
        print("❌ Имя файла не может быть пустым!")
        return None

    # Добавляем расширение если его нет
    if not filename.lower().endswith(('.pcap', '.pcapng')):
        filename += '.pcap'

    if not os.path.exists(filename):
        print(f"❌ Файл '{filename}' не найден!")
        return None

    return filename


def main():
    print("\n" + "=" * 70)
    print("           NETWORK ANOMALY DETECTOR v1.0")
    print("  Система обнаружения сетевых аномалий и кибератак")
    print("=" * 70)

    # Создаем тестовый файл если нет ни одного PCAP
    pcap_files = [f for f in os.listdir('.') if f.lower().endswith(('.pcap', '.pcapng'))]

    if not pcap_files:
        print("⚠️  В текущей директории нет PCAP файлов")
        create_test = input("Создать тестовый файл с примерами атак? (да/нет): ").strip().lower()
        if create_test in ['да', 'yes', 'y', 'д']:
            try:
                from scapy.all import Ether, IP, TCP, wrpcap
                import random

                packets = []

                # Нормальный трафик
                for i in range(80):
                    p = Ether() / IP(src=f"192.168.1.{random.randint(1, 50)}",
                                     dst=f"10.0.0.{random.randint(1, 10)}") / TCP(dport=80,
                                                                                  sport=random.randint(1024, 65535))
                    packets.append(p)

                # Сканирование портов
                for port in range(1, 101):
                    p = Ether() / IP(src="192.168.1.100", dst="10.0.0.1") / TCP(dport=port, flags="S")
                    packets.append(p)

                # DDoS атака
                for i in range(200):
                    p = Ether() / IP(src=f"10.1.1.{random.randint(1, 254)}", dst="192.168.1.1") / TCP(dport=80,
                                                                                                      flags="S")
                    packets.append(p)

                wrpcap("demo_attack.pcap", packets)
                print("✅ Создан демонстрационный файл: demo_attack.pcap")
                print("   Содержит: нормальный трафик + сканирование портов + DDoS")
                pcap_files = ['demo_attack.pcap']

            except ImportError:
                print("⚠️  Не удалось создать тестовый файл (требуется Scapy)")

    detector = NetworkAnomalyDetector()

    while True:
        filename = get_pcap_file()
        if filename:
            # Выбор модели
            print(f"\n🤖 ВЫБОР МОДЕЛИ АНАЛИЗА для файла '{filename}':")
            print("   1. Random Forest (рекомендуется) - классификация типов атак")
            print("   2. Isolation Forest - обнаружение неизвестных аномалий")

            model_choice = input("Ваш выбор (1 или 2): ").strip()
            model_type = 'random_forest' if model_choice == '1' else 'isolation_forest'

            # Запуск анализа
            print(f"\n🚀 ЗАПУСК АНАЛИЗА...")
            results = detector.analyze_pcap(filename, model_type)

            if results is not None:
                another = input("\n🔁 Проанализировать другой файл? (да/нет): ").strip().lower()
                if another not in ['да', 'yes', 'y', 'д']:
                    print("\n👋 ЗАВЕРШЕНИЕ РАБОТЫ.")
                    break
            else:
                retry = input("\n🔄 Попробовать другой файл? (да/нет): ").strip().lower()
                if retry not in ['да', 'yes', 'y', 'д']:
                    print("\n👋 ЗАВЕРШЕНИЕ РАБОТЫ.")
                    break
        else:
            exit_choice = input("\n🚪 Выйти из программы? (да/нет): ").strip().lower()
            if exit_choice in ['да', 'yes', 'y', 'д']:
                print("\n👋 ЗАВЕРШЕНИЕ РАБОТЫ.")
                break


if __name__ == "__main__":
    main()