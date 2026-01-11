from main import NetworkAnomalyDetector

filename = input("Введите имя PCAP файла (например: 1.pcap): ").strip()

detector = NetworkAnomalyDetector()
results = detector.analyze_pcap(filename, model_type='random_forest')

# Анализируем с Isolation Forest (если нужно)
# results = detector.analyze_pcap(filename, model_type='isol1p2ation_forest')