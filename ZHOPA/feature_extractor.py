# feature_extractor.py
from scapy.all import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import collections

from scapy.layers.inet import TCP, IP


class NetworkFeatureExtractor:
    def __init__(self, window_size=60):  # окно в 60 секунд
        self.window_size = window_size
        self.current_window = {}
        self.features_history = []

    def process_packet(self, packet):
        """Обработка одного пакета"""
        if IP in packet:
            ip_src = packet[IP].src
            ip_dst = packet[IP].dst
            timestamp = packet.time

            # Инициализация записи для IP, если её нет
            if ip_src not in self.current_window:
                self.current_window[ip_src] = {
                    'packet_count': 0,
                    'total_size': 0,
                    'dst_ports': set(),
                    'syn_count': 0,
                    'syn_ack_count': 0,
                    'first_seen': timestamp,
                    'last_seen': timestamp
                }

            # Обновление статистики
            self.current_window[ip_src]['packet_count'] += 1
            self.current_window[ip_src]['total_size'] += len(packet)
            self.current_window[ip_src]['last_seen'] = timestamp

            # Анализ TCP флагов
            if TCP in packet:
                dst_port = packet[TCP].dport
                self.current_window[ip_src]['dst_ports'].add(dst_port)

                flags = packet[TCP].flags
                if flags == 'S':  # SYN
                    self.current_window[ip_src]['syn_count'] += 1
                elif flags == 'SA':  # SYN-ACK
                    self.current_window[ip_src]['syn_ack_count'] += 1

    def extract_features(self, ip):
        """Извлечение признаков для IP-адреса"""
        data = self.current_window[ip]
        time_span = data['last_seen'] - data['first_seen']

        # Преобразуем timestamp в float для корректной работы с datetime
        last_seen_timestamp = float(data['last_seen'])

        features = {
            'ip': ip,
            'timestamp': datetime.fromtimestamp(last_seen_timestamp),  # Исправлено здесь
            'packet_count': data['packet_count'],
            'total_bytes': data['total_size'],
            'avg_packet_size': data['total_size'] / data['packet_count'] if data['packet_count'] > 0 else 0,
            'unique_dst_ports': len(data['dst_ports']),
            'syn_count': data['syn_count'],
            'syn_ack_ratio': data['syn_ack_count'] / data['syn_count'] if data['syn_count'] > 0 else 1,
            'packets_per_second': data['packet_count'] / time_span if time_span > 0 else data['packet_count'],
            'bytes_per_second': data['total_size'] / time_span if time_span > 0 else data['total_size'],
            'port_scan_score': len(data['dst_ports']) / data['packet_count'] if data['packet_count'] > 0 else 0
        }
        return features

    def process_pcap_file(self, pcap_file):
        """Обработка pcap файла"""
        print(f"Обработка файла {pcap_file}...")

        def packet_handler(packet):
            self.process_packet(packet)

        sniff(offline=pcap_file, prn=packet_handler, store=0)

        # Извлечение признаков для всех IP в текущем окне
        for ip in self.current_window.keys():
            features = self.extract_features(ip)
            self.features_history.append(features)

        return pd.DataFrame(self.features_history)