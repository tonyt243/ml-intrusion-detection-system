from scapy.all import sniff, IP, TCP, UDP, ICMP
from datetime import datetime
import threading
from collections import defaultdict

class PacketCapturer:
    def __init__(self):
        self.is_capturing = False
        self.capture_thread = None
        self.packet_callback = None
        self.connection_tracker = defaultdict(lambda: {
            'count': 0,
            'src_bytes': 0,
            'dst_bytes': 0,
            'timestamps': []
        })
        
    def extract_features(self, packet):
        """
        Extract basic features from a packet
        Simplified version - extracts core features only
        """
        if not packet.haslayer(IP):
            return None
            
        features = {
            'duration': 0,
            'src_bytes': len(packet),
            'dst_bytes': 0,
            'land': 0,
            'wrong_fragment': 0,
            'urgent': 0,
            'hot': 0,
            'num_failed_logins': 0,
            'logged_in': 0,
            'num_compromised': 0,
            'root_shell': 0,
            'su_attempted': 0,
            'num_root': 0,
            'num_file_creations': 0,
            'num_shells': 0,
            'num_access_files': 0,
            'num_outbound_cmds': 0,
            'is_host_login': 0,
            'is_guest_login': 0,
            'count': 1,
            'srv_count': 1,
            'serror_rate': 0.0,
            'srv_serror_rate': 0.0,
            'rerror_rate': 0.0,
            'srv_rerror_rate': 0.0,
            'same_srv_rate': 1.0,
            'diff_srv_rate': 0.0,
            'srv_diff_host_rate': 0.0,
            'dst_host_count': 1,
            'dst_host_srv_count': 1,
            'dst_host_same_srv_rate': 1.0,
            'dst_host_diff_srv_rate': 0.0,
            'dst_host_same_src_port_rate': 0.0,
            'dst_host_srv_diff_host_rate': 0.0,
            'dst_host_serror_rate': 0.0,
            'dst_host_srv_serror_rate': 0.0,
            'dst_host_rerror_rate': 0.0,
            'dst_host_srv_rerror_rate': 0.0
        }
        
        # Protocol type
        if packet.haslayer(TCP):
            features['protocol_type'] = 'tcp'
            tcp_layer = packet[TCP]
            features['flag'] = self._get_tcp_flag(tcp_layer)
            features['urgent'] = 1 if tcp_layer.flags.U else 0
        elif packet.haslayer(UDP):
            features['protocol_type'] = 'udp'
            features['flag'] = 'SF'
        elif packet.haslayer(ICMP):
            features['protocol_type'] = 'icmp'
            features['flag'] = 'SF'
        else:
            features['protocol_type'] = 'other'
            features['flag'] = 'OTH'
            
        # Service (simplified - based on port)
        if packet.haslayer(TCP) or packet.haslayer(UDP):
            dst_port = packet[TCP].dport if packet.haslayer(TCP) else packet[UDP].dport
            features['service'] = self._port_to_service(dst_port)
        else:
            features['service'] = 'other'
            
        return features
        
    def _get_tcp_flag(self, tcp_layer):
        """Convert TCP flags to NSL-KDD flag format"""
        flags = tcp_layer.flags
        
        if flags.S and not flags.A:
            return 'S0'  # SYN without ACK
        elif flags.S and flags.A:
            return 'S1'  # SYN-ACK
        elif flags.F:
            return 'SF'  # Normal termination
        elif flags.R:
            return 'REJ'  # Connection rejected
        elif flags.P:
            return 'SF'  # Push flag
        else:
            return 'OTH'
            
    def _port_to_service(self, port):
        """Map port number to service name"""
        port_map = {
            20: 'ftp_data', 21: 'ftp', 22: 'ssh', 23: 'telnet',
            25: 'smtp', 53: 'domain', 80: 'http', 110: 'pop_3',
            143: 'imap', 443: 'https', 3306: 'mysql', 5432: 'postgresql'
        }
        return port_map.get(port, 'private')
        
    def process_packet(self, packet):
        """Process captured packet"""
        features = self.extract_features(packet)
        
        if features and self.packet_callback:
            # Add packet metadata
            packet_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'src_ip': packet[IP].src if packet.haslayer(IP) else 'unknown',
                'dst_ip': packet[IP].dst if packet.haslayer(IP) else 'unknown',
                'protocol': features['protocol_type'],
                'size': len(packet),
                'features': features
            }
            self.packet_callback(packet_data)
            
    def start_capture(self, interface=None, callback=None):
        """Start packet capture in background thread"""
        if self.is_capturing:
            return False
            
        self.is_capturing = True
        self.packet_callback = callback
        
        def capture_loop():
            try:
                sniff(
                    iface=interface,
                    prn=self.process_packet,
                    store=False,
                    stop_filter=lambda x: not self.is_capturing
                )
            except Exception as e:
                print(f"Capture error: {e}")
                self.is_capturing = False
                
        self.capture_thread = threading.Thread(target=capture_loop, daemon=True)
        self.capture_thread.start()
        return True
        
    def stop_capture(self):
        """Stop packet capture"""
        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        return True
        
    def get_status(self):
        """Get capture status"""
        return {
            'is_capturing': self.is_capturing,
            'active': self.capture_thread.is_alive() if self.capture_thread else False
        }