import numpy as np
from numpy.random import default_rng
from scipy.special import erfc
from scipy.stats import chi2
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import struct
import time
import hashlib
from typing import Tuple, Optional, List, Dict
from .mathematics import MathematicalCore

# encryption.py
import hashlib
import time
from .mathematics import MathematicalCore

class EnhancedTimeline:
    def __init__(self):
        self.markers = {}
        self.checksum_history = []
        self.mathematical_core = MathematicalCore()

    def create_marker(self, seed: int, message_id: int, message: bytes, entropy: float, layer: int = 1) -> dict:
        """Create enhanced timeline marker with specific layer and entropy."""
        timeline = self.mathematical_core.generate_entry_timeline(seed)
        layer_value = self.mathematical_core.compute_layer(seed, layer)
        
        marker = {
            'seed': seed,
            'id': message_id,
            'timestamp': time.time(),
            'entropy': entropy,
            'layer': layer_value,
            'timeline': timeline,
            'checksum': self.generate_checksum(seed, message_id, message)
        }
        self.markers[seed] = marker
        return marker

    def verify_marker(self, seed: int, message_id: int, message: bytes) -> bool:
        """Verify if a given marker's checksum matches calculated checksum."""
        if seed not in self.markers:
            return False
        marker = self.markers[seed]
        expected_checksum = self.generate_checksum(seed, message_id, message)
        return marker['checksum'] == expected_checksum        

    def generate_checksum(self, seed: int, message_id: int, message: bytes) -> str:
        """Generate a checksum for a marker, including timeline properties."""
        timeline_str = str(self.mathematical_core.generate_entry_timeline(seed))
        checksum_data = f"{seed}{message_id}{message}{timeline_str}".encode()
        checksum = hashlib.sha256(checksum_data).hexdigest()
        self.checksum_history.append(checksum)
        return checksum

    def verify_cumulative_hash(self, cumulative_hash: str) -> bool:
        """Verify cumulative hash by comparing it to combined checksums only if there are markers."""
        if not self.markers:
            print("No markers available to verify cumulative hash.")
            return False

        combined_checksums = ''.join(self.markers[seed]['checksum'] for seed in sorted(self.markers))
        expected_hash = hashlib.sha256(combined_checksums.encode()).hexdigest()
        return cumulative_hash == expected_hash


    def get_visualization_data(self) -> Dict[str, List[float]]:
        """Provide data for visualization in the format expected by VisualizationTab."""
        if not self.markers:
            return {
                'timestamps': [],
                'layers': [],
                'entropies': [],
                'depths': [],
                'seeds': [],
                'checksum_counts': []
            }

        visualization_data = {
            'timestamps': [],
            'layers': [],
            'entropies': [],
            'depths': [],
            'seeds': [],
            'checksum_counts': []
        }

        for seed, marker in self.markers.items():
            visualization_data['timestamps'].append(marker['timestamp'])
            visualization_data['layers'].append(marker['layer'])
            visualization_data['entropies'].append(marker['entropy'])
            visualization_data['depths'].append(len(marker['timeline']))
            visualization_data['seeds'].append(seed)
            visualization_data['checksum_counts'].append(len(self.checksum_history))

        return visualization_data

    def get_layer_statistics(self) -> Dict[str, float]:
        """Calculate statistical properties of the layer history."""
        layers = [marker['layer'] for marker in self.markers.values()]
        
        if not layers:
            return {
                'mean_layer': 0,
                'std_layer': 0,
                'min_layer': 0,
                'max_layer': 0,
                'total_layers': 0
            }
        
        return {
            'mean_layer': float(np.mean(layers)),
            'std_layer': float(np.std(layers)),
            'min_layer': int(np.min(layers)),
            'max_layer': int(np.max(layers)),
            'total_layers': len(set(layers))
        }
    
 
    
class QuantumStackEncryption:
    def __init__(self):
        self.messages = []
        self.perfect_seeds = []
        self.encryption_data = []
        self.timeline = EnhancedTimeline()
        self.entropy_history = []
        self.math_core = MathematicalCore()
        self.grid_values = np.zeros((10, 10, 10))
        self.used_seeds = set()  # Track used seeds
        self._initialize_grid()

    def _initialize_grid(self):
        """Initialize the grid with mathematical patterns"""
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    # Based on provided grid formula
                    self.grid_values[i, j, k] = (j - i) % 10

    def generate_adaptive_key(self, seed: int, message_length: int) -> bytes:
        """Generate encryption key"""
        rng = default_rng(seed)
        return rng.integers(0, 256, size=32, dtype=np.uint8).tobytes()

    def encrypt_with_seed(self, message: bytes, seed: int) -> Tuple[bytes, bytes]:
        """Encrypt message using seed"""
        key = self.generate_adaptive_key(seed, len(message))
        cipher = AES.new(key, AES.MODE_CBC)
        iv = cipher.iv
        ciphertext = cipher.encrypt(pad(message, AES.block_size))
        return iv, ciphertext

    def decrypt_with_seed(self, ciphertext: bytes, seed: int, iv: bytes) -> bytes:
        """Decrypt message using seed"""
        key = self.generate_adaptive_key(seed, len(ciphertext))
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return unpad(cipher.decrypt(ciphertext), AES.block_size)

    def calculate_entropy(self, bits: np.ndarray) -> float:
        """Calculate entropy of bit sequence"""
        _, counts = np.unique(bits, return_counts=True)
        probabilities = counts / len(bits)
        entropy = -np.sum(np.fromiter((p * np.log2(p) for p in probabilities if p > 0), dtype=float))
        return entropy / np.log2(2)

    def find_perfect_entropy_seed(self, message: bytes, max_attempts: int = 100000) -> Tuple[Optional[int], Optional[bytes], Optional[bytes], Optional[float]]:
        """Find seed that produces perfect entropy"""
        print(f"\nSearching for perfect entropy seed...")
        best_entropy = 0
        best_data = None

        for seed in range(1, max_attempts):
            # Skip if seed already used
            if seed in self.used_seeds:
                continue

            if seed % 10000 == 0:
                print(f"Trying seed: {seed}, Best entropy so far: {best_entropy:.10f}")
                
            iv, ciphertext = self.encrypt_with_seed(message, seed)
            ciphertext_bits = np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8))
            entropy = self.calculate_entropy(ciphertext_bits)
            
            if entropy > best_entropy:
                if best_data is None or seed not in self.used_seeds:
                    best_entropy = entropy
                    best_data = (seed, iv, ciphertext, entropy)
                
            if abs(entropy - 1.0) < 0.0000001 and seed not in self.used_seeds:
                print(f"Found perfect entropy seed: {seed}")
                self.used_seeds.add(seed)
                return best_data

        # If we found any valid seed, mark it as used
        if best_data:
            self.used_seeds.add(best_data[0])
        
        return best_data if best_data else (None, None, None, None)

    def add_message(self, message: bytes, seed: Optional[int] = None) -> Tuple[bool, float]:
        """Add a message using either provided seed or finding perfect entropy seed"""
        message_id = len(self.messages)
    
        if seed is not None and seed not in self.used_seeds:
            self.used_seeds.add(seed)
            iv, ciphertext = self.encrypt_with_seed(message, seed)
            ciphertext_bits = np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8))
            entropy = self.calculate_entropy(ciphertext_bits)
            self.store_message(message, seed, iv, ciphertext, entropy, message_id)
            return True, entropy
    
        seed, iv, ciphertext, entropy = self.find_perfect_entropy_seed(message)
        if seed is not None:
            self.store_message(message, seed, iv, ciphertext, entropy, message_id)
            return True, entropy
        return False, 0

    def store_message(self, message: bytes, seed: int, iv: bytes, ciphertext: bytes, entropy: float, message_id: int) -> None:
        """Store message data and create timeline marker."""
        self.messages.append(message)
        self.perfect_seeds.append(seed)
        self.encryption_data.append((iv, ciphertext, entropy))  # Storing encryption data for decryption
        self.timeline.create_marker(seed, message_id, message, entropy)
        self.entropy_history.append(entropy)

    def combine_messages(self) -> Optional[bytes]:
        """Combine all messages with timeline markers"""
        if not self.messages:
            return None

        combined = b''
        for i, (message, seed, (iv, ciphertext, entropy)) in enumerate(zip(
                self.messages, self.perfect_seeds, self.encryption_data)):
            marker = struct.pack('>Q', seed)
            size = struct.pack('>I', len(ciphertext))
            msg_id = struct.pack('>I', i)
            timestamp = struct.pack('>Q', int(time.time()))
            checksum = self.timeline.markers[seed]['checksum'].encode()[:32]

            combined += marker + size + msg_id + timestamp + iv + checksum + ciphertext

        return combined

    def extract_message(self, combined_data: bytes, seed: int) -> Tuple[Optional[bytes], Optional[int]]:
        """Extract specific message using seed"""
        try:
            marker = struct.pack('>Q', seed)
            pos = combined_data.find(marker)

            if pos == -1:
                return None, None

            pos += 8  # Skip seed
            size = struct.unpack('>I', combined_data[pos:pos + 4])[0]
            pos += 4  # Skip size
            msg_id = struct.unpack('>I', combined_data[pos:pos + 4])[0]
            pos += 4  # Skip message ID
            timestamp = struct.unpack('>Q', combined_data[pos:pos + 8])[0]
            pos += 8  # Skip timestamp
            iv = combined_data[pos:pos + 16]
            pos += 16  # Skip IV
            checksum = combined_data[pos:pos + 32]
            pos += 32  # Skip checksum
            ciphertext = combined_data[pos:pos + size]

            message = self.decrypt_with_seed(ciphertext, seed, iv)
            
            if not self.timeline.verify_marker(seed, msg_id, message):
                return None, None

            return message, timestamp

        except Exception as e:
            print(f"Error extracting message: {str(e)}")
            return None, None

    def format_hash(self, combined_data: bytes) -> str:
        """Format hash for display"""
        return combined_data.hex()

    def verify_hash(self, hash_data: str) -> bool:
        """Verify hash integrity by recomputing from stored data"""
        try:
            combined_data = bytes.fromhex(hash_data)
            recomputed_hash = self.format_hash(self.combine_messages())
            return recomputed_hash == hash_data
        except Exception as e:
            print(f"Error verifying hash: {str(e)}")
            return False


    def monobit_test(self, data: np.ndarray) -> float:
        """Perform the Monobit Test on the binary data"""
        ones = np.count_nonzero(data)
        zeros = len(data) - ones
        total_bits = len(data)
        s = abs(ones - zeros) / np.sqrt(total_bits)
        p_value = erfc(s / np.sqrt(2))
        return p_value

    def runs_test(self, data: np.ndarray) -> float:
        """Perform the Runs Test on the binary data"""
        ones = np.count_nonzero(data)
        zeros = len(data) - ones
        pi = ones / len(data)

        if abs(pi - 0.5) >= (2 / np.sqrt(len(data))):
            return 0.0

        vobs = 1
        for i in range(1, len(data)):
            if data[i] != data[i - 1]:
                vobs += 1

        p_value = erfc(abs(vobs - (2 * len(data) * pi * (1 - pi))) /
                      (2 * np.sqrt(2 * len(data)) * pi * (1 - pi)))
        return p_value

    def chi_squared_test(self, data: np.ndarray) -> float:
        """Perform the Chi-Squared Test"""
        ones = np.count_nonzero(data)
        zeros = len(data) - ones
        expected = len(data) / 2
        chi_squared = ((zeros - expected) ** 2 + (ones - expected) ** 2) / expected
        p_value = 1 - chi2.cdf(chi_squared, df=1)
        return p_value

    def avalanche_test(self, message: bytes, seed: int) -> float:
        """Perform the Avalanche Test"""
        key = self.generate_adaptive_key(seed, len(message))
        cipher = AES.new(key, AES.MODE_CBC)
        iv = cipher.iv
        ciphertext1 = cipher.encrypt(pad(message, AES.block_size))

        flipped_message = bytearray(message)
        flipped_message[0] ^= 0x01
        cipher = AES.new(key, AES.MODE_CBC, iv)
        ciphertext2 = cipher.encrypt(pad(flipped_message, AES.block_size))

        bits1 = np.unpackbits(np.frombuffer(ciphertext1, dtype=np.uint8))
        bits2 = np.unpackbits(np.frombuffer(ciphertext2, dtype=np.uint8))
        differing_bits = np.sum(bits1 != bits2)
        total_bits = len(bits1)
        
        return differing_bits / total_bits
