import numpy as np
from typing import List, Tuple, Dict, Union
import math
import hashlib

class MathematicalCore:
    def __init__(self):
        self.layer_cache = {}
        self.timeline_cache = {}
        self.layer_states = {}
        self.layer_transitions = []
        self.grid_values = np.zeros((10, 10, 10))
        self._initialize_grid()
        self.previous_layer = None

    def _initialize_grid(self):
        """Initialize the grid with mathematical patterns"""
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    # Based on provided grid formula
                    self.grid_values[i, j, k] = (j - i) % 10

    def generate_entry_timeline(self, n: int) -> List[Tuple[int, int]]:
        """Generate Entry Timeline for a given number"""
        if n in self.timeline_cache:
            return self.timeline_cache[n]
            
        str_n = str(n)
        k = len(str_n)
        timeline = [(int(str_n[i]), k - i) for i in range(k)]
        self.timeline_cache[n] = timeline
        return timeline

    def compute_layer(self, value: int, layer: int = 1) -> int:
        """Compute layer based on the number of digits in the value."""
        num_digits = len(str(value))
        return num_digits

    def _smooth_layer_transition(self, old_layer: int, new_layer: int) -> int:
        """Ensure smooth transition between layers"""
        if abs(new_layer - old_layer) > 1:
            return old_layer + (1 if new_layer > old_layer else -1)
        return new_layer

    def layer_function(self, n: float, k: int) -> float:
        """Compute layer function f_k(n) with caching"""
        if n <= 1 or k < 1:
            return 0
            
        cache_key = (n, k)
        if cache_key in self.layer_cache:
            return self.layer_cache[cache_key]
            
        log_n = math.log(n)
        result = (n ** (1/k)) * (log_n ** ((k-1)/k))
        self.layer_cache[cache_key] = result
        return result

    def wave_function_hybrid(self, x: float, k: int, omega: float) -> float:
        """Compute Wave-Function Hybrid with validation"""
        if x <= 0:
            return 0
            
        # Prevent potential numerical instabilities
        x = min(x, 1e300)  # Limit x to prevent overflow
        k = min(k, 1000)   # Reasonable limit for k
        omega = min(omega, 1000)  # Limit frequency
        
        power_term = x ** (1/k)
        log_term = (math.log(x)) ** ((k-1)/k)
        wave_term = math.sin(omega * x)
        
        return power_term * log_term * wave_term

    def logarithmic_ratio(self, n: float, k: int) -> float:
        """Compute logarithmic ratio between consecutive layers with validation"""
        if n <= 1 or k < 1:
            return 0
            
        try:
            f_k = self.layer_function(n, k)
            f_k_plus_1 = self.layer_function(n, k + 1)
            
            if f_k_plus_1 == 0:
                return float('inf')
                
            ratio = (n / math.log(n)) ** (1/(k * (k+1)))
            
            # Validate ratio bounds
            if not (0 < ratio < float('inf')):
                return 0
                
            return ratio
        except (ValueError, OverflowError):
            return 0

    def adaptive_noise(self, layer: int, index: int, grid_values: np.ndarray) -> float:
        """Generate adaptive noise based on layer and grid values"""
        # Get grid value based on layer and index
        G_L_i = self._select_grid_value(grid_values, layer, index)
        
        # Generate Gaussian and exponential components
        gaussian_noise = np.random.normal(0, 0.3) * np.log1p(abs(G_L_i))
        exp_noise = np.random.exponential(0.5) * np.log1p(abs(G_L_i))
        
        # Dynamic weights based on entropy feedback
        w_g, w_e = self._compute_weights(layer, index)
        
        # Combine noises with weights
        hybrid_noise = w_g * gaussian_noise + w_e * exp_noise
        return hybrid_noise

    def _select_grid_value(self, grid: np.ndarray, layer: int, index: int) -> float:
        """Select value from grid based on layer and index"""
        dimensions = grid.shape
        coords = [(index ** (i+1)) % dim for i, dim in enumerate(dimensions)]
        return grid[tuple(coords)]

    def _compute_weights(self, layer: int, index: int) -> Tuple[float, float]:
        """Compute dynamic weights for noise combination"""
        # Start with balanced weights
        w_g = 0.5
        w_e = 0.5
        
        # Adjust based on layer depth
        layer_factor = math.exp(-layer/10)
        w_g = min(w_g + layer_factor, 1.0)
        w_e = 1.0 - w_g
        
        return w_g, w_e

    def _manage_layer_state(self, seed: int, new_layer: int) -> int:
        """Manage layer state transitions"""
        if seed in self.layer_states:
            old_layer = self.layer_states[seed]
            transition_layer = self._smooth_layer_transition(old_layer, new_layer)
            self.layer_states[seed] = transition_layer
            self.layer_transitions.append((seed, old_layer, transition_layer))
            return transition_layer
        
        self.layer_states[seed] = new_layer
        return new_layer

    def validate_layer_sequence(self, layers: List[int]) -> bool:
        """Validate sequence of layer transitions"""
        if not layers:
            return True
        
        for i in range(1, len(layers)):
            if abs(layers[i] - layers[i-1]) > 1:
                return False
        return True

    def get_layer_statistics(self) -> Dict:
        """Calculate statistical properties of layers"""
        if not self.layer_transitions:
            return {}
            
        layers = [t[2] for t in self.layer_transitions]
        return {
            'mean_layer': np.mean(layers),
            'std_layer': np.std(layers),
            'min_layer': min(layers),
            'max_layer': max(layers),
            'total_transitions': len(self.layer_transitions)
        }

    def clear_caches(self):
        """Clear all caches and reset states"""
        self.layer_cache.clear()
        self.timeline_cache.clear()
        self.layer_states.clear()
        self.layer_transitions.clear()
        self.previous_layer = None