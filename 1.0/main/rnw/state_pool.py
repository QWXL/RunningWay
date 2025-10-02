from torch import nn
import torch

class StateMemoryPool(nn.Module):
    """
    RunningWay Stage 1: State Memory Pool with Dynamic Allocation
    """
    def __init__(self, total_dim: int, n_layer: int, n_head: int, head_size: int, 
                 device=None, dtype=torch.float32):
        super().__init__()
        self.total_dim = total_dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.head_size = head_size
        self.device = device
        self.dtype = dtype
        
        # Learnable allocator MLP
        self.allocator = nn.Sequential(
            nn.Linear(33, 64),  # 1 (has_sys) + 32 (task_emb)
            nn.SiLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )
        
        # System state encoding projection
        self.system_proj = nn.ModuleList([
            nn.Linear(total_dim, total_dim // 3) for _ in range(n_layer)
        ])
        
        # Initialize with default allocation [0.3, 0.4, 0.3]
        with torch.no_grad():
            self.allocator[2].weight.data.zero_()
            self.allocator[2].bias.data.copy_(torch.tensor([0.3, 0.4, 0.3]))
        
        # State buffers
        self.system_state = None
        self.has_system_prompt = False
        
        # Pre-allocate fixed-size state pools
        # RNN state: decay-based long-term memory
        self.register_buffer('rnn_state', torch.zeros(n_layer, n_head, head_size, 
                                                     device=device, dtype=dtype))
        # Window state: sliding window cache (recent tokens)
        self.register_buffer('window_state', torch.zeros(n_layer, n_head, head_size, 
                                                        device=device, dtype=dtype))
        # System state: frozen system prompt encoding
        self.register_buffer('system_state_buffer', torch.zeros(n_layer, n_head, head_size, 
                                                               device=device, dtype=dtype))
        
        # Window management
        self.window_size = 1024
        self.window_positions = [0] * n_layer
        self.window_caches = [
            torch.zeros(1024, n_head, head_size, device=device, dtype=dtype) 
            for _ in range(n_layer)
        ]
        
        # Current allocation ratios
        self.alpha_sys = 0.3
        self.alpha_rnn = 0.4
        self.alpha_win = 0.3
        
        print(f"[StatePool] Initialized: {n_layer} layers, {n_head} heads, {head_size} head_size")
    
    def set_system_prompt(self, system_emb: torch.Tensor, model=None):
        """
        Encode system prompt and initialize system states for all layers
        """
        if system_emb is None:
            self.reset_for_no_system()
            return
            
        self.has_system_prompt = True
        B, T, C = system_emb.shape
        assert B == 1, "System prompt should have batch size 1"
        
        # Encode system prompt through all layers to get layer-wise system states
        current_states = []
        
        # Simplified system encoding - in practice, you'd run through actual model layers
        for layer_idx in range(self.n_layer):
            layer_system_state = self.system_proj[layer_idx](system_emb.mean(dim=1, keepdim=True))
            layer_system_state = layer_system_state.view(1, self.n_head, self.head_size)
            current_states.append(layer_system_state.squeeze(0))  # [n_head, head_size]
        
        # Store system states
        for layer_idx, system_state in enumerate(current_states):
            self.system_state_buffer[layer_idx] = system_state.detach()
        
        # Update allocation to prioritize system state
        self.alpha_sys, self.alpha_rnn, self.alpha_win = 0.3, 0.4, 0.3
        print(f"[StatePool] System prompt set, allocation: sys={self.alpha_sys}, "
              f"rnn={self.alpha_rnn}, win={self.alpha_win}")
    
    def reset_for_no_system(self):
        """Reset for sessions without system prompt"""
        self.has_system_prompt = False
        self.system_state_buffer.zero_()
        self.alpha_sys, self.alpha_rnn, self.alpha_win = 0.0, 0.5, 0.5
        print(f"[StatePool] No system prompt, allocation: rnn={self.alpha_rnn}, win={self.alpha_win}")
    
    def get_state_slices(self, layer_idx: int):
        """
        Get state slices for a given layer with current allocation weights
        Returns:
            rnn_state: [n_head, head_size]
            window_state: [n_head, head_size] 
            system_state: [n_head, head_size] or None
        """
        rnn_state = self.rnn_state[layer_idx] * self.alpha_rnn
        window_state = self.window_state[layer_idx] * self.alpha_win
        
        if self.has_system_prompt:
            system_state = self.system_state_buffer[layer_idx] * self.alpha_sys
        else:
            system_state = None
            
        return rnn_state, window_state, system_state
    
    def update_rnn_state(self, layer_idx: int, new_state: torch.Tensor, decay: float = 0.9):
        """Update RNN state with decay"""
        self.rnn_state[layer_idx] = (decay * self.rnn_state[layer_idx] + 
                                   (1 - decay) * new_state)
    
    def update_window_state(self, layer_idx: int, new_state: torch.Tensor):
        """Update sliding window state"""
        pos = self.window_positions[layer_idx]
        self.window_caches[layer_idx][pos] = new_state.detach()
        self.window_positions[layer_idx] = (pos + 1) % self.window_size
        
        # Update window state as mean of recent window
        start = max(0, pos - 64)  # last 64 tokens
        self.window_state[layer_idx] = self.window_caches[layer_idx][start:pos+1].mean(dim=0)
    
    def reset_all_states(self):
        """Reset all states to zero"""
        self.rnn_state.zero_()
        self.window_state.zero_()
        self.window_positions = [0] * self.n_layer
        for cache in self.window_caches:
            cache.zero_()
        print("[StatePool] All states reset")