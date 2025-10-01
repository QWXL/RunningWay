# src/model_runningway/state_pool.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class StateMemoryPool(nn.Module):
    """
    RunningWay Stage 1: State Memory Pool with Dynamic Allocation
    
    Total state dimension is fixed (e.g., 2048). At session start,
    it is split into:
        - system_state (optional, frozen)
        - rnn_state (long-term, decay-based)
        - window_state (short-term, sliding KV cache)
    
    Usage:
        pool = StateMemoryPool(total_dim=2048, n_layer=24)
        pool.set_system_prompt(system_emb)  # optional
        rnn_state, window_state, system_state = pool.get_states()
    """
    def __init__(self, total_dim: int, n_layer: int, device=None, dtype=torch.float32):
        super().__init__()
        self.total_dim = total_dim
        self.n_layer = n_layer
        self.device = device
        self.dtype = dtype
        
        # Learnable allocator MLP: [has_sys, task_emb] -> [alpha_sys, alpha_rnn, alpha_win]
        # Input: 1 (has_sys) + 32 (task_emb) = 33
        self.allocator = nn.Sequential(
            nn.Linear(33, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)  # outputs [alpha_sys, alpha_rnn, alpha_win]
        )
        
        # System prompt embedding (will be set externally)
        self.system_prompt_emb = None
        self.system_state = None  # Will be initialized when set_system_prompt is called
        
        # Pre-allocate fixed-size state pool for all layers
        # Shape: [n_layer, total_dim]
        self.register_buffer('state_pool', torch.zeros(n_layer, total_dim, device=device, dtype=dtype))
        
        # Store current allocation
        self.alpha_sys = 0.0
        self.alpha_rnn = 0.0
        self.alpha_win = 0.0
        self.sys_dim = 0
        self.rnn_dim = 0
        self.win_dim = 0
        
    def set_system_prompt(self, system_emb: torch.Tensor, task_emb: torch.Tensor = None):
        """
        Initialize system state from system prompt embedding.
        This should be called ONCE at the beginning of a session.
        
        Args:
            system_emb: [1, hidden_size] - encoded system prompt
            task_emb: [32] - optional task embedding (e.g., from tokenizer)
        """
        assert system_emb.shape[0] == 1, "System emb must be batch=1"
        assert len(system_emb.shape) == 2, "Shape should be [1, hidden_size]"
        
        if task_emb is None:
            task_emb = torch.zeros(32, device=system_emb.device, dtype=system_emb.dtype)
        else:
            assert task_emb.shape == (32,), "Task emb must be [32]"
        
        # Allocate state dimensions
        has_sys = torch.ones(1, device=system_emb.device, dtype=system_emb.dtype)
        allocator_input = torch.cat([has_sys, task_emb.unsqueeze(0)], dim=-1)  # [1, 33]
        alphas = self.allocator(allocator_input).squeeze(0)  # [3]
        
        # Enforce minimum for system state if present
        if alphas[0] < 0.2:
            alphas[0] = 0.2
            alphas[1:] = alphas[1:] * (0.8 / alphas[1:].sum())
        
        self.alpha_sys, self.alpha_rnn, self.alpha_win = alphas.tolist()
        self.sys_dim = max(1, int(self.alpha_sys * self.total_dim))
        self.rnn_dim = max(1, int(self.alpha_rnn * self.total_dim))
        self.win_dim = self.total_dim - self.sys_dim - self.rnn_dim
        if self.win_dim < 1:
            self.win_dim = 1
            self.rnn_dim = self.total_dim - self.sys_dim - self.win_dim
        
        # Initialize system state by encoding the system prompt
        # In practice, you'd run the system prompt through the model's TimeMixing
        # For now, we use a simple projection
        hidden_size = system_emb.shape[-1]
        self.system_proj = nn.Linear(hidden_size, self.sys_dim, device=system_emb.device, dtype=system_emb.dtype)
        with torch.no_grad():
            self.system_state = self.system_proj(system_emb).squeeze(0)  # [sys_dim]
            self.system_state = self.system_state.detach()  # Freeze
            self.system_state.requires_grad = False
        
        print(f"[StatePool] Allocated: sys={self.sys_dim}, rnn={self.rnn_dim}, win={self.win_dim} (total={self.total_dim})")
    
    def reset_for_no_system(self, task_emb: torch.Tensor = None):
        """Reset pool for sessions without system prompt."""
        if task_emb is None:
            task_emb = torch.zeros(32, device=self.state_pool.device, dtype=self.state_pool.dtype)
        has_sys = torch.zeros(1, device=self.state_pool.device, dtype=self.state_pool.dtype)
        allocator_input = torch.cat([has_sys, task_emb.unsqueeze(0)], dim=-1)
        alphas = self.allocator(allocator_input).squeeze(0)
        alphas[0] = 0.0  # Force no system state
        alphas[1:] = alphas[1:] / alphas[1:].sum()
        
        self.alpha_sys, self.alpha_rnn, self.alpha_win = alphas.tolist()
        self.sys_dim = 0
        self.rnn_dim = max(1, int(self.alpha_rnn * self.total_dim))
        self.win_dim = self.total_dim - self.rnn_dim
        if self.win_dim < 1:
            self.win_dim = 1
            self.rnn_dim = self.total_dim - self.win_dim
        
        self.system_state = None
        print(f"[StatePool] No system prompt. Allocated: rnn={self.rnn_dim}, win={self.win_dim}")
    
    def get_state_slices(self, layer_idx: int):
        """
        Get state slices for a given layer.
        Returns:
            rnn_state: [rnn_dim]
            window_state: [win_dim]
            system_state: [sys_dim] or None
        """
        pool_slice = self.state_pool[layer_idx]  # [total_dim]
        
        if self.sys_dim > 0:
            sys_state = pool_slice[:self.sys_dim]
            rnn_start = self.sys_dim
        else:
            sys_state = None
            rnn_start = 0
        
        rnn_state = pool_slice[rnn_start:rnn_start + self.rnn_dim]
        win_state = pool_slice[rnn_start + self.rnn_dim:rnn_start + self.rnn_dim + self.win_dim]
        
        return rnn_state, win_state, sys_state
    
    def update_state_pool(self, layer_idx: int, rnn_update: torch.Tensor, win_update: torch.Tensor):
        """
        Update the state pool with new RNN and Window states.
        Assumes updates are already computed and shaped correctly.
        """
        pool_slice = self.state_pool[layer_idx]
        
        start = 0
        if self.sys_dim > 0:
            # System state is frozen, do not update
            start = self.sys_dim
        
        # Update RNN state
        pool_slice[start:start + self.rnn_dim] = rnn_update
        # Update Window state
        pool_slice[start + self.rnn_dim:start + self.rnn_dim + self.win_dim] = win_update