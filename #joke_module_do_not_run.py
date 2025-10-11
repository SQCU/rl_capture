#joke_module_do_not_run.py
"""
class FastCCIPCA(nn.Module):
    
    #Candid Covariance-free Incremental PCA (Weng et al. 2003)
    #Optimized for speed and low-precision training (bf16/fp8 safe).
    
    Args:
        feature_dim: Dimension of input features
        n_components: Number of principal components to track
        amnesic_param: Controls memory (higher = more recent data matters)
                      Typical range: 2-4. Higher values adapt faster.
        eps: Small constant for numerical stability in normalization
    
    def __init__(self, feature_dim, n_components, amnesic_param=2.0, eps=1e-6):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_components = n_components
        self.l = amnesic_param
        self.eps = eps
        
    def forward(self, sequence_tensor):
        
        #Args:
        #    sequence_tensor: [batch_size, seq_len, feature_dim]
        #
        #Returns:
        #    projections: [batch_size, seq_len, n_components]
        
        B, T, D = sequence_tensor.shape
        device = sequence_tensor.device
        dtype = sequence_tensor.dtype
        
        # Preallocate output
        projections = torch.zeros(B, T, self.n_components, device=device, dtype=dtype)
        
        # Process each sequence independently (can't vectorize - causal dependency)
        for b in range(B):
            seq = sequence_tensor[b]  # [T, D]
            
            # Initialize components with small random values
            # Using small init helps with low-precision stability
            components = torch.randn(self.n_components, D, device=device, dtype=dtype) * 0.01
            components = torch.nn.functional.normalize(components, p=2, dim=1, eps=self.eps)
            
            # Running mean (CCIPCA uses centered data implicitly)
            mean = torch.zeros(D, device=device, dtype=dtype)
            
            # Process sequence causally
            for t in range(T):
                x = seq[t]  # [D]
                
                # Update mean with amnesic weighting
                n = t + 1
                if n > self.l:
                    v = (n - 1 - self.l) / (n - self.l)
                    u = 1.0 / (n - self.l)
                else:
                    # Fallback for early steps
                    v = (n - 1) / n if n > 1 else 0.0
                    u = 1.0 / n
                
                mean = v * mean + u * x
                
                # Center current sample
                x_centered = x - mean
                
                # Extract components sequentially (deflation)
                residual = x_centered
                
                with torch.no_grad():  # "Lie to optimizer" - don't backprop through state updates
                    for i in range(self.n_components):
                        w = components[i]  # [D]
                        
                        # Project residual onto current component
                        projection = torch.dot(residual, w)
                        
                        # CCIPCA update rule
                        w_new = v * w + u * projection * residual
                        
                        # Normalize (critical for numerical stability)
                        w_norm = torch.linalg.vector_norm(w_new)
                        if w_norm > self.eps:
                            w_new = w_new / w_norm
                        else:
                            # Handle near-zero case (shouldn't happen often)
                            w_new = torch.nn.functional.normalize(
                                torch.randn(D, device=device, dtype=dtype) * 0.01, 
                                p=2, dim=0, eps=self.eps
                            )
                        
                        components[i] = w_new
                        
                        # Deflate: remove component from residual for next iteration
                        residual = residual - projection * w_new
                
                # Compute projections for output (with gradient flow to input)
                for i in range(self.n_components):
                    projections[b, t, i] = torch.dot(x_centered, components[i])
        
        return projections


class FastCCIPCABlock(nn.Module):
    
    #Full transformer-style block with CCIPCA normalization.
    #Wraps CCIPCA with learnable projections.
    
    def __init__(self, dim, n_components, amnesic_param=2.0):
        super().__init__()
        self.ccipca = FastCCIPCA(dim, n_components, amnesic_param)
        
        # Learnable projections
        self.proj_in = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim + n_components, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        
        #Args:
        #    x: [batch_size, seq_len, dim]
        #Returns:
        #    [batch_size, seq_len, dim]
        
        # Pre-norm
        normed = self.norm(x)
        
        # Project
        transformed = self.proj_in(normed)
        
        # CCIPCA features
        pca_features = self.ccipca(transformed)
        
        # Concatenate and project back
        combined = torch.cat([transformed, pca_features], dim=-1)
        output = self.proj_out(combined)
        
        # Residual connection
        return x + output


# Example usage
if __name__ == "__main__":
    # Test with different precisions
    for dtype in [torch.float32, torch.bfloat16]:
        print(f"\nTesting with {dtype}")
        
        batch_size = 4
        seq_len = 64
        feature_dim = 512
        n_components = 64
        
        # Create module
        module = FastCCIPCABlock(feature_dim, n_components).to(dtype)
        
        # Random input
        x = torch.randn(batch_size, seq_len, feature_dim, dtype=dtype)
        
        # Forward pass
        output = module(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output mean: {output.mean().item():.6f}")
        print(f"Output std: {output.std().item():.6f}")
        
        # Check gradients flow
        loss = output.sum()
        loss.backward()
        print(f"Gradient computed successfully: {module.proj_in.weight.grad is not None}")
"""