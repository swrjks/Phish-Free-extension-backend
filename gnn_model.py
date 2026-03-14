
from __future__ import annotations

import os
import math
import time
import warnings
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

# Try to import torch and PyG; if fails, set flags and provide fallbacks
_HAS_TORCH = True
_HAS_PYG = True
_HAS_TORCH_CLUSTER = True

try:
    import torch
except Exception as e:
    torch = None
    _HAS_TORCH = False
    _HAS_PYG = False
    _HAS_TORCH_CLUSTER = False
    warnings.warn(f"torch not available: {e}. GNN functionality disabled.", UserWarning)

if _HAS_TORCH:
    try:
        # Try PyG imports
        from torch_geometric.data import Data
        from torch_geometric.nn import SAGEConv, GATConv
        from sklearn.metrics import roc_auc_score
    except Exception as e:
        # PyG partially failed (compiled extensions missing) — mark as unavailable
        Data = None
        SAGEConv = None
        GATConv = None
        _HAS_PYG = False
        warnings.warn(f"torch_geometric import partially failed: {e}. Falling back to lightweight GraphEngine. "
                      "Full GNN support requires installing torch-geometric wheels that match your torch+cuda.", UserWarning)

# ---------------------------
# If full PyG available, define GraphSAGEModel + helpers
# ---------------------------
if _HAS_PYG and torch is not None and Data is not None and SAGEConv is not None:
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv, GATConv

    class GraphSAGEModel(torch.nn.Module):
        def __init__(self, in_channels: int, hidden: int = 128, num_layers: int = 2, dropout: float = 0.2, use_gat: bool = False):
            super().__init__()
            conv = GATConv if use_gat else SAGEConv
            self.convs = torch.nn.ModuleList()
            self.convs.append(conv(in_channels, hidden))
            for _ in range(num_layers - 1):
                self.convs.append(conv(hidden, hidden))
            self.classifier = torch.nn.Linear(hidden, 1)
            self.dropout = dropout

        def forward(self, x, edge_index):
            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            logits = self.classifier(x).squeeze(-1)
            return torch.sigmoid(logits)

    def build_graph(node_features_np: np.ndarray, edge_index_np: np.ndarray, labels_np: Optional[np.ndarray] = None):
        """
        Build a PyG Data object from numpy arrays.
        node_features_np: [N, F]
        edge_index_np: [2, E]
        labels_np: optional [N]
        """
        x = torch.tensor(node_features_np, dtype=torch.float)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        if labels_np is not None:
            data.y = torch.tensor(labels_np, dtype=torch.float)
        return data

    def train_gnn(data, train_idx, val_idx=None, device: Optional[str] = None, out_dir: str = "./gnn_ckpt",
                  epochs: int = 80, lr: float = 1e-3, weight_decay: float = 1e-5, hidden: int = 128, num_layers: int = 2, use_gat: bool = False):
        """
        Full-graph training routine (simple). For huge graphs use neighbor sampling / NeighborLoader.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(out_dir, exist_ok=True)
        model = GraphSAGEModel(in_channels=data.num_node_features, hidden=hidden, num_layers=num_layers, use_gat=use_gat).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=(device != "cpu"))

        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        y = (data.y.to(device) if hasattr(data, "y") else None)

        best_auc = 0.0
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device != "cpu")):
                preds = model(x, edge_index)
                loss = torch.nn.functional.binary_cross_entropy(preds[train_idx], y[train_idx])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            # validation
            model.eval()
            with torch.no_grad():
                all_preds = model(x, edge_index).cpu().numpy()
                if val_idx is not None:
                    y_true = y[val_idx].cpu().numpy()
                    y_score = all_preds[val_idx]
                    try:
                        auc = roc_auc_score(y_true, y_score)
                    except Exception:
                        auc = 0.5
                    if auc > best_auc:
                        best_auc = auc
                        torch.save(model.state_dict(), os.path.join(out_dir, "best_gnn.pth"))
                    if epoch % 5 == 0:
                        print(f"[GNN] Epoch {epoch} loss {loss.item():.4f} val_auc {auc:.4f}")
        print("[GNN] Training complete. Best val AUC:", best_auc)
        return model, best_auc

# ---------------------------
# Always provide GraphEngine class (fallback-capable)
# ---------------------------
class GraphEngine:
    """
    Compatibility GraphEngine that works whether or not PyG is fully installed.
    Public methods:
      - load_graph(node_features_np, edge_index_np)
      - build_graph_from_edges(list_of_edges)
      - compute_node2vec_embeddings(dimensions=64, ...)
      - load_model(model_ckpt_path)  (if real model available)
      - predict_node_score(domain_or_idx) -> float or None
      - predict_all() -> np.ndarray (probabilities) or None
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")
        self.raw_model = None
        self.node_features = None  # numpy array shape [N,F]
        self.edge_index = None     # numpy array shape [2,E]
        self.labels = None
        self.embeddings = {}       # dict domain->vector or idx->vector
        self._nodes_map = {}       # domain->idx mapping if built from edges
        self._reverse_nodes_map = {}
        self._pyg_enabled = (_HAS_PYG and torch is not None and SAGEConv is not None) if 'SAGEConv' in globals() else False
        # if a real model checkpoint path given and PyG supports model loading, try to load later via load_model
        self.model_path = model_path

    # --------------------
    # Graph loading / building
    # --------------------
    def load_graph(self, node_features_np: np.ndarray, edge_index_np: np.ndarray, labels_np: Optional[np.ndarray] = None):
        """
        Load numeric graph arrays (numpy). This sets internal arrays used for inference.
        """
        if node_features_np is None or edge_index_np is None:
            raise ValueError("node_features_np and edge_index_np must be provided")
        self.node_features = np.asarray(node_features_np)
        self.edge_index = np.asarray(edge_index_np)
        if labels_np is not None:
            self.labels = np.asarray(labels_np)
        # build trivial index mapping if nodes not present
        N = self.node_features.shape[0]
        # if _nodes_map empty, map "0..N-1"
        if not self._nodes_map:
            self._nodes_map = {str(i): i for i in range(N)}
            self._reverse_nodes_map = {i: str(i) for i in range(N)}
        # clear embeddings so they may be recomputed
        self.embeddings = {}

    def build_graph_from_edges(self, edges: List[Tuple[str, str]]):
        """
        Build a node index from edge list [(src,dst)...] and create minimal node_features and edge_index arrays.
        This is a light-weight builder (no PyG required).
        """
        if edges is None:
            return False
        nodes = {}
        idx = 0
        for a, b in edges:
            if a not in nodes:
                nodes[a] = idx; idx += 1
            if b not in nodes:
                nodes[b] = idx; idx += 1
        self._nodes_map = nodes
        self._reverse_nodes_map = {v: k for k, v in nodes.items()}
        N = len(nodes)
        # minimal node features zeros
        self.node_features = np.zeros((N, 8), dtype=np.float32)
        # build integer edge_index
        if N == 0:
            self.edge_index = np.zeros((2, 0), dtype=np.int64)
        else:
            u = []
            v = []
            for a, b in edges:
                try:
                    u.append(nodes[a]); v.append(nodes[b])
                except KeyError:
                    continue
            if len(u) == 0:
                self.edge_index = np.zeros((2, 0), dtype=np.int64)
            else:
                self.edge_index = np.vstack([np.array(u, dtype=np.int64), np.array(v, dtype=np.int64)])
        # clear embeddings
        self.embeddings = {}
        return True

    # --------------------
    # Embedding / Node2Vec fallback (lightweight)
    # --------------------
    def compute_node2vec_embeddings(self, dimensions: int = 32, walk_length: int = 5, num_walks: int = 20):
        """
        If PyG & a real node2vec implementation is available, prefer that. Otherwise compute a deterministic lightweight embedding:
        embedding = [deg, log(1+deg), sin(deg*k), cos(deg*k) ...] truncated/padded to `dimensions`.
        """
        # If underlying raw_model has node2vec implementation, try to call it
        try:
            if hasattr(self.raw_model, "compute_node2vec") or hasattr(self.raw_model, "node2vec"):
                # attempt best-effort call
                try:
                    if hasattr(self.raw_model, "compute_node2vec"):
                        self.raw_model.compute_node2vec(dimensions=dimensions, walk_length=walk_length, num_walks=num_walks)
                    else:
                        self.raw_model.node2vec(dimensions=dimensions, walk_length=walk_length, num_walks=num_walks)
                    self.embeddings = getattr(self.raw_model, "embeddings", {}) or {}
                    return True
                except Exception:
                    pass
        except Exception:
            pass

        # Fallback path
        if self.edge_index is None or self.node_features is None:
            # nothing to compute
            self.embeddings = {}
            return False

        N = int(self.node_features.shape[0])
        deg = np.zeros((N,), dtype=np.float32)
        if self.edge_index.size:
            # edge_index shape [2,E]
            u = self.edge_index[0].astype(np.int64)
            for uu in u:
                if 0 <= int(uu) < N:
                    deg[int(uu)] += 1.0
        emb_map = {}
        for name, idx in self._nodes_map.items():
            base = float(deg[int(idx)]) if int(idx) < len(deg) else 0.0
            vec = np.zeros((dimensions,), dtype=np.float32)
            vec[0] = base
            vec[1] = math.log1p(base)
            for i in range(2, dimensions):
                # deterministic transform
                vec[i] = math.sin(base * (i + 1)) if base != 0 else 0.0
            emb_map[name] = vec
        self.embeddings = emb_map
        # Attach embeddings to raw_model if possible
        try:
            setattr(self.raw_model, "embeddings", self.embeddings)
        except Exception:
            pass
        return True

    # --------------------
    # Model loading (best-effort)
    # --------------------
    def load_model(self, model_ckpt_path: str):
        """
        Try to load a PyTorch model if environment supports. If not available, store path for manual use.
        """
        if model_ckpt_path is None:
            return False
        self.model_path = model_ckpt_path
        # If PyG + torch available and GraphSAGEModel defined, try to load
        if _HAS_PYG and torch is not None and 'GraphSAGEModel' in globals():
            try:
                in_ch = int(self.node_features.shape[1]) if self.node_features is not None else 16
                model = GraphSAGEModel(in_channels=in_ch)
                model.load_state_dict(torch.load(model_ckpt_path, map_location=self.device))
                model.to(self.device).eval()
                self.raw_model = model
                return True
            except Exception as e:
                warnings.warn(f"Failed to load GNN model checkpoint: {e}", UserWarning)
                self.raw_model = None
                return False
        # fallback: can't load model in this env
        return False

    # --------------------
    # Inference helpers
    # --------------------
    def predict_node_score(self, domain_or_idx) -> Optional[float]:
        """
        domain_or_idx: either domain string (if _nodes_map created with domain keys) or integer node index.
        Returns probability in 0..1 or None if not available.
        """
        # If raw_model exists and is a PyTorch model, run forward
        try:
            if self.raw_model is not None and torch is not None:
                # prepare x and edge_index tensors
                if self.node_features is None or self.edge_index is None:
                    return 0.0  # Default score for unknown domains
                x = torch.tensor(self.node_features, dtype=torch.float32, device=self.device)
                edge_index = torch.tensor(self.edge_index, dtype=torch.long, device=self.device)
                self.raw_model.eval()
                with torch.no_grad():
                    out = self.raw_model(x, edge_index)
                out_np = out.cpu().numpy() if hasattr(out, "cpu") else np.asarray(out)
                if isinstance(domain_or_idx, str):
                    if domain_or_idx not in self._nodes_map:
                        return 0.0  # Default score for unknown domains
                    idx = int(self._nodes_map[domain_or_idx])
                else:
                    idx = int(domain_or_idx)
                if 0 <= idx < len(out_np):
                    return float(out_np[idx])
                else:
                    return 0.0  # Default score for out of range
        except Exception:
            # fall through to embedding heuristic
            pass

        # If embeddings present, use embedding norm heuristic
        if isinstance(domain_or_idx, str):
            key = domain_or_idx
        else:
            key = str(domain_or_idx)
        if key in self.embeddings:
            vec = self.embeddings[key]
            try:
                norm = float(np.linalg.norm(vec))
                score = min(1.0, norm / (1.0 + norm))
                return float(score)
            except Exception:
                return 0.0  # Default score on error

        # If provided an integer index and embeddings are keyed by index names
        if isinstance(domain_or_idx, int):
            idx = domain_or_idx
            name = self._reverse_nodes_map.get(idx, str(idx))
            if name in self.embeddings:
                vec = self.embeddings[name]
                norm = float(np.linalg.norm(vec))
                return float(min(1.0, norm / (1.0 + norm)))
        
        # Default score for unknown domains (instead of None)
        return 0.0

    def predict_all(self) -> Optional[np.ndarray]:
        """
        Return array of node probabilities if raw_model present; otherwise try to compute using embedding heuristic.
        """
        try:
            if self.raw_model is not None and torch is not None:
                x = torch.tensor(self.node_features, dtype=torch.float32, device=self.device)
                edge_index = torch.tensor(self.edge_index, dtype=torch.long, device=self.device)
                with torch.no_grad():
                    out = self.raw_model(x, edge_index)
                out_np = out.cpu().numpy() if hasattr(out, "cpu") else np.asarray(out)
                return out_np
        except Exception:
            pass

        # embedding heuristic: map norms to 0..1
        if self.embeddings:
            N = len(self._nodes_map)
            probs = np.zeros((N,), dtype=np.float32)
            for name, idx in self._nodes_map.items():
                vec = self.embeddings.get(name)
                if vec is None:
                    probs[idx] = 0.0
                else:
                    nrm = float(np.linalg.norm(vec))
                    probs[idx] = min(1.0, nrm / (1.0 + nrm))
            return probs
        return None

# ---------------------------
# Module-level convenience: create a default engine if desired
# ---------------------------
def make_default_engine():
    return GraphEngine()
 
# Provide GraphEngine symbol for `from gnn_model import GraphEngine`
__all__ = ["GraphEngine", "make_default_engine"]
