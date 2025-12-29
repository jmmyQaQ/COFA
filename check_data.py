import torch
from dataset import load_fairness_dataset
import sys

datasets_to_check = ['dblp', 'pokec_z', 'pokec_n']

print("=== Data Loading Check ===")
for name in datasets_to_check:
    print(f"\nChecking {name}...")
    try:
        data = load_fairness_dataset(name)
        print(f"✅ SUCCESS: {name}")
        print(f"   - Nodes: {data.num_nodes}")
        print(f"   - Edges: {data.edge_index.shape[1]}")
        print(f"   - Sensitive Attr Shape: {data.sens.shape}")
    except Exception as e:
        print(f"❌ FAILED: {name}")
        print(f"   Error: {e}")