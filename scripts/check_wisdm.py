import numpy as np

d = np.load('data/wisdm/processed/wisdm_processed.npz')
data = d['data']
print('Data stats:')
print(f'Min: {data.min():.4f}, Max: {data.max():.4f}')
print(f'Mean: {data.mean():.4f}, Std: {data.std():.4f}')
print(f'Shape: {data.shape}')
