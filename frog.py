import numpy as np

from sklearn.decomposition import PCA

sample_tensor = np.random.rand(1, 1024)

dim = 512

pca = PCA(n_components=dim)

reduced = pca.fit_transform(sample_tensor)

print(sample_tensor.shape)
print(reduced.shape)