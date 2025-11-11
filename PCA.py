import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Generate synthetic 2D data
np.random.seed(0)
X = np.dot(np.random.rand(2, 2), np.random.randn(2, 200)).T

# 2. Center & scale the data
X_scaled = StandardScaler().fit_transform(X)

# 3. Apply PCA (2D → 1D)
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)

# 4. Reconstruct projected data (back to 2D for visualization)
X_projected = pca.inverse_transform(X_pca)

# 5. Plot original and projected data
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.3, label="Original Data")
plt.scatter(X_projected[:, 0], X_projected[:, 1], alpha=0.8, label="Projected (1D) → Reconstructed", color='red')
plt.axis('equal')
plt.legend()
plt.title("PCA as Linear Transformation: 2D → 1D → 2D")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
