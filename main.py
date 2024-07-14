
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.linalg import svds
import numpy as np


df = pd.read_csv('data/ratings.csv')
ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')
print(ratings_matrix.head())


ratings_matrix_filled = ratings_matrix.fillna(2.5)
R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

u, sigma, v_t = svds(R_demeaned, k=3)
print(u)


u_subset = u[:100, :]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(u_subset[:, 0], u_subset[:, 1], u_subset[:, 2], marker='o')

ax.set_title('Preferences')
ax.set_xlabel('Feat 1')
ax.set_ylabel('Feat 2')
ax.set_zlabel('Feat 3')

plt.show()

vt_subset = v_t.T[:500, :]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(vt_subset[:, 0], vt_subset[:, 1], vt_subset[:, 2], marker='o')

ax.set_title('Movie Features')
ax.set_xlabel('Feat 1')
ax.set_ylabel('Feat 2')
ax.set_zlabel('Feat 3')

plt.show()
