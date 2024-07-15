
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
import numpy as np
from tabulate import tabulate

from svd_result import SvdResult


def do_svd(r_demeaned, k):
    u, sigma, v_t = svds(r_demeaned, k=k)
    # print(u)

    u_subset = u[:100, :]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(u_subset[:, 0], u_subset[:, 1], u_subset[:, 2], marker='o')
    ax.set_title(f'Preferences k={k}')
    ax.set_xlabel('Feat 1')
    ax.set_ylabel('Feat 2')
    ax.set_zlabel('Feat 3')
    plt.show()

    vt_subset = v_t.T[:500, :]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(vt_subset[:, 0], vt_subset[:, 1], vt_subset[:, 2], marker='o')
    ax.set_title(f'Movie Features k={k}')
    ax.set_xlabel('Feat 1')
    ax.set_ylabel('Feat 2')
    ax.set_zlabel('Feat 3')
    plt.show()
    return SvdResult(u, sigma, v_t)


def print_df(df, n=10):
    df_to_print = df.iloc[:n, :n]
    print(tabulate(df_to_print.head(n), headers='keys', tablefmt='psql'))


df = pd.read_csv('data/ratings.csv')
ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')
print_df(ratings_matrix)


def compute_predicted_ratings(u, sigma, v_t):
    all_user_predicted_ratings = np.dot(np.dot(u, sigma), v_t) + user_ratings_mean.reshape(-1, 1)
    predicts_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_matrix.columns, index=ratings_matrix.index)
    return predicts_df


ratings_matrix_filled = ratings_matrix.fillna(2.5)
R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

# do_svd(R_demeaned, 3)
# do_svd(R_demeaned, 12)
svd_result = do_svd(R_demeaned, 7)
predictions_df = compute_predicted_ratings(svd_result.u, np.diag(svd_result.sigma), svd_result.v_t)
print_df(predictions_df)


def remove_original_ratings(preds_df, original_df):
    preds_only_df = preds_df.copy()
    mask = ~original_df.isna()
    preds_only_df[mask] = np.nan
    return preds_only_df


preds_only_df_res = remove_original_ratings(predictions_df, ratings_matrix)
print_df(preds_only_df_res)


movies_df = pd.read_csv('data/movies.csv')


def get_recommendations(user_id, predicts_df, movies_df, n=10):
    sorted_user_predictions = predicts_df.iloc[user_id - 1].sort_values(ascending=False)

    recommendations = (movies_df[movies_df['movieId'].isin(sorted_user_predictions.index)]
                       .set_index('movieId')
                       .loc[sorted_user_predictions.index]
                       .assign(predicted_rating=sorted_user_predictions.values)
                       .sort_values('predicted_rating', ascending=False))

    return recommendations.head(n)


user_id = 1
user_recommendations = get_recommendations(user_id, preds_only_df_res, movies_df)
print(tabulate(user_recommendations, headers='keys', tablefmt='psql'))




