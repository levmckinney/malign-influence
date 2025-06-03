import numpy as np
import pandas as pd
from datasets import Dataset
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from tqdm import tqdm


def n_gram_preprocess(
    input_ids: list[NDArray[np.int64]], n: int, hash: int | None = None
) -> tuple[list[NDArray[np.int64]], int]:
    """Preprocess the input ids into n-grams and possibly hash them for preformance.

    Args:
        input_ids: List of input ids of shape (seq_len,).
        n: The length of the n-gram.
        hash: The number of buckets to hash the n-grams into. If None, no hashing is done.

    Returns:
        The input ids converted into n-grams.
        The largest token in the n-grams.
    """

    max_token = np.max([np.max(ids) for ids in tqdm(input_ids, desc="finding max token")])

    if (max_token + 1) ** n > np.iinfo(np.int64).max:
        raise ValueError(f"Token range {max_token} is too large for n={n}")

    n_gram_dataset = []

    largest_token = 0
    for ids in tqdm(input_ids, desc=f"preprocessing into {n}-grams"):
        seq_len = len(ids)
        new_seq_len = seq_len - n + 1
        n_grams = np.zeros(new_seq_len, dtype=np.int64)
        for j in range(n):
            n_grams = n_grams * (max_token + 1) + ids[j : new_seq_len + j]

        if hash is not None:
            n_grams = n_grams % hash

        max_n_gram = int(np.max(n_grams))
        if max_n_gram > largest_token:
            largest_token = max_n_gram

        n_gram_dataset.append(n_grams)

    return n_gram_dataset, largest_token


def get_tfidf_matrix(input_ids: list[NDArray[np.int64]], largest_token: int) -> coo_matrix:
    # Reference
    # https://courses.cs.washington.edu/courses/cse573/12sp/lectures/17-ir.pdf
    num_docs = len(input_ids)
    token_ids = []
    document_ids = []
    for i, ids in enumerate(input_ids):
        token_ids.append(ids.flatten())
        document_ids.append(np.full(len(ids), i, dtype=np.int64))

    token_ids = np.concatenate(token_ids)
    document_ids = np.concatenate(document_ids)

    values = np.ones_like(document_ids, dtype=float)

    tf = coo_matrix(
        (values, (document_ids, token_ids)),
        (num_docs, largest_token + 1),
        dtype=float,
    )

    m = tf.max(axis=1).toarray()
    tf = tf / m.reshape(-1, 1)

    df = tf._getnnz(axis=0)
    eps = 1e-6
    idf = np.where(df > 0, np.log(num_docs) - np.log2(df + eps) + 1, 0)
    tfidf = tf.multiply(idf.reshape(1, -1))
    norm = np.sqrt(tfidf.power(2).sum(axis=1))
    tfidf = tfidf / norm
    return tfidf


def get_tfidf_scores(
    queries: Dataset,
    dataset: Dataset,
    n_gram_length: int = 1,
    max_value: int | None = None,
) -> pd.DataFrame:
    """Get the tfidf scores of our queries a"""
    train_and_eval_ids = list(dataset["input_ids"]) + list(queries["input_ids"])
    train_and_eval_ids = [np.array(ids, dtype=np.int64) for ids in train_and_eval_ids]
    print("Calculating n-grams")
    train_and_eval_ids, largest_token = n_gram_preprocess(train_and_eval_ids, n_gram_length, max_value)
    print(f"Calculating tfidf scores {len(train_and_eval_ids)=} {largest_token=}")
    tfidf_mat = get_tfidf_matrix(train_and_eval_ids, largest_token)
    where_ds = np.where(tfidf_mat.row < len(dataset))[0]
    where_query = np.where(tfidf_mat.row >= len(dataset))[0]
    vector_size = tfidf_mat.shape[1]  # type: ignore
    ds_tfidf_mat = coo_matrix(
        (
            tfidf_mat.data[where_ds],
            (tfidf_mat.row[where_ds], tfidf_mat.col[where_ds]),
        ),
        (len(dataset), vector_size),
    )
    query_tfidf_mat = coo_matrix(
        (
            tfidf_mat.data[where_query],
            (
                tfidf_mat.row[where_query] - len(dataset),
                tfidf_mat.col[where_query],
            ),
        ),
        (len(queries), vector_size),
    )

    similarity_matrix = (query_tfidf_mat @ ds_tfidf_mat.T).toarray()

    train_ids = dataset["data_index"]
    query_ids = queries["data_index"]

    train_ids, query_ids = np.meshgrid(train_ids, query_ids)
    scores = similarity_matrix.flatten()

    df = pd.DataFrame(
        {
            "train_data_index": train_ids.flatten(),
            "query_data_index": query_ids.flatten(),
            "score": scores,
        }
    )

    return df
