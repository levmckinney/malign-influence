import numpy as np
import pandas as pd
from datasets import Dataset
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from tqdm import tqdm


def n_gram_preprocess(
    input_ids: list[NDArray[np.int64]], n: int, hash: int | None = None
) -> tuple[list[NDArray[np.int64]], int]:
    """Preprocess the input ids into n-grams and possibly hash them for performance.

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

        # Handle documents shorter than n-gram length
        if new_seq_len <= 0:
            # Create empty array for documents too short for n-grams
            n_grams = np.array([], dtype=np.int64)
        else:
            n_grams = np.zeros(new_seq_len, dtype=np.int64)
            for j in range(n):
                n_grams = n_grams * (max_token + 1) + ids[j : new_seq_len + j]

            if hash is not None:
                n_grams = n_grams % hash

            if len(n_grams) > 0:
                max_n_gram = int(np.max(n_grams))
                if max_n_gram > largest_token:
                    largest_token = max_n_gram

        n_gram_dataset.append(n_grams)

    return n_gram_dataset, largest_token


def get_tfidf_vectors(input_ids: list[NDArray[np.int64]], largest_token: int) -> coo_matrix:
    """Convert each of the input documents into a tf-idf vector.

    Args:
        input_ids: List of input ids of shape (seq_len,).
        largest_token: The largest token in the n-grams.

    Returns:
        A coo_matrix of the tf-idf vectors.
    """
    # Reference
    # https://courses.cs.washington.edu/courses/cse573/12sp/lectures/17-ir.pdf
    num_docs = len(input_ids)
    token_ids = []
    document_ids = []
    for i, ids in enumerate(input_ids):
        if len(ids) > 0:  # Only process non-empty documents
            token_ids.append(ids.flatten())
            document_ids.append(np.full(len(ids), i, dtype=np.int64))

    # Handle case where all documents are empty
    if len(token_ids) == 0:
        return coo_matrix((num_docs, largest_token + 1), dtype=float)

    token_ids = np.concatenate(token_ids)
    document_ids = np.concatenate(document_ids)

    values = np.ones_like(document_ids, dtype=float)

    tf = coo_matrix(
        (values, (document_ids, token_ids)),
        (num_docs, largest_token + 1),
        dtype=float,
    )

    # Handle division by zero for documents with no terms
    m = tf.max(axis=1).toarray()
    # Avoid division by zero - documents with no terms will remain zero
    m = np.where(m == 0, 1, m)
    tf = tf / m.reshape(-1, 1)

    df = tf._getnnz(axis=0)  # type: ignore
    eps = 1e-6
    idf = np.where(df > 0, np.log(num_docs) - np.log2(df + eps) + 1, 0)
    tfidf = tf.multiply(idf.reshape(1, -1))  # type: ignore

    # Handle normalization for documents with zero vectors
    norm = np.sqrt(tfidf.power(2).sum(axis=1))
    norm = np.where(norm == 0, 1, norm)  # Avoid division by zero
    tfidf = tfidf / norm

    return tfidf


def get_tfidf_scores(
    queries: Dataset,
    dataset: Dataset,
    n_gram_length: int = 1,
    max_value: int | None = None,
) -> pd.DataFrame:
    """Get the tfidf scores of each of our queries to each of of training examples

    Args:
        queries: The queries to score with a "data_index" column and an "input_ids" column.
        dataset: The dataset to score the queries against with a "data_index" column and an "input_ids" column.
        n_gram_length: The length of the n-grams to use.
        max_value: The maximum value of the n-grams.

    Returns:
        A dataframe with a "train_data_index" column, a "query_data_index" column, and a "score" column.
    """
    train_and_eval_ids = list(dataset["input_ids"]) + list(queries["input_ids"])
    train_and_eval_ids = [np.array(ids, dtype=np.int64) for ids in train_and_eval_ids]
    print("Calculating n-grams")
    train_and_eval_ids, largest_token = n_gram_preprocess(train_and_eval_ids, n_gram_length, max_value)
    print(f"Calculating tfidf scores {len(train_and_eval_ids)=} {largest_token=}")
    tfidf_mat = get_tfidf_vectors(train_and_eval_ids, largest_token)
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

    train_ids = dataset["id"]
    query_ids = queries["id"]

    train_ids, query_ids = np.meshgrid(train_ids, query_ids)
    scores = similarity_matrix.flatten()

    df = pd.DataFrame(
        {
            "query_id": query_ids.flatten(),
            "train_id": train_ids.flatten(),
            "influence_score": scores,
        }
    )

    return df
