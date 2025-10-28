import numpy as np

class simple_splitter:
    def shuffle_and_split_data(data, test_ratio):

        shuffled_indices = np.random.permutation(len(data))
        test_set_size = int(len(data) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]

        return data.iloc[train_indices], data.iloc[test_indices]

from zlib import crc32

class split_with_identifier:

    def is_id_in_test_set(identifier, test_ratio):
        return crc32(np.int64(identifier)) < test_ratio * 2**32

    def split_data_with_id_hash(data, test_ratio, id_column):

        ids = data[id_column]
        in_test_set = ids.apply(lambda id_: split_with_identifier.is_id_in_test_set(id_, test_ratio))

        return data.loc[~in_test_set], data.loc[in_test_set]