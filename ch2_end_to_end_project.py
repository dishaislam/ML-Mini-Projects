from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")

    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)

        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
            
    return pd.read_csv(Path("datasets/housing/housing.csv"))
            

housing = load_housing_data()

# print(housing.head())

# print(housing.info())

# print(housing["ocean_proximity"].value_counts())

# print(housing.describe())


#plotting histogram for all numerical values 

# import matplotlib.pyplot as plt
# housing.hist(bins=50, figsize=(12, 8))
# plt.show()



from train_test_splitter_demo import split_with_identifier,simple_splitter
train_set, test_set = simple_splitter.shuffle_and_split_data(housing, 0.2)

print( len(train_set))
print(len(test_set))

housing_with_id = housing.reset_index() # adds an `index` column
train_set_demo, test_set_demo =split_with_identifier.split_data_with_id_hash(housing_with_id, 0.2, "index")

print(len(train_set_demo))
print(len(test_set_demo))