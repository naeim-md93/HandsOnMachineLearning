import os
import urllib
import numpy as np
import pandas as pd
from src.utils import fetch_housing_data, load_housing_data, \
    split_train_test, split_train_test_by_id, income_cat_proportions

### Settings ###
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

split_method = 'sklearn' # sklearn, custom, id, stratified
### End Settings ###

# Download Data
fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)

# Load Data
housing = load_housing_data(housing_path=HOUSING_PATH)

# Print data
print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())

# Plot Histogram of all numerical columns
housing.hist(bins=50, figsize=(15,10))
plt.show()

if split_method == 'sklearn':
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

elif split_method == 'custom':
    train_set, test_set = split_train_test(data=housing, test_ratio=0.2)

elif split_method == 'id':
    housing_with_id = housing.reset_index()  # adds an `index` column
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

elif split_method == 'stratified':
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    from sklearn.model_selection import StratifiedShuffleSplit
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        train_set = housing.loc[train_index]
        test_set = housing.loc[test_index]


else:
    housing["id"] = housing["longitude"] * 1000 + housing["latitude"]
    train_set, test_set = split_train_test_by_id(data=housing, test_ratio=0.2, id_column="id")


# compare_props = pd.DataFrame({
#     "Overall": income_cat_proportions(housing),
#     "Stratified": income_cat_proportions(strat_test_set),
#     "Random": income_cat_proportions(test_set),
# }).sort_index()
#
# compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
# compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100

# Visualize data
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()
plt.show()

# Download the California image
images_path = os.path.join(os.getcwd(), "images", "end_to_end_project")
os.makedirs(images_path, exist_ok=True)
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
filename = "california.png"
print("Downloading", filename)
url = DOWNLOAD_ROOT + "images/end_to_end_project/" + filename
urllib.request.urlretrieve(url, os.path.join(images_path, filename))

# Correlation
corr_matrix = housing.corr()
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()


housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
plt.axis([0, 16, 0, 550000])
plt.show()

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()