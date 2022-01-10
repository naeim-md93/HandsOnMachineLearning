import pandas as pd
import urllib.request
import os


def prepare_country_stats(datapath):
    """
    This function just merges the OECD's life satisfaction data and the IMF's GDP per capita data.
    :param oecd_bli: (pd.DataFrame) OECD_BLI data
    :param gdp_per_capita: (pd.DataFrame) GDP_per_CAPITA data
    :return: (pd.DataFrame) life Satisfaction data
    """
    # read data
    oecd_bli = pd.read_csv(filepath_or_buffer=datapath + "oecd_bli_2015.csv",
                           thousands=',')
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")

    gdp_per_capita = pd.read_csv(filepath_or_buffer=datapath + "gdp_per_capita.csv",
                                 thousands=',',
                                 delimiter='\t',
                                 encoding='latin1',
                                 na_values="n/a")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)

    full_country_stats = pd.merge(left=oecd_bli,
                                  right=gdp_per_capita,
                                  left_index=True,
                                  right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))

    sample_data = full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]
    missing_data = full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[remove_indices]

    return sample_data, missing_data, full_country_stats, gdp_per_capita, oecd_bli


def download_data(datapath):
    """
    Download the data
    :param datapath:
    :return:
    """
    PATH_ROOT = os.getcwd() + '/'
    PATH_DATA = PATH_ROOT + datapath

    if not (os.path.exists(os.path.join(PATH_DATA, 'oecd_bli_2015.csv')) or os.path.exists(os.path.join(PATH_DATA, 'gdp_per_capita.csv'))):
        DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
        os.makedirs(datapath, exist_ok=True)
        for filename in ("oecd_bli_2015.csv", "gdp_per_capita.csv"):
            print("Downloading", filename)
            url = DOWNLOAD_ROOT + "datasets/lifesat/" + filename
            urllib.request.urlretrieve(url, datapath + filename)

    else:
        print('Files Already exist')