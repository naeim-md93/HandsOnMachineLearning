import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from sklearn import pipeline

from src.utils import download_data, prepare_country_stats

### Settings ###
datapath = os.path.join("datasets", "lifesat", "")

pred_model = 'LinearRegression'

# To plot pretty figures directly within Jupyter
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
### End Settings ###


# Download data
download_data(datapath)


# Prepare the data
sample_data, missing_data, full_country_stats, gdp_per_capita, oecd_bli = prepare_country_stats(datapath)

# Visualize the data
sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()


X = np.c_[sample_data["GDP per capita"]]
y = np.c_[sample_data["Life satisfaction"]]

if pred_model == 'LinearRegression':
    # Select a linear model
    model = LinearRegression()

    # Train the model
    model.fit(X, y)


elif pred_model == 'KNeighborsRegressor':

    model = KNeighborsRegressor(n_neighbors=3)

    # Train the model
    model.fit(X, y)

# Cyprus' GDP per capita
X_new = [[22587]]

# Make a prediction for Cyprus
# LR: [[ 5.96242338]], KNN: [[5.76666667]]
print(model.predict(X_new))

# Money Happy Scatterplot
sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))
plt.axis([0, 60000, 0, 10])
position_text = {
    "Hungary": (5000, 1),
    "Korea": (18000, 1.7),
    "France": (29000, 2.4),
    "Australia": (40000, 3.0),
    "United States": (52000, 3.8),
}
for country, pos_text in position_text.items():
    pos_data_x, pos_data_y = sample_data.loc[country]
    country = "U.S." if country == "United States" else country
    plt.annotate(country, xy=(pos_data_x, pos_data_y), xytext=pos_text,
            arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=5))
    plt.plot(pos_data_x, pos_data_y, "ro")
plt.xlabel("GDP per capita (USD)")
plt.show()

# Tweaking model params
sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))
plt.xlabel("GDP per capita (USD)")
plt.axis([0, 60000, 0, 10])
X=np.linspace(0, 60000, 1000)
plt.plot(X, 2*X/100000, "r")
plt.text(40000, 2.7, r"$\theta_0 = 0$", fontsize=14, color="r")
plt.text(40000, 1.8, r"$\theta_1 = 2 \times 10^{-5}$", fontsize=14, color="r")
plt.plot(X, 8 - 5*X/100000, "g")
plt.text(5000, 9.1, r"$\theta_0 = 8$", fontsize=14, color="g")
plt.text(5000, 8.2, r"$\theta_1 = -5 \times 10^{-5}$", fontsize=14, color="g")
plt.plot(X, 4 + 5*X/100000, "b")
plt.text(5000, 3.5, r"$\theta_0 = 4$", fontsize=14, color="b")
plt.text(5000, 2.6, r"$\theta_1 = 5 \times 10^{-5}$", fontsize=14, color="b")
plt.show()

# Plotting Linear Regression Model
if pred_model == 'LinearRegression':
    t0, t1 = model.intercept_[0], model.coef_[0][0]
    sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5, 3))
    plt.xlabel("GDP per capita (USD)")
    plt.axis([0, 60000, 0, 10])
    X = np.linspace(0, 60000, 1000)
    plt.plot(X, t0 + t1 * X, "b")
    plt.text(5000, 3.1, r"$\theta_0 = 4.85$", fontsize=14, color="b")
    plt.text(5000, 2.2, r"$\theta_1 = 4.91 \times 10^{-5}$", fontsize=14, color="b")
    plt.show()

    # Cyprus Prediction Plot
    cyprus_gdp_per_capita = gdp_per_capita.loc["Cyprus"]["GDP per capita"]
    cyprus_predicted_life_satisfaction = model.predict([[cyprus_gdp_per_capita]])[0][0]
    sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5, 3), s=1)
    plt.xlabel("GDP per capita (USD)")
    X = np.linspace(0, 60000, 1000)
    plt.plot(X, t0 + t1 * X, "b")
    plt.axis([0, 60000, 0, 10])
    plt.text(5000, 7.5, r"$\theta_0 = 4.85$", fontsize=14, color="b")
    plt.text(5000, 6.6, r"$\theta_1 = 4.91 \times 10^{-5}$", fontsize=14, color="b")
    plt.plot([cyprus_gdp_per_capita, cyprus_gdp_per_capita], [0, cyprus_predicted_life_satisfaction], "r--")
    plt.text(25000, 5.0, r"Prediction = 5.96", fontsize=14, color="b")
    plt.plot(cyprus_gdp_per_capita, cyprus_predicted_life_satisfaction, "ro")
    plt.show()

    # Representative Training Data Scatterplot
    position_text2 = {
        "Brazil": (1000, 9.0),
        "Mexico": (11000, 9.0),
        "Chile": (25000, 9.0),
        "Czech Republic": (35000, 9.0),
        "Norway": (60000, 3),
        "Switzerland": (72000, 3.0),
        "Luxembourg": (90000, 3.0),
    }

    sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(8, 3))
    plt.axis([0, 110000, 0, 10])

    for country, pos_text in position_text2.items():
        pos_data_x, pos_data_y = missing_data.loc[country]
        plt.annotate(country, xy=(pos_data_x, pos_data_y), xytext=pos_text,
                     arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=5))
        plt.plot(pos_data_x, pos_data_y, "rs")

    X = np.linspace(0, 110000, 1000)
    plt.plot(X, t0 + t1 * X, "b:")

    lin_reg_full = LinearRegression()
    Xfull = np.c_[full_country_stats["GDP per capita"]]
    yfull = np.c_[full_country_stats["Life satisfaction"]]
    lin_reg_full.fit(Xfull, yfull)

    t0full, t1full = lin_reg_full.intercept_[0], lin_reg_full.coef_[0][0]
    X = np.linspace(0, 110000, 1000)
    plt.plot(X, t0full + t1full * X, "k")
    plt.xlabel("GDP per capita (USD)")
    plt.show()

    # Overfitting model plot
    full_country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(8, 3))
    plt.axis([0, 110000, 0, 10])

    poly = preprocessing.PolynomialFeatures(degree=30, include_bias=False)
    scaler = preprocessing.StandardScaler()
    lin_reg2 = LinearRegression()

    pipeline_reg = pipeline.Pipeline([('poly', poly), ('scal', scaler), ('lin', lin_reg2)])
    pipeline_reg.fit(Xfull, yfull)
    curve = pipeline_reg.predict(X[:, np.newaxis])
    plt.plot(X, curve)
    plt.xlabel("GDP per capita (USD)")
    plt.show()

    # Ridge Model Plot
    plt.figure(figsize=(8, 3))
    plt.xlabel("GDP per capita")
    plt.ylabel('Life satisfaction')

    plt.plot(list(sample_data["GDP per capita"]), list(sample_data["Life satisfaction"]), "bo")
    plt.plot(list(missing_data["GDP per capita"]), list(missing_data["Life satisfaction"]), "rs")

    X = np.linspace(0, 110000, 1000)
    plt.plot(X, t0full + t1full * X, "r--", label="Linear model on all data")
    plt.plot(X, t0 + t1 * X, "b:", label="Linear model on partial data")

    ridge = Ridge(alpha=10 ** 9.5)
    Xsample = np.c_[sample_data["GDP per capita"]]
    ysample = np.c_[sample_data["Life satisfaction"]]
    ridge.fit(Xsample, ysample)
    t0ridge, t1ridge = ridge.intercept_[0], ridge.coef_[0][0]
    plt.plot(X, t0ridge + t1ridge * X, "b", label="Regularized linear model on partial data")

    plt.legend(loc="lower right")
    plt.axis([0, 110000, 0, 10])
    plt.xlabel("GDP per capita (USD)")
    plt.show()