import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


df = pd.read_csv("polynomial.csv", sep=";")

a = df[['deneyim']]
y = df['maas']


pol_reg = PolynomialFeatures(degree=4)
x_pol = pol_reg.fit_transform(a)

reg = LinearRegression()
reg.fit(x_pol, y)

y_head = reg.predict(x_pol)

plt.scatter(a, y)
plt.plot(a, y_head)
plt.show()
