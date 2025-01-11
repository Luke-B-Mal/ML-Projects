import pandas as pd
import matplotlib.pyplot as plt

# bad data set to use, would work better (and therefore give more a more accurate predictive analysis if logic.
# regression was used.
df = pd.read_csv("/Users/beaumaldonado/PycharmProjects/Personal Projects/data cleaning/cleaned_adm_data.csv")


# not too useful as it doesn't give the partial derivative that is used to minimize the error function.
def msef(m, b, pts):
    errortot = 0
    for i in range(len(pts)):
        x = pts.iloc[i].CGPA
        y = pts.iloc[i].ChanceofAdmit
        errortot += (y - (m * x + b)) ** 2
    errortot = errortot / float(len(pts.CGPA))


# the msef derived and applied in such a way that gives the values of m and b that minimizes the error of the line.
def graddesc(m, b, pts, learn):
    mgrad = 0
    bgrad = 0
    n = len(pts)

    for i in range(n):
        x = pts.iloc[i].CGPA
        y = pts.iloc[i].ChanceofAdmit

        mgrad += -(2 / n) * x * (y - (m * x + b))
        bgrad += -(2 / n) * (y - (m * x + b))

    nm = m - mgrad * learn
    nb = b - bgrad * learn
    return nm, nb

m = 0
b = 0
learn = 0.001
epochs = 1000

for i in range(epochs):
    m, b = graddesc(m, b, df, learn)

print(m, b)

plt.scatter(df.CGPA, df.ChanceofAdmit, color="blue")
plt.plot(list(range(7, 11)), [m * x + b for x in range(7, 11)], color="red")
plt.show()
