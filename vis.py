import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# DANE
# =====================================================

df = pd.read_excel("dane1.xlsx", index_col=0)
df = df.replace(",", ".", regex=True).astype(float)

df = df.drop(columns=["X6", "X8", "X11", "X12"])

types = ["+", "+", "+", "-", "-", "+", "-", "-"]

# =====================================================
# HELLWIG
# =====================================================

Z = (df - df.mean()) / df.std(ddof=0)
wzorzec = Z.max()
d = np.sqrt(((Z - wzorzec) ** 2).sum(axis=1))
d0 = d.mean() + 2 * d.std(ddof=0)

hellwig = 1 - d / d0
hellwig = np.clip(hellwig, 0, None)

# =====================================================
# TOPSIS
# =====================================================

X = df.values
R = X / np.sqrt((X**2).sum(axis=0))

V = R

ideal_best = np.array([
    V[:, j].max() if types[j] == "+" else V[:, j].min()
    for j in range(V.shape[1])
])

ideal_worst = np.array([
    V[:, j].min() if types[j] == "+" else V[:, j].max()
    for j in range(V.shape[1])
])

d_pos = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
d_neg = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

topsis = d_neg / (d_pos + d_neg)

# =====================================================
# COPRAS
# =====================================================

Xc = df.values
norm = Xc / Xc.sum(axis=0)

S_plus = np.zeros(len(df))
S_minus = np.zeros(len(df))

for j in range(norm.shape[1]):
    if types[j] == "+":
        S_plus += norm[:, j]
    else:
        S_minus += norm[:, j]

Q = S_plus + (S_minus.min() * S_minus.sum()) / S_minus
Q = Q / Q.max()

# =====================================================
# DATAFRAME PORÓWNAWCZY
# =====================================================

final = pd.DataFrame({
    "wojewodztwo": df.index,
    "Hellwig": hellwig,
    "TOPSIS": topsis,
    "COPRAS": Q
})

final["mean_score"] = final[["Hellwig", "TOPSIS", "COPRAS"]].mean(axis=1)

# ranking (im wyżej tym lepiej)
final = final.sort_values("mean_score", ascending=False)
final["rank"] = range(1, len(final) + 1)

# =====================================================
# TOP 3 i WORST 3
# =====================================================

top3 = final.head(3)["wojewodztwo"].values
worst3 = final.tail(3)["wojewodztwo"].values

# =====================================================
# KOLORY
# =====================================================

colors = []

for w in final["wojewodztwo"]:
    if w in top3:
        colors.append("red")
    elif w in worst3:
        colors.append("blue")
    else:
        colors.append("lightgray")

# =====================================================
# WYKRES
# =====================================================

plt.figure(figsize=(12, 7))

bars = plt.barh(final["wojewodztwo"], final["mean_score"], color=colors)

for bar in bars:
    v = bar.get_width()
    plt.text(
        v / 2,
        bar.get_y() + bar.get_height() / 2,
        f"{v:.3f}",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold"
    )

plt.gca().invert_yaxis()

plt.title("PORÓWNANIE METOD MCDM (HELLWIG / TOPSIS / COPRAS)")
plt.xlabel("Średni wynik (ranking uśredniony)")
plt.ylabel("Województwo")

plt.tight_layout()
plt.show()