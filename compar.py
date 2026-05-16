import pandas as pd
import numpy as np

# =====================================================
# WCZYTANIE DANYCH
# =====================================================

df = pd.read_excel("dane1.xlsx", index_col=0)
df = df.replace(",", ".", regex=True).astype(float)

# usunięcie kolumn jak w Twoim modelu
df = df.drop(columns=["X6", "X8", "X11", "X12"])

# typy kryteriów
types = ["+", "+", "+", "-", "-", "+", "-", "-"]

# =====================================================
# =============== 1. HELLWIG ==========================
# =====================================================

Z = (df - df.mean()) / df.std(ddof=0)

wzorzec = Z.max()
d = np.sqrt(((Z - wzorzec) ** 2).sum(axis=1))

d0 = d.mean() + 2 * d.std(ddof=0)

hellwig = 1 - (d / d0)
hellwig = np.where(hellwig < 0, 0, hellwig)

hellwig_df = pd.DataFrame({
    "wojewodztwo": df.index,
    "Hellwig": hellwig
}).sort_values("Hellwig", ascending=False)

hellwig_df["rank_H"] = range(1, len(df) + 1)

# =====================================================
# =============== 2. TOPSIS ===========================
# =====================================================

X = df.values.astype(float)

norm = np.sqrt((X ** 2).sum(axis=0))
R = X / norm

V = R

ideal_best = np.zeros(V.shape[1])
ideal_worst = np.zeros(V.shape[1])

for j in range(V.shape[1]):
    if types[j] == "+":
        ideal_best[j] = V[:, j].max()
        ideal_worst[j] = V[:, j].min()
    else:
        ideal_best[j] = V[:, j].min()
        ideal_worst[j] = V[:, j].max()

d_pos = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
d_neg = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

topsis = d_neg / (d_pos + d_neg)

topsis_df = pd.DataFrame({
    "wojewodztwo": df.index,
    "TOPSIS": topsis
}).sort_values("TOPSIS", ascending=False)

topsis_df["rank_T"] = range(1, len(df) + 1)

# =====================================================
# =============== 3. COPRAS ===========================
# =====================================================

Xc = df.values.astype(float)
norm_c = Xc / Xc.sum(axis=0)

S_plus = np.zeros(len(df))
S_minus = np.zeros(len(df))

for j in range(norm_c.shape[1]):
    if types[j] == "+":
        S_plus += norm_c[:, j]
    else:
        S_minus += norm_c[:, j]

Q = S_plus + (S_minus.min() * S_minus.sum()) / S_minus
Q = Q / Q.max()

copras_df = pd.DataFrame({
    "wojewodztwo": df.index,
    "COPRAS": Q
}).sort_values("COPRAS", ascending=False)

copras_df["rank_C"] = range(1, len(df) + 1)

# =====================================================
# =============== PORÓWNANIE ==========================
# =====================================================

final = hellwig_df[["wojewodztwo", "Hellwig", "rank_H"]] \
    .merge(topsis_df[["wojewodztwo", "TOPSIS", "rank_T"]], on="wojewodztwo") \
    .merge(copras_df[["wojewodztwo", "COPRAS", "rank_C"]], on="wojewodztwo")

# średni ranking
final["mean_rank"] = final[["rank_H", "rank_T", "rank_C"]].mean(axis=1)

final = final.sort_values("mean_rank")

print(final)

# =====================================================
# ZAPIS DO EXCELA
# =====================================================

final.to_excel("porownanie_metod.xlsx", index=False)

print("\nZapisano plik: porownanie_metod.xlsx")