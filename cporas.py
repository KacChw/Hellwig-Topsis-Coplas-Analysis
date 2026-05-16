import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# WCZYTANIE DANYCH
# =====================================================

plik = "dane1.xlsx"
df = pd.read_excel(plik, index_col=0)

df = df.replace(",", ".", regex=True).astype(float)

# =====================================================
# USUNIĘCIE KOLUMN (jak w R)
# =====================================================

df = df.drop(columns=["X6", "X8", "X11", "X12"])

print("DANE PO REDUKCJI:")
print(df)
print()

# =====================================================
# TYPOLOGIA KRYTERIÓW
# s s s d d s d d
# =====================================================

types = ["+", "+", "+", "-", "-", "+", "-", "-"]

# =====================================================
# NORMALIZACJA COPRAS
# x_ij / sum(x_j)
# =====================================================

X = df.values.astype(float)

norm_matrix = X / X.sum(axis=0)

# =====================================================
# WAGI (jak w R)
# =====================================================

w = np.array([1] * norm_matrix.shape[1])

weighted = norm_matrix * w

# =====================================================
# SUMY S+ i S-
# =====================================================

S_plus = np.zeros(len(df))
S_minus = np.zeros(len(df))

for j in range(weighted.shape[1]):

    if types[j] == "+":
        S_plus += weighted[:, j]
    else:
        S_minus += weighted[:, j]

# =====================================================
# COPRAS Qi
# =====================================================

S_minus_sum = S_minus.sum()
S_minus_min = S_minus.min()

Q = S_plus + (S_minus_min * S_minus.sum()) / S_minus

# =====================================================
# NORMALIZACJA Q (jak w literaturze COPRAS)
# =====================================================

Q = Q / Q.max()

# =====================================================
# RANKING
# =====================================================

wyniki = pd.DataFrame({
    "wojewodztwo": df.index,
    "score": Q
})

wyniki = wyniki.sort_values(by="score", ascending=False)
wyniki["rank"] = range(1, len(wyniki) + 1)

wyniki["score"] = wyniki["score"].round(4)

print("RANKING COPRAS:")
print(wyniki)

# =====================================================
# ZAPIS
# =====================================================

wyniki.to_excel("copras_wyniki.xlsx", index=False)

# =====================================================
# WYKRES (ggplot-like)
# =====================================================

colors = ["red" if r == 1 else "steelblue" for r in wyniki["rank"]]

plt.figure(figsize=(10, 6))

bars = plt.barh(wyniki["wojewodztwo"], wyniki["score"], color=colors)

for bar in bars:
    w_val = bar.get_width()
    plt.text(
        w_val / 2,
        bar.get_y() + bar.get_height() / 2,
        f"{w_val:.3f}",
        ha="center",
        va="center",
        fontweight="bold"
    )

plt.gca().invert_yaxis()
plt.title("METODA COPRAS")
plt.xlabel("Miara COPRAS")
plt.ylabel("Województwo")

plt.tight_layout()
plt.show()