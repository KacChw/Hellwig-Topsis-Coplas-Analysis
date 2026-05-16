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
# KIERUNKI KRYTERIÓW
# s s s d d s d d
# =====================================================

types = ["+", "+", "+", "-", "-", "+", "-", "-"]

# =====================================================
# MACIERZ DECYZYJNA
# =====================================================

X = df.copy().values.astype(float)

# =====================================================
# NORMALIZACJA WEKTOROWA (TOPSIS)
# =====================================================

norm = np.sqrt((X ** 2).sum(axis=0))
R = X / norm

# =====================================================
# WAGI (jak w R)
# =====================================================

w = np.array([1] * R.shape[1])
V = R * w

# =====================================================
# IDEAL POSITIVE / NEGATIVE
# =====================================================

ideal_best = np.zeros(V.shape[1])
ideal_worst = np.zeros(V.shape[1])

for j in range(V.shape[1]):

    if types[j] == "+":
        ideal_best[j] = V[:, j].max()
        ideal_worst[j] = V[:, j].min()

    else:
        ideal_best[j] = V[:, j].min()
        ideal_worst[j] = V[:, j].max()

# =====================================================
# ODLEGŁOŚCI
# =====================================================

d_pos = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
d_neg = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

# =====================================================
# TOPSIS SCORE
# =====================================================

score = d_neg / (d_pos + d_neg)

# =====================================================
# RANKING
# =====================================================

wyniki = pd.DataFrame({
    "wojewodztwo": df.index,
    "score": score
})

wyniki = wyniki.sort_values(by="score", ascending=False)
wyniki["rank"] = range(1, len(wyniki) + 1)

wyniki["score"] = wyniki["score"].round(4)

print("RANKING TOPSIS:")
print(wyniki)

# =====================================================
# ZAPIS
# =====================================================

wyniki.to_excel("topsis_wyniki.xlsx", index=False)

# =====================================================
# WYKRES (jak ggplot2)
# =====================================================

colors = ["red" if r == 1 else "steelblue" for r in wyniki["rank"]]

plt.figure(figsize=(10, 6))

bars = plt.barh(wyniki["wojewodztwo"], wyniki["score"], color=colors)

for bar in bars:
    w = bar.get_width()
    plt.text(
        w / 2,
        bar.get_y() + bar.get_height() / 2,
        f"{w:.3f}",
        ha="center",
        va="center",
        fontweight="bold"
    )

plt.gca().invert_yaxis()
plt.title("METODA TOPSIS")
plt.xlabel("Miara TOPSIS")
plt.ylabel("Województwo")

plt.tight_layout()
plt.show()