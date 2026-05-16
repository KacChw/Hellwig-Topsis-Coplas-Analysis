import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# WCZYTANIE DANYCH
# =====================================================

plik = "dane1.xlsx"
df = pd.read_excel(plik, index_col=0)

# zamiana przecinków na kropki
df = df.replace(",", ".", regex=True).astype(float)

# =====================================================
# USUNIĘCIE KOLUMN (jak w R)
# dane <- dane[, -c(6, 8, 11:12)]
# =====================================================

df = df.drop(columns=["X6", "X8", "X11", "X12"])

print("DANE PO REDUKCJI:")
print(df)
print()

# =====================================================
# TYPOLOGIA ZMIENNYCH (po redukcji!)
# s s s d d s d d
# =====================================================

stymulanty = ["X1", "X2", "X3", "X7"]
destymulanty = ["X4", "X5", "X9", "X10"]

# =====================================================
# NORMALIZACJA (Z-SCORE - Excel-like)
# ddof=0 -> POPULACYJNE odchylenie (ważne!)
# =====================================================

Z = pd.DataFrame(index=df.index)

for col in df.columns:

    if col in stymulanty:
        Z[col] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

    elif col in destymulanty:
        Z[col] = (df[col].mean() - df[col]) / df[col].std(ddof=0)

# =====================================================
# WZORZEC ROZWOJU
# =====================================================

wzorzec = Z.max()

# =====================================================
# ODLEGŁOŚĆ OD WZORCA
# =====================================================

odleglosci = np.sqrt(((Z - wzorzec) ** 2).sum(axis=1))

# =====================================================
# PARAMETR d0 (HELLWIG)
# =====================================================

d_srednie = odleglosci.mean()
s_d = odleglosci.std(ddof=0)

d0 = d_srednie + 2 * s_d

print("d0 =", round(d0, 6))
print()

# =====================================================
# MIERNIK HELWIGA
# =====================================================

score = 1 - (odleglosci / d0)
score = np.where(score < 0, 0, score)

# =====================================================
# RANKING
# =====================================================

wyniki = pd.DataFrame({
    "wojewodztwo": df.index,
    "score": score
})

wyniki = wyniki.sort_values(by="score", ascending=False)
wyniki["rank"] = range(1, len(wyniki) + 1)

wyniki["score"] = wyniki["score"].round(3)

print("RANKING:")
print(wyniki)

# =====================================================
# ZAPIS DO EXCELA
# =====================================================

wyniki.to_excel("ranking_hellwiga.xlsx", index=False)

print("\nZapisano: ranking_hellwiga.xlsx")

# =====================================================
# WYKRES
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
plt.title("METODA WZORCA HELLWIGA")
plt.xlabel("Miernik rozwoju")
plt.ylabel("Województwo")

plt.tight_layout()
plt.show()