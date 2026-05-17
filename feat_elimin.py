import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# =====================================================
# 1. WCZYTANIE DANYCH
# =====================================================

df = pd.read_excel("dane1.xlsx", index_col=0)
df = df.replace(",", ".", regex=True).astype(float)

df_num = df.copy()

# =====================================================
# PARAMETR
# =====================================================

PROG_KORELACJI = 0.8

# =====================================================
# 2. ELIMINACJA KORELACYJNA (HELLWIG STYLE)
# =====================================================

def eliminacja_hellwig(data, r_kryt):

    zmienne = list(data.columns)
    historia = []
    iteracja = 0

    while len(zmienne) >= 2:

        iteracja += 1

        R = data[zmienne].corr().abs()
        np.fill_diagonal(R.values, 0)

        max_r = R.values.max()

        if max_r < r_kryt:
            historia.append({
                "iteracja": iteracja,
                "zmienne": zmienne.copy(),
                "max_r": max_r,
                "para_A": None,
                "para_B": None,
                "usunieta": None,
                "koniec": True
            })
            break

        i, j = np.unravel_index(np.argmax(R.values), R.shape)

        cecha_A = zmienne[i]
        cecha_B = zmienne[j]

        pozostale = [z for z in zmienne if z not in [cecha_A, cecha_B]]

        if len(pozostale) > 0:

            suma_A = data[cecha_A].corr(data[pozostale]).abs().sum()
            suma_B = data[cecha_B].corr(data[pozostale]).abs().sum()

            do_usuniecia = cecha_A if suma_A >= suma_B else cecha_B

        else:
            do_usuniecia = cecha_B
            suma_A = suma_B = np.nan

        historia.append({
            "iteracja": iteracja,
            "zmienne": zmienne.copy(),
            "max_r": max_r,
            "para_A": cecha_A,
            "para_B": cecha_B,
            "usunieta": do_usuniecia,
            "koniec": False
        })

        zmienne.remove(do_usuniecia)

    return zmienne, historia


zmienne_finalne, historia = eliminacja_hellwig(df_num, PROG_KORELACJI)

# =====================================================
# 3. MACIERZ KORELACJI PRZED
# =====================================================

plt.figure(figsize=(7,5))
sns.heatmap(df_num.corr(), cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Macierz korelacji PRZED eliminacją")
plt.show()

# =====================================================
# 4. SCHEMAT ELIMINACJI
# =====================================================

rows = []

for h in historia:
    for z in h["zmienne"]:
        rows.append({
            "Iteracja": h["iteracja"],
            "Zmienna": z,
            "Status": "Aktywna"
        })

    if h["usunieta"]:
        rows.append({
            "Iteracja": h["iteracja"],
            "Zmienna": h["usunieta"],
            "Status": "Usuwana"
        })

df_hist = pd.DataFrame(rows)

plt.figure(figsize=(10,6))
sns.scatterplot(data=df_hist, x="Iteracja", y="Zmienna", hue="Status")
plt.title("Schemat eliminacji zmiennych")
plt.show()

# =====================================================
# 5. MAX KORELACJA W ITERACJACH
# =====================================================

hist_df = pd.DataFrame([
    {
        "Iteracja": h["iteracja"],
        "Max_r": h["max_r"],
        "Usunieta": h.get("usunieta")
    }
    for h in historia
])

plt.figure(figsize=(8,5))
plt.bar(hist_df["Iteracja"], hist_df["Max_r"])
plt.axhline(PROG_KORELACJI, color="red", linestyle="--")
plt.title("Maksymalna korelacja w iteracjach")
plt.show()

# =====================================================
# 6. MACIERZ PO ELIMINACJI
# =====================================================

df_final = df_num[zmienne_finalne]

plt.figure(figsize=(6,5))
sns.heatmap(df_final.corr(), cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Macierz PO eliminacji")
plt.show()

# =====================================================
# 7. ZMIENNOŚĆ (CV)
# =====================================================

cv = (df_final.std() / df_final.mean()) * 100

df_cv = pd.DataFrame({
    "Cecha": cv.index,
    "V": cv.values
})

df_cv["Status"] = np.where(df_cv["V"] >= 10, "Zachowana", "Odrzucona")

plt.figure(figsize=(8,5))
sns.barplot(data=df_cv, x="V", y="Cecha", hue="Status")
plt.axvline(10, color="red", linestyle="--")
plt.title("Analiza zmienności cech (CV)")
plt.show()

# =====================================================
# 8. CECHY FINALNE
# =====================================================

cechy_finalne = cv[cv >= 10].index.tolist()

print("Zmienne po korelacji:", zmienne_finalne)
print("Zmienne finalne (CV >= 10%):", cechy_finalne)