
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 110


# -----------------------------------------------------------------------------
# VISÃO GERAL
# -----------------------------------------------------------------------------

def resumo(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna um resumo com dtype, nulos, % nulos e únicos por coluna."""
    r = pd.DataFrame({
        "dtype":    df.dtypes,
        "nulos":    df.isnull().sum(),
        "% nulos":  (df.isnull().sum() / len(df) * 100).round(2),
        "unicos":   df.nunique(),
    })
    return r.sort_values("nulos", ascending=False)


# -----------------------------------------------------------------------------
# DISTRIBUIÇÕES
# -----------------------------------------------------------------------------

def plot_histogramas(df: pd.DataFrame, features: list, bins: int = 40,
                     cols: int = 3) -> None:
    """Histogramas com KDE para uma lista de variáveis numéricas."""
    n = len(features)
    rows = -(-n // cols)  # ceil division
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()
    for i, col in enumerate(features):
        axes[i].hist(df[col].dropna(), bins=bins,
                     color="#7F77DD", edgecolor="white", alpha=0.85, density=True)
        df[col].dropna().plot.kde(ax=axes[i], color="#D85A30", linewidth=1.5)
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Densidade")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Distribuições — histograma + KDE", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_boxplots(df: pd.DataFrame, features: list, cols: int = 3) -> None:
    """Boxplots lado a lado para detectar outliers."""
    n = len(features)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()
    for i, col in enumerate(features):
        sns.boxplot(
            y=df[col], ax=axes[i],
            color="#5DCAA5", linewidth=1.1,
            flierprops=dict(marker="o", markerfacecolor="#D85A30",
                            markersize=3, alpha=0.45)
        )
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_ylabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Boxplots — distribuição e outliers", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# CORRELAÇÃO
# -----------------------------------------------------------------------------

def plot_correlacao(df: pd.DataFrame, features: list = None,
                    metodo: str = "pearson") -> None:
    """Heatmap de correlação (triângulo inferior)."""
    data = df[features] if features else df.select_dtypes(include="number")
    corr = data.corr(method=metodo)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.8),
                                    max(6, len(corr) * 0.7)))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, linewidths=0.4,
        ax=ax, annot_kws={"size": 8}
    )
    ax.set_title(f"Correlação ({metodo})", fontsize=12)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# PAIRPLOT  ←  BUG CORRIGIDO AQUI
# -----------------------------------------------------------------------------

def plot_pair(df: pd.DataFrame, features: list, hue: str = None) -> None:
    """
    Pairplot entre as features selecionadas.

    CORREÇÃO: plt.figure(12, 6) estava sendo chamado com dois ints separados,
    fazendo o matplotlib interpretar 12 como num (id da figura) e 6 como dpi.
    O correto é passar figsize como tupla nomeada — mas o pairplot do seaborn
    cria a figura internamente, então basta chamar sns.pairplot diretamente.
    """
    cols = features + ([hue] if hue else [])
    g = sns.pairplot(
        df[cols],
        hue=hue,
        diag_kind="kde",
        plot_kws=dict(alpha=0.4, s=12),
        height=2.2,
        aspect=1.0,
    )
    g.figure.suptitle("Pairplot — relações entre variáveis", y=1.01, fontsize=12)
    plt.show()


# -----------------------------------------------------------------------------
# ANÁLISE DE NULOS
# -----------------------------------------------------------------------------

def plot_nulos(df: pd.DataFrame) -> None:
    """Barras horizontais com % de nulos por coluna."""
    nulos = df.isnull().mean() * 100
    nulos = nulos[nulos > 0].sort_values(ascending=True)
    if nulos.empty:
        print("Nenhum valor nulo encontrado.")
        return
    fig, ax = plt.subplots(figsize=(8, max(3, len(nulos) * 0.5)))
    nulos.plot(kind="barh", ax=ax, color="#D85A30", alpha=0.85)
    ax.set_xlabel("% de nulos")
    ax.set_title("Valores nulos por coluna")
    for i, v in enumerate(nulos):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# OUTLIERS — CONTAGEM POR IQR
# -----------------------------------------------------------------------------

def contar_outliers(df: pd.DataFrame, features: list = None) -> pd.DataFrame:
    """
    Conta outliers via IQR (< Q1-1.5*IQR ou > Q3+1.5*IQR).
    Retorna DataFrame com contagem e % por variável.
    """
    data = df[features] if features else df.select_dtypes(include="number")
    resultado = []
    for col in data.columns:
        s = data[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        n_out = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
        resultado.append({
            "variavel": col,
            "n_outliers": n_out,
            "pct_outliers": round(n_out / len(s) * 100, 2)
        })
    return (pd.DataFrame(resultado)
              .sort_values("pct_outliers", ascending=False)
              .reset_index(drop=True))


# -----------------------------------------------------------------------------
# TRANSFORMAÇÃO LOG
# -----------------------------------------------------------------------------

def aplicar_log(df: pd.DataFrame, features: list,
                sufixo: str = "_LOG") -> pd.DataFrame:
    """
    Aplica log1p nas colunas indicadas e adiciona como novas colunas.
    log1p = log(1 + x), evita log(0) quando x = 0.
    """
    df = df.copy()
    for col in features:
        df[col + sufixo] = np.log1p(df[col])
    print(f"Colunas criadas: {[c + sufixo for c in features]}")
    return df

