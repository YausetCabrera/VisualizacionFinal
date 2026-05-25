"""
lab_secciones.py
================
Script de PROTOTIPADO independiente (sin Dagster).
Ejecutar directamente para exploración y desarrollo:

  $ python lab_secciones.py

Útil para:
  - Explorar rápidamente los datasets
  - Prototipar visualizaciones antes de añadirlas al pipeline
  - Generar estadísticas descriptivas y análisis exploratorio
  - Depurar transformaciones en local

Nota académica:
  El prototipado en laboratorio (EDA) es la primera fase de DataOps.
  Permite validar hipótesis sobre los datos antes de formalizar el pipeline.
  Las funciones aquí desarrolladas migran luego a assets_secciones.py.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from plotnine import (
    ggplot, aes, geom_point, geom_line, geom_bar, geom_boxplot,
    geom_histogram, geom_text, geom_label,
    facet_wrap, facet_grid,
    scale_fill_cmap, scale_fill_manual, scale_color_manual,
    scale_x_continuous, scale_y_continuous, scale_color_brewer,
    coord_sf, labs, theme, theme_minimal, theme_void,
    element_text, element_blank, element_line, element_rect,
    guide_colorbar, guides,
    stat_smooth,
)
from plotnine.scales import scale_fill_gradientn, scale_fill_gradient

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

AÑOS = [2021, 2022, 2023]

# ─────────────────────────────────────────────────────────────────
# SECCIÓN 1 — CARGA Y EXPLORACIÓN INICIAL
# ─────────────────────────────────────────────────────────────────

def cargar_dataset(nombre: str) -> pd.DataFrame:
    """Carga un CSV con detección automática de separador."""
    ruta = DATA_DIR / nombre
    with open(ruta, encoding="utf-8-sig") as f:
        primera = f.readline()
    sep = ";" if primera.count(";") > primera.count(",") else ","
    df = pd.read_csv(ruta, sep=sep, encoding="utf-8-sig", thousands=".", decimal=",")
    print(f"\n[LOAD] {nombre}: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"       Columnas: {list(df.columns)}")
    return df


def explorar_dataset(df: pd.DataFrame, nombre: str = "dataset") -> None:
    """Análisis exploratorio completo de un DataFrame."""
    print(f"\n{'='*60}")
    print(f"EDA: {nombre}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"\nTipos de datos:")
    print(df.dtypes.to_string())
    print(f"\nEstadísticas descriptivas (numéricas):")
    print(df.describe().to_string())
    print(f"\nNulos por columna:")
    nulos = df.isnull().sum()
    print(nulos[nulos > 0].to_string() if nulos.any() else "  Sin valores nulos")
    print(f"\nPrimeras filas:")
    print(df.head(3).to_string())


# ─────────────────────────────────────────────────────────────────
# SECCIÓN 2 — TRANSFORMACIONES DE PROTOTIPO
# ─────────────────────────────────────────────────────────────────

def normalizar_seccion(s: pd.Series) -> pd.Series:
    """Normaliza código de sección censal a 10 dígitos."""
    return s.astype(str).str.strip().str.replace(r"\s+", "", regex=True).str.zfill(10)


def limpiar_renta(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de limpieza de renta media (prototipo)."""
    df = df.copy()

    # Detectar columnas dinámicamente
    cols = {c.lower().strip(): c for c in df.columns}

    rename = {}
    for cand in ["periodo", "año", "anyo", "year"]:
        if cand in cols:
            rename[cols[cand]] = "anio"
            break
    for cand in ["sección censal", "seccion censal", "cusec", "seccion"]:
        if cand in cols:
            rename[cols[cand]] = "seccion"
            break
    for cand in ["renta neta media por persona", "renta media", "media"]:
        if cand in cols:
            rename[cols[cand]] = "renta_media"
            break
    for cand in ["renta neta mediana por persona", "renta mediana", "mediana"]:
        if cand in cols:
            rename[cols[cand]] = "renta_mediana"
            break

    df = df.rename(columns=rename)

    if "anio" in df.columns:
        df["anio"] = pd.to_numeric(df["anio"], errors="coerce")
        df = df[df["anio"].isin(AÑOS)]

    if "seccion" in df.columns:
        df["seccion"] = normalizar_seccion(df["seccion"])

    for col in ["renta_media", "renta_mediana"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                        .str.replace(".", "", regex=False)
                        .str.replace(",", ".", regex=False)
                        .replace({"": np.nan, "nd": np.nan, "N/A": np.nan})
                        .pipe(pd.to_numeric, errors="coerce")
            )

    df = df.drop_duplicates(["seccion", "anio"])
    print(f"\n[CLEAN renta] Shape final: {df.shape}")
    print(f"  Renta media — min: {df['renta_media'].min():.0f}€, max: {df['renta_media'].max():.0f}€")
    return df.reset_index(drop=True)


def limpiar_ingresos(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de limpieza de distribución de ingresos (prototipo)."""
    df = df.copy()
    cols = {c.lower().strip(): c for c in df.columns}

    rename = {}
    for cand in ["periodo", "año", "anyo", "year"]:
        if cand in cols:
            rename[cols[cand]] = "anio"
            break
    for cand in ["sección censal", "seccion censal", "cusec", "seccion"]:
        if cand in cols:
            rename[cols[cand]] = "seccion"
            break

    fuentes = {
        "salarios": ["salarios", "rentas salariales", "trabajo"],
        "pensiones": ["pensiones", "jubilación", "jubilacion"],
        "prestaciones": ["prestaciones", "desempleo"],
        "rendimientos_cap": ["rendimientos del capital", "capital"],
        "otros": ["otros", "otras fuentes"],
    }
    for canon, candidatos in fuentes.items():
        for cand in candidatos:
            if cand in cols and cols[cand] not in rename.values():
                rename[cols[cand]] = canon
                break

    df = df.rename(columns=rename)

    if "anio" in df.columns:
        df["anio"] = pd.to_numeric(df["anio"], errors="coerce")
        df = df[df["anio"].isin(AÑOS)]

    if "seccion" in df.columns:
        df["seccion"] = normalizar_seccion(df["seccion"])

    cols_fuentes = [c for c in ["salarios", "pensiones", "prestaciones", "rendimientos_cap", "otros"]
                    if c in df.columns]
    for col in cols_fuentes:
        df[col] = (
            df[col].astype(str)
                    .str.replace(".", "", regex=False)
                    .str.replace(",", ".", regex=False)
                    .replace({"": np.nan, "nd": np.nan})
                    .pipe(pd.to_numeric, errors="coerce")
        )

    total = df[cols_fuentes].sum(axis=1).replace(0, np.nan)
    for col in cols_fuentes:
        df[f"pct_{col}"] = df[col] / total * 100

    pct_cols = [f"pct_{c}" for c in cols_fuentes]
    if pct_cols:
        df["ingreso_dominante"] = (
            df[pct_cols].idxmax(axis=1)
                        .str.replace("pct_", "")
                        .str.replace("_", " ")
                        .str.title()
        )

    df = df.drop_duplicates(["seccion", "anio"]).reset_index(drop=True)
    print(f"\n[CLEAN ingresos] Shape final: {df.shape}")
    print(f"  Fuentes detectadas: {cols_fuentes}")
    return df


def limpiar_actividad(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de limpieza de actividad laboral (prototipo)."""
    df = df.copy()
    cols = {c.lower().strip(): c for c in df.columns}

    rename = {}
    for cand in ["periodo", "año", "anyo", "year"]:
        if cand in cols:
            rename[cols[cand]] = "anio"
            break
    for cand in ["sección censal", "seccion censal", "cusec", "seccion"]:
        if cand in cols:
            rename[cols[cand]] = "seccion"
            break

    campos = {
        "ocupados": ["ocupados", "empleo"],
        "parados": ["parados", "desempleados"],
        "inactivos": ["inactivos"],
        "activos": ["activos", "población activa"],
        "poblacion": ["población total", "poblacion total", "poblacion"],
    }
    for canon, candidatos in campos.items():
        for cand in candidatos:
            if cand in cols and cols[cand] not in rename.values():
                rename[cols[cand]] = canon
                break

    df = df.rename(columns=rename)

    if "anio" in df.columns:
        df["anio"] = pd.to_numeric(df["anio"], errors="coerce")
        df = df[df["anio"].isin(AÑOS)]

    if "seccion" in df.columns:
        df["seccion"] = normalizar_seccion(df["seccion"])

    for col in ["ocupados", "parados", "inactivos", "activos", "poblacion"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                        .str.replace(".", "", regex=False)
                        .str.replace(",", ".", regex=False)
                        .replace({"": np.nan, "nd": np.nan})
                        .pipe(pd.to_numeric, errors="coerce")
            )

    if "ocupados" in df.columns and "activos" in df.columns:
        df["tasa_ocupacion"] = (df["ocupados"] / df["activos"] * 100).clip(0, 100)
    if "parados" in df.columns and "activos" in df.columns:
        df["tasa_paro"] = (df["parados"] / df["activos"] * 100).clip(0, 100)
    if "activos" in df.columns and "poblacion" in df.columns:
        df["tasa_actividad"] = (df["activos"] / df["poblacion"] * 100).clip(0, 100)

    df = df.drop_duplicates(["seccion", "anio"]).reset_index(drop=True)
    print(f"\n[CLEAN actividad] Shape final: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────
# SECCIÓN 3 — ANÁLISIS EXPLORATORIO VISUAL
# ─────────────────────────────────────────────────────────────────

def eda_distribucion_renta(df_renta: pd.DataFrame) -> None:
    """
    Análisis exploratorio de la distribución de renta.
    Gramática: geom_histogram, facet_wrap(~anio), escala libre
    """
    df = df_renta[df_renta["renta_media"].notna()].copy()
    df["anio"] = df["anio"].astype(int).astype(str)

    p = (
        ggplot(df, aes(x="renta_media"))
        + geom_histogram(aes(fill="anio"), bins=50, alpha=0.75, color="white")
        + facet_wrap("anio", ncol=3)
        + scale_fill_manual(
            values={"2021": "#1565C0", "2022": "#00897B", "2023": "#E53935"},
            guide=False,
        )
        + scale_x_continuous(labels=lambda l: [f"{int(v/1000)}k€" for v in l])
        + labs(
            title="[EDA] Distribución de renta media por sección censal",
            subtitle="Histograma de frecuencias (nº de secciones) por año",
            x="Renta media neta anual (€)",
            y="Número de secciones",
        )
        + theme_minimal()
        + theme(plot_title=element_text(size=12, face="bold"))
    )
    ruta = str(OUTPUT_DIR / "lab_eda_hist_renta.png")
    p.save(ruta, width=12, height=5, dpi=120, verbose=False)
    print(f"[EDA] Guardado histograma renta: {ruta}")


def eda_renta_vs_actividad(df_integrado: pd.DataFrame) -> None:
    """
    Scatter plot exploratorio: renta vs. tasa de ocupación.
    Gramática: geom_point, aes(color=ingreso_dominante), stat_smooth (lm)
    """
    col_tasa = next((c for c in ["tasa_ocupacion", "tasa_actividad"] if c in df_integrado.columns), None)
    if col_tasa is None:
        print("[EDA] Sin tasa de actividad disponible.")
        return

    df = (
        df_integrado[
            df_integrado["anio"] == 2023
        ][["renta_media", col_tasa, "ingreso_dominante"]]
        .dropna()
        .copy()
    )

    p = (
        ggplot(df, aes(x=col_tasa, y="renta_media", color="ingreso_dominante"))
        + geom_point(alpha=0.45, size=1.8)
        + stat_smooth(method="lm", color="#333333", linetype="dashed", size=0.8, se=False)
        + scale_y_continuous(labels=lambda l: [f"{int(v/1000)}k€" for v in l])
        + labs(
            title=f"[EDA] Renta media vs. {col_tasa.replace('_', ' ')} (2023)",
            subtitle="Cada punto = una sección censal. Línea punteada = tendencia lineal",
            x=col_tasa.replace("_", " ").capitalize() + " (%)",
            y="Renta media neta (€)",
            color="Ingreso dominante",
        )
        + theme_minimal()
        + theme(
            plot_title=element_text(size=12, face="bold"),
            legend_position="right",
        )
    )
    ruta = str(OUTPUT_DIR / "lab_eda_scatter_renta_actividad.png")
    p.save(ruta, width=9, height=6, dpi=120, verbose=False)
    print(f"[EDA] Guardado scatter: {ruta}")


def eda_boxplot_ingresos(df_ingresos: pd.DataFrame) -> None:
    """
    Boxplot de la distribución de proporciones por fuente de ingreso.
    Gramática: geom_boxplot, aes(x=fuente, y=pct), facet_wrap(~anio)
    """
    pct_cols = [c for c in df_ingresos.columns if c.startswith("pct_")]
    if not pct_cols:
        print("[EDA] Sin columnas pct_ disponibles.")
        return

    # Reshape a formato largo (tidy)
    df_long = df_ingresos[["seccion", "anio"] + pct_cols].melt(
        id_vars=["seccion", "anio"],
        value_vars=pct_cols,
        var_name="fuente",
        value_name="pct",
    ).dropna()
    df_long["fuente"] = df_long["fuente"].str.replace("pct_", "").str.replace("_", " ").str.title()
    df_long["anio"] = df_long["anio"].astype(int).astype(str)

    p = (
        ggplot(df_long, aes(x="fuente", y="pct", fill="fuente"))
        + geom_boxplot(outlier_alpha=0.2, outlier_size=0.8)
        + facet_wrap("anio", ncol=3)
        + scale_fill_manual(
            values={"Salarios": "#2196F3", "Pensiones": "#FF9800",
                    "Prestaciones": "#9C27B0", "Rendimientos Cap": "#4CAF50", "Otros": "#F44336"},
            guide=False,
        )
        + labs(
            title="[EDA] Distribución de proporciones por fuente de ingresos",
            subtitle="% sobre el total de ingresos de los hogares por sección censal",
            x="Fuente de ingresos",
            y="Proporción (%)",
        )
        + theme_minimal()
        + theme(
            plot_title=element_text(size=12, face="bold"),
            axis_text_x=element_text(angle=30, ha="right", size=8),
        )
    )
    ruta = str(OUTPUT_DIR / "lab_eda_boxplot_ingresos.png")
    p.save(ruta, width=12, height=6, dpi=120, verbose=False)
    print(f"[EDA] Guardado boxplot ingresos: {ruta}")


def eda_elbow_kmeans(df_integrado: pd.DataFrame) -> None:
    """
    Método del codo para determinar el número óptimo de clusters K-Means.
    Gramática: geom_line + geom_point sobre inertia vs. k
    """
    vars_cluster = [v for v in [
        "renta_media", "tasa_ocupacion", "tasa_paro",
        "pct_salarios", "pct_pensiones", "pct_prestaciones",
    ] if v in df_integrado.columns]

    df_ref = df_integrado[df_integrado["anio"] == 2023].dropna(subset=vars_cluster)
    X = StandardScaler().fit_transform(df_ref[vars_cluster])

    inertias = []
    ks = range(2, 10)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append({"k": k, "inertia": km.inertia_})

    df_elbow = pd.DataFrame(inertias)
    p = (
        ggplot(df_elbow, aes(x="k", y="inertia"))
        + geom_line(color="#1565C0", size=1.2)
        + geom_point(color="#E53935", size=3.5)
        + scale_x_continuous(breaks=list(ks))
        + labs(
            title="[EDA] Método del codo — Selección de K en K-Means",
            subtitle=f"Variables: {vars_cluster}. Datos año 2023",
            x="Número de clusters (k)",
            y="Inercia (suma de distancias cuadráticas intra-cluster)",
        )
        + theme_minimal()
        + theme(plot_title=element_text(size=12, face="bold"))
    )
    ruta = str(OUTPUT_DIR / "lab_eda_elbow_kmeans.png")
    p.save(ruta, width=8, height=5, dpi=120, verbose=False)
    print(f"[EDA] Guardado elbow plot: {ruta}")


def eda_pca_clusters(df_integrado: pd.DataFrame) -> None:
    """
    Proyección PCA de los clusters (dimensión reducida para inspección visual).
    Gramática: geom_point, aes(x=PC1, y=PC2, color=cluster)
    """
    vars_cluster = [v for v in [
        "renta_media", "tasa_ocupacion", "tasa_paro",
        "pct_salarios", "pct_pensiones", "pct_prestaciones",
    ] if v in df_integrado.columns]

    if "cluster" not in df_integrado.columns:
        print("[EDA] Columna 'cluster' no disponible. Ejecuta el pipeline completo primero.")
        return

    df_ref = df_integrado[df_integrado["anio"] == 2023].dropna(subset=vars_cluster + ["cluster"])
    X = StandardScaler().fit_transform(df_ref[vars_cluster])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    df_pca = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "cluster": df_ref["cluster"].values,
    })
    var_exp = pca.explained_variance_ratio_ * 100

    palette = {"1": "#1A237E", "2": "#42A5F5", "3": "#A5D6A7", "4": "#FFD54F", "5": "#E53935"}

    p = (
        ggplot(df_pca, aes(x="PC1", y="PC2", color="cluster"))
        + geom_point(alpha=0.65, size=2)
        + scale_color_manual(values=palette, name="Cluster")
        + labs(
            title="[EDA] Proyección PCA de clusters territoriales (2023)",
            subtitle=f"PC1: {var_exp[0]:.1f}% var. | PC2: {var_exp[1]:.1f}% var.",
            x=f"Componente Principal 1 ({var_exp[0]:.1f}%)",
            y=f"Componente Principal 2 ({var_exp[1]:.1f}%)",
        )
        + theme_minimal()
        + theme(
            plot_title=element_text(size=12, face="bold"),
            legend_position="right",
        )
    )
    ruta = str(OUTPUT_DIR / "lab_eda_pca_clusters.png")
    p.save(ruta, width=8, height=6, dpi=120, verbose=False)
    print(f"[EDA] Guardado PCA clusters: {ruta}")


# ─────────────────────────────────────────────────────────────────
# SECCIÓN 4 — ESTADÍSTICAS DESCRIPTIVAS DE REFERENCIA
# ─────────────────────────────────────────────────────────────────

def resumen_estadistico(df_integrado: pd.DataFrame) -> pd.DataFrame:
    """
    Genera tabla de estadísticos descriptivos por año.
    Útil para el informe académico.
    """
    vars_num = [v for v in [
        "renta_media", "renta_mediana", "tasa_ocupacion",
        "tasa_paro", "tasa_actividad",
        "pct_salarios", "pct_pensiones", "pct_prestaciones",
    ] if v in df_integrado.columns]

    resumen = (
        df_integrado.groupby("anio")[vars_num]
        .agg(["mean", "median", "std", "min", "max"])
        .round(2)
    )
    print("\n[STATS] Resumen estadístico por año:")
    print(resumen.to_string())
    return resumen


def indice_gini_aproximado(df_renta: pd.DataFrame) -> pd.DataFrame:
    """
    Aproximación del índice de Gini por año a partir de la distribución
    de renta media por sección censal.
    (Aproximación: no equivale al Gini individual, sino al Gini inter-seccional)
    """
    resultados = []
    for anio in AÑOS:
        renta = df_renta[df_renta["anio"] == anio]["renta_media"].dropna().sort_values().values
        n = len(renta)
        if n == 0:
            continue
        # Fórmula de Brown para el Gini
        indices = np.arange(1, n + 1)
        gini = (2 * np.sum(indices * renta)) / (n * np.sum(renta)) - (n + 1) / n
        resultados.append({"anio": anio, "gini_interseccional": round(gini, 4)})

    df_gini = pd.DataFrame(resultados)
    print("\n[STATS] Índice de Gini inter-seccional aproximado:")
    print(df_gini.to_string(index=False))
    return df_gini


# ─────────────────────────────────────────────────────────────────
# MAIN — EJECUCIÓN DEL LABORATORIO
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("LAB SECCIONES — Análisis exploratorio socioeconómico Tenerife")
    print("=" * 70)

    # ── 1. Carga ──────────────────────────────────────────────────
    try:
        df_renta_raw = cargar_dataset("rentamedia-sc-3.csv")
        df_ingresos_raw = cargar_dataset("distribucion-renta-ingresos.csv")
        df_actividad_raw = cargar_dataset("actividad-sc-3.csv")
    except FileNotFoundError as e:
        print(f"\n⚠ ERROR: {e}")
        print("Asegúrate de colocar los archivos CSV en la carpeta 'data/'.")
        return

    # ── 2. EDA inicial ────────────────────────────────────────────
    explorar_dataset(df_renta_raw, "rentamedia-sc-3.csv")
    explorar_dataset(df_ingresos_raw, "distribucion-renta-ingresos.csv")
    explorar_dataset(df_actividad_raw, "actividad-sc-3.csv")

    # ── 3. Limpieza ───────────────────────────────────────────────
    df_renta = limpiar_renta(df_renta_raw)
    df_ingresos = limpiar_ingresos(df_ingresos_raw)
    df_actividad = limpiar_actividad(df_actividad_raw)

    # ── 4. Integración tabular (sin geo, solo para EDA) ───────────
    df_merge = df_renta.merge(df_ingresos, on=["seccion", "anio"], how="outer")
    df_merge = df_merge.merge(df_actividad, on=["seccion", "anio"], how="outer")
    print(f"\n[MERGE] Dataset integrado tabular: {df_merge.shape}")

    # ── 5. Estadísticas descriptivas ──────────────────────────────
    resumen_estadistico(df_merge)
    indice_gini_aproximado(df_renta)

    # ── 6. Visualizaciones exploratorias ─────────────────────────
    eda_distribucion_renta(df_renta)
    eda_boxplot_ingresos(df_ingresos)
    eda_renta_vs_actividad(df_merge)
    eda_elbow_kmeans(df_merge)

    # ── 7. Clustering exploratorio para PCA ───────────────────────
    vars_cluster = [v for v in [
        "renta_media", "tasa_ocupacion", "tasa_paro",
        "pct_salarios", "pct_pensiones",
    ] if v in df_merge.columns]

    if vars_cluster:
        df_ref = df_merge[df_merge["anio"] == 2023].dropna(subset=vars_cluster).copy()
        X = StandardScaler().fit_transform(df_ref[vars_cluster])
        km = KMeans(n_clusters=5, random_state=42, n_init=20)
        df_merge.loc[df_merge["anio"] == 2023, "cluster"] = (km.fit_predict(X) + 1).astype(str)
        eda_pca_clusters(df_merge)

    print(f"\n{'='*70}")
    print(f"Laboratorio completado. Outputs en: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()