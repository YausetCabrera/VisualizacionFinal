"""
Pipeline: Análisis socioeconómico por sección censal
Provincia de Santa Cruz de Tenerife -> 2021-2023
 
Arquitectura de assets Dagster:
  Capa RAW         -> ingesta directa desde CSV/GeoJSON
  Capa CLEAN       -> limpieza, normalización y tipado
  Capa INTEGRATED  -> joins y construcción del dataset analítico + clustering
  Capa VIZ         -> visualizaciones ggplot (plotnine, matplotlib puntualmente)
 
Autor: Yauset Cabrera Aparicio (2026, Visualización, Máster Universitario en Ciberseguridad e Inteligencia de Datos)
"""
 
import os
import json
import warnings
from pathlib import Path
 
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_text,
    facet_wrap, scale_fill_gradient, scale_fill_gradientn,
    scale_fill_manual, scale_color_manual, scale_color_brewer,
    labs, theme, theme_void, theme_minimal,
    element_text, element_rect, element_line, element_blank,
    scale_x_continuous, scale_y_continuous,
    guides, guide_legend, guide_colorbar,
    annotate, geom_vline, geom_hline, geom_map
)

from plotnine.scales import scale_fill_cmap
from plotnine import geom_polygon
from sklearn.metrics import silhouette_score
 
import dagster as dg
from dagster import (
    asset, AssetIn, MetadataValue, Output,
    asset_check, AssetCheckResult, AssetCheckSeverity,
)
 
warnings.filterwarnings("ignore")
 
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
 
AÑOS = [2021, 2022, 2023]
 
PALETTE_INGRESOS = {
    "Salarios":          "#2196F3",   # azul
    "Pensiones":         "#FF9800",   # naranja
    "Prestaciones":      "#9C27B0",   # violeta
    "Rendimientos_cap":  "#4CAF50",   # verde
    "Otros":             "#F44336",   # rojo
}
 
PALETTE_CLUSTER = {
    "1": "#1A237E",   # Azul muy oscuro – zonas deprimidas
    "2": "#42A5F5",   # Azul medio – zonas medias-bajas
    "3": "#A5D6A7",   # Verde claro – zonas medias-altas
    "4": "#FFD54F",   # Amarillo – zonas acomodadas
    "5": "#E53935",   # Rojo – zonas de alta renta
}
 
# Helpers de carga
 
def _leer_csv(nombre: str, sep: str = None) -> pd.DataFrame:
    """Carga un CSV detectando automáticamente el separador (`;` o `,`)."""
    ruta = DATA_DIR / nombre
    if not ruta.exists():
        raise FileNotFoundError(
            f"Archivo no encontrado: {ruta}\n"
            f"Coloca el CSV en la carpeta 'data/'."
        )
    if sep is None:
        with open(ruta, encoding="utf-8-sig") as f:
            primera = f.readline()
        sep = ";" if primera.count(";") > primera.count(",") else ","
    return pd.read_csv(ruta, sep=sep, encoding="utf-8-sig", thousands=".", decimal=",")
 
 
def _leer_geojson(nombre: str) -> gpd.GeoDataFrame:
    """Carga un GeoJSON y garantiza CRS ETRS89 (EPSG:4326)."""
    ruta = DATA_DIR / nombre
    if not ruta.exists():
        raise FileNotFoundError(f"GeoJSON no encontrado: {ruta}")
    gdf = gpd.read_file(ruta)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf
 
 
def _normalizar_seccion(s: pd.Series) -> pd.Series:
    """
    Normaliza el código de sección censal a formato INE de 10 dígitos:
    PPMMMSSS ->  PPMMMSSSCC (donde CC = código de sección sin padding histórico)
    Acepta strings y enteros; asegura dtype str con ceros a la izquierda.
    """
    return (
        s.astype(str)
         .str.strip()
         .str.replace(r"\s+", "", regex=True)
         .str.zfill(10)
    )
 


# Assets de carga (RAW)
 
@asset(
    group_name="raw",
    description="Carga del CSV de renta media y mediana por sección censal (2021-2023).",
    compute_kind="pandas",
)
def raw_renta_media(context: dg.AssetExecutionContext) -> Output[pd.DataFrame]:
    df = _leer_csv("rentamedia-sc-3.csv")
    context.log.info(f"raw_renta_media cargado: {df.shape}")

    context.log.info(df["OBS_VALUE"].describe())
    context.log.info(df["OBS_VALUE"].head(10))
    context.log.info(df.dtypes)
    return Output(
        df,
        metadata={
            "n_filas": MetadataValue.int(len(df)),
            "n_columnas": MetadataValue.int(len(df.columns)),
            "columnas": MetadataValue.text(str(list(df.columns))),
            "preview": MetadataValue.md(df.head(5).to_markdown()),
        },
    )
 
 
@asset(
    group_name="raw",
    description="Carga del CSV de distribución de ingresos por fuente (2021-2023).",
    compute_kind="pandas",
)
def raw_distribucion_ingresos(context: dg.AssetExecutionContext) -> Output[pd.DataFrame]:
    df = _leer_csv("distribucion-renta-ingresos.csv")
    context.log.info(f"raw_distribucion_ingresos cargado: {df.shape}")
    return Output(
        df,
        metadata={
            "n_filas": MetadataValue.int(len(df)),
            "n_columnas": MetadataValue.int(len(df.columns)),
            "columnas": MetadataValue.text(str(list(df.columns))),
            "preview": MetadataValue.md(df.head(5).to_markdown()),
        },
    )
 
 
@asset(
    group_name="raw",
    description="Carga del CSV de actividad (ocupación, paro, inactividad) (2021-2023).",
    compute_kind="pandas",
)
def raw_actividad(context: dg.AssetExecutionContext) -> Output[pd.DataFrame]:
    df = _leer_csv("actividad-sc-3.csv")
    context.log.info(f"raw_actividad cargado: {df.shape}")
    return Output(
        df,
        metadata={
            "n_filas": MetadataValue.int(len(df)),
            "n_columnas": MetadataValue.int(len(df.columns)),
            "columnas": MetadataValue.text(str(list(df.columns))),
            "preview": MetadataValue.md(df.head(5).to_markdown()),
        },
    )
 
 
@asset(
    group_name="raw",
    description="Carga de los GeoJSON de secciones censales (multi-año).",
    compute_kind="geopandas",
)
def raw_geojson_secciones(context: dg.AssetExecutionContext) -> Output[gpd.GeoDataFrame]:

    geojson_files = sorted(DATA_DIR.glob("secciones_*_tenerife.json"))

    if not geojson_files:
        raise FileNotFoundError("No se encontraron GeoJSON en data/")

    gdfs = []

    for path in geojson_files:
        gdf = gpd.read_file(path)
        año = int(path.name.split("_")[1][:4])

        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")

        gdf["anio"] = año
        gdfs.append(gdf)

        context.log.info(f"Cargado {path.name}: {gdf.shape}")

    gdf_all = pd.concat(gdfs, ignore_index=True)

    return Output(
        gdf_all,
        metadata={
            "n_total": MetadataValue.int(len(gdf_all)),
            "anios": MetadataValue.text(str(sorted(gdf_all["anio"].unique().tolist()))),
        },
    )
 


# Assets de limpieza (CLEAN)
 
@asset(
    group_name="clean",
    ins={"raw": AssetIn("raw_renta_media")},
    description=(
        "Limpieza de renta media: normalización de códigos de sección."),
    compute_kind="pandas",
)
def clean_renta_media(
    context: dg.AssetExecutionContext,
    raw: pd.DataFrame,
) -> Output[pd.DataFrame]:
    df = raw.copy()
    col_map = _detectar_columnas_renta(df)
    context.log.info(f"Mapeo de columnas detectado: {col_map}")

    df = df.rename(columns=col_map)
    context.log.info(f"Columnas tras rename: {df.columns.tolist()}")

    if "OBS_VALUE" in df.columns:
        df = df.rename(columns={"OBS_VALUE": "renta_media"})
    elif "renta_media" not in df.columns:
        raise ValueError(
            "No se encontró ni 'OBS_VALUE' ni 'renta_media' en el dataset de renta media."
        )

    if "anio" not in df.columns:
        raise ValueError(f"No se encontró columna 'anio'. Columnas: {df.columns.tolist()}")

    if "seccion" not in df.columns:
        raise ValueError(f"No se encontró columna 'seccion'. Columnas: {df.columns.tolist()}")

    df["anio"] = pd.to_numeric(df["anio"], errors="coerce")
    df = df[df["anio"].isin(AÑOS)].copy()
    df["seccion"] = _normalizar_seccion(df["seccion"])

    for col in ["renta_media", "renta_mediana"]:
        if col in df.columns:

            if pd.api.types.is_numeric_dtype(df[col]):
                continue

            df[col] = (
                df[col]
                .astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
                .replace({"": np.nan, "nd": np.nan, "N/A": np.nan})
                .pipe(pd.to_numeric, errors="coerce")
            )

    n_antes = len(df)
    df = df.drop_duplicates(subset=["seccion", "anio"])
    n_dupl = n_antes - len(df)

    if n_dupl > 0:
        context.log.warning(f"Se eliminaron {n_dupl} duplicados en renta_media.")

    df = df.reset_index(drop=True)

    return Output(
        df,
        metadata={
            "n_filas": MetadataValue.int(len(df)),
            "n_nulos_renta": MetadataValue.int(int(df["renta_media"].isna().sum()))
            if "renta_media" in df.columns else MetadataValue.int(0),
            "anios_presentes": MetadataValue.text(str(sorted(df["anio"].unique().tolist()))),
            "n_secciones_unicas": MetadataValue.int(df["seccion"].nunique()),
            "preview": MetadataValue.md(df.head(5).to_markdown()),
        },
    )
 
 
@asset(
    group_name="clean",
    ins={"raw": AssetIn("raw_distribucion_ingresos")},
    description=(
        "Limpieza de distribución de ingresos."),
    compute_kind="pandas",
)
def clean_distribucion_ingresos(
    context: dg.AssetExecutionContext,
    raw: pd.DataFrame,
) -> Output[pd.DataFrame]:

    df = raw.copy()

    col_map = _detectar_columnas_ingresos(df)
    context.log.info(f"Mapeo columnas ingresos: {col_map}")
    df = df.rename(columns=col_map)

    context.log.info(f"Columnas disponibles: {df.columns.tolist()}")

    if "seccion" not in df.columns or "anio" not in df.columns:
        raise ValueError("Faltan columnas base: seccion o anio")

    df["anio"] = pd.to_numeric(df["anio"], errors="coerce")
    df = df[df["anio"].isin(AÑOS)].copy()

    value_col = None
    for c in ["OBS_VALUE", "valor", "value", "importe"]:
        if c in df.columns:
            value_col = c
            break

    if value_col is None:
        raise ValueError("No se encontró columna de valores (OBS_VALUE)")

    if "MEDIDAS#es" in df.columns:
        cat_col = "MEDIDAS#es"
    elif "medidas" in df.columns:
        cat_col = "medidas"
    else:
        raise ValueError("No se encontró columna de categoría (MEDIDAS)")

    df[value_col] = (
        df[value_col]
        .astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"": np.nan, "nd": np.nan, "N/A": np.nan})
    )

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    df_wide = (
        df.pivot_table(
            index=["seccion", "anio"],
            columns=cat_col,
            values=value_col,
            aggfunc="sum",
        )
        .reset_index()
    )

    df_wide.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in df_wide.columns
    ]

    cols_fuentes = [
        c for c in df_wide.columns
        if c not in ["seccion", "anio"]
    ]

    total = df_wide[cols_fuentes].sum(axis=1).replace(0, np.nan)

    for c in cols_fuentes:
        df_wide[f"pct_{c}"] = df_wide[c] / total * 100

    pct_cols = [f"pct_{c}" for c in cols_fuentes]

    df_wide["ingreso_dominante"] = (
        df_wide[pct_cols]
        .idxmax(axis=1)
        .str.replace("pct_", "", regex=False)
    )

    return Output(
        df_wide,
        metadata={
            "n_filas": dg.MetadataValue.int(len(df_wide)),
            "fuentes_detectadas": dg.MetadataValue.text(str(cols_fuentes)),
            "preview": dg.MetadataValue.md(df_wide.head(5).to_markdown()),
        },
    )
 
 
@asset(
    group_name="clean",
    description=(
        "Limpieza de actividad."),
    ins={"raw": AssetIn("raw_actividad")},
    compute_kind="pandas",
)
def clean_actividad(
    context: dg.AssetExecutionContext,
    raw: pd.DataFrame,
) -> Output[pd.DataFrame]:

    df = raw.copy()
    col_map = _detectar_columnas_actividad(df)
    context.log.info(f"Mapeo columnas actividad: {col_map}")
    df = df.rename(columns=col_map)

    context.log.info(f"Columnas disponibles: {df.columns.tolist()}")
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce")
    df = df[df["anio"].isin(AÑOS)].copy()

    if "geocode" not in df.columns:
        raise ValueError("No existe columna 'geocode'")

    df["seccion"] = (
        df["geocode"]
        .astype(str)
        .str.extract(r'_(\d{5}_D\d{2}_S\d{3})$')[0]
    )

    df["seccion"] = _normalizar_seccion(df["seccion"])
    df["num_casos"] = (
        df["num_casos"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"": np.nan, "nd": np.nan})
        .pipe(pd.to_numeric, errors="coerce")
    )

    df = (
        df.groupby(
            ["seccion", "anio", "Sexo", "Actividad económica"],
            as_index=False
        )["num_casos"]
        .sum()
    )

    df["total_seccion_anio"] = df.groupby(
        ["seccion", "anio"]
    )["num_casos"].transform("sum")

    df["pct_seccion"] = df["num_casos"] / df["total_seccion_anio"] * 100
    df = df.drop_duplicates(["seccion", "anio", "Sexo", "Actividad económica"])
    df = df.reset_index(drop=True)

    return Output(
        df,
        metadata={
            "n_filas": MetadataValue.int(len(df)),
            "n_secciones": MetadataValue.int(df["seccion"].nunique()),
            "actividades": MetadataValue.text(
                str(df["Actividad económica"].nunique())
            ),
            "preview": MetadataValue.md(df.head(5).to_markdown()),
        },
    )
 
@asset(
    group_name="clean",
    ins={"raw": AssetIn("raw_geojson_secciones")},
    description="Limpieza multi-año del GeoJSON: Normalización de sección, Con Validación Geométrica.",
    compute_kind="geopandas",
)
def clean_geojson(
    context: dg.AssetExecutionContext,
    raw: gpd.GeoDataFrame,
) -> Output[gpd.GeoDataFrame]:

    gdf = raw.copy()

    if "anio" not in gdf.columns:
        raise ValueError("El GeoJSON debe incluir columna 'anio' (multi-año)")

    if "geocode" not in gdf.columns:
        raise ValueError(f"No existe 'geocode'. Columnas: {list(gdf.columns)}")

    # municipio (5 dígitos: ya incluye provincia)
    gdf["municipio"] = gdf["gcd_municipio"].astype(str).str.zfill(5)

    # distrito (2 dígitos)
    gdf["distrito"] = gdf["geocode"].str.extract(r'_D(\d{2})_')[0]

    # sección (3 dígitos)
    gdf["seccion_num"] = gdf["geocode"].str.extract(r'_S(\d{3})$')[0]

    gdf["seccion"] = (
        gdf["municipio"]
        + gdf["distrito"]
        + gdf["seccion_num"]
    )

    gdf["seccion"] = _normalizar_seccion(gdf["seccion"])

    if gdf["seccion"].isna().any():
        raise ValueError("Error generando códigos de sección en GeoJSON")

    n_invalidas = (~gdf.geometry.is_valid).sum()

    if n_invalidas > 0:
        context.log.warning(f"{n_invalidas} geometrías inválidas → buffer(0)")
        gdf["geometry"] = gdf["geometry"].buffer(0)

    gdf = gdf[~gdf.geometry.is_empty].copy()

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    gdf = gdf.drop_duplicates(subset=["seccion", "anio"])

    context.log.info(
        f"clean_geojson: {len(gdf)} filas | años: {sorted(gdf['anio'].unique())}"
    )

    context.log.info(f"Ejemplo secciones geo: {gdf['seccion'].head(5).tolist()}")

    return Output(
        gdf,
        metadata={
            "n_filas": MetadataValue.int(len(gdf)),
            "anios": MetadataValue.text(str(sorted(gdf["anio"].unique().tolist()))),
            "n_secciones": MetadataValue.int(gdf["seccion"].nunique()),
            "n_invalidas_corregidas": MetadataValue.int(int(n_invalidas)),
        },
    )
 



# Assets de integración (INTEGRATED)
 
@asset(
    group_name="integrated",
    ins={
        "renta": AssetIn("clean_renta_media"),
        "ingresos": AssetIn("clean_distribucion_ingresos"),
        "actividad": AssetIn("clean_actividad"),
        "geo": AssetIn("clean_geojson"),
    },
    compute_kind="pandas",
    description= "Se consolida un dataframe común con todas las variables a emplear en esta versión del pipeline. No se han incluido los datos de actividad."
)
def dataset_integrado(
    context: dg.AssetExecutionContext,
    renta: pd.DataFrame,
    ingresos: pd.DataFrame,
    actividad: pd.DataFrame,
    geo: gpd.GeoDataFrame,
) -> Output[gpd.GeoDataFrame]:

    renta_df, ingresos_df, actividad_df, geo_df = renta.copy(), ingresos.copy(), actividad.copy(), geo.copy()
    geo_df = geo_df.to_crs("EPSG:32628")

    def _extract_sufijo_seccion(series):
        return series.astype(str).str.replace(r"[^0-9]", "", regex=True).str[-3:]

    for df in [renta_df, ingresos_df, actividad_df, geo_df]:
        df["sec_id"] = _extract_sufijo_seccion(df["seccion"])
        df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype(int)

    mapa_desfase = {2021: 2022, 2022: 2023, 2023: 2024}
    for df in [renta_df, ingresos_df, actividad_df]:
        df_filtered = df[df["anio"].isin(mapa_desfase.keys())].copy()
        df_filtered["anio_geo"] = df_filtered["anio"].map(mapa_desfase)
        if df is renta_df: renta_df = df_filtered
        elif df is ingresos_df: ingresos_df = df_filtered
        elif df is actividad_df: actividad_df = df_filtered

    join_keys = ["gcd_isla", "sec_id", "anio_geo"]
    if "gcd_isla" not in actividad_df.columns:
        context.log.warning("Columna 'gcd_isla' no encontrada. Saltando agrupación por isla.")
        join_keys = ["sec_id", "anio_geo"]

    actividad_agg = actividad_df.groupby(join_keys, as_index=False).agg({"num_casos": "sum", "pct_seccion": "mean"})
    cols_ingresos_base = ["sueldos_y_salarios", "pensiones", "prestaciones_por_desempleo", "otras_prestaciones", "otros_ingresos"]
    cols_a_traer = list(set(["seccion", "anio", "ingreso_dominante"] + join_keys + 
                       [c for c in ingresos_df.columns if c.startswith("pct_") or c in cols_ingresos_base]))
    
    cols_a_traer = [c for c in cols_a_traer if c in ingresos_df.columns]
    ingresos_limpio = ingresos_df[cols_a_traer].drop_duplicates(subset=join_keys).copy()

    # Estandarizado común
    for df_to_clean in [renta_df, ingresos_limpio, actividad_agg]:
        for col in join_keys:
            df_to_clean[col] = df_to_clean[col].astype(str).str.strip()
            if col == "sec_id":
                df_to_clean[col] = df_to_clean[col].str.zfill(3)

    df_tab = renta_df.merge(ingresos_limpio, on=join_keys, how="left")
    df_tab = df_tab.merge(actividad_agg, on=join_keys, how="left")
    
    context.log.info(f"Suma tras merge: {df_tab['sueldos_y_salarios'].sum()}")
    join_keys_final = ["sec_id", "anio_geo"]
    
    geo_df["sec_id"] = geo_df["sec_id"].astype(str).str.strip().str.zfill(3)
    geo_df["anio_geo"] = geo_df["anio"].map(mapa_desfase).fillna(0).astype(int)
    
    df_tab["sec_id"] = df_tab["sec_id"].astype(str).str.strip().str.zfill(3)
    df_tab["anio_geo"] = df_tab["anio_geo"].astype(int)

    cols_a_evitar = [c for c in df_tab.columns if c in geo_df.columns and c not in join_keys_final]
    df_tab_clean = df_tab.drop(columns=cols_a_evitar)
    gdf = geo_df.merge(df_tab_clean, on=join_keys_final, how="left")
    
    context.log.info(f"Registros en GDF tras merge: {len(gdf)}")
    context.log.info(f"Registros con datos de renta: {gdf['renta_media'].notna().sum()}")

    for col in cols_ingresos_base:
        gdf[col] = pd.to_numeric(gdf[col], errors='coerce').fillna(0)

    gdf["ingreso_dominante"] = gdf[cols_ingresos_base].idxmax(axis=1)
    
    # Debug
    # context.log.info(f"Suma de sueldos: {gdf['sueldos_y_salarios'].sum()}")
    # context.log.info(f"Muestra numérica: {gdf[cols_ingresos_base].head(2).to_string()}")

    stats = gdf.groupby("anio").apply(lambda x: (x["renta_media"].notna().sum() / len(x)) if len(x) > 0 else 0).to_dict()

    return Output(
        gdf, 
        metadata={
            "n_registros": MetadataValue.int(len(gdf)), 
            "coverage": MetadataValue.text(str(stats)),
            "preview": MetadataValue.md(gdf.drop(columns=["geometry"], errors="ignore").head(5).to_markdown())
        }
    )
 
 
@asset(
    group_name="integrated",
    ins={"base": AssetIn("dataset_integrado")},
    description=(
        "Clustering territorial por año con selección automática de k (2–5) "
        "mediante silhouette score. Genera métricas por año."
    ),
    compute_kind="sklearn",
)
def clusters_territoriales(
    context: dg.AssetExecutionContext,
    base: gpd.GeoDataFrame,
) -> Output[gpd.GeoDataFrame]:

    import numpy as np
    import pandas as pd
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import dagster as dg

    def _seleccionar_vars_cluster(df):
        # Las variables aquí se pueden modificar.
        cols = [
            "renta_media", 
            "pct_sueldos_y_salarios", 
            "pct_pensiones", 
            "pct_prestaciones_por_desempleo"
        ]
        return [c for c in cols if c in df.columns]

    resultados_global = {}
    outputs = []
    años = sorted(base["anio"].dropna().unique())

    for año in años:
        context.log.info(f"--- Clustering año {año} ---")
        df_year = base[base["anio"] == año].copy()

        vars_cluster = _seleccionar_vars_cluster(df_year)
        df_year_clean = df_year.dropna(subset=vars_cluster).copy()

        if len(df_year_clean) < 10:
            context.log.warning(f"Año {año} con pocos datos ({len(df_year_clean)}) → se omite")
            continue

        scaler = StandardScaler()
        X = scaler.fit_transform(df_year_clean[vars_cluster])

        resultados = {}
        for k in [2, 3, 4, 5]:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
                labels = kmeans.fit_predict(X)
                score = silhouette_score(X, labels)
                resultados[k] = {
                    "labels": labels,
                    "silhouette": float(score),
                    "inertia": float(kmeans.inertia_),
                }
            except Exception as e:
                context.log.warning(f"Año {año} | k={k} falló: {e}")

        if not resultados: continue

        best_k = max(resultados, key=lambda k: resultados[k]["silhouette"])
        best = resultados[best_k]

        df_year_clean["cluster"] = (best["labels"] + 1).astype(str)
        orden = df_year_clean.groupby("cluster")["renta_media"].mean().sort_values().index.tolist()
        mapping = {c: str(i + 1) for i, c in enumerate(orden)}
        df_year_clean["cluster"] = df_year_clean["cluster"].map(mapping)

        resultados_global[int(año)] = {
            "best_k": int(best_k),
            "silhouette": round(best["silhouette"], 4),
            "distribucion": df_year_clean["cluster"].value_counts(normalize=True).to_dict()
        }

        outputs.append(df_year_clean[["seccion", "anio", "cluster"]])

    if outputs:
        clusters_df = pd.concat(outputs, ignore_index=True)
        # Merge usando las llaves consistentes del dataset_integrado
        out = base.merge(clusters_df, on=["seccion", "anio"], how="left")
    else:
        out = base.copy()
        out["cluster"] = np.nan

    out = out[out["cluster"].notna()].copy() 
    out["anio"] = out["anio"].astype(int)
    out["cluster"] = out["cluster"].astype(str)

    return Output(
        out,
        metadata={
            "años_clusterizados": MetadataValue.text(str(sorted(out["anio"].unique().tolist()))),
            "resultados_resumen": MetadataValue.json(resultados_global),
        },
    )



# Assets de Visualización (VIS)
 
@asset(
    group_name="visualization",
    ins={"base": AssetIn("dataset_integrado")},
    compute_kind="geopandas",
    description= "(VIZ 1) Visualización de Mapa de Renta Media"
)
def viz_mapa_renta_media(context, base):
    import geopandas as gpd
    import matplotlib.pyplot as plt

    import geopandas as gpd
    import matplotlib.pyplot as plt

    df = base.copy()
    if df.crs is None or df.crs.to_epsg() != 32628:
        df = df.to_crs("EPSG:32628")
    
    anios_deseados = [2021, 2022, 2023]
    anios_mapa = [a for a in sorted(df["anio"].dropna().unique().astype(int)) if a in anios_deseados]
    context.log.info(f"Años a representar en la matriz: {anios_mapa}")
    
    islas_ordenadas = sorted(df["gcd_isla"].dropna().unique())
    n_columnas = len(anios_mapa)
    
    fig, axes = plt.subplots(len(islas_ordenadas), n_columnas, 
                             figsize=(16, 12), squeeze=False)
    
    vmin, vmax = df["renta_media"].min(), df["renta_media"].max()

    for r, isla in enumerate(islas_ordenadas):
        for c, anio in enumerate(anios_mapa):
            ax = axes[r, c]
            sub = df[(df["gcd_isla"] == isla) & (df["anio"] == anio)]
            
            if not sub.empty:
                sub.plot(column="renta_media", cmap="viridis", vmin=vmin, vmax=vmax, 
                         edgecolor="white", linewidth=0.05, ax=ax)
                ax.set_xlim(sub.total_bounds[0], sub.total_bounds[2])
                ax.set_ylim(sub.total_bounds[1], sub.total_bounds[3])
            
            ax.axis("off")
            if r == 0: ax.set_title(f"Año {anio}", fontsize=12, fontweight="bold")
            if c == 0: ax.text(-0.1, 0.5, str(isla), fontsize=10, fontweight="bold", 
                               va="center", ha="right", transform=ax.transAxes, rotation=90)

    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    cax = fig.add_axes([0.25, 0.02, 0.5, 0.015])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Renta Bruta Media (€)")

    ruta = str(OUTPUT_DIR / "viz_mapa_renta_media.png")
    plt.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return Output(ruta)
 
 
@asset(
    group_name="visualization",
    ins={"base": AssetIn("dataset_integrado")},
    description="(VIZ 2) Mapa de Especialización con todas las categorías.",
    compute_kind="geopandas",
)
def viz_mapa_ingreso_dominante(context, base):
    import matplotlib.pyplot as plt
    import geopandas as gpd
    import pandas as pd
    from matplotlib.patches import Patch

    df = base.copy()
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce")
    anios = [2021, 2022, 2023]
    islas = sorted(df["gcd_isla"].dropna().unique())
    
    pct_cols = [
        "pct_sueldos_y_salarios", 
        "pct_pensiones", 
        "pct_prestaciones_por_desempleo", 
        "pct_otras_prestaciones", 
        "pct_otros_ingresos"
    ]

    def calcular_divergencia(sub_df):
        medias = sub_df[pct_cols].mean()
        sub_df["ingreso_dominante"] = (sub_df[pct_cols] - medias).idxmax(axis=1)
        return sub_df
    
    df = df.groupby(["gcd_isla", "anio"], group_keys=False).apply(calcular_divergencia)

    # Uso otros colores para esta visualización
    color_map = {
        "Salarios": "#87CEEB",
        "Pensiones": "#8B4513",
        "Desempleo": "#98FF98",
        "Otras Prestaciones": "#006400",
        "Otros Ingresos": "#DDA0DD"
    }

    def get_cat(col_name):
        s = str(col_name).lower()
        
        if "sueldos_y_salarios" in s: return "Salarios"
        if "pensiones" in s: return "Pensiones"
        if "prestaciones_por_desempleo" in s: return "Desempleo"
        if "otras_prestaciones" in s: return "Otras Prestaciones"
        if "otros_ingresos" in s: return "Otros Ingresos"
        
        return "Otros Ingresos" # Valor por defecto para casos no mapeados
        
    df["categoria"] = df["ingreso_dominante"].astype(str).apply(get_cat)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(len(islas), len(anios), height_ratios=[1, 1, 1, 2.5], wspace=0.02, hspace=0.05) 
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)

    for r, isla in enumerate(islas):
        for c, anio in enumerate(anios):
            ax = fig.add_subplot(gs[r, c])
            sub = df[(df["gcd_isla"] == isla) & (df["anio"] == anio)]
            
            if not sub.empty:
                for cat, color in color_map.items():
                    data = sub[sub["categoria"] == cat]
                    if not data.empty:
                        data.plot(color=color, ax=ax, edgecolor="black", linewidth=0.1)
                
                ax.set_xlim(sub.total_bounds[0], sub.total_bounds[2])
                ax.set_ylim(sub.total_bounds[1], sub.total_bounds[3])
            
            ax.set_axis_off()
            if r == 0: ax.set_title(f"Año {anio}", fontsize=12, fontweight="bold")
            if c == 0: ax.text(-0.15, 0.5, str(isla), transform=ax.transAxes, fontweight="bold", rotation=90, va="center", ha="center")

    legend_elements = [Patch(facecolor=c, label=l) for l, c in color_map.items()]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=8, title="Fuente con mayor desviación")
    
    plt.figtext(0.95, 0.02, "Fuente: Atlas de Distribución de Renta, INE / AEAT", horizontalalignment='right', fontsize=9, color='gray')

    ruta = str(OUTPUT_DIR / "viz_ingreso_dominante.png")
    plt.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return Output(ruta)


@asset(
    group_name="visualization",
    ins={"base": AssetIn("clusters_territoriales")},
    description="(VIZ 3) Mapa de síntesis de clústers.",
    compute_kind="plotnine",
)
def viz_mapa_clusters(
    context: dg.AssetExecutionContext,
    base: gpd.GeoDataFrame,
) -> Output[str]:
    from plotnine import (
        ggplot, aes, geom_map, scale_fill_manual, facet_wrap, 
        labs, theme_void, theme, element_text, element_rect
    )

    df = base[base["anio"].isin([2021, 2022, 2023]) & base["cluster"].notna()].copy()
    
    labels_cluster = {
        "1": "1 — Renta muy baja",
        "2": "2 — Renta baja",
        "3": "3 — Renta media",
        "4": "4 — Renta media-alta",
        "5": "5 — Renta alta",
    }
    df["cluster_label"] = df["cluster"].map(labels_cluster).fillna("Sin clasificar")
    
    palette = {
        "1 — Renta muy baja": "#d73027",
        "2 — Renta baja": "#fc8d59",
        "3 — Renta media": "#fee08b",
        "4 — Renta media-alta": "#91cf60",
        "5 — Renta alta": "#1a9850"
    }

    p = (
        ggplot(df)
        + geom_map(aes(fill="cluster_label"), color="white", size=0.05)
        + facet_wrap("~anio", ncol=3)
        + scale_fill_manual(values=palette, name="Tipología")
        + labs(
            title="Evolución de Tipologías Socioeconómicas (2021-2023)",
            subtitle="Clustering K-Means basado en renta y estructura de ingresos",
            caption="Fuente: INE / AEAT · Elaboración propia"
        )
        + theme_void()
        + theme(
            plot_background=element_rect(fill="white"),
            strip_text=element_text(size=12, face="bold"),
            legend_position="bottom",
            legend_title=element_text(size=10, face="bold"),
            legend_text=element_text(size=9),
            plot_title=element_text(size=16, face="bold"),
            figure_size=(24, 8) 
        )
    )

    ruta = str(OUTPUT_DIR / "viz_evolucion_clusters.png")
    p.save(ruta, dpi=250, verbose=False)
    
    return Output(ruta, metadata={"ruta": dg.MetadataValue.path(ruta)})


@asset(
    group_name="visualization",
    ins={"base": AssetIn("clusters_territoriales")},
    description="(VIZ 4) Gráfico de barras con la distribución porcentual de secciones por cluster y año.",
    compute_kind="plotnine",
)
def viz_distribucion_clusters(
    context: dg.AssetExecutionContext,
    base: gpd.GeoDataFrame,
) -> Output[str]:
    from plotnine import (
        ggplot, aes, geom_col, position_fill, scale_fill_manual, 
        labs, theme_minimal, theme, element_text, element_rect
    )
    import pandas as pd

    df_counts = base.groupby(["anio", "cluster"]).size().reset_index(name="count")
    
    labels_cluster = {
        "1": "1 — Renta muy baja",
        "2": "2 — Renta baja",
        "3": "3 — Renta media",
        "4": "4 — Renta media-alta",
        "5": "5 — Renta alta",
    }
    df_counts["cluster_label"] = df_counts["cluster"].map(labels_cluster)
    
    palette = {
        "1 — Renta muy baja": "#d73027",
        "2 — Renta baja": "#fc8d59",
        "3 — Renta media": "#fee08b",
        "4 — Renta media-alta": "#91cf60",
        "5 — Renta alta": "#1a9850"
    }

    p = (
        ggplot(df_counts, aes(x="factor(anio)", y="count", fill="cluster_label"))
        + geom_col(position="fill") 
        + scale_fill_manual(values=palette, name="Tipología")
        + labs(
            title="Distribución Anual de Tipologías Socioeconómicas",
            subtitle="Porcentaje de secciones censales por cluster (2021-2023)",
            x="Año",
            y="Proporción de secciones",
            caption="Fuente: INE / AEAT · Elaboración propia"
        )
        + theme_minimal()
        + theme(
            legend_position="bottom",
            plot_title=element_text(size=14, face="bold"),
            figure_size=(10, 6)
        )
    )

    ruta = str(OUTPUT_DIR / "viz_distribucion_clusters.png")
    p.save(ruta, dpi=250, verbose=False)
    
    return Output(ruta, metadata={"ruta": dg.MetadataValue.path(ruta)})

 
@asset(
    group_name="visualization",
    ins={"base": AssetIn("clusters_territoriales")},
    description="(VIZ 5) Perfil Dinámico: Evolución conjunta de Renta, Salarios y Pensiones por Cluster.",
    compute_kind="plotnine",
)
def viz_perfil_dinamico_clusters(
    context: dg.AssetExecutionContext,
    base: gpd.GeoDataFrame,
) -> Output[str]:
    from plotnine import (
        ggplot, aes, geom_line, geom_point, facet_wrap, 
        labs, theme_bw, theme, element_text, scale_color_brewer
    )
    import pandas as pd

    vars_to_plot = ["renta_media", "pct_sueldos_y_salarios", "pct_pensiones"]
    for var in vars_to_plot:
        base[f"{var}_norm"] = (base[var] - base[var].min()) / (base[var].max() - base[var].min())

    df_perfil = base.groupby(["anio", "cluster"])[
        ["renta_media_norm", "pct_sueldos_y_salarios_norm", "pct_pensiones_norm"]
    ].mean().reset_index()

    df_long = df_perfil.melt(
        id_vars=["anio", "cluster"], 
        value_vars=["renta_media_norm", "pct_sueldos_y_salarios_norm", "pct_pensiones_norm"],
        var_name="variable", value_name="valor"
    )

    p = (
        ggplot(df_long, aes(x="factor(anio)", y="valor", color="variable", group="variable"))
        + geom_line(size=1.5)
        + geom_point(size=3)
        + facet_wrap("~cluster", labeller="label_both")
        + labs(
            title="Evolución del Perfil Socioeconómico por Cluster",
            subtitle="Comparativa normalizada de Renta, Salarios y Pensiones",
            x="Año", y="Valor Normalizado (0-1)"
        )
        + theme_bw()
    )

    ruta = str(OUTPUT_DIR / "viz_perfil_dinamico.png")
    p.save(ruta, dpi=250, verbose=False)
    
    return Output(ruta, metadata={"ruta": dg.MetadataValue.path(ruta)})


@asset(
    group_name="visualization",
    ins={"base": AssetIn("clusters_territoriales")},
    description=(
        "(VIZ 6) Análisis de Contribución: Correlación entre Renta Media y Salarios."),
    compute_kind="plotnine",
)
def viz_analisis_contribucion(
    context: dg.AssetExecutionContext,
    base: gpd.GeoDataFrame,
) -> Output[str]:
    from plotnine import (
        ggplot, aes, geom_point, geom_smooth, facet_wrap, 
        labs, theme_bw, theme, element_text, scale_x_continuous
    )

    p = (
        ggplot(base, aes(x="pct_sueldos_y_salarios", y="renta_media", color="factor(anio)"))
        + geom_point(alpha=0.5, size=1.5)
        + geom_smooth(method="lm", color="black", linetype="dashed")
        + facet_wrap("~anio")
        + labs(
            title="Motor de la Movilidad: Salarios vs. Renta",
            subtitle="Correlación lineal entre la especialización salarial y el nivel de renta",
            x="Peso de Sueldos y Salarios (%)",
            y="Renta Media Bruta (€)",
            color="Año",
            caption="Fuente: INE / AEAT · Elaboración propia"
        )
        + theme_bw()
        + theme(
            figure_size=(12, 6),
            strip_text=element_text(face="bold")
        )
    )

    ruta = str(OUTPUT_DIR / "viz_analisis_contribucionclusters.png")
    p.save(ruta, dpi=250, verbose=False)
    
    return Output(ruta, metadata={"ruta": dg.MetadataValue.path(ruta)})


@asset(
    group_name="visualization",
    ins={"base": AssetIn("dataset_integrado")},
    description="(VIZ 7) Composición de ingresos: Peso relativo de fuentes (sin Renta Media).",
    compute_kind="plotnine",
)
def viz_perfil(
    context: dg.AssetExecutionContext,
    base: gpd.GeoDataFrame,
) -> Output[str]:
    import plotnine as pn
    import pandas as pd

    mapping_islas = {
        'ES709': 'Tenerife',
        'ES706': 'La Gomera',
        'ES707': 'La Palma',
        'ES703': 'El Hierro'
    }

    df_data = base[base["anio"] == 2023].copy()
    idx = df_data.groupby("gcd_isla")["renta_media"].idxmax()
    df_top = df_data.loc[idx].copy()

    cols_ingresos = [
        "pct_sueldos_y_salarios", 
        "pct_prestaciones_por_desempleo", 
        "pct_pensiones", 
        "pct_otros_ingresos", 
        "pct_otras_prestaciones"
    ]
    
    df_top[cols_ingresos] = df_top[cols_ingresos].fillna(0)
    
    suma_ingresos = df_top[cols_ingresos].sum(axis=1)
    for col in cols_ingresos:
        df_top[f"{col}_prop"] = df_top[col] / suma_ingresos

    df_top["nombre_mun_isla"] = (
        df_top["etiqueta"].str.split("-").str[-1].str.strip() + 
        " (" + df_top["gcd_isla"].map(mapping_islas) + ")"
    )
    
    nombres_legibles = {
        "pct_sueldos_y_salarios_prop": "Sueldos y Salarios",
        "pct_prestaciones_por_desempleo_prop": "Desempleo",
        "pct_pensiones_prop": "Pensiones",
        "pct_otros_ingresos_prop": "Otros Ingresos",
        "pct_otras_prestaciones_prop": "Otras Prestaciones"
    }
    
    df_long = df_top.melt(
        id_vars=["nombre_mun_isla"], 
        value_vars=list(nombres_legibles.keys()),
        var_name="indicador", value_name="proporcion"
    )
    df_long["indicador"] = df_long["indicador"].replace(nombres_legibles)

    p = (
        pn.ggplot(df_long, pn.aes(x="indicador", y="proporcion", color="nombre_mun_isla"))
        + pn.geom_point(size=4)
        + pn.geom_segment(pn.aes(xend="indicador", yend=0), size=1.2)
        + pn.facet_wrap("~nombre_mun_isla")
        + pn.theme_minimal()
        + pn.theme(
            figure_size=(14, 8),
            axis_text_x=pn.element_text(rotation=45, hjust=1),
            strip_text=pn.element_text(size=10, weight="bold")
        )
        + pn.labs(
            title="Estructura de Ingresos: Fuentes principales en municipios líderes (2023)",
            subtitle="Peso relativo (%) de cada fuente sobre el total de ingresos de actividad y protección",
            x="", y="Proporción sobre el total"
        )
    )

    ruta = str(OUTPUT_DIR / "viz_perfil.png")
    p.save(ruta, dpi=250, verbose=False)
    
    return Output(ruta, metadata={"ruta": dg.MetadataValue.path(ruta)})


@asset(
    group_name="visualization",
    ins={"base": AssetIn("dataset_integrado")},
    description="(VIZ 8) Raincloud Plot: Distribución y desigualdad de la renta media por isla (2021-2023).",
    compute_kind="plotnine",
)
def viz_raincloud_renta(
    context: dg.AssetExecutionContext,
    base: gpd.GeoDataFrame,
) -> Output[str]:
    import plotnine as pn
    import pandas as pd

    df = base.dropna(subset=["renta_media", "anio", "gcd_isla"]).copy()
    
    mapping_islas = {'ES709': 'Tenerife', 'ES706': 'La Gomera', 'ES707': 'La Palma', 'ES703': 'El Hierro'}
    df["nombre_isla"] = df["gcd_isla"].map(mapping_islas)
    df["anio"] = df["anio"].astype(str)
    df = df.rename(columns={"anio": "Año"})

    p = (
        pn.ggplot(df, pn.aes(x="Año", y="renta_media", fill="Año"))
        + pn.geom_violin(alpha=0.3, width=0.7, show_legend=False)
        + pn.geom_boxplot(width=0.1, alpha=0.8, show_legend=False)
        + pn.geom_jitter(width=0.05, height=0, alpha=0.3, size=1)
        + pn.facet_wrap("~nombre_isla", scales="free_y")
        + pn.theme_minimal()
        + pn.guides(fill=pn.guide_legend(override_aes={'size': 10}))
        + pn.theme(
            figure_size=(12, 8),
            strip_text=pn.element_text(size=12, weight="bold")
        )
        + pn.labs(
            title="Dispersión y Desigualdad de la Renta Media",
            subtitle="Distribución de municipios por isla (2021-2023)",
            x="Año", y="Renta Media Bruta (€)",
            caption="Fuente: INE / AEAT · Elaboración propia"
        )
    )

    ruta = str(OUTPUT_DIR / "viz_raincloud_renta.png")
    p.save(ruta, dpi=250, verbose=False)
    
    return Output(ruta, metadata={"ruta": dg.MetadataValue.path(ruta)})


@asset(
    group_name="visualization",
    ins={"base": AssetIn("dataset_integrado")},
    description="(VIZ 9) Mapa de calor por evolución temporal en Tenerife.",
    compute_kind="matplotlib",
)
def viz_mapa_dinamica_tenerife(
    context: dg.AssetExecutionContext,
    base: gpd.GeoDataFrame,
) -> Output[str]:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import griddata
    from shapely.geometry import Point

    df = base[base["gcd_isla"] == "ES709"].copy()
    isla_geom = df.geometry.unary_union
    
    df_2021 = df[df["anio"] == 2021][["etiqueta", "renta_media"]].rename(columns={"renta_media": "r2021"})
    df_2023 = df[df["anio"] == 2023].merge(df_2021, on="etiqueta")
    df_2023["delta"] = df_2023["renta_media"] - df_2023["r2021"]
    
    x, y, z = df_2023.geometry.centroid.x, df_2023.geometry.centroid.y, df_2023["delta"]
    
    xi = np.linspace(x.min(), x.max(), 800)
    yi = np.linspace(y.min(), y.max(), 800)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    
    import matplotlib.path as mpltPath
    path = mpltPath.Path(np.array(isla_geom.exterior.coords))
    mask = path.contains_points(np.vstack((xi.flatten(), yi.flatten())).T).reshape(xi.shape)
    zi[~mask] = np.nan 
    
    fig, ax = plt.subplots(figsize=(12, 8))
    contour = ax.contourf(xi, yi, zi, levels=50, cmap="RdYlGn", alpha=0.8)
    
    ax.set_title("Evolución de la Renta Media en Tenerife (2021-2023)", 
                 fontsize=16, fontweight="bold", pad=20)
    
    cbar = plt.colorbar(contour, shrink=0.7)
    cbar.set_label("Variación de Renta (€)", rotation=270, labelpad=20)
    
    plt.text(1.0, -0.05, "Fuente: Elaboración propia a partir de datos INE", 
             transform=ax.transAxes, ha="right", fontsize=9, style="italic")
    
    ax.axis("off")
    
    ruta = str(OUTPUT_DIR / "viz_mapa_dinamico_tenerife.png")
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    return Output(ruta, metadata={"ruta": dg.MetadataValue.path(ruta)})



# Helpers internos
 
def _detectar_columnas_renta(df: pd.DataFrame) -> dict:
    """
    Detecta y mapea columnas del CSV de renta media a nombres canónicos.
    Acepta variantes habituales del INE/AEAT en español.
    """
    mapping = {}
    cols_lower = {c.lower().strip(): c for c in df.columns}
 
    for cand in ["periodo", "año", "anyo", "year", "anio"]:
        if cand in cols_lower:
            mapping[cols_lower[cand]] = "anio"
            break
 
    for col in df.columns:
        c = col.lower().strip()

        if any(k in c for k in ["secc", "cusec", "sec"]):
            mapping[col] = "seccion"
            break
 
    for cand in [
        "renta neta media por persona", "renta media", "renta_media",
        "media", "renta neta media", "renta_neta_media",
    ]:
        if cand in cols_lower and cols_lower[cand] not in mapping.values():
            mapping[cols_lower[cand]] = "renta_media"
            break
 
    for cand in [
        "renta neta mediana por persona", "renta mediana", "renta_mediana",
        "mediana", "renta neta mediana",
    ]:
        if cand in cols_lower and cols_lower[cand] not in mapping.values():
            mapping[cols_lower[cand]] = "renta_mediana"
            break
 
    return mapping
 
 
def _detectar_columnas_ingresos(df: pd.DataFrame) -> dict:
    mapping = {}
    cols_lower = {c.lower().strip(): c for c in df.columns}
 
    for cand in ["periodo", "año", "anyo", "year", "anio"]:
        if cand in cols_lower:
            mapping[cols_lower[cand]] = "anio"
            break
    for cand in ["sección censal", "seccion censal", "cusec", "seccion", "cod_sec"]:
        if cand in cols_lower:
            mapping[cols_lower[cand]] = "seccion"
            break
 
    fuentes = {
        "salarios": ["salarios", "rentas salariales", "renta salarial", "trabajo"],
        "pensiones": ["pensiones", "jubilación", "jubilacion", "prestaciones contributivas"],
        "prestaciones": ["prestaciones", "prestaciones no contributivas", "desempleo"],
        "rendimientos_cap": ["rendimientos del capital", "capital", "rendimientos capital"],
        "otros": ["otros", "otras fuentes", "resto"],
    }
    for canon, candidatos in fuentes.items():
        for cand in candidatos:
            if cand in cols_lower and cols_lower[cand] not in mapping.values():
                mapping[cols_lower[cand]] = canon
                break
 
    return mapping
 
 
def _detectar_columnas_actividad(df: pd.DataFrame) -> dict:
    mapping = {}
    cols_lower = {c.lower().strip(): c for c in df.columns}
 
    for cand in ["periodo", "año", "anyo", "year", "anio"]:
        if cand in cols_lower:
            mapping[cols_lower[cand]] = "anio"
            break
    for cand in ["sección censal", "seccion censal", "cusec", "seccion", "cod_sec"]:
        if cand in cols_lower:
            mapping[cols_lower[cand]] = "seccion"
            break
 
    campos = {
        "ocupados": ["ocupados", "empleo", "empleados"],
        "parados": ["parados", "desempleados", "en paro"],
        "inactivos": ["inactivos", "inactividad"],
        "activos": ["activos", "población activa", "poblacion activa"],
        "poblacion": ["población total", "poblacion total", "poblacion", "total"],
    }
    for canon, candidatos in campos.items():
        for cand in candidatos:
            if cand in cols_lower and cols_lower[cand] not in mapping.values():
                mapping[cols_lower[cand]] = canon
                break
 
    return mapping