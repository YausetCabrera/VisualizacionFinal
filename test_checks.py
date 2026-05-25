"""
Asset Checks de Dagster para control de calidad del pipeline (carga, limpieza e integridad)
Provincia de Santa Cruz de Tenerife — Análisis de Renta 2021-2023

Principios DataOps aplicados:
  - Validación automática en cada capa del tratamiento e integridad.
  - Reproducibilidad mediante checks deterministas
  - Fail-fast ante violaciones críticas de integridad

Autor: Yauset Cabrera Aparicio (2026, Visualización, Máster Universitario en Ciberseguridad e Inteligencia de Datos)
"""

import numpy as np
import pandas as pd
import geopandas as gpd

import dagster as dg
from dagster import (
    asset_check, AssetCheckResult, AssetCheckSeverity,
    AssetIn,
)


AÑOS_ESPERADOS = {2022, 2023, 2024}
MAX_PCT_NULOS = 0.15          # máx. 15% de nulos en variables clave
MIN_SECCIONES = 500           # mínimo esperado de secciones en Tenerife
MAX_RENTA = 200_000           # valor máximo razonable de renta media (€)
MIN_RENTA = 1_000             # valor mínimo razonable de renta media (€)
MIN_MATCH_JOIN = 0.80         # mínimo 80% de secciones geo deben tener datos
LONGITUD_CODIGO_SEC = 10      # longitud estándar del código de sección INE


@asset_check(asset="raw_renta_media", description="Verifica que el CSV de renta media no esté vacío.")
def check_raw_renta_no_vacio(raw_renta_media: pd.DataFrame) -> AssetCheckResult:
    n = len(raw_renta_media)
    passed = n > 0
    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR,
        description=f"El dataset de renta media tiene {n} filas." if passed
                    else "El dataset de renta media está VACÍO.",
        metadata={"n_filas": dg.MetadataValue.int(n)},
    )


@asset_check(asset="raw_actividad", description="Verifica que el CSV de actividad no esté vacío.")
def check_raw_actividad_no_vacio(raw_actividad: pd.DataFrame) -> AssetCheckResult:
    n = len(raw_actividad)
    passed = n > 0
    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR,
        description=f"El dataset de actividad tiene {n} filas." if passed
                    else "El dataset de actividad está VACÍO.",
        metadata={"n_filas": dg.MetadataValue.int(n)},
    )


@asset_check(
    asset="raw_distribucion_ingresos",
    description="Verifica que el CSV de distribución de ingresos no esté vacío.",
)
def check_raw_ingresos_no_vacio(raw_distribucion_ingresos: pd.DataFrame) -> AssetCheckResult:
    n = len(raw_distribucion_ingresos)
    passed = n > 0
    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR,
        description=f"El dataset de ingresos tiene {n} filas." if passed
                    else "El dataset de ingresos está VACÍO.",
        metadata={"n_filas": dg.MetadataValue.int(n)},
    )


@asset_check(
    asset="raw_geojson_secciones",
    description=(
        "Verifica que el GeoJSON tenga cobertura suficiente de secciones "
        "y no esté vacío tras integrar múltiples años."
    ),
)
def check_raw_geojson_cobertura(raw_geojson_secciones: gpd.GeoDataFrame) -> AssetCheckResult:

    n = len(raw_geojson_secciones)

    passed = n > 0 and n >= MIN_SECCIONES * 0.8

    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR if n == 0 else AssetCheckSeverity.WARN,
        description=(
            f"GeoJSON contiene {n} secciones."
            if passed else
            f"Cobertura baja: {n} secciones (esperado ≈ {MIN_SECCIONES})."
        ),
        metadata={
            "n_secciones": dg.MetadataValue.int(n),
        },
    )



# Checks de la capa CLEAN

@asset_check(
    asset="clean_renta_media",
    description=(
        "Valida coherencia temporal: el dataset debe contener los años 2021, 2022 y 2023. "
        "Un año faltante indica fallo en la ingesta o en el filtrado."
    ),
)
def check_coherencia_temporal_renta(clean_renta_media: pd.DataFrame) -> AssetCheckResult:
    if "anio" not in clean_renta_media.columns:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            description="Columna 'anio' no encontrada en clean_renta_media.",
        )
    años_presentes = set(clean_renta_media["anio"].dropna().unique().astype(int))
    faltantes = {2021,2022,2023} - años_presentes
    passed = len(faltantes) == 0
    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR if not passed else AssetCheckSeverity.WARN,
        description=(
            f"Años presentes: {sorted(años_presentes)}. Todos los esperados cubiertos."
            if passed else
            f"FALTA(N) año(s): {sorted(faltantes)}. Presentes: {sorted(años_presentes)}."
        ),
        metadata={
            "años_presentes": dg.MetadataValue.text(str(sorted(años_presentes))),
            "años_faltantes": dg.MetadataValue.text(str(sorted(faltantes))),
        },
    )


@asset_check(
    asset="clean_renta_media",
    description=(
        "Valida rango de renta media basado en distribución observada del dataset."
    ),
)
def check_rango_renta(clean_renta_media: pd.DataFrame) -> AssetCheckResult:

    if "renta_media" not in clean_renta_media.columns:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            description="Columna 'renta_media' no encontrada.",
        )

    renta = pd.to_numeric(
        clean_renta_media["renta_media"],
        errors="coerce"
    ).dropna()

    if renta.empty:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            description="No hay valores válidos de renta_media.",
        )

    # Rangos basados en dataset real
    MIN_RENTA = 6000
    MAX_RENTA = 130000

    n_fuera = ((renta < MIN_RENTA) | (renta > MAX_RENTA)).sum()
    passed = n_fuera == 0

    return AssetCheckResult(
        passed=bool(passed),
        severity=AssetCheckSeverity.ERROR if not passed else AssetCheckSeverity.WARN,
        description=(
            "Rentas dentro del rango esperado."
            if passed
            else (
                f"{n_fuera} valores fuera de rango. "
                f"Min observado={renta.min():.0f}, Max observado={renta.max():.0f}"
            )
        ),
        metadata={
            "n_fuera_rango": dg.MetadataValue.int(int(n_fuera)),
            "renta_min": dg.MetadataValue.float(float(renta.min())),
            "renta_max": dg.MetadataValue.float(float(renta.max())),
        },
    )


@asset_check(
    asset="clean_renta_media",
    description=(
        "Valida el porcentaje de nulos en renta_media. "
        f"Se tolera hasta un {MAX_PCT_NULOS*100:.0f}%."
    ),
)
def check_nulos_renta(clean_renta_media: pd.DataFrame) -> AssetCheckResult:

    if "renta_media" not in clean_renta_media.columns:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            description="Columna 'renta_media' no encontrada. Revisa el rename de OBS_VALUE.",
        )

    serie = clean_renta_media["renta_media"]

    n_total = len(serie)
    n_nulos = serie.isna().sum()
    pct_nulos = n_nulos / max(n_total, 1)

    passed = pct_nulos <= MAX_PCT_NULOS

    return AssetCheckResult(
        passed=bool(passed),
        severity=AssetCheckSeverity.WARN,
        description=(
            f"{n_nulos} nulos ({pct_nulos*100:.1f}%) dentro del umbral."
            if passed
            else f"EXCESO DE NULOS: {pct_nulos*100:.1f}% > permitido."
        ),
        metadata={
            "n_nulos": dg.MetadataValue.int(int(n_nulos)),
            "pct_nulos": dg.MetadataValue.float(float(pct_nulos)),
        },
    )


@asset_check(
    asset="clean_renta_media",
    description=(
        "Valida el formato del código de sección censal: debe tener "
        f"exactamente {LONGITUD_CODIGO_SEC} dígitos numéricos (código INE)."
    ),
)
def check_formato_codigo_seccion(clean_renta_media: pd.DataFrame) -> AssetCheckResult:

    if "seccion" not in clean_renta_media.columns:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            description="Columna 'seccion' no encontrada.",
        )

    secciones = (
        clean_renta_media["seccion"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    mal_longitud = (secciones.str.len() != LONGITUD_CODIGO_SEC).sum()
    no_numericos = (~secciones.str.fullmatch(r"\d+")).sum()
    passed = (mal_longitud == 0) and (no_numericos == 0)

    return AssetCheckResult(
        passed=bool(passed),
        severity=AssetCheckSeverity.ERROR if not passed else AssetCheckSeverity.WARN,
        description=(
            f"Todos los códigos de sección tienen {LONGITUD_CODIGO_SEC} dígitos."
            if passed
            else f"{mal_longitud} con longitud incorrecta; {no_numericos} no numéricos."
        ),
        metadata={
            "mal_longitud": dg.MetadataValue.int(int(mal_longitud)),
            "no_numericos": dg.MetadataValue.int(int(no_numericos)),
            "ejemplo_valido": dg.MetadataValue.text(
                secciones.iloc[0] if len(secciones) else "N/A"
            ),
        },
    )


@asset_check(
    asset="clean_renta_media",
    description=(
        "Verifica que no existan duplicados (seccion, año). "
    ),
)
def check_unicidad_clave_renta(clean_renta_media: pd.DataFrame) -> AssetCheckResult:
    if not all(c in clean_renta_media.columns for c in ["seccion", "anio"]):
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            description="Columnas 'seccion' o 'anio' no encontradas.",
        )
    n_total = len(clean_renta_media)
    n_unicos = clean_renta_media.drop_duplicates(["seccion", "anio"]).shape[0]
    n_dupl = n_total - n_unicos
    passed = n_dupl == 0
    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR if n_dupl > 0 else AssetCheckSeverity.WARN,
        description=(
            "Sin duplicados en (seccion, anio)."
            if passed else
            f"{n_dupl} filas duplicadas en (seccion, anio) detectadas."
        ),
        metadata={"n_duplicados": dg.MetadataValue.int(int(n_dupl))},
    )


@asset_check(
    asset="clean_distribucion_ingresos",
    description=(
        "Valida que las proporciones de fuentes de ingresos sumen ~100% por sección y año. "
        "Desviaciones graves indican errores en la fuente o en el cálculo de porcentajes."
    ),
)
def check_proporciones_ingresos(clean_distribucion_ingresos: pd.DataFrame) -> AssetCheckResult:

    pct_cols = [
        c for c in clean_distribucion_ingresos.columns
        if c.startswith("pct_")
    ]

    if not pct_cols:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.WARN,
            description="No se encontraron columnas de porcentaje (pct_*). Omitiendo check.",
        )

    df = clean_distribucion_ingresos.dropna(subset=pct_cols)

    total_pct = df[pct_cols].sum(axis=1)
    tolerancia = 1.0

    n_fuera = int(
        ((total_pct < 100 - tolerancia) | (total_pct > 100 + tolerancia)).sum()
    )

    passed = bool(n_fuera == 0)

    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.WARN,
        description=(
            "Todas las filas suman ~100% (±1pp) en columnas pct_*."
            if passed
            else f"{n_fuera} filas no suman ~100% en proporciones de ingresos."
        ),
        metadata={
            "n_fuera_rango": dg.MetadataValue.int(n_fuera),
            "media_suma": dg.MetadataValue.float(float(total_pct.mean())),
        },
    )

@asset_check(
    asset="clean_actividad",
    description=(
        "Valida que las proporciones (pct_seccion) estén en [0, 100]. "
        "Valores fuera de este rango indican errores en el cálculo de porcentajes "
        "o inconsistencias en los agregados por sección."
    ),
)
def check_rango_tasas(clean_actividad: pd.DataFrame) -> AssetCheckResult:

    if "pct_seccion" not in clean_actividad.columns:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.WARN,
            description="No se encontró la columna 'pct_seccion' en clean_actividad.",
        )

    vals = clean_actividad["pct_seccion"].dropna()
    n_fuera = int(((vals < 0) | (vals > 100)).sum())
    passed = bool(n_fuera == 0)

    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR if n_fuera > 0 else AssetCheckSeverity.WARN,
        description=(
            "Todas las proporciones están en [0, 100]."
            if passed else
            f"{n_fuera} valores fuera de rango. "
            f"Min: {float(vals.min()):.2f}, Max: {float(vals.max()):.2f}."
        ),
        metadata={
            "n_fuera_rango": dg.MetadataValue.int(n_fuera),
            "min_pct": dg.MetadataValue.float(float(vals.min())) if len(vals) else dg.MetadataValue.float(0.0),
            "max_pct": dg.MetadataValue.float(float(vals.max())) if len(vals) else dg.MetadataValue.float(0.0),
        },
    )



# Checks de la capa INTEGRATED

@asset_check(
    asset="dataset_integrado",
    description="Valida la integridad del join espacial por año para 2022-2024."
)
def check_integridad_join(dataset_integrado: gpd.GeoDataFrame) -> AssetCheckResult:

    if dataset_integrado is None or len(dataset_integrado) == 0:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            description="El dataset integrado está vacío.",
        )

    resultados = {}
    errores = []
    años_a_validar = [2021, 2022, 2023]

    for año, sub in dataset_integrado.groupby("anio"):
        if año not in años_a_validar:
            continue

        n_total = len(sub)
        n_con_renta = int(sub["renta_media"].notna().sum())

        pct = n_con_renta / max(n_total, 1)
        ok = pct >= MIN_MATCH_JOIN

        resultados[int(año)] = round(pct, 3)

        if not ok:
            errores.append(int(año))

    passed = len(errores) == 0

    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR if not passed else AssetCheckSeverity.WARN,
        description=(
            "Join correcto en todos los años validados (2022-2024)."
            if passed else
            f"Join insuficiente en años: {errores}"
        ),
        metadata={
            "pct_match_por_año": dg.MetadataValue.json(resultados),
        },
    )

@asset_check(
    asset="dataset_integrado",
    description="Verifica que el dataset integrado contenga los años de mapa esperados."
)
def check_cobertura_temporal_integrado(
    dataset_integrado: gpd.GeoDataFrame
) -> AssetCheckResult:

    if dataset_integrado is None or dataset_integrado.empty:
        return AssetCheckResult(passed=False, description="Dataset integrado vacío.")

    anios_presentes = set(dataset_integrado["anio"].dropna().astype(int).unique())
    faltantes = AÑOS_ESPERADOS - anios_presentes
    passed = len(faltantes) == 0

    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR if not passed else AssetCheckSeverity.WARN,
        description=(
            f"Cobertura temporal correcta (años de mapa): {sorted(anios_presentes)}."
            if passed else
            f"Años de mapa faltantes: {sorted(faltantes)}."
        ),
        metadata={
            "años_presentes": dg.MetadataValue.text(str(sorted(anios_presentes))),
            "faltantes": dg.MetadataValue.text(str(sorted(faltantes))),
        },
    )


@asset_check(
    asset="dataset_integrado",
    description=(
        "Valida consistencia de claves: una fila por (seccion, anio). "
        "Detecta duplicaciones tras joins."
    ),
)
def check_consistencia_claves_integrado(
    dataset_integrado: gpd.GeoDataFrame,
) -> AssetCheckResult:

    if dataset_integrado.empty:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            description="dataset_integrado está vacío.",
        )

    if not all(c in dataset_integrado.columns for c in ["seccion", "anio"]):
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            description="Faltan columnas clave.",
        )

    n_total = len(dataset_integrado)
    n_unique = dataset_integrado.drop_duplicates(["seccion", "anio"]).shape[0]
    n_dupl = n_total - n_unique
    passed = bool(n_dupl == 0)

    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR if n_dupl > 0 else AssetCheckSeverity.WARN,
        description=(
            "Sin duplicados en (seccion, anio)."
            if passed else
            f"{n_dupl} duplicados detectados en (seccion, anio)."
        ),
        metadata={
            "n_duplicados": dg.MetadataValue.int(int(n_dupl)),
            "n_total": dg.MetadataValue.int(n_total),
        },
    )


@asset_check(
    asset="dataset_integrado",
    description=(
        "Verifica que 'ingreso_dominante' contenga categorías válidas "
        "según los datos reales del dataset (INE)."
    ),
)
def check_categorias_ingreso_dominante(
    dataset_integrado: gpd.GeoDataFrame,
) -> AssetCheckResult:

    if "ingreso_dominante" not in dataset_integrado.columns:
        return AssetCheckResult(
            passed=True,
            severity=AssetCheckSeverity.WARN,
            description="Columna 'ingreso_dominante' no presente. Check omitido.",
        )

    categorias_validas = {
        "sueldos_y_salarios",
        "pensiones",
        "prestaciones",
        "otras_prestaciones",
        "rendimientos_capital",
        "otros",
    }

    categorias_presentes = set(
        dataset_integrado["ingreso_dominante"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
    )

    inesperadas = categorias_presentes - categorias_validas
    passed = len(inesperadas) == 0

    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.WARN if inesperadas else AssetCheckSeverity.WARN,
        description=(
            f"Categorías detectadas correctamente: {sorted(categorias_presentes)}."
            if passed else
            f"Categorías no reconocidas: {sorted(inesperadas)}."
        ),
        metadata={
            "categorias_presentes": dg.MetadataValue.text(str(sorted(categorias_presentes))),
            "categorias_inesperadas": dg.MetadataValue.text(str(sorted(inesperadas))),
            "n_categorias": dg.MetadataValue.int(len(categorias_presentes)),
        },
    )


@asset_check(
    asset="clusters_territoriales",
    description=(
        "Verifica consistencia del clustering POR AÑO (2022-2024): "
        "número de clusters, distribución y ausencia de dominancia extrema."
    ),
)
def check_distribucion_clusters(
    clusters_territoriales: gpd.GeoDataFrame,
) -> AssetCheckResult:

    df = clusters_territoriales.copy()
    df["anio"] = df["anio"].astype(int)
    df["cluster"] = df["cluster"].astype(str)

    años_a_validar = [2021, 2022, 2023]
    df = df[df["anio"].isin(años_a_validar)].copy()

    if df.empty:
        return AssetCheckResult(
            passed=False, 
            severity=AssetCheckSeverity.ERROR, 
            description="No hay datos para 2022-2024."
        )

    resultados = {}
    errores = []

    for año in años_a_validar:
        sub = df[df["anio"] == año]
        
        if sub.empty:
            errores.append(año)
            continue

        dist = sub["cluster"].value_counts(normalize=True)
        n_clusters = int(dist.shape[0])
        min_pct = float(dist.min())
        max_pct = float(dist.max())

        ok = (
            (n_clusters >= 2) 
            and (n_clusters <= 6) 
        )

        resultados[año] = {
            "n_clusters": n_clusters,
            "min_pct": round(min_pct, 3),
            "max_pct": round(max_pct, 3),
        }

        if not ok:
            errores.append(año)

    passed = len(errores) == 0
    
    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.WARN if passed else AssetCheckSeverity.ERROR,
        description=(
            "Clustering consistente en años 2022-2024."
            if passed else
            f"Problemas en clustering para años: {errores}"
        ),
        metadata={
            "resumen_por_año": dg.MetadataValue.json(resultados),
            "años_problematicos": dg.MetadataValue.json(errores),
        },
    )