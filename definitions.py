"""
definitions.py
==============
Punto de entrada de Dagster para el pipeline DataOps de análisis
socioeconómico por sección censal — Tenerife 2021-2023.

Arquitectura DataOps:
  ┌─────────────────────────────────────────────────────┐
  │                  DAGSTER PIPELINE                   │
  │                                                     │
  │  RAW ──► CLEAN ──► INTEGRATED ──► VISUALIZATION     │
  │                        │                            │
  │                    CLUSTERING                       │
  └─────────────────────────────────────────────────────┘

Para lanzar el servidor de Dagster:
  $ dagster dev -f definitions.py

Para materializar todos los assets:
  $ dagster asset materialize --select '*' -f definitions.py
"""

import dagster as dg
from dagster import (
    Definitions,
    define_asset_job,
    AssetSelection,
    ScheduleDefinition,
)

# ── Importar todos los assets ──────────────────────────────────────
from assets_secciones import (
    # Capa RAW
    raw_renta_media,
    raw_distribucion_ingresos,
    raw_actividad,
    raw_geojson_secciones,
    # Capa CLEAN
    clean_renta_media,
    clean_distribucion_ingresos,
    clean_actividad,
    clean_geojson,
    # Capa INTEGRATED
    dataset_integrado,
    clusters_territoriales,
    # Capa VISUALIZATION
    viz_mapa_renta_media,
    viz_mapa_ingreso_dominante,
    viz_mapa_clusters,
    viz_distribucion_clusters,
    viz_perfil_dinamico_clusters,
    viz_analisis_contribucion,
    viz_perfil,
    viz_raincloud_renta,
    viz_mapa_dinamica_tenerife,
)

# ── Importar todos los checks ──────────────────────────────────────
from test_checks import (
    # RAW
    check_raw_renta_no_vacio,
    check_raw_actividad_no_vacio,
    check_raw_ingresos_no_vacio,
    check_raw_geojson_cobertura,
    # CLEAN
    check_coherencia_temporal_renta,
    check_rango_renta,
    check_nulos_renta,
    check_formato_codigo_seccion,
    check_unicidad_clave_renta,
    check_proporciones_ingresos,
    check_rango_tasas,
    # INTEGRATED
    check_integridad_join,
    check_cobertura_temporal_integrado,
    check_consistencia_claves_integrado,
    check_categorias_ingreso_dominante,
    check_distribucion_clusters,
)

# ─────────────────────────────────────────────────────────────────
# DEFINICIÓN DE JOBS
# ─────────────────────────────────────────────────────────────────

# Job completo: todo el pipeline de extremo a extremo
job_pipeline_completo = define_asset_job(
    name="pipeline_completo_tenerife",
    selection=AssetSelection.all(),
    description=(
        "Ejecuta el pipeline completo DataOps: "
        "ingesta → limpieza → integración → clustering → visualizaciones."
    ),
)

# Job solo ingesta y limpieza (útil para validar nuevos datos)
job_ingesta_limpieza = define_asset_job(
    name="job_ingesta_limpieza",
    selection=AssetSelection.groups("raw", "clean"),
    description="Solo ingesta y limpieza. Útil para validar nuevos ficheros CSV.",
)

# Job solo visualizaciones (cuando los datos ya están integrados)
job_visualizaciones = define_asset_job(
    name="job_visualizaciones",
    selection=AssetSelection.groups("visualization"),
    description="Regenera todas las visualizaciones ggplot a partir del dataset integrado.",
)

# Job de integración + clustering
job_analisis = define_asset_job(
    name="job_analisis",
    selection=AssetSelection.groups("integrated"),
    description="Integración de datos y clustering territorial.",
)

# ─────────────────────────────────────────────────────────────────
# DEFINICIÓN DE DAGSTER
# ─────────────────────────────────────────────────────────────────

defs = Definitions(
    assets=[
        # ── Capa RAW ────────────────────────────────────────────────
        raw_renta_media,
        raw_distribucion_ingresos,
        raw_actividad,
        raw_geojson_secciones,
        # ── Capa CLEAN ──────────────────────────────────────────────
        clean_renta_media,
        clean_distribucion_ingresos,
        clean_actividad,
        clean_geojson,
        # ── Capa INTEGRATED ─────────────────────────────────────────
        dataset_integrado,
        clusters_territoriales,
        # ── Capa VISUALIZATION ──────────────────────────────────────
        viz_mapa_renta_media,
        viz_mapa_ingreso_dominante,
        viz_mapa_clusters,
        viz_distribucion_clusters,
        viz_perfil_dinamico_clusters,
        viz_analisis_contribucion,
        viz_perfil,
        viz_raincloud_renta,
        viz_mapa_dinamica_tenerife,
    ],
    asset_checks=[
        # ── Checks RAW ──────────────────────────────────────────────
        check_raw_renta_no_vacio,
        check_raw_actividad_no_vacio,
        check_raw_ingresos_no_vacio,
        check_raw_geojson_cobertura,
        # ── Checks CLEAN ────────────────────────────────────────────
        check_coherencia_temporal_renta,
        check_rango_renta,
        check_nulos_renta,
        check_formato_codigo_seccion,
        check_unicidad_clave_renta,
        check_proporciones_ingresos,
        check_rango_tasas,
        # ── Checks INTEGRATED ───────────────────────────────────────
        check_integridad_join,
        check_cobertura_temporal_integrado,
        check_consistencia_claves_integrado,
        check_categorias_ingreso_dominante,
        check_distribucion_clusters,
    ],
    jobs=[
        job_pipeline_completo,
        job_ingesta_limpieza,
        job_visualizaciones,
        job_analisis,
    ],
    schedules=[],   # Añadir CronSchedule si se desea ejecución periódica
    sensors=[],     # Añadir sensores de fichero si los CSVs llegan por FTP/S3
)