"""
evaluar_test.py
---------------
Evaluación final del agente SAC sobre el split de test.

Flujo:
  1. Carga config_ganadora_hpo.json (ganadores de Fase B del HPO)
  2. Concatena train + validation para formar el split de entrenamiento final
  3. Calcula covarianzas del split concatenado sin leakage
  4. Para cada perfil: sobreescribe target_entropy según perfil, entrena con
     train+val y evalúa en test
  5. Guarda métricas (JSON), curvas (CSV) y gráfica (PNG) en resultados/

Uso:
  python evaluar_test.py
  python evaluar_test.py --config otra_config.json
  python evaluar_test.py --pasos 40000 --seed 42
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ------------------------------------------------------------------
# Target entropy por perfil
# ------------------------------------------------------------------
# El HPO usó -26 para todos los perfiles (riesgo=1.0 en Fase A).
# Con perfiles más arriesgados necesitamos más exploración (menos negativo)
# para evitar que el agente colapse a bonos/cash.
# Razonamiento: con 16 activos invertibles, entropía máxima ≈ log(16) ≈ 2.77.
# -26 es extremadamente determinista. Relajamos progresivamente con el riesgo.
TARGET_ENTROPY_POR_PERFIL = {
    "muy_conservador": -26,   # máximo determinismo — quiere baja vol y pocos activos
    "conservador":     -24,
    "normal":          -22,
    "arriesgado":      -20,
    "muy_arriesgado":  -18,   # más exploración — necesita diversificar en RV
}

# ------------------------------------------------------------------
# Rutas
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTADOS_DIR = PROJECT_ROOT / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Entrenamiento_SAC import ConfigEntrenamiento, entrenar_sac
from entorno_cartera import EntornoCartera


# ------------------------------------------------------------------
# Utilidades generales
# ------------------------------------------------------------------
def fijar_semillas(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encontrar_carpeta_datos() -> Path:
    for candidato in ["Datos", "datos", "datos_procesados"]:
        p = PROJECT_ROOT / candidato
        if p.exists():
            return p
    raise FileNotFoundError("No se encuentra la carpeta de datos. Revisa PROJECT_ROOT.")


def cargar_split(carpeta_base: Path, nombre: str):
    """Carga datos_estado, retornos y rf_semanal de un split."""
    carpeta = carpeta_base / nombre.capitalize()
    datos_estado = pd.read_csv(
        carpeta / f"datos_estado_{nombre}.csv", index_col=0, parse_dates=True
    )
    retornos = pd.read_csv(
        carpeta / f"retornos_{nombre}.csv", index_col=0, parse_dates=True
    )
    rf_semanal = pd.read_csv(
        carpeta / f"rf_semanal_{nombre}.csv", index_col=0, parse_dates=True
    ).squeeze("columns")
    return datos_estado, retornos, rf_semanal


def cargar_covarianzas(carpeta_base: Path, nombre: str) -> pd.DataFrame:
    carpeta = carpeta_base / nombre.capitalize()
    return pd.read_csv(carpeta / f"covarianzas_{nombre}.csv", index_col=0)


# ------------------------------------------------------------------
# Concatenar train + validation
# ------------------------------------------------------------------
def concatenar_train_val(carpeta_datos: Path):
    """
    Une train y validation en orden cronológico.
    Las covarianzas se recalculan sobre la ventana completa sin leakage:
    se usa la covarianza histórica hasta cada fecha (expanding window).
    """
    datos_estado_train, retornos_train, rf_train = cargar_split(carpeta_datos, "train")
    datos_estado_val,   retornos_val,   rf_val   = cargar_split(carpeta_datos, "validation")

    # Concatenar en orden cronológico
    datos_estado = pd.concat([datos_estado_train, datos_estado_val]).sort_index()
    retornos     = pd.concat([retornos_train,     retornos_val    ]).sort_index()
    rf_semanal   = pd.concat([rf_train,           rf_val          ]).sort_index()

    # Eliminar duplicados de índice si los hubiera (por solapamiento de fechas)
    datos_estado = datos_estado[~datos_estado.index.duplicated(keep="last")]
    retornos     = retornos    [~retornos.index.duplicated(keep="last")]
    rf_semanal   = rf_semanal  [~rf_semanal.index.duplicated(keep="last")]

    # Covarianzas recalculadas: media de covarianzas train y val
    # (forma simple y sin leakage al no usar datos de test)
    cov_train = cargar_covarianzas(carpeta_datos, "train")
    cov_val   = cargar_covarianzas(carpeta_datos, "validation")
    # Promedio ponderado por número de semanas de cada split
    n_train = len(retornos_train)
    n_val   = len(retornos_val)
    cov_trainval = (cov_train * n_train + cov_val * n_val) / (n_train + n_val)

    print(f"  Train:      {len(retornos_train):>5} semanas")
    print(f"  Validation: {len(retornos_val):>5} semanas")
    print(f"  Total:      {len(retornos):>5} semanas")

    return datos_estado, retornos, rf_semanal, cov_trainval


# ------------------------------------------------------------------
# Construcción de entorno y config
# ------------------------------------------------------------------
def construir_entorno(
    datos_estado: pd.DataFrame,
    retornos: pd.DataFrame,
    rf_semanal: pd.Series,
    riesgo: float,
    covarianzas_iniciales: pd.DataFrame | None = None,
    **lambda_kwargs,
) -> EntornoCartera:
    kwargs = dict(
        datos_estado=datos_estado,
        retornos_semanales=retornos,
        rf_semanal=rf_semanal,
        coste_transaccion=0.001,
        valor_inicial=1000.0,
        covarianzas_iniciales=covarianzas_iniciales,
    )
    firma = inspect.signature(EntornoCartera)
    for nombre in ["riesgo", "perfil_riesgo", "riesgo_usuario", "score_riesgo", "nivel_riesgo"]:
        if nombre in firma.parameters:
            kwargs[nombre] = float(riesgo)
            break
    for k, v in lambda_kwargs.items():
        if k in firma.parameters:
            kwargs[k] = v
    return EntornoCartera(**kwargs)


def build_train_config(config: dict) -> ConfigEntrenamiento:
    return ConfigEntrenamiento(
        semilla=int(config["semilla"]),
        pasos_totales=int(config["pasos_totales"]),
        gamma=float(config["gamma"]),
        tau=float(config["tau"]),
        lr_actor=float(config["lr_actor"]),
        lr_criticos=float(config["lr_criticos"]),
        lr_alpha=float(config["lr_alpha"]),
        tamano_batch=int(config["tamano_batch"]),
        tamano_buffer=int(config["tamano_buffer"]),
        pasos_warmup=int(config["pasos_warmup"]),
        frecuencia_actualizacion=int(config["frecuencia_actualizacion"]),
        actualizaciones_por_step=int(config["actualizaciones_por_step"]),
        reward_scale=float(config["reward_scale"]),
        target_entropy=(
            None if config.get("target_entropy") is None
            else float(config["target_entropy"])
        ),
        offset_target_entropy=float(config["offset_target_entropy"]),
        max_concentracion_total_extra=float(config["max_concentracion_total_extra"]),
        ventana_log_recompensa=int(config["ventana_log_recompensa"]),
        frecuencia_log=int(config["frecuencia_log"]),
    )


def politica_determinista(agente, device):
    def fn(estado_np):
        estado_t = torch.as_tensor(
            estado_np, dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            accion = agente.seleccionar_accion(estado_t, determinista=True)
        return accion.squeeze(0).detach().cpu().numpy().astype(np.float64)
    return fn


# ------------------------------------------------------------------
# Métricas financieras
# ------------------------------------------------------------------
def calcular_metricas_financieras(
    valor_cartera: pd.Series,
    rf_semanal: pd.Series,
    pesos: pd.DataFrame = None,
) -> dict:
    valor_cartera = valor_cartera.astype(float).dropna()
    rend = valor_cartera.pct_change().dropna()

    nan_dict = {k: np.nan for k in [
        "valor_final", "retorno_total", "cagr", "volatilidad_anual",
        "sharpe", "sortino", "max_drawdown", "calmar",
        "turnover", "peso_cash_medio", "exposicion_media", "n_activos_medio",
    ]}

    if valor_cartera.empty or len(valor_cartera) < 2 or rend.empty:
        return nan_dict

    rf_alineado = rf_semanal.reindex(rend.index).ffill().fillna(0.0)
    exceso = rend - rf_alineado

    valor_inicial = float(valor_cartera.iloc[0])
    valor_final   = float(valor_cartera.iloc[-1])
    retorno_total = valor_final / valor_inicial - 1.0
    n      = len(rend)
    cagr   = (valor_final / valor_inicial) ** (52 / n) - 1.0
    vol    = rend.std() * np.sqrt(52)
    sharpe = exceso.mean() / exceso.std() * np.sqrt(52) if exceso.std() > 0 else np.nan

    downside = rend[rend < 0]
    sortino  = rend.mean() / downside.std() * np.sqrt(52) if len(downside) > 0 else np.nan

    acumulado = (1 + rend).cumprod()
    max_dd    = (acumulado / acumulado.cummax() - 1.0).min()
    calmar    = cagr / abs(max_dd) if max_dd != 0 else np.nan

    turnover = peso_cash_medio = exposicion_media = n_activos_medio = np.nan
    if pesos is not None:
        turnover = pesos.diff().abs().sum(axis=1).mean()
        if "cash" in pesos.columns:
            peso_cash_medio  = pesos["cash"].mean()
            exposicion_media = 1 - peso_cash_medio
        n_activos_medio = (pesos > 1e-3).sum(axis=1).mean()

    return {
        "valor_final": valor_final, "retorno_total": retorno_total,
        "cagr": cagr, "volatilidad_anual": vol,
        "sharpe": sharpe, "sortino": sortino,
        "max_drawdown": max_dd, "calmar": calmar,
        "turnover": turnover, "peso_cash_medio": peso_cash_medio,
        "exposicion_media": exposicion_media, "n_activos_medio": n_activos_medio,
    }


# ------------------------------------------------------------------
# Entrenamiento + evaluación de un perfil
# ------------------------------------------------------------------
PARAMS_LAMBDA = [
    "lambda_dd_min", "lambda_dd_max",
    "lambda_varianza_min", "lambda_varianza_max",
    "lambda_correlacion_min", "lambda_correlacion_max",
    "correlacion_objetivo_min", "correlacion_objetivo_max",
]


def entrenar_y_evaluar(
    config: dict,
    datos_estado_trainval: pd.DataFrame,
    retornos_trainval: pd.DataFrame,
    rf_trainval: pd.Series,
    cov_trainval: pd.DataFrame,
    datos_estado_test: pd.DataFrame,
    retornos_test: pd.DataFrame,
    rf_test: pd.Series,
    cov_test: pd.DataFrame,
    cartera_inicial: np.ndarray,
) -> tuple[pd.Series, dict]:
    fijar_semillas(int(config["semilla"]))

    lambda_kwargs = {k: config[k] for k in PARAMS_LAMBDA if k in config}

    # Entrenar sobre train+val
    entorno_train = construir_entorno(
        datos_estado=datos_estado_trainval,
        retornos=retornos_trainval,
        rf_semanal=rf_trainval,
        riesgo=config["riesgo"],
        covarianzas_iniciales=cov_trainval,
        **lambda_kwargs,
    )
    cfg_entrenamiento = build_train_config(config)
    _, agente = entrenar_sac(
        entorno=entorno_train,
        config=cfg_entrenamiento,
        devolver_agente=True,
        riesgo=config["riesgo"],
    )

    # Evaluar en test
    entorno_test = construir_entorno(
        datos_estado=datos_estado_test,
        retornos=retornos_test,
        rf_semanal=rf_test,
        riesgo=config["riesgo"],
        covarianzas_iniciales=cov_test,
        **lambda_kwargs,
    )
    device   = next(agente.actor.parameters()).device
    fn_pesos = politica_determinista(agente, device)

    backtest_df = entorno_test.ejecutar_backtest(
        funcion_pesos=fn_pesos,
        pesos_iniciales=cartera_inicial,
    )

    col_valor = next(
        (c for c in ["valor_cartera", "valor", "portfolio_value"] if c in backtest_df.columns),
        backtest_df.columns[0],
    )
    serie_valor = backtest_df[col_valor]

    pesos_cols = [c for c in backtest_df.columns if c != col_valor]
    pesos_df   = backtest_df[pesos_cols] if pesos_cols else None

    metricas = calcular_metricas_financieras(serie_valor, rf_test, pesos_df)
    return serie_valor, metricas


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(args):
    print("=" * 70)
    print("EVALUACIÓN FINAL — TEST")
    print("=" * 70)

    # Cargar configuraciones ganadoras del HPO
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"No se encuentra {config_path}")
    with open(config_path) as f:
        configs_hpo = json.load(f)
    print(f"\nConfiguración cargada: {config_path}")
    print(f"Perfiles: {list(configs_hpo.keys())}\n")

    # Cargar datos
    carpeta_datos = encontrar_carpeta_datos()
    print("Cargando datos...")

    print("\n[Train + Validation]")
    datos_estado_tv, retornos_tv, rf_tv, cov_tv = concatenar_train_val(carpeta_datos)

    print("\n[Test]")
    datos_estado_test, retornos_test, rf_test = cargar_split(carpeta_datos, "test")
    cov_test = cargar_covarianzas(carpeta_datos, "test")
    print(f"  Test: {len(retornos_test):>5} semanas")

    # Cartera inicial equiponderada
    n_activos = retornos_tv.shape[1] + 1
    cartera_inicial = np.ones(n_activos) / n_activos

    # Colores por perfil para la gráfica
    colores = {
        "muy_conservador": "steelblue",
        "conservador":     "green",
        "normal":          "orange",
        "arriesgado":      "red",
        "muy_arriesgado":  "purple",
    }

    # ------------------------------------------------------------------
    # Entrenar y evaluar cada perfil
    # ------------------------------------------------------------------
    resultados   = {}   # perfil -> metricas
    curvas       = {}   # perfil -> serie_valor

    for perfil, cfg_base in configs_hpo.items():
        print("\n" + "=" * 70)
        print(f"PERFIL: {perfil}  |  riesgo={cfg_base['riesgo']}")
        print("=" * 70)

        cfg = copy.deepcopy(cfg_base)

        # Sobreescribir target_entropy según perfil
        # (el JSON tiene -26 para todos, heredado del HPO con riesgo=1.0)
        te_perfil = TARGET_ENTROPY_POR_PERFIL.get(perfil, cfg.get("target_entropy", -26))
        cfg["target_entropy"] = te_perfil
        print(f"  target_entropy: {te_perfil}  (JSON tenía: {cfg_base.get('target_entropy', -26)})")

        # Override de pasos y semilla si se pasan por argumento
        if args.pasos is not None:
            cfg["pasos_totales"] = args.pasos
            print(f"  pasos_totales sobreescrito: {args.pasos}")
        if args.seed is not None:
            cfg["semilla"] = args.seed

        serie, metricas = entrenar_y_evaluar(
            config=cfg,
            datos_estado_trainval=datos_estado_tv,
            retornos_trainval=retornos_tv,
            rf_trainval=rf_tv,
            cov_trainval=cov_tv,
            datos_estado_test=datos_estado_test,
            retornos_test=retornos_test,
            rf_test=rf_test,
            cov_test=cov_test,
            cartera_inicial=cartera_inicial,
        )

        resultados[perfil] = metricas
        curvas[perfil]     = serie

        print(f"  CAGR:       {metricas['cagr']:>8.2%}")
        print(f"  Sharpe:     {metricas['sharpe']:>8.3f}")
        print(f"  MDD:        {metricas['max_drawdown']:>8.2%}")
        print(f"  Volatilidad:{metricas['volatilidad_anual']:>8.2%}")
        if not np.isnan(metricas.get('peso_cash_medio', np.nan)):
            print(f"  Cash medio: {metricas['peso_cash_medio']:>8.2%}")

    # ------------------------------------------------------------------
    # Tabla resumen
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESUMEN FINAL — TEST")
    print(f"{'Perfil':<20} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'Vol':>8}")
    print("-" * 60)
    for perfil in configs_hpo:
        m = resultados[perfil]
        print(
            f"{perfil:<20} "
            f"{m['cagr']:>8.2%} "
            f"{m['sharpe']:>8.3f} "
            f"{m['max_drawdown']:>8.2%} "
            f"{m['volatilidad_anual']:>8.2%}"
        )

    # Verificar jerarquía
    perfiles_ordenados = sorted(configs_hpo.keys(), key=lambda p: configs_hpo[p]["riesgo"])
    cagrs = [resultados[p]["cagr"] for p in perfiles_ordenados]
    jerarquia_ok = all(cagrs[i] <= cagrs[i + 1] for i in range(len(cagrs) - 1))
    print(f"\n{'✅' if jerarquia_ok else '⚠️ '} Jerarquía CAGR correcta: {jerarquia_ok}")

    # ------------------------------------------------------------------
    # Guardar métricas en JSON
    # ------------------------------------------------------------------
    metricas_path = RESULTADOS_DIR / "metricas_test.json"
    metricas_serializables = {
        perfil: {
            k: (float(v) if isinstance(v, (float, np.floating)) and not np.isnan(v)
                else None if isinstance(v, float) and np.isnan(v)
                else v)
            for k, v in m.items()
        }
        for perfil, m in resultados.items()
    }
    with open(metricas_path, "w") as f:
        json.dump(metricas_serializables, f, indent=2)
    print(f"\n✅ Métricas guardadas en: {metricas_path}")

    # ------------------------------------------------------------------
    # Guardar curvas en CSV
    # ------------------------------------------------------------------
    curvas_df = pd.DataFrame(curvas)
    curvas_path = RESULTADOS_DIR / "curvas_test.csv"
    curvas_df.to_csv(curvas_path)
    print(f"✅ Curvas guardadas en:   {curvas_path}")

    # ------------------------------------------------------------------
    # Gráfica única con los 5 perfiles
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))

    for perfil in perfiles_ordenados:
        serie  = curvas[perfil]
        m      = resultados[perfil]
        color  = colores.get(perfil, None)
        ax.plot(
            serie.index, serie.values,
            lw=2,
            color=color,
            label=f"{perfil}  |  Sharpe={m['sharpe']:.2f}  CAGR={m['cagr']:.1%}  MDD={m['max_drawdown']:.1%}",
        )

    ax.set_title("Evaluación final — Test (entrenado con Train+Validation)", fontsize=13)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor cartera")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    grafica_path = RESULTADOS_DIR / "graficas_test.png"
    plt.savefig(grafica_path, dpi=150)
    plt.show()
    print(f"✅ Gráfica guardada en:   {grafica_path}")
    print("\nEvaluación completada.")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación final del agente SAC en test.")
    parser.add_argument(
        "--config",
        default="config_ganadora_hpo.json",
        help="Ruta al JSON con las configuraciones ganadoras del HPO (default: config_ganadora_hpo.json)",
    )
    parser.add_argument(
        "--pasos",
        type=int,
        default=None,
        help="Sobreescribe pasos_totales de todas las configs (opcional)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sobreescribe la semilla de todas las configs (opcional)",
    )
    args = parser.parse_args()
    main(args)