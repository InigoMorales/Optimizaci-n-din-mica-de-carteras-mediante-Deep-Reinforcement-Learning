"""
Test_C_reentrenar_en_train_val_y_evaluar_test.py
─────────────────────────────────────────────────
Usa los hiperparámetros ganadores del HPO (config_ganadora_hpo.json) para
REENTRENAR cada perfil sobre TRAIN + VALIDATION, y luego evalúa el agente
resultante en TEST.

Este es el flujo "producción": se aprovecha todo el dato histórico disponible
antes del test para reentrenar, y el test queda como evaluación out-of-sample
definitiva.

Flujo:
  1. Lee config_ganadora_hpo.json
  2. Concatena train + validation como datos de entrenamiento
  3. Reentrena cada perfil con pasos_totales aumentados (configurable)
  4. Guarda los agentes reentrenados en agentes_finales/
  5. Evalúa en test y guarda en resultados/C_reentrenado_test/
       - metricas_C_reentrenado_test.json
       - curvas_C_reentrenado_test.csv
       - grafica_C_reentrenado_test.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ── Rutas ──────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent
AGENTES_DIR    = PROJECT_ROOT / "agentes_finales"    # agentes reentrenados aquí
RESULTADOS_DIR = PROJECT_ROOT / "resultados" / "C_reentrenado_test"
AGENTES_DIR.mkdir(parents=True, exist_ok=True)
RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Agente_SAC import AgenteSAC
from Entrenamiento_SAC import ConfigEntrenamiento, entrenar_sac
from entorno_cartera import EntornoCartera

# ── Constantes ─────────────────────────────────────────────────────────────
ORDEN_PERFILES = ["muy_conservador", "conservador", "normal", "arriesgado", "muy_arriesgado"]
COLORES = {
    "muy_conservador": "steelblue",
    "conservador":     "mediumseagreen",
    "normal":          "orange",
    "arriesgado":      "tomato",
    "muy_arriesgado":  "mediumpurple",
}

# Multiplicador sobre los pasos_totales del HPO.
# 1.0 → mismos pasos que en HPO (train solo).
# 1.3 → 30 % más pasos, razonable dado que el conjunto es ~30 % mayor.
FACTOR_PASOS_EXTRA = 3.0

# Semilla base para el reentrenamiento (distinta del HPO para independencia)
SEMILLA_REENTRENAMIENTO = 98


# ── Helpers ────────────────────────────────────────────────────────────────

def encontrar_carpeta_datos() -> Path:
    for nombre in ["Datos", "datos", "datos_procesados"]:
        p = PROJECT_ROOT / nombre
        if p.exists():
            return p
    raise FileNotFoundError("No se encuentra la carpeta de datos.")


def cargar_split(carpeta_base: Path, nombre: str):
    carpeta = carpeta_base / nombre.capitalize()
    de  = pd.read_csv(carpeta / f"datos_estado_{nombre}.csv",  index_col=0, parse_dates=True)
    ret = pd.read_csv(carpeta / f"retornos_{nombre}.csv",      index_col=0, parse_dates=True)
    rf  = pd.read_csv(carpeta / f"rf_semanal_{nombre}.csv",    index_col=0, parse_dates=True
                      ).squeeze("columns")
    return de, ret, rf


def cargar_cov(carpeta_base: Path, nombre: str) -> pd.DataFrame:
    return pd.read_csv(
        carpeta_base / nombre.capitalize() / f"covarianzas_{nombre}.csv", index_col=0
    )


def politica_det(agente: AgenteSAC, device: torch.device):
    def fn(estado_np: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(estado_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a = agente.seleccionar_accion(t, determinista=True)
        return a.squeeze(0).detach().cpu().numpy().astype(np.float64)
    return fn


def calcular_metricas(valor: pd.Series, rf: pd.Series) -> dict:
    valor = valor.astype(float).dropna()
    rend  = valor.pct_change().dropna()
    nan_d = {k: np.nan for k in [
        "valor_final", "retorno_total", "cagr", "volatilidad_anual",
        "sharpe", "sortino", "max_drawdown", "calmar"
    ]}
    if valor.empty or len(valor) < 2:
        return nan_d
    rf_al  = rf.reindex(rend.index).ffill().fillna(0.0)
    exceso = rend - rf_al
    v0, vf = float(valor.iloc[0]), float(valor.iloc[-1])
    n      = len(rend)
    cagr   = (vf / v0) ** (52 / n) - 1.0
    vol    = rend.std() * np.sqrt(52)
    sharpe = exceso.mean() / exceso.std() * np.sqrt(52) if exceso.std() > 0 else np.nan
    ds     = rend[rend < 0]
    sortino = rend.mean() / ds.std() * np.sqrt(52) if len(ds) > 0 else np.nan
    acum   = (1 + rend).cumprod()
    mdd    = (acum / acum.cummax() - 1.0).min()
    calmar = cagr / abs(mdd) if mdd != 0 else np.nan
    return {
        "valor_final": vf, "retorno_total": vf / v0 - 1,
        "cagr": cagr, "volatilidad_anual": vol,
        "sharpe": sharpe, "sortino": sortino,
        "max_drawdown": mdd, "calmar": calmar,
    }


def construir_config_entrenamiento(cfg: dict, pasos_totales: int) -> ConfigEntrenamiento:
    """Construye ConfigEntrenamiento desde el dict del JSON del HPO."""
    return ConfigEntrenamiento(
        semilla=SEMILLA_REENTRENAMIENTO,
        pasos_totales=pasos_totales,
        tamano_buffer=int(cfg.get("tamano_buffer", 50_000)),
        tamano_batch=int(cfg.get("tamano_batch", 256)),
        pasos_warmup=int(cfg.get("pasos_warmup", 1_000)),
        gamma=float(cfg.get("gamma", 0.99)),
        tau=float(cfg.get("tau", 0.02)),
        lr_actor=float(cfg.get("lr_actor", 1e-4)),
        lr_criticos=float(cfg.get("lr_criticos", 3e-4)),
        lr_alpha=float(cfg.get("lr_alpha", 1e-4)),
        target_entropy=(
            float(cfg["target_entropy"]) if cfg.get("target_entropy") else None
        ),
        reward_scale=float(cfg.get("reward_scale", 20.0)),
        offset_target_entropy=float(cfg.get("offset_target_entropy", 0.0)),
        max_concentracion_total_extra=float(cfg.get("max_concentracion_total_extra", 5.0)),
        frecuencia_actualizacion=int(cfg.get("frecuencia_actualizacion", 2)),
        actualizaciones_por_step=int(cfg.get("actualizaciones_por_step", 2)),
        ventana_log_recompensa=int(cfg.get("ventana_log_recompensa", 500)),
        frecuencia_log=int(cfg.get("frecuencia_log", 5_000)),
        frecuencia_snapshot_cartera=int(cfg.get("frecuencia_snapshot_cartera", 5_000)),
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("TEST C — Reentrenamiento en TRAIN+VALIDATION → Evaluación en TEST")
    print(f"Factor de pasos extra: {FACTOR_PASOS_EXTRA}×")
    print("=" * 70)

    config_path = PROJECT_ROOT / "config_ganadora_hpo.json"
    with open(config_path) as f:
        configs_hpo = json.load(f)

    carpeta_datos = encontrar_carpeta_datos()

    # ── Cargar splits ──────────────────────────────────────────────────────
    de_train, ret_train, rf_train = cargar_split(carpeta_datos, "train")
    de_val,   ret_val,   rf_val   = cargar_split(carpeta_datos, "validation")
    de_test,  ret_test,  rf_test  = cargar_split(carpeta_datos, "test")

    # Covarianzas precalculadas sobre train (sin leakage)
    cov_train = cargar_cov(carpeta_datos, "train")

    # Concatenar train + validation para el reentrenamiento
    de_trainval  = pd.concat([de_train,  de_val],  axis=0).sort_index()
    ret_trainval = pd.concat([ret_train, ret_val],  axis=0).sort_index()
    rf_trainval  = pd.concat([rf_train,  rf_val],   axis=0).sort_index()
    rf_trainval  = rf_trainval[~rf_trainval.index.duplicated(keep="first")]

    print(f"\nTrain:      {len(ret_train)} semanas")
    print(f"Validation: {len(ret_val)} semanas")
    print(f"Train+Val:  {len(ret_trainval)} semanas "
          f"({ret_trainval.index[0].date()} → {ret_trainval.index[-1].date()})")
    print(f"Test:       {len(ret_test)} semanas "
          f"({ret_test.index[0].date()} → {ret_test.index[-1].date()})")

    n_activos       = ret_test.shape[1] + 1
    cartera_inicial = np.zeros(n_activos)
    cartera_inicial[-1] = 1.0  # 100 % cash al arrancar

    resultados: dict = {}
    curvas: dict     = {}

    for perfil in ORDEN_PERFILES:
        if perfil not in configs_hpo:
            print(f"\n⚠️  Perfil '{perfil}' no encontrado en config_ganadora_hpo.json, se omite.")
            continue

        print(f"\n{'═'*70}")
        print(f"PERFIL: {perfil}")
        print(f"{'═'*70}")
        cfg = configs_hpo[perfil]
        riesgo = float(cfg["riesgo"])

        # Calcular pasos totales con factor de escala
        pasos_base  = int(cfg.get("pasos_totales", 100_000))
        pasos_final = int(pasos_base * FACTOR_PASOS_EXTRA)
        print(f"  Pasos base HPO: {pasos_base:,}  →  Reentrenamiento: {pasos_final:,}")

        # ── Entorno de entrenamiento (train + validation) ──────────────────
        entorno_trainval = EntornoCartera(
            datos_estado=de_trainval,
            retornos_semanales=ret_trainval,
            rf_semanal=rf_trainval,
            coste_transaccion=0.001,
            valor_inicial=1_000.0,
            covarianzas_iniciales=cov_train,
            riesgo=riesgo,
            lambda_dd=float(cfg.get("lambda_dd", 0.0)),
            lambda_varianza=float(cfg.get("lambda_varianza", 0.20)),
            lambda_correlacion=float(cfg.get("lambda_correlacion", 0.10)),
        )

        config_train = construir_config_entrenamiento(cfg, pasos_final)

        print(f"  Entrenando {perfil} sobre train+val ({len(ret_trainval)} semanas)…")
        t0 = time.time()

        _, agente = entrenar_sac(
            entorno=entorno_trainval,
            config=config_train,
            riesgo=riesgo,
            devolver_agente=True,
        )

        elapsed = time.time() - t0
        print(f"  ⏱  Entrenamiento completado en {elapsed/60:.1f} min")

        # ── Guardar agente reentrenado ─────────────────────────────────────
        device     = next(agente.actor.parameters()).device
        dim_estado = agente.actor.obs_dim
        dim_accion = agente.actor.act_dim

        ruta_agente = AGENTES_DIR / f"agente_final_{perfil}.pt"
        torch.save({
            "actor_state_dict":   agente.actor.state_dict(),
            "critic1_state_dict": agente.critic1.state_dict(),
            "critic2_state_dict": agente.critic2.state_dict(),
            "log_alpha":          agente.log_alpha.detach().cpu(),
            "perfil":             perfil,
            "config":             cfg,
            "dim_estado":         dim_estado,
            "dim_accion":         dim_accion,
            "pasos_reentrenamiento": pasos_final,
            "semilla_reentrenamiento": SEMILLA_REENTRENAMIENTO,
        }, ruta_agente)
        print(f"  💾 Agente guardado: {ruta_agente.name}")

        # ── Evaluación en test ─────────────────────────────────────────────
        entorno_test = EntornoCartera(
            datos_estado=de_test,
            retornos_semanales=ret_test,
            rf_semanal=rf_test,
            coste_transaccion=0.001,
            valor_inicial=1_000.0,
            covarianzas_iniciales=cov_train,
            riesgo=riesgo,
            lambda_dd=float(cfg.get("lambda_dd", 0.0)),
            lambda_varianza=float(cfg.get("lambda_varianza", 0.20)),
            lambda_correlacion=float(cfg.get("lambda_correlacion", 0.10)),
        )

        bt    = entorno_test.ejecutar_backtest(
            funcion_pesos=politica_det(agente, device),
            pesos_iniciales=cartera_inicial,
        )
        serie = bt["valor_cartera"] if "valor_cartera" in bt.columns else bt.iloc[:, 0]
        met   = calcular_metricas(serie, rf_test)

        resultados[perfil] = met
        curvas[perfil]     = serie

        print(f"  [Test]  Sharpe={met['sharpe']:.3f}  "
              f"CAGR={met['cagr']:.1%}  "
              f"MDD={met['max_drawdown']:.1%}  "
              f"Vol={met['volatilidad_anual']:.1%}")

        # Actualizar checkpoint con métricas de test
        ck = torch.load(ruta_agente, map_location=device)
        ck["test_sharpe"] = met["sharpe"]
        ck["test_cagr"]   = met["cagr"]
        torch.save(ck, ruta_agente)

    # ── Resumen ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESUMEN FINAL — TEST C (reentrenado en train+val, evaluado en test)")
    print(f"{'Perfil':<20} {'Sharpe':>8} {'CAGR':>8} {'MDD':>10} {'Vol':>8}")
    print("-" * 60)
    for p in ORDEN_PERFILES:
        if p not in resultados:
            continue
        m = resultados[p]
        print(f"{p:<20} {m['sharpe']:>8.3f} {m['cagr']:>8.1%} "
              f"{m['max_drawdown']:>10.1%} {m['volatilidad_anual']:>8.1%}")

    cagrs = [resultados[p]["cagr"] for p in ORDEN_PERFILES if p in resultados]
    vols  = [resultados[p]["volatilidad_anual"] for p in ORDEN_PERFILES if p in resultados]
    jer_c = all(cagrs[i] <= cagrs[i + 1] for i in range(len(cagrs) - 1))
    jer_v = all(vols[i]  <= vols[i + 1]  for i in range(len(vols) - 1))
    print(f"\n{'✅' if jer_c else '⚠️ '} Jerarquía CAGR: {'correcta' if jer_c else 'VIOLADA'}")
    print(f"{'✅' if jer_v else '⚠️ '} Jerarquía VOL:  {'correcta' if jer_v else 'VIOLADA'}")

    # ── Guardar resultados ─────────────────────────────────────────────────
    with open(RESULTADOS_DIR / "metricas_C_reentrenado_test.json", "w") as f:
        json.dump({
            p: {k: (float(v) if isinstance(v, (float, np.floating)) and not np.isnan(v) else None)
                for k, v in m.items()}
            for p, m in resultados.items()
        }, f, indent=2)

    pd.DataFrame(curvas).to_csv(RESULTADOS_DIR / "curvas_C_reentrenado_test.csv")

    # ── Gráfica ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    for p in ORDEN_PERFILES:
        if p not in curvas:
            continue
        m = resultados[p]
        ax.plot(
            curvas[p].index, curvas[p].values,
            lw=2, color=COLORES.get(p),
            label=(f"{p}  Sharpe={m['sharpe']:.3f}  "
                   f"CAGR={m['cagr']:.1%}  MDD={m['max_drawdown']:.1%}"),
        )
    ax.set_title(
        f"Test C — Reentrenado en Train+Val (×{FACTOR_PASOS_EXTRA} pasos), evaluado en Test",
        fontsize=13,
    )
    ax.set_ylabel("Valor cartera (base 1 000)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    grafica = RESULTADOS_DIR / "grafica_C_reentrenado_test.png"
    plt.savefig(grafica, dpi=150)
    plt.show()
    plt.close()

    print(f"\n✅ Métricas:  {RESULTADOS_DIR / 'metricas_C_reentrenado_test.json'}")
    print(f"✅ Curvas:    {RESULTADOS_DIR / 'curvas_C_reentrenado_test.csv'}")
    print(f"✅ Gráfica:   {grafica}")
    print(f"✅ Agentes:   {AGENTES_DIR}/agente_final_<perfil>.pt")
    print("\nTest C completado.")


if __name__ == "__main__":
    main()