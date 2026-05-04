"""
Test_B_evaluar_en_val_y_test.py
────────────────────────────────
Carga los agentes guardados del HPO y los evalúa DIRECTAMENTE en el periodo
VALIDATION + TEST concatenados (2018–2026), sin ningún reentrenamiento.

Útil para ver el comportamiento continuo desde el final del train hasta hoy,
e identificar si hay degradación entre validación y test.

Flujo:
  1. Lee config_ganadora_hpo.json
  2. Carga agentes_ganadores/agente_<perfil>.pt
  3. Concatena datos de validation y test
  4. Ejecuta backtest determinista sobre el periodo completo
  5. Guarda en resultados/B_val_y_test/
       - metricas_B_val_y_test.json
       - curvas_B_val_y_test.csv
       - grafica_B_val_y_test.png  (con línea divisoria val|test)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import torch

# ── Rutas ──────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent
AGENTES_DIR    = PROJECT_ROOT / "agentes_ganadores"
RESULTADOS_DIR = PROJECT_ROOT / "resultados" / "B_val_y_test"
RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Agente_SAC import AgenteSAC
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


def cargar_agente(perfil: str, cfg: dict) -> tuple[AgenteSAC, int, int]:
    ruta = AGENTES_DIR / f"agente_{perfil}.pt"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}\n"
            "Ejecuta primero la celda de persistencia del notebook HPO."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ruta, map_location=device)

    dim_estado = ck["actor_state_dict"]["net.0.weight"].shape[1]
    dim_accion = ck["actor_state_dict"]["mu_head.weight"].shape[0]
    print(f"  dim_estado={dim_estado}  dim_accion={dim_accion}  (leídos del checkpoint)")

    agente = AgenteSAC(
        dimension_estado=dim_estado,
        dimension_accion=dim_accion,
        dispositivo=device,
        gamma=float(cfg.get("gamma", 0.99)),
        tau=float(cfg.get("tau", 0.02)),
        tasa_aprendizaje_actor=float(cfg.get("lr_actor", 1e-4)),
        tasa_aprendizaje_criticos=float(cfg.get("lr_criticos", 3e-4)),
        tasa_aprendizaje_alpha=float(cfg.get("lr_alpha", 1e-4)),
        target_entropy=float(cfg["target_entropy"]) if cfg.get("target_entropy") else None,
        reward_scale=float(cfg.get("reward_scale", 20.0)),
        offset_target_entropy=float(cfg.get("offset_target_entropy", 0.0)),
    )
    agente.actor.load_state_dict(ck["actor_state_dict"])
    agente.critic1.load_state_dict(ck["critic1_state_dict"])
    agente.critic2.load_state_dict(ck["critic2_state_dict"])
    agente.critic1_target.load_state_dict(agente.critic1.state_dict())
    agente.critic2_target.load_state_dict(agente.critic2.state_dict())
    with torch.no_grad():
        agente.log_alpha.copy_(ck["log_alpha"].to(device))

    print(f"  ✅ {ck.get('config_id', '?')}  "
          f"val_sharpe={ck.get('val_sharpe', float('nan')):.3f}  "
          f"val_cagr={ck.get('val_cagr', float('nan')):.1%}")
    return agente, dim_estado, dim_accion


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("TEST B — Agente HPO evaluado en VALIDATION + TEST (sin reentrenamiento)")
    print("Periodo completo: 2018–2026")
    print("=" * 70)

    config_path = PROJECT_ROOT / "config_ganadora_hpo.json"
    with open(config_path) as f:
        configs_hpo = json.load(f)

    carpeta_datos = encontrar_carpeta_datos()

    # Cargar los dos splits
    de_val,  ret_val,  rf_val  = cargar_split(carpeta_datos, "validation")
    de_test, ret_test, rf_test = cargar_split(carpeta_datos, "test")

    # Concatenar — el entorno ve todo el periodo de corrido
    de_completo  = pd.concat([de_val,  de_test],  axis=0).sort_index()
    ret_completo = pd.concat([ret_val, ret_test],  axis=0).sort_index()
    rf_completo  = pd.concat([rf_val,  rf_test],   axis=0).sort_index()
    rf_completo  = rf_completo[~rf_completo.index.duplicated(keep="first")]

    # Covarianzas del split de train — sin leakage
    cov_train = cargar_cov(carpeta_datos, "train")

    # Fecha de inicio del test (para la línea divisoria en la gráfica)
    fecha_inicio_test = ret_test.index[0]

    print(f"\nValidation: {len(ret_val)} semanas "
          f"({ret_val.index[0].date()} → {ret_val.index[-1].date()})")
    print(f"Test:       {len(ret_test)} semanas "
          f"({ret_test.index[0].date()} → {ret_test.index[-1].date()})")
    print(f"Total:      {len(ret_completo)} semanas")

    n_activos       = ret_completo.shape[1] + 1
    cartera_inicial = np.zeros(n_activos)
    cartera_inicial[-1] = 1.0  # 100 % cash al arrancar

    resultados:      dict = {}
    curvas:          dict = {}
    metricas_val:    dict = {}
    metricas_test_s: dict = {}

    for perfil in ORDEN_PERFILES:
        if perfil not in configs_hpo:
            print(f"\n⚠️  Perfil '{perfil}' no encontrado en config_ganadora_hpo.json, se omite.")
            continue

        print(f"\n{'─'*70}\nPERFIL: {perfil}\n{'─'*70}")
        cfg = configs_hpo[perfil]

        agente, dim_estado, dim_accion = cargar_agente(perfil, cfg)

        entorno_completo = EntornoCartera(
            datos_estado=de_completo,
            retornos_semanales=ret_completo,
            rf_semanal=rf_completo,
            coste_transaccion=0.001,
            valor_inicial=1_000.0,
            covarianzas_iniciales=cov_train,
            riesgo=float(cfg["riesgo"]),
            lambda_dd=float(cfg.get("lambda_dd", 0.0)),
            lambda_varianza=float(cfg.get("lambda_varianza", 0.20)),
            lambda_correlacion=float(cfg.get("lambda_correlacion", 0.10)),
        )

        device = next(agente.actor.parameters()).device
        bt     = entorno_completo.ejecutar_backtest(
            funcion_pesos=politica_det(agente, device),
            pesos_iniciales=cartera_inicial,
        )
        serie = bt["valor_cartera"] if "valor_cartera" in bt.columns else bt.iloc[:, 0]

        # Métricas sobre el periodo completo
        met_total = calcular_metricas(serie, rf_completo)

        # Métricas separadas por sub-periodo (para análisis de degradación)
        serie_val_part  = serie[serie.index < fecha_inicio_test]
        serie_test_part = serie[serie.index >= fecha_inicio_test]

        # Re-basar el sub-periodo test al valor en la frontera para métricas aisladas
        met_val_part  = calcular_metricas(serie_val_part,  rf_val)
        met_test_part = calcular_metricas(serie_test_part, rf_test)

        resultados[perfil]      = met_total
        curvas[perfil]          = serie
        metricas_val[perfil]    = met_val_part
        metricas_test_s[perfil] = met_test_part

        print(f"  [Total val+test]")
        print(f"    Sharpe:      {met_total['sharpe']:>8.3f}")
        print(f"    CAGR:        {met_total['cagr']:>8.1%}")
        print(f"    MDD:         {met_total['max_drawdown']:>8.1%}")
        print(f"    Volatilidad: {met_total['volatilidad_anual']:>8.1%}")
        print(f"  [Solo validation]  CAGR={met_val_part['cagr']:>6.1%}  "
              f"Sharpe={met_val_part['sharpe']:.3f}")
        print(f"  [Solo test]        CAGR={met_test_part['cagr']:>6.1%}  "
              f"Sharpe={met_test_part['sharpe']:.3f}")

    # ── Resumen ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESUMEN FINAL — TEST B (agente HPO, periodo val+test)")
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
    def serializar(d: dict) -> dict:
        return {
            p: {k: (float(v) if isinstance(v, (float, np.floating)) and not np.isnan(v) else None)
                for k, v in m.items()}
            for p, m in d.items()
        }

    with open(RESULTADOS_DIR / "metricas_B_val_y_test.json", "w") as f:
        json.dump({
            "total_val_test": serializar(resultados),
            "solo_validation": serializar(metricas_val),
            "solo_test":       serializar(metricas_test_s),
        }, f, indent=2)

    pd.DataFrame(curvas).to_csv(RESULTADOS_DIR / "curvas_B_val_y_test.csv")

    # ── Benchmarks por perfil ──────────────────────────────────────────────
    # Cada perfil se compara contra su benchmark equivalente:
    #   muy_conservador → SHY  (T-bills 1-3y, riesgo mínimo)
    #   conservador     → AGG  (aggregate bonds, riesgo bajo)
    #   normal          → 60/40 SPY+AGG
    #   arriesgado      → 80/20 SPY+AGG
    #   muy_arriesgado  → SPY  (100% renta variable)
    BENCHMARK_POR_PERFIL = {
        "muy_conservador": ("SHY",  "SHY (T-bills)",    "black",   ":"),
        "conservador":     ("AGG",  "AGG (bonds)",       "black",   ":"),
        "normal":          (None,   "50/50 SPY+AGG",     "black",   ":"),
        "arriesgado":      (None,   "70/30 SPY+AGG",     "black",   ":"),
        "muy_arriesgado":  ("SPY",  "SPY (100% RV)",     "black",   ":"),
    }

    print("\nDescargando benchmarks (SPY, AGG, SHY)...")
    px_bench: pd.DataFrame = pd.DataFrame()
    try:
        import yfinance as yf
        fecha_inicio_bench = ret_completo.index[0].strftime("%Y-%m-%d")
        fecha_fin_bench    = (ret_completo.index[-1] + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        raw = yf.download(
            ["SPY", "AGG", "SHY"],
            start=fecha_inicio_bench,
            end=fecha_fin_bench,
            auto_adjust=True,
            progress=False,
        )
        px_bench = raw["Close"] if "Close" in raw else raw["Adj Close"]
        px_bench = px_bench.resample("W-FRI").last().ffill()
        px_bench = px_bench.reindex(ret_completo.index, method="ffill").dropna()
        print("  Benchmarks descargados.")
    except Exception as e:
        print(f"  No se pudieron descargar benchmarks: {e}")

    def _bench_curva(perfil: str, base: float = 1_000.0):
        """Devuelve (curva, label) del benchmark para cada perfil."""
        if px_bench.empty:
            return None, None
        ticker, label, _, _ = BENCHMARK_POR_PERFIL[perfil]
        ret_spy = px_bench["SPY"].pct_change().fillna(0.0)
        ret_agg = px_bench["AGG"].pct_change().fillna(0.0)
        if perfil == "muy_conservador":
            serie = px_bench["SHY"]
            curva = serie / serie.iloc[0] * base
        elif perfil == "conservador":
            serie = px_bench["AGG"]
            curva = serie / serie.iloc[0] * base
        elif perfil == "normal":
            curva = (1 + 0.50 * ret_spy + 0.50 * ret_agg).cumprod() * base
        elif perfil == "arriesgado":
            curva = (1 + 0.70 * ret_spy + 0.30 * ret_agg).cumprod() * base
        elif perfil == "muy_arriesgado":
            serie = px_bench["SPY"]
            curva = serie / serie.iloc[0] * base
        else:
            return None, None
        return curva, label

    # ── Gráfica: un subplot por perfil ────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, p in enumerate(ORDEN_PERFILES):
        if p not in curvas:
            continue
        ax = axes[idx]
        m  = resultados[p]

        # Curva del modelo
        ax.plot(
            curvas[p].index, curvas[p].values,
            lw=2.5, color=COLORES.get(p),
            label=f"Modelo  Sh={m['sharpe']:.2f}  CAGR={m['cagr']:.1%}  MDD={m['max_drawdown']:.1%}",
        )

        # Benchmark equivalente
        curva_b, label_b = _bench_curva(p)
        if curva_b is not None:
            mb = calcular_metricas(curva_b, rf_completo)
            _, _, color_b, ls_b = BENCHMARK_POR_PERFIL[p]
            ax.plot(
                curva_b.index, curva_b.values,
                lw=1.8, color=color_b, ls=ls_b,
                label=f"{label_b}  Sh={mb['sharpe']:.2f}  CAGR={mb['cagr']:.1%}  MDD={mb['max_drawdown']:.1%}",
            )

        # Línea val/test
        ax.axvline(fecha_inicio_test, color="gray", lw=1.0, ls="--", alpha=0.5)
        ax.set_title(p.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_ylabel("Valor (base 1 000)", fontsize=8)
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.25)

    # Ocultar el subplot sobrante (2x3 = 6 celdas, 5 perfiles)
    axes[5].set_visible(False)

    fig.suptitle(
        "Agente HPO vs Benchmark equivalente por perfil (Validation + Test, 2018–2026)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    grafica = RESULTADOS_DIR / "grafica_B_val_y_test.png"
    plt.savefig(grafica, dpi=150)
    plt.show()
    plt.close()

    print(f"\n Metricas:  {RESULTADOS_DIR / 'metricas_B_val_y_test.json'}")
    print(f"✅ Curvas:    {RESULTADOS_DIR / 'curvas_B_val_y_test.csv'}")
    print(f"✅ Gráfica:   {grafica}")
    print("\nTest B completado.")


if __name__ == "__main__":
    main()