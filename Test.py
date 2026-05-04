"""
Test.py
-------
Evaluación final sobre el split de test.

Flujo:
  1. Carga config_ganadora_hpo.json
  2. Carga agentes_ganadores/agente_<perfil>.pt  (pesos del agente ganador del HPO)
  3. Reconstruye la arquitectura desde las dimensiones del propio checkpoint
  4. Evalúa directamente en test SIN reentrenar
  5. Guarda métricas (JSON), curvas (CSV) y gráfica (PNG) en resultados/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT   = Path(__file__).resolve().parent
RESULTADOS_DIR = PROJECT_ROOT / "resultados"
AGENTES_DIR    = PROJECT_ROOT / "agentes_ganadores"
RESULTADOS_DIR.mkdir(exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Agente_SAC import AgenteSAC
from entorno_cartera import EntornoCartera


def cargar_split(carpeta_base: Path, nombre: str):
    carpeta = carpeta_base / nombre.capitalize()
    de  = pd.read_csv(carpeta / f"datos_estado_{nombre}.csv",  index_col=0, parse_dates=True)
    ret = pd.read_csv(carpeta / f"retornos_{nombre}.csv",      index_col=0, parse_dates=True)
    rf  = pd.read_csv(carpeta / f"rf_semanal_{nombre}.csv",    index_col=0, parse_dates=True).squeeze("columns")
    return de, ret, rf


def cargar_cov(carpeta_base: Path, nombre: str) -> pd.DataFrame:
    return pd.read_csv(
        carpeta_base / nombre.capitalize() / f"covarianzas_{nombre}.csv", index_col=0
    )


def encontrar_carpeta_datos() -> Path:
    for c in ["Datos", "datos", "datos_procesados"]:
        p = PROJECT_ROOT / c
        if p.exists():
            return p
    raise FileNotFoundError("No se encuentra la carpeta de datos.")


def politica_det(agente, device):
    def fn(s):
        t = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a = agente.seleccionar_accion(t, determinista=True)
        return a.squeeze(0).detach().cpu().numpy().astype(np.float64)
    return fn


def calcular_metricas(valor: pd.Series, rf: pd.Series) -> dict:
    valor = valor.astype(float).dropna()
    rend  = valor.pct_change().dropna()
    if valor.empty or len(valor) < 2:
        return {k: np.nan for k in ["valor_final","retorno_total","cagr",
                "volatilidad_anual","sharpe","sortino","max_drawdown","calmar"]}
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
    return {"valor_final": vf, "retorno_total": vf/v0-1, "cagr": cagr,
            "volatilidad_anual": vol, "sharpe": sharpe, "sortino": sortino,
            "max_drawdown": mdd, "calmar": calmar}


def cargar_agente(perfil: str, cfg: dict) -> tuple[AgenteSAC, int, int]:
    """
    Carga el agente desde el checkpoint del HPO.
    Las dimensiones se leen del propio checkpoint — no se recalculan.
    """
    ruta = AGENTES_DIR / f"agente_{perfil}.pt"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}\n"
            f"Ejecuta primero la celda de persistencia del notebook HPO."
        )

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ruta, map_location=dispositivo)

    # Leer dimensiones directamente del checkpoint
    # net.0.weight tiene shape [hidden, dim_estado] → dim_estado = shape[1]
    dim_estado = ck["actor_state_dict"]["net.0.weight"].shape[1]
    # mu_head.weight tiene shape [dim_accion, hidden] → dim_accion = shape[0]
    dim_accion = ck["actor_state_dict"]["mu_head.weight"].shape[0]

    print(f"  dim_estado={dim_estado}  dim_accion={dim_accion}  "
          f"(leídos del checkpoint)")

    agente = AgenteSAC(
        dimension_estado=dim_estado,
        dimension_accion=dim_accion,
        dispositivo=dispositivo,
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
        agente.log_alpha.copy_(ck["log_alpha"].to(dispositivo))

    print(f"  ✅ {ck.get('config_id','?')}  "
          f"val_sharpe={ck.get('val_sharpe', float('nan')):.3f}  "
          f"val_cagr={ck.get('val_cagr', float('nan')):.1%}")
    return agente, dim_estado, dim_accion


def main():
    print("=" * 70)
    print("EVALUACIÓN FINAL — TEST (sin reentrenamiento)")
    print("=" * 70)

    config_path = PROJECT_ROOT / "config_ganadora_hpo.json"
    with open(config_path) as f:
        configs_hpo = json.load(f)
    print(f"\nPerfiles: {list(configs_hpo.keys())}")

    carpeta_datos = encontrar_carpeta_datos()
    de_test, ret_test, rf_test = cargar_split(carpeta_datos, "test")
    cov_train = cargar_cov(carpeta_datos, "train")
    print(f"Test: {len(ret_test)} semanas")

    n_activos       = ret_test.shape[1] + 1
    cartera_inicial = np.zeros(n_activos)
    cartera_inicial[-1] = 1.0  # 100% cash

    ORDEN   = ["muy_conservador","conservador","normal","arriesgado","muy_arriesgado"]
    colores = {
        "muy_conservador": "steelblue",   "conservador": "mediumseagreen",
        "normal":          "orange",       "arriesgado":  "tomato",
        "muy_arriesgado":  "mediumpurple",
    }

    resultados = {}
    curvas     = {}

    for perfil in ORDEN:
        if perfil not in configs_hpo:
            continue

        print(f"\n{'='*70}\nPERFIL: {perfil}\n{'='*70}")
        cfg = configs_hpo[perfil]

        # Cargar agente — dimensiones desde el checkpoint
        agente, dim_estado, dim_accion = cargar_agente(perfil, cfg)

        # Entorno de test con cov_train (sin leakage)
        entorno_test = EntornoCartera(
            datos_estado=de_test,
            retornos_semanales=ret_test,
            rf_semanal=rf_test,
            coste_transaccion=0.001,
            valor_inicial=1_000.0,
            covarianzas_iniciales=cov_train,
            riesgo=float(cfg["riesgo"]),
            lambda_dd=float(cfg.get("lambda_dd", 0.0)),
            lambda_varianza=float(cfg.get("lambda_varianza", 0.20)),
            lambda_correlacion=float(cfg.get("lambda_correlacion", 0.10)),
        )

        device = next(agente.actor.parameters()).device
        bt     = entorno_test.ejecutar_backtest(
            funcion_pesos=politica_det(agente, device),
            pesos_iniciales=cartera_inicial,
        )
        col   = "valor_cartera" if "valor_cartera" in bt.columns else bt.columns[0]
        serie = bt[col]
        met   = calcular_metricas(serie, rf_test)

        resultados[perfil] = met
        curvas[perfil]     = serie

        print(f"  Sharpe:      {met['sharpe']:>8.3f}")
        print(f"  CAGR:        {met['cagr']:>8.1%}")
        print(f"  MDD:         {met['max_drawdown']:>8.1%}")
        print(f"  Volatilidad: {met['volatilidad_anual']:>8.1%}")

        # Guardar agente final para la app
        ruta_final = RESULTADOS_DIR / f"agente_final_{perfil}.pt"
        torch.save({
            "actor_state_dict":   agente.actor.state_dict(),
            "critic1_state_dict": agente.critic1.state_dict(),
            "critic2_state_dict": agente.critic2.state_dict(),
            "log_alpha":          agente.log_alpha.detach().cpu(),
            "perfil":             perfil,
            "config":             cfg,
            "dim_estado":         dim_estado,
            "dim_accion":         dim_accion,
            "test_sharpe":        met["sharpe"],
            "test_cagr":          met["cagr"],
        }, ruta_final)
        print(f"  💾 {ruta_final.name}")

    # Tabla resumen
    print("\n" + "="*70)
    print("RESUMEN FINAL — TEST")
    print(f"{'Perfil':<20} {'Sharpe':>8} {'CAGR':>8} {'MDD':>8} {'Vol':>8}")
    print("-"*60)
    for p in ORDEN:
        if p not in resultados: continue
        m = resultados[p]
        print(f"{p:<20} {m['sharpe']:>8.3f} {m['cagr']:>8.1%} "
              f"{m['max_drawdown']:>8.1%} {m['volatilidad_anual']:>8.1%}")

    cagrs = [resultados[p]["cagr"] for p in ORDEN if p in resultados]
    vols  = [resultados[p]["volatilidad_anual"] for p in ORDEN if p in resultados]
    jer_c = all(cagrs[i] <= cagrs[i+1] for i in range(len(cagrs)-1))
    jer_v = all(vols[i]  <= vols[i+1]  for i in range(len(vols)-1))
    print(f"\n{'✅' if jer_c else '⚠️ '} Jerarquía CAGR: {'OK' if jer_c else 'VIOLADA'}")
    print(f"{'✅' if jer_v else '⚠️ '} Jerarquía VOL:  {'OK' if jer_v else 'VIOLADA'}")

    # Guardar JSON y CSV
    with open(RESULTADOS_DIR / "metricas_test.json", "w") as f:
        json.dump({
            p: {k: (float(v) if isinstance(v, (float, np.floating)) and not np.isnan(v) else None)
                for k, v in m.items()}
            for p, m in resultados.items()
        }, f, indent=2)
    pd.DataFrame(curvas).to_csv(RESULTADOS_DIR / "curvas_test.csv")
    print(f"\n✅ Métricas: {RESULTADOS_DIR / 'metricas_test.json'}")
    print(f"✅ Curvas:   {RESULTADOS_DIR / 'curvas_test.csv'}")

    # Gráfica
    fig, ax = plt.subplots(figsize=(14, 6))
    for p in ORDEN:
        if p not in curvas: continue
        m = resultados[p]
        ax.plot(curvas[p].index, curvas[p].values, lw=2, color=colores.get(p),
                label=(f"{p}  Sharpe={m['sharpe']:.3f}  "
                       f"CAGR={m['cagr']:.1%}  MDD={m['max_drawdown']:.1%}"))
    ax.set_title("Ganador de cada perfil — Test (2021–2026)", fontsize=13)
    ax.set_ylabel("Valor cartera")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    grafica = RESULTADOS_DIR / "grafica_test.png"
    plt.savefig(grafica, dpi=150)
    plt.show()
    print(f"✅ Gráfica:  {grafica}")
    print("\nEvaluación completada.")


if __name__ == "__main__":
    main()