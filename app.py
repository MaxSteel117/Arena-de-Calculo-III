from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from functools import lru_cache
from pathlib import Path
from flask import Flask, Response, render_template_string, request
import plotly
from calculo.integrales import (
    calcular_campo_vectorial,
    calcular_descenso_gradiente,
    calcular_doble,
    calcular_linea_cerrada,
    calcular_simple,
    calcular_triple,
)
from graficos.recorrido import generar_grafica_html

BASE_DIR = Path(__file__).resolve().parent
PLANTILLA_INICIO = (BASE_DIR / "interfaz" / "inicio.html").read_text(encoding="utf-8")
PLANTILLA_HTML = (BASE_DIR / "interfaz" / "index.html").read_text(encoding="utf-8")

app = Flask(__name__)

# Pool persistente de hilos: SymPy, NumPy y Plotly ya están en memoria (0s de arranque)
EXECUTOR = ThreadPoolExecutor(max_workers=4)
PLOTLY_BYTES = (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_bytes()


@lru_cache(maxsize=64)
def _calcular_cacheado(modo: str, items_ordenados: tuple):
    kwargs = dict(items_ordenados)
    if modo == "simple":
        res = calcular_simple(**kwargs)
    elif modo == "doble":
        res = calcular_doble(**kwargs)
    elif modo == "triple":
        res = calcular_triple(**kwargs)
    elif modo == "linea":
        res = calcular_linea_cerrada(**kwargs)
    elif modo == "campo":
        res = calcular_campo_vectorial(**kwargs)
    elif modo == "gradiente":
        res = calcular_descenso_gradiente(**kwargs)
    else:
        return {"error": f"El tipo '{modo}' no es válido."}

    if "error" not in res:
        datos_graf = res.pop("_datos_grafica")
        res["grafica_html"] = generar_grafica_html(modo, datos_graf)

    return res


def ejecutar_rapido(modo: str, kwargs: dict, timeout_seg: float = 5.0):
    items = tuple(sorted(kwargs.items()))
    futuro = EXECUTOR.submit(_calcular_cacheado, modo, items)
    try:
        return futuro.result(timeout=timeout_seg)
    except FuturesTimeout:
        return {
            "error": (
                "La integral simbólica es demasiado compleja para resolverse en menos de 5 segundos. "
                "Prueba una expresión más directa."
            )
        }
    except Exception as exc:
        return {"error": f"Error al procesar la expresión: {exc}"}


@app.route("/plotly.js", methods=["GET"])
def servir_plotly_local():
    return Response(
        PLOTLY_BYTES,
        mimetype="application/javascript",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.route("/", methods=["GET"])
def inicio():
    return render_template_string(PLANTILLA_INICIO)


@app.route("/IntegralesMatematicas", methods=["GET", "POST"])
def index():
    modo = "simple"

    # Simple
    expr_simple, diff_simple, a_simple, b_simple = "cos(teta)", "dteta", "0", "pi/2"

    # Doble
    coord_doble = "cartesianas"
    expr_doble, diff_doble = "x * sen(y)", "dy dx"
    l1_inf_doble, l1_sup_doble, l2_inf_doble, l2_sup_doble = "0", "x", "0", "pi"

    # Triple
    coord_triple = "cartesianas"
    expr_triple, diff_triple = "x * y * z", "dz dy dx"
    l1_inf_triple, l1_sup_triple = "0", "x + y"
    l2_inf_triple, l2_sup_triple = "0", "x"
    l3_inf_triple, l3_sup_triple = "0", "1"

    # Línea / Cerrada (Teorema de Green)
    m_linea, n_linea = "-y", "x"
    xt_linea, yt_linea = "2*cos(t)", "2*sin(t)"
    t0_linea, t1_linea = "0", "2*pi"

    # Campo Vectorial
    p_campo, q_campo, r_campo = "x^2 * z", "2 * y^3 * z^2", "x * y^2 * z"

    # Descenso del Gradiente 3D
    expr_grad = "x^2 + 2*y^2"
    x0_grad, y0_grad = "3", "-2.5"
    lr_grad, pasos_grad = "0.15", "25"

    error = None
    resultado_data = None

    if request.method == "POST":
        modo = request.form.get("modo", "simple").strip().lower()

        expr_simple = request.form.get("expr_simple", expr_simple)
        diff_simple = request.form.get("diff_simple", diff_simple)
        a_simple = request.form.get("a_simple", a_simple)
        b_simple = request.form.get("b_simple", b_simple)

        coord_doble = request.form.get("coord_doble", coord_doble)
        expr_doble = request.form.get("expr_doble", expr_doble)
        diff_doble = request.form.get("diff_doble", diff_doble)
        l1_inf_doble = request.form.get("l1_inf_doble", l1_inf_doble)
        l1_sup_doble = request.form.get("l1_sup_doble", l1_sup_doble)
        l2_inf_doble = request.form.get("l2_inf_doble", l2_inf_doble)
        l2_sup_doble = request.form.get("l2_sup_doble", l2_sup_doble)

        coord_triple = request.form.get("coord_triple", coord_triple)
        expr_triple = request.form.get("expr_triple", expr_triple)
        diff_triple = request.form.get("diff_triple", diff_triple)
        l1_inf_triple = request.form.get("l1_inf_triple", l1_inf_triple)
        l1_sup_triple = request.form.get("l1_sup_triple", l1_sup_triple)
        l2_inf_triple = request.form.get("l2_inf_triple", l2_inf_triple)
        l2_sup_triple = request.form.get("l2_sup_triple", l2_sup_triple)
        l3_inf_triple = request.form.get("l3_inf_triple", l3_inf_triple)
        l3_sup_triple = request.form.get("l3_sup_triple", l3_sup_triple)

        m_linea = request.form.get("m_linea", m_linea)
        n_linea = request.form.get("n_linea", n_linea)
        xt_linea = request.form.get("xt_linea", xt_linea)
        yt_linea = request.form.get("yt_linea", yt_linea)
        t0_linea = request.form.get("t0_linea", t0_linea)
        t1_linea = request.form.get("t1_linea", t1_linea)

        p_campo = request.form.get("p_campo", p_campo)
        q_campo = request.form.get("q_campo", q_campo)
        r_campo = request.form.get("r_campo", r_campo)

        expr_grad = request.form.get("expr_grad", expr_grad)
        x0_grad = request.form.get("x0_grad", x0_grad)
        y0_grad = request.form.get("y0_grad", y0_grad)
        lr_grad = request.form.get("lr_grad", lr_grad)
        pasos_grad = request.form.get("pasos_grad", pasos_grad)

        if modo == "simple":
            kwargs = {"expr_raw": expr_simple, "diff_raw": diff_simple, "a_raw": a_simple, "b_raw": b_simple}
        elif modo == "doble":
            kwargs = {
                "expr_raw": expr_doble, "diff_raw": diff_doble,
                "l1_inf": l1_inf_doble, "l1_sup": l1_sup_doble,
                "l2_inf": l2_inf_doble, "l2_sup": l2_sup_doble,
                "coord": coord_doble,
            }
        elif modo == "triple":
            kwargs = {
                "expr_raw": expr_triple, "diff_raw": diff_triple,
                "l1_inf": l1_inf_triple, "l1_sup": l1_sup_triple,
                "l2_inf": l2_inf_triple, "l2_sup": l2_sup_triple,
                "l3_inf": l3_inf_triple, "l3_sup": l3_sup_triple,
                "coord": coord_triple,
            }
        elif modo == "linea":
            kwargs = {
                "m_raw": m_linea, "n_raw": n_linea,
                "xt_raw": xt_linea, "yt_raw": yt_linea,
                "t0_raw": t0_linea, "t1_raw": t1_linea,
            }
        elif modo == "campo":
            kwargs = {"p_raw": p_campo, "q_raw": q_campo, "r_raw": r_campo}
        elif modo == "gradiente":
            kwargs = {
                "expr_raw": expr_grad, "x0_raw": x0_grad, "y0_raw": y0_grad,
                "lr_raw": lr_grad, "pasos_raw": pasos_grad,
            }
        else:
            kwargs = {}

        res = ejecutar_rapido(modo, kwargs, timeout_seg=5.0)

        if "error" in res:
            error = res["error"]
        else:
            resultado_data = res

    return render_template_string(
        PLANTILLA_HTML,
        modo=modo,
        expr_simple=expr_simple, diff_simple=diff_simple, a_simple=a_simple, b_simple=b_simple,
        coord_doble=coord_doble, expr_doble=expr_doble, diff_doble=diff_doble,
        l1_inf_doble=l1_inf_doble, l1_sup_doble=l1_sup_doble, l2_inf_doble=l2_inf_doble, l2_sup_doble=l2_sup_doble,
        coord_triple=coord_triple, expr_triple=expr_triple, diff_triple=diff_triple,
        l1_inf_triple=l1_inf_triple, l1_sup_triple=l1_sup_triple,
        l2_inf_triple=l2_inf_triple, l2_sup_triple=l2_sup_triple,
        l3_inf_triple=l3_inf_triple, l3_sup_triple=l3_sup_triple,
        m_linea=m_linea, n_linea=n_linea, xt_linea=xt_linea, yt_linea=yt_linea,
        t0_linea=t0_linea, t1_linea=t1_linea,
        p_campo=p_campo, q_campo=q_campo, r_campo=r_campo,
        expr_grad=expr_grad, x0_grad=x0_grad, y0_grad=y0_grad,
        lr_grad=lr_grad, pasos_grad=pasos_grad,
        error=error, resultado_data=resultado_data,
    )


if __name__ == "__main__":
    print("Abre en tu navegador: http://127.0.0.1:5000")
    app.run(debug=False, port=5000, threaded=True)