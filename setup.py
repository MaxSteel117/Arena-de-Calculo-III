from pathlib import Path

BASE = Path(__file__).resolve().parent

INTEGRALES_PY = r'''import re
import numpy as np
from scipy import integrate
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

TRANSFORMACIONES = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

PROHIBIDOS = [
    "__", "import", "exec", "eval", "open", "os", "sys", "subprocess",
    "lambda", "dy/dx", "dx/dy", "dz/dx", "dz/dy", "d/dx", "d/dy", "d/dz", "=",
]

OPERADORES_SIMBOLICOS = {
    "dy/dx", "dx/dy", "dz/dx", "dz/dy", "d/dx", "d/dy", "d/dz", "=",
}


def _preprocesar_texto(texto: str):
    if not texto or not str(texto).strip():
        return False, "", "Debes ingresar una expresión o límite válido."
    limpio = str(texto).strip().lower()
    for token in PROHIBIDOS:
        coincide = (token in limpio) if token in OPERADORES_SIMBOLICOS else bool(
            re.search(rf"\b{re.escape(token)}\b", limpio)
        )
        if coincide:
            return (
                False,
                "",
                f"Expresión no soportada ('{token}'). Este sistema es exclusivo para integrales simples, dobles o triples.",
            )
    limpio = re.sub(r"\bteta\b", "theta", limpio)
    limpio = re.sub(r"\bsen\b", "sin", limpio)
    limpio = re.sub(r"\bln\b", "log", limpio)
    return True, limpio, ""


def _simbolos_base():
    x, y, z, theta = sp.symbols("x y z theta", real=True)
    return {
        "x": x, "y": y, "z": z, "theta": theta,
        "pi": sp.pi, "e": sp.E,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
        "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
        "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs,
        "log": sp.log, "log10": lambda u: sp.log(u, 10),
    }


def _parsear_sympy(texto: str, extras_simbolos: dict | None = None):
    ok, limpio, err = _preprocesar_texto(texto)
    if not ok:
        return None, err
    ns = _simbolos_base()
    if extras_simbolos:
        ns.update(extras_simbolos)
    try:
        expr = parse_expr(limpio, local_dict=ns, transformations=TRANSFORMACIONES)
        return expr, ""
    except Exception as exc:
        return None, f"No se pudo interpretar '{texto}': {exc}"


def _ltx(expr_sym) -> str:
    return sp.latex(expr_sym, fold_short_frac=False)


def _fmt_lim_latex(val_raw: str, expr_sym) -> str:
    if str(val_raw).strip():
        s, err = _parsear_sympy(val_raw)
        if not err and s is not None:
            return _ltx(s)
    return _ltx(expr_sym)


def parsear_diferenciales(diff_raw: str, esperados: int):
    if not diff_raw or not str(diff_raw).strip():
        return False, [], "Debes indicar los diferenciales (ej. dx, dy dx, dz dy dx)."
    texto = str(diff_raw).strip().lower().replace("teta", "theta")
    matches = re.findall(r"\bd(theta|[a-z])\b", texto)
    if not matches:
        compacto = re.sub(r"\s+", "", texto)
        matches = re.findall(r"d(theta|[a-z])", compacto)
    if len(matches) != esperados:
        return False, [], f"Se esperaban {esperados} diferencial(es), pero se detectaron {len(matches)} en '{diff_raw}'."
    if len(set(matches)) != len(matches):
        return False, [], f"Los diferenciales en '{diff_raw}' no deben repetirse."
    return True, matches, ""


def _validar_variables_expr(expr_sym, permitidas: set[str], contexto: str):
    libres = {str(s) for s in expr_sym.free_symbols}
    extra = sorted(libres - permitidas)
    if extra:
        return f"Variable(s) no permitida(s) en {contexto}: {', '.join(extra)}."
    return ""


def _a_num_float(val_sym) -> float:
    return float(sp.N(val_sym))


def calcular_simple(expr_raw: str, diff_raw: str, a_raw: str, b_raw: str):
    ok_d, diffs, err_d = parsear_diferenciales(diff_raw, 1)
    if not ok_d:
        return {"error": err_d}
    var_name = diffs[0]
    sym_var = sp.Symbol(var_name, real=True)

    expr_sym, err = _parsear_sympy(expr_raw)
    if err:
        return {"error": err}
    err_v = _validar_variables_expr(expr_sym, {var_name, "x", "theta"}, "la integral simple")
    if err_v:
        return {"error": err_v}

    if var_name == "theta" and sp.Symbol("x", real=True) in expr_sym.free_symbols:
        expr_sym = expr_sym.subs(sp.Symbol("x", real=True), sym_var)
    if var_name == "x" and sp.Symbol("theta", real=True) in expr_sym.free_symbols:
        expr_sym = expr_sym.subs(sp.Symbol("theta", real=True), sym_var)

    a_sym, ea = _parsear_sympy(a_raw)
    b_sym, eb = _parsear_sympy(b_raw)
    if ea or eb:
        return {"error": ea or eb}
    if a_sym.free_symbols or b_sym.free_symbols:
        return {"error": "En una integral simple los límites deben ser constantes."}

    a_num, b_num = _a_num_float(a_sym), _a_num_float(b_sym)
    if not (np.isfinite(a_num) and np.isfinite(b_num)) or a_num >= b_num:
        return {"error": "Límites inválidos: se requiere límite inferior < superior."}

    fn_num = sp.lambdify(sym_var, expr_sym, modules=["numpy", "math"])
    resultado_num, error_num = integrate.quad(fn_num, a_num, b_num)

    primitiva = sp.integrate(expr_sym, sym_var)
    eval_sup = sp.simplify(primitiva.subs(sym_var, b_sym))
    eval_inf = sp.simplify(primitiva.subs(sym_var, a_sym))
    res_simbolico = sp.simplify(eval_sup - eval_inf)

    a_ltx = _fmt_lim_latex(a_raw, a_sym)
    b_ltx = _fmt_lim_latex(b_raw, b_sym)
    v_ltx = _ltx(sym_var)
    f_ltx = _ltx(expr_sym)
    prim_ltx = _ltx(primitiva)

    latex_integral = rf"\int_{{{a_ltx}}}^{{{b_ltx}}} ({f_ltx})\,d{v_ltx}"

    pasos = [
        {"titulo": "1. Integral planteada", "latex": latex_integral},
        {"titulo": f"2. Antiderivada respecto a {var_name}", "latex": rf"F({v_ltx}) = {prim_ltx}"},
        {
            "titulo": f"3. Evaluar desde {a_raw} hasta {b_raw}",
            "latex": rf"({prim_ltx})\Big|_{{{a_ltx}}}^{{{b_ltx}}} = ({_ltx(eval_sup)}) - ({_ltx(eval_inf)})",
        },
        {"titulo": "4. Simplificación", "latex": rf"{_ltx(res_simbolico)} \approx {resultado_num:.6f}"},
    ]

    return {
        "tipo_titulo": "Integral Simple",
        "latex_integral": latex_integral,
        "resultado_simbolico": _ltx(res_simbolico),
        "resultado": f"{resultado_num:.6f}",
        "detalle_extra": f"Error numérico estimado: {error_num:.2e}",
        "pasos": pasos,
        "_datos_grafica": {
            "fn": fn_num, "var_name": var_name, "expr_raw": expr_raw,
            "a": a_num, "b": b_num, "resultado": resultado_num,
        },
    }


def calcular_doble(expr_raw: str, diff_raw: str, l1_inf: str, l1_sup: str, l2_inf: str, l2_sup: str):
    ok_d, diffs, err_d = parsear_diferenciales(diff_raw, 2)
    if not ok_d:
        return {"error": err_d}

    v_int, v_ext = diffs[0], diffs[1]
    s_int = sp.Symbol(v_int, real=True)
    s_ext = sp.Symbol(v_ext, real=True)

    expr_sym, err = _parsear_sympy(expr_raw)
    if err:
        return {"error": err}
    err_v = _validar_variables_expr(expr_sym, {"x", "y"}, "la integral doble")
    if err_v:
        return {"error": err_v}

    l2a_sym, e2a = _parsear_sympy(l2_inf)
    l2b_sym, e2b = _parsear_sympy(l2_sup)
    if e2a or e2b:
        return {"error": e2a or e2b}
    if l2a_sym.free_symbols or l2b_sym.free_symbols:
        return {"error": f"Los límites de la primera ∫ ({v_ext}) deben ser constantes."}

    l1a_sym, e1a = _parsear_sympy(l1_inf)
    l1b_sym, e1b = _parsear_sympy(l1_sup)
    if e1a or e1b:
        return {"error": e1a or e1b}
    if _validar_variables_expr(l1a_sym, {v_ext}, f"límite inferior de {v_int}") or _validar_variables_expr(l1b_sym, {v_ext}, f"límite superior de {v_int}"):
        return {"error": f"Los límites de la segunda ∫ ({v_int}) solo pueden depender de {v_ext}."}

    ext_a, ext_b = _a_num_float(l2a_sym), _a_num_float(l2b_sym)
    if not (np.isfinite(ext_a) and np.isfinite(ext_b)) or ext_a >= ext_b:
        return {"error": f"Límites de {v_ext} inválidos."}

    fn_int_a = sp.lambdify(s_ext, l1a_sym, modules=["numpy", "math"])
    fn_int_b = sp.lambdify(s_ext, l1b_sym, modules=["numpy", "math"])

    def _eval_lim_int(fn_l, val):
        return float(np.asarray(fn_l(val), dtype=float).item())

    for val_ext in np.linspace(ext_a, ext_b, 6):
        ia = _eval_lim_int(fn_int_a, val_ext)
        ib = _eval_lim_int(fn_int_b, val_ext)
        if not (np.isfinite(ia) and np.isfinite(ib)) or ia > ib + 1e-9:
            return {"error": f"En {v_ext}={val_ext:.3f}, el límite inferior de {v_int} supera al superior."}

    sx, sy = sp.Symbol("x", real=True), sp.Symbol("y", real=True)
    fn_num = sp.lambdify((sx, sy), expr_sym, modules=["numpy", "math"])

    def _f_x_y(xv, yv):
        if v_int == "y" and v_ext == "x":
            return fn_num(xv, yv)
        if v_int == "x" and v_ext == "y":
            return fn_num(yv, xv)
        return fn_num(xv, yv) if v_int == "y" else fn_num(yv, xv)

    resultado_num, error_num = integrate.dblquad(
        lambda u_int, u_ext: _f_x_y(u_ext, u_int) if v_ext == "x" else _f_x_y(u_int, u_ext),
        ext_a,
        ext_b,
        lambda u_ext: _eval_lim_int(fn_int_a, u_ext),
        lambda u_ext: _eval_lim_int(fn_int_b, u_ext),
    )

    prim_int = sp.integrate(expr_sym, s_int)
    res_int_sup = sp.simplify(prim_int.subs(s_int, l1b_sym))
    res_int_inf = sp.simplify(prim_int.subs(s_int, l1a_sym))
    interior_simplificado = sp.simplify(res_int_sup - res_int_inf)

    prim_ext = sp.integrate(interior_simplificado, s_ext)
    res_ext_sup = sp.simplify(prim_ext.subs(s_ext, l2b_sym))
    res_ext_inf = sp.simplify(prim_ext.subs(s_ext, l2a_sym))
    res_simbolico = sp.simplify(res_ext_sup - res_ext_inf)

    l1a_ltx = _fmt_lim_latex(l1_inf, l1a_sym)
    l1b_ltx = _fmt_lim_latex(l1_sup, l1b_sym)
    l2a_ltx = _fmt_lim_latex(l2_inf, l2a_sym)
    l2b_ltx = _fmt_lim_latex(l2_sup, l2b_sym)

    f_ltx = _ltx(expr_sym)
    vi_ltx = _ltx(s_int)
    ve_ltx = _ltx(s_ext)
    prim_int_ltx = _ltx(prim_int)
    int_simp_ltx = _ltx(interior_simplificado)
    prim_ext_ltx = _ltx(prim_ext)

    latex_integral = rf"\int_{{{l2a_ltx}}}^{{{l2b_ltx}}} \int_{{{l1a_ltx}}}^{{{l1b_ltx}}} ({f_ltx})\,d{vi_ltx}\,d{ve_ltx}"

    pasos = [
        {"titulo": "1. Integral doble planteada", "latex": latex_integral},
        {
            "titulo": f"2. Integrar respecto a {v_int} desde {l1_inf} hasta {l1_sup}",
            "latex": rf"\int_{{{l1a_ltx}}}^{{{l1b_ltx}}} ({f_ltx})\,d{vi_ltx} = ({prim_int_ltx})\Big|_{{{l1a_ltx}}}^{{{l1b_ltx}}}",
        },
        {"titulo": f"3. Simplificar integral respecto a {v_int}", "latex": int_simp_ltx},
        {
            "titulo": f"4. Integrar respecto a {v_ext} desde {l2_inf} hasta {l2_sup}",
            "latex": rf"\int_{{{l2a_ltx}}}^{{{l2b_ltx}}} ({int_simp_ltx})\,d{ve_ltx} = ({prim_ext_ltx})\Big|_{{{l2a_ltx}}}^{{{l2b_ltx}}}",
        },
        {"titulo": "5. Simplificación final", "latex": _ltx(res_simbolico)},
    ]

    return {
        "tipo_titulo": "Integral Doble",
        "latex_integral": latex_integral,
        "resultado_simbolico": _ltx(res_simbolico),
        "resultado": f"{resultado_num:.6f}",
        "detalle_extra": f"Error numérico estimado: {error_num:.2e}",
        "pasos": pasos,
        "_datos_grafica": {
            "fn": lambda gx, gy: fn_num(gx, gy),
            "v_int": v_int,
            "v_ext": v_ext,
            "diff_raw": diff_raw,
            "expr_raw": expr_raw,
            "ext_a": ext_a,
            "ext_b": ext_b,
            "lim_int_a": lambda **kw: _eval_lim_int(fn_int_a, kw[v_ext]),
            "lim_int_b": lambda **kw: _eval_lim_int(fn_int_b, kw[v_ext]),
        },
    }


def calcular_triple(expr_raw: str, diff_raw: str, l1_inf: str, l1_sup: str, l2_inf: str, l2_sup: str, l3_inf: str, l3_sup: str):
    ok_d, diffs, err_d = parsear_diferenciales(diff_raw, 3)
    if not ok_d:
        return {"error": err_d}

    v1, v2, v3 = diffs[0], diffs[1], diffs[2]
    s1, s2, s3 = sp.Symbol(v1, real=True), sp.Symbol(v2, real=True), sp.Symbol(v3, real=True)

    expr_sym, err = _parsear_sympy(expr_raw)
    if err:
        return {"error": err}
    err_v = _validar_variables_expr(expr_sym, {"x", "y", "z"}, "la integral triple")
    if err_v:
        return {"error": err_v}

    l3a_sym, e3a = _parsear_sympy(l3_inf)
    l3b_sym, e3b = _parsear_sympy(l3_sup)
    if e3a or e3b:
        return {"error": e3a or e3b}
    if l3a_sym.free_symbols or l3b_sym.free_symbols:
        return {"error": f"Los límites de la primera ∫ ({v3}) deben ser constantes."}

    l2a_sym, e2a = _parsear_sympy(l2_inf)
    l2b_sym, e2b = _parsear_sympy(l2_sup)
    if e2a or e2b:
        return {"error": e2a or e2b}
    if _validar_variables_expr(l2a_sym, {v3}, f"límite inferior de {v2}") or _validar_variables_expr(l2b_sym, {v3}, f"límite superior de {v2}"):
        return {"error": f"Los límites de la segunda ∫ ({v2}) solo pueden depender de {v3}."}

    l1a_sym, e1a = _parsear_sympy(l1_inf)
    l1b_sym, e1b = _parsear_sympy(l1_sup)
    if e1a or e1b:
        return {"error": e1a or e1b}
    if _validar_variables_expr(l1a_sym, {v3, v2}, f"límite inferior de {v1}") or _validar_variables_expr(l1b_sym, {v3, v2}, f"límite superior de {v1}"):
        return {"error": f"Los límites de la tercera ∫ ({v1}) solo pueden depender de {v3} y {v2}."}

    a3, b3 = _a_num_float(l3a_sym), _a_num_float(l3b_sym)
    if not (np.isfinite(a3) and np.isfinite(b3)) or a3 >= b3:
        return {"error": f"Límites de {v3} inválidos."}

    fn_l2a = sp.lambdify(s3, l2a_sym, modules=["numpy", "math"])
    fn_l2b = sp.lambdify(s3, l2b_sym, modules=["numpy", "math"])
    fn_l1a = sp.lambdify((s3, s2), l1a_sym, modules=["numpy", "math"])
    fn_l1b = sp.lambdify((s3, s2), l1b_sym, modules=["numpy", "math"])

    def _f1(f_l, u3):
        return float(np.asarray(f_l(u3), dtype=float).item())

    def _f2(f_l, u3, u2):
        return float(np.asarray(f_l(u3, u2), dtype=float).item())

    sx, sy, sz = sp.Symbol("x", real=True), sp.Symbol("y", real=True), sp.Symbol("z", real=True)
    fn_xyz = sp.lambdify((sx, sy, sz), expr_sym, modules=["numpy", "math"])

    def _eval_en_orden(val1, val2, val3):
        mapa = {v1: val1, v2: val2, v3: val3}
        return fn_xyz(mapa["x"], mapa["y"], mapa["z"])

    resultado_num, error_num = integrate.tplquad(
        lambda u1, u2, u3: _eval_en_orden(u1, u2, u3),
        a3,
        b3,
        lambda u3: _f1(fn_l2a, u3),
        lambda u3: _f1(fn_l2b, u3),
        lambda u3, u2: _f2(fn_l1a, u3, u2),
        lambda u3, u2: _f2(fn_l1b, u3, u2),
    )

    prim1 = sp.integrate(expr_sym, s1)
    paso1_simp = sp.simplify(prim1.subs(s1, l1b_sym) - prim1.subs(s1, l1a_sym))

    prim2 = sp.integrate(paso1_simp, s2)
    paso2_simp = sp.simplify(prim2.subs(s2, l2b_sym) - prim2.subs(s2, l2a_sym))

    prim3 = sp.integrate(paso2_simp, s3)
    res_simbolico = sp.simplify(prim3.subs(s3, l3b_sym) - prim3.subs(s3, l3a_sym))

    l1a_ltx, l1b_ltx = _fmt_lim_latex(l1_inf, l1a_sym), _fmt_lim_latex(l1_sup, l1b_sym)
    l2a_ltx, l2b_ltx = _fmt_lim_latex(l2_inf, l2a_sym), _fmt_lim_latex(l2_sup, l2b_sym)
    l3a_ltx, l3b_ltx = _fmt_lim_latex(l3_inf, l3a_sym), _fmt_lim_latex(l3_sup, l3b_sym)

    f_ltx = _ltx(expr_sym)
    s1_ltx, s2_ltx, s3_ltx = _ltx(s1), _ltx(s2), _ltx(s3)

    latex_integral = rf"\int_{{{l3a_ltx}}}^{{{l3b_ltx}}} \int_{{{l2a_ltx}}}^{{{l2b_ltx}}} \int_{{{l1a_ltx}}}^{{{l1b_ltx}}} ({f_ltx})\,d{s1_ltx}\,d{s2_ltx}\,d{s3_ltx}"

    pasos = [
        {"titulo": "1. Integral triple planteada", "latex": latex_integral},
        {
            "titulo": f"2. Integrar respecto a {v1} desde {l1_inf} hasta {l1_sup}",
            "latex": rf"({_ltx(prim1)})\Big|_{{{l1a_ltx}}}^{{{l1b_ltx}}} = {_ltx(paso1_simp)}",
        },
        {
            "titulo": f"3. Integrar respecto a {v2} desde {l2_inf} hasta {l2_sup}",
            "latex": rf"({_ltx(prim2)})\Big|_{{{l2a_ltx}}}^{{{l2b_ltx}}} = {_ltx(paso2_simp)}",
        },
        {
            "titulo": f"4. Integrar respecto a {v3} desde {l3_inf} hasta {l3_sup}",
            "latex": rf"({_ltx(prim3)})\Big|_{{{l3a_ltx}}}^{{{l3b_ltx}}} = {_ltx(res_simbolico)}",
        },
        {"titulo": "5. Simplificación final", "latex": _ltx(res_simbolico)},
    ]

    return {
        "tipo_titulo": "Integral Triple",
        "latex_integral": latex_integral,
        "resultado_simbolico": _ltx(res_simbolico),
        "resultado": f"{resultado_num:.6f}",
        "detalle_extra": f"Error numérico estimado: {error_num:.2e}",
        "pasos": pasos,
        "_datos_grafica": {
            "eval_fn_orden": _eval_en_orden,
            "diffs": diffs,
            "diff_raw": diff_raw,
            "expr_raw": expr_raw,
            "a3": a3,
            "b3": b3,
            "lim2_a": lambda **kw: _f1(fn_l2a, kw[v3]),
            "lim2_b": lambda **kw: _f1(fn_l2b, kw[v3]),
            "lim1_a": lambda **kw: _f2(fn_l1a, kw[v3], kw[v2]),
            "lim1_b": lambda **kw: _f2(fn_l1b, kw[v3], kw[v2]),
            "v1": v1,
            "v2": v2,
            "v3": v3,
        },
    }
'''

RECORRIDO_PY = r'''import numpy as np
import plotly.graph_objects as go


def generar_grafica_html(modo: str, datos: dict) -> str:
    if modo == "simple":
        fn = datos["fn"]
        var_name = datos["var_name"]
        expr_raw = datos["expr_raw"]
        a, b = datos["a"], datos["b"]
        resultado = datos["resultado"]

        margen = max(0.25, 0.1 * (b - a))
        xg = np.linspace(a - margen, b + margen, 260)
        yg = fn(xg)
        xa = np.linspace(a, b, 130)
        ya = fn(xa)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=xg, y=yg, mode="lines",
                name=f"f({var_name}) = {expr_raw}",
                line=dict(width=3.5, color="#8c5a3c"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([[xa[0]], xa, [xa[-1]]]),
                y=np.concatenate([[0.0], ya, [0.0]]),
                fill="toself",
                fillcolor="rgba(196, 154, 122, 0.35)",
                line=dict(color="#a67352", width=1.5),
                name=f"Integral = {resultado:.4f}",
            )
        )
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#fffdf9",
            plot_bgcolor="#faf5ef",
            font=dict(color="#4a3525"),
            title=f"Integral Simple: {expr_raw} en [{a:.2f}, {b:.2f}]",
            xaxis_title=var_name,
            yaxis_title=f"f({var_name})",
            height=450,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    if modo == "doble":
        fn = datos["fn"]
        v_int, v_ext = datos["v_int"], datos["v_ext"]
        diff_raw = datos["diff_raw"]
        expr_raw = datos["expr_raw"]
        ext_a, ext_b = datos["ext_a"], datos["ext_b"]
        lim_int_a, lim_int_b = datos["lim_int_a"], datos["lim_int_b"]

        exts = np.linspace(ext_a, ext_b, 48)
        filas = [np.linspace(lim_int_a(**{v_ext: ve}), lim_int_b(**{v_ext: ve}), 48) for ve in exts]
        G_int = np.array(filas)
        G_ext = np.tile(exts[:, None], (1, 48))
        G_z = fn(G_ext, G_int)

        fig = go.Figure(
            data=[
                go.Surface(
                    x=G_ext, y=G_int, z=G_z,
                    colorscale="Peach", opacity=0.94, name=expr_raw,
                )
            ]
        )
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#fffdf9",
            font=dict(color="#4a3525"),
            title=f"Superficie 3D ({diff_raw}): {expr_raw}",
            scene=dict(xaxis_title=v_ext, yaxis_title=v_int, zaxis_title="f"),
            height=500,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    eval_fn_orden = datos["eval_fn_orden"]
    diff_raw = datos["diff_raw"]
    expr_raw = datos["expr_raw"]
    a3, b3 = datos["a3"], datos["b3"]
    lim2_a, lim2_b = datos["lim2_a"], datos["lim2_b"]
    lim1_a, lim1_b = datos["lim1_a"], datos["lim1_b"]
    v1, v2, v3 = datos["v1"], datos["v2"], datos["v3"]

    px, py, pz, pw = [], [], [], []
    for c3 in np.linspace(a3, b3, 5):
        a2, b2 = lim2_a(**{v3: c3}), lim2_b(**{v3: c3})
        for c2 in np.linspace(a2, b2, 5):
            a1, b1 = lim1_a(**{v3: c3, v2: c2}), lim1_b(**{v3: c3, v2: c2})
            for c1 in np.linspace(a1, b1, 5):
                mapa = {v1: c1, v2: c2, v3: c3}
                px.append(mapa.get("x", c3))
                py.append(mapa.get("y", c2))
                pz.append(mapa.get("z", c1))
                pw.append(eval_fn_orden(c1, c2, c3))

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=px, y=py, z=pz,
                mode="markers",
                marker=dict(size=5.5, color=pw, colorscale="Peach", showscale=True, colorbar=dict(title="f")),
                name=expr_raw,
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#fffdf9",
        font=dict(color="#4a3525"),
        title=f"Región y Recorrido 3D ({diff_raw}): {expr_raw}",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        height=500,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")
'''

INDEX_HTML = r'''<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>IntegralesMatematicas</title>
  <script>
    window.MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']]
      }
    };
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
      background: linear-gradient(155deg, #fbf6f0 0%, #f3e8dc 55%, #ede0d1 100%);
      color: #3f2e21;
      margin: 0;
      padding: 28px 18px;
      min-height: 100vh;
    }
    .container { max-width: 1020px; margin: auto; }
    .header { margin-bottom: 22px; }
    h1 {
      margin: 0 0 6px 0;
      font-size: 32px;
      letter-spacing: -0.5px;
      color: #5c3d28;
    }
    .sub { color: #7a5c46; margin: 0; font-size: 15px; }
    .card {
      background: #fffdfb;
      border: 1px solid #e4d2c0;
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 22px;
      box-shadow: 0 10px 26px rgba(120, 86, 58, 0.09);
    }
    .menu-tabs { display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0 18px 0; }
    .tab-btn {
      padding: 10px 18px;
      border-radius: 12px;
      border: 1px solid #dcc4ad;
      background: #faf3eb;
      color: #6b4c35;
      cursor: pointer;
      font-weight: 600;
      font-size: 14px;
      transition: all 0.15s ease;
    }
    .tab-btn:hover { border-color: #b5835a; background: #f3e5d6; }
    .tab-btn.active {
      background: #b5835a;
      border-color: #9e6d46;
      color: #ffffff;
      box-shadow: 0 4px 12px rgba(158, 109, 70, 0.25);
    }
    .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }
    label { display: block; font-size: 13px; color: #593f2b; margin-top: 12px; margin-bottom: 6px; font-weight: 600; }
    input {
      width: 100%;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid #d9c1aa;
      background: #fffaf5;
      color: #3f2e21;
      font-size: 14px;
      outline: none;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    input:focus { border-color: #a67352; box-shadow: 0 0 0 3px rgba(166, 115, 82, 0.16); background: #ffffff; }

    .int-row {
      display: flex;
      align-items: center;
      gap: 16px;
      margin-top: 16px;
      padding: 16px;
      border-radius: 14px;
      background: #faf3eb;
      border: 1px solid #e6d4c3;
      flex-wrap: wrap;
    }
    .int-glyphs {
      display: inline-flex;
      align-items: center;
      gap: 2px;
      font-size: 34px;
      line-height: 1;
      font-weight: 700;
      color: #9c6644;
      background: #ffffff;
      border: 1px solid #e2cbb5;
      border-radius: 12px;
      padding: 6px 12px;
      min-width: 56px;
      justify-content: center;
    }
    .int-fields { flex: 1; min-width: 240px; }

    .submit-btn {
      margin-top: 22px;
      padding: 12px 26px;
      background: #9c6644;
      color: #ffffff;
      font-weight: 700;
      border: none;
      border-radius: 12px;
      cursor: pointer;
      font-size: 15px;
      box-shadow: 0 5px 14px rgba(156, 102, 68, 0.25);
      transition: transform 0.12s ease, background 0.15s ease;
    }
    .submit-btn:hover { background: #855335; transform: translateY(-1px); }
    .loading-text {
      margin-left: 12px;
      font-size: 13px;
      color: #855335;
      font-weight: 600;
    }
    .error-box {
      background: #fdf2f0;
      border: 1px solid #e29587;
      color: #8c2d19;
      padding: 14px 16px;
      border-radius: 12px;
      margin-bottom: 20px;
    }
    .paso-item {
      background: #faf4ed;
      border-left: 4px solid #b5835a;
      border-radius: 10px;
      padding: 12px 16px;
      margin-bottom: 10px;
      opacity: 0;
      transform: translateY(6px);
      transition: opacity 0.35s ease, transform 0.35s ease;
      overflow-x: auto;
    }
    .paso-item.visible { opacity: 1; transform: translateY(0); }
    .paso-titulo { font-size: 14px; font-weight: 700; color: #6e472b; margin-bottom: 6px; }
    .resultado-box {
      background: #faf4ed;
      border: 1px solid #dec7b0;
      border-radius: 14px;
      padding: 18px;
      margin-top: 12px;
    }
    .resultado-num { font-size: 28px; font-weight: 800; color: #7f4f24; margin: 8px 0 4px 0; }
    .grafica { margin-top: 12px; border-radius: 12px; overflow: hidden; border: 1px solid #e2cbb5; }
    .hidden { display: none; }
  </style>
</head>
<body>
  <div class="header">
      <h1>IntegralesMatematicas</h1>
    </div>

    <form method="POST" class="card" onsubmit="mostrarCargando()">
      <label>1. Selecciona el tipo de integral:</label>
      <div class="menu-tabs">
        <button type="button" class="tab-btn {% if modo == 'simple' %}active{% endif %}" onclick="cambiarModo('simple')">Integral Simple</button>
        <button type="button" class="tab-btn {% if modo == 'doble' %}active{% endif %}" onclick="cambiarModo('doble')">Integral Doble</button>
        <button type="button" class="tab-btn {% if modo == 'triple' %}active{% endif %}" onclick="cambiarModo('triple')">Integral Triple</button>
      </div>
      <input type="hidden" id="modo" name="modo" value="{{ modo }}" />

      <div id="sec-simple" class="{% if modo != 'simple' %}hidden{% endif %}">
        <div class="grid-2">
          <div><label>Expresión / Integrando:</label><input type="text" name="expr_simple" value="{{ expr_simple }}" /></div>
          <div><label>Diferencial (ej. dx, dy, dz o dteta):</label><input type="text" name="diff_simple" value="{{ diff_simple }}" /></div>
        </div>
        <div class="int-row">
          <div class="int-glyphs"><span>∫</span></div>
          <div class="int-fields">
            <div class="grid-2">
              <div><label style="margin-top:0;">Límite inferior:</label><input type="text" name="a_simple" value="{{ a_simple }}" /></div>
              <div><label style="margin-top:0;">Límite superior:</label><input type="text" name="b_simple" value="{{ b_simple }}" /></div>
            </div>
          </div>
        </div>
      </div>

      <div id="sec-doble" class="{% if modo != 'doble' %}hidden{% endif %}">
        <div class="grid-2">
          <div><label>Expresión / Integrando:</label><input type="text" name="expr_doble" value="{{ expr_doble }}" /></div>
          <div><label>Diferenciales (ej. dy dx o dx dy):</label><input type="text" name="diff_doble" value="{{ diff_doble }}" /></div>
        </div>

        <div class="int-row">
          <div class="int-glyphs"><span>∫</span></div>
          <div class="int-fields">
            <div class="grid-2">
              <div><label style="margin-top:0;">Límite inferior:</label><input type="text" name="l2_inf_doble" value="{{ l2_inf_doble }}" /></div>
              <div><label style="margin-top:0;">Límite superior:</label><input type="text" name="l2_sup_doble" value="{{ l2_sup_doble }}" /></div>
            </div>
          </div>
        </div>

        <div class="int-row">
          <div class="int-glyphs"><span>∫</span><span>∫</span></div>
          <div class="int-fields">
            <div class="grid-2">
              <div><label style="margin-top:0;">Límite inferior:</label><input type="text" name="l1_inf_doble" value="{{ l1_inf_doble }}" /></div>
              <div><label style="margin-top:0;">Límite superior:</label><input type="text" name="l1_sup_doble" value="{{ l1_sup_doble }}" /></div>
            </div>
          </div>
        </div>
      </div>

      <div id="sec-triple" class="{% if modo != 'triple' %}hidden{% endif %}">
        <div class="grid-2">
          <div><label>Expresión / Integrando:</label><input type="text" name="expr_triple" value="{{ expr_triple }}" /></div>
          <div><label>Diferenciales (ej. dz dy dx o dx dy dz):</label><input type="text" name="diff_triple" value="{{ diff_triple }}" /></div>
        </div>

        <div class="int-row">
          <div class="int-glyphs"><span>∫</span></div>
          <div class="int-fields">
            <div class="grid-2">
              <div><label style="margin-top:0;">Límite inferior:</label><input type="text" name="l3_inf_triple" value="{{ l3_inf_triple }}" /></div>
              <div><label style="margin-top:0;">Límite superior:</label><input type="text" name="l3_sup_triple" value="{{ l3_sup_triple }}" /></div>
            </div>
          </div>
        </div>

        <div class="int-row">
          <div class="int-glyphs"><span>∫</span><span>∫</span></div>
          <div class="int-fields">
            <div class="grid-2">
              <div><label style="margin-top:0;">Límite inferior:</label><input type="text" name="l2_inf_triple" value="{{ l2_inf_triple }}" /></div>
              <div><label style="margin-top:0;">Límite superior:</label><input type="text" name="l2_sup_triple" value="{{ l2_sup_triple }}" /></div>
            </div>
          </div>
        </div>

        <div class="int-row">
          <div class="int-glyphs"><span>∫</span><span>∫</span><span>∫</span></div>
          <div class="int-fields">
            <div class="grid-2">
              <div><label style="margin-top:0;">Límite inferior:</label><input type="text" name="l1_inf_triple" value="{{ l1_inf_triple }}" /></div>
              <div><label style="margin-top:0;">Límite superior:</label><input type="text" name="l1_sup_triple" value="{{ l1_sup_triple }}" /></div>
            </div>
          </div>
        </div>
      </div>

      <button type="submit" id="btn-calcular" class="submit-btn">Calcular y Graficar</button>
      <span id="txt-cargando" class="loading-text hidden">Calculando desarrollo, resultado y gráfica...</span>
    </form>

    {% if error %}
      <div class="error-box"><strong>Operación no reconocida o no soportada:</strong> {{ error }}</div>
    {% endif %}

    {% if resultado_data %}
      <div class="card">
        <h2>Desarrollo</h2>
        <div id="pasos-contenedor">
          {% for p in resultado_data.pasos %}
            <div class="paso-item">
              <div class="paso-titulo">{{ p.titulo }}</div>
              <div>$${{ p.latex }}$$</div>
            </div>
          {% endfor %}
        </div>

        <h2 style="margin-top:22px;">Resultado</h2>
        <div class="resultado-box">
          <div style="font-size:17px; overflow-x:auto;">$${{ resultado_data.latex_integral }}$$</div>
          <div class="resultado-num">$${{ resultado_data.resultado_simbolico }}$$ &nbsp;({{ resultado_data.resultado }})</div>
          <div style="color:#7a5c46; font-size:13px;">{{ resultado_data.detalle_extra }}</div>
        </div>

        <h2 style="margin-top:22px;">Gráfica</h2>
        <div class="grafica">{{ resultado_data.grafica_html | safe }}</div>
      </div>
    {% endif %}
  </div>

  <script>
    function cambiarModo(nuevoModo) {
      document.getElementById("modo").value = nuevoModo;
      document.getElementById("sec-simple").classList.toggle("hidden", nuevoModo !== "simple");
      document.getElementById("sec-doble").classList.toggle("hidden", nuevoModo !== "doble");
      document.getElementById("sec-triple").classList.toggle("hidden", nuevoModo !== "triple");
      const btns = document.querySelectorAll(".tab-btn");
      btns.forEach((btn, idx) => btn.classList.toggle("active", ["simple", "doble", "triple"][idx] === nuevoModo));
    }

    function mostrarCargando() {
      const btn = document.getElementById("btn-calcular");
      const txt = document.getElementById("txt-cargando");
      if (btn) { btn.disabled = true; btn.style.opacity = "0.75"; }
      if (txt) { txt.classList.remove("hidden"); }
    }

    document.addEventListener("DOMContentLoaded", () => {
      const pasos = document.querySelectorAll(".paso-item");
      pasos.forEach((el, idx) => {
        setTimeout(() => el.classList.add("visible"), 160 * (idx + 1));
      });
    });
  </script>
</body>
</html>
'''

APP_PY = r'''from pathlib import Path
from flask import Flask, render_template_string, request
from calculo.integrales import calcular_doble, calcular_simple, calcular_triple
from graficos.recorrido import generar_grafica_html

BASE_DIR = Path(__file__).resolve().parent
PLANTILLA_HTML = (BASE_DIR / "interfaz" / "index.html").read_text(encoding="utf-8")

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
@app.route("/IntegralesMatematicas", methods=["GET", "POST"])
def index():
    modo = "simple"
    expr_simple, diff_simple, a_simple, b_simple = "cos(teta)", "dteta", "0", "pi/2"
    expr_doble, diff_doble = "x * sen(y)", "dy dx"
    l1_inf_doble, l1_sup_doble, l2_inf_doble, l2_sup_doble = "0", "x", "0", "pi"
    expr_triple, diff_triple = "z * e^(x + y) * cos(z)", "dz dy dx"
    l1_inf_triple, l1_sup_triple = "0", "x + y"
    l2_inf_triple, l2_sup_triple = "0", "x"
    l3_inf_triple, l3_sup_triple = "0", "1"
    error = None
    resultado_data = None

    if request.method == "POST":
        modo = request.form.get("modo", "simple").strip().lower()
        expr_simple = request.form.get("expr_simple", expr_simple)
        diff_simple = request.form.get("diff_simple", diff_simple)
        a_simple = request.form.get("a_simple", a_simple)
        b_simple = request.form.get("b_simple", b_simple)

        expr_doble = request.form.get("expr_doble", expr_doble)
        diff_doble = request.form.get("diff_doble", diff_doble)
        l1_inf_doble = request.form.get("l1_inf_doble", l1_inf_doble)
        l1_sup_doble = request.form.get("l1_sup_doble", l1_sup_doble)
        l2_inf_doble = request.form.get("l2_inf_doble", l2_inf_doble)
        l2_sup_doble = request.form.get("l2_sup_doble", l2_sup_doble)

        expr_triple = request.form.get("expr_triple", expr_triple)
        diff_triple = request.form.get("diff_triple", diff_triple)
        l1_inf_triple = request.form.get("l1_inf_triple", l1_inf_triple)
        l1_sup_triple = request.form.get("l1_sup_triple", l1_sup_triple)
        l2_inf_triple = request.form.get("l2_inf_triple", l2_inf_triple)
        l2_sup_triple = request.form.get("l2_sup_triple", l2_sup_triple)
        l3_inf_triple = request.form.get("l3_inf_triple", l3_inf_triple)
        l3_sup_triple = request.form.get("l3_sup_triple", l3_sup_triple)

        try:
            if modo == "simple":
                res = calcular_simple(expr_simple, diff_simple, a_simple, b_simple)
            elif modo == "doble":
                res = calcular_doble(
                    expr_doble, diff_doble, l1_inf_doble, l1_sup_doble, l2_inf_doble, l2_sup_doble
                )
            elif modo == "triple":
                res = calcular_triple(
                    expr_triple, diff_triple, l1_inf_triple, l1_sup_triple,
                    l2_inf_triple, l2_sup_triple, l3_inf_triple, l3_sup_triple,
                )
            else:
                res = {"error": f"El tipo '{modo}' no es válido."}
        except Exception as exc:
            res = {"error": f"No se pudo evaluar la integral: {exc}"}

        if "error" in res:
            error = res["error"]
        else:
            datos_graf = res.pop("_datos_grafica")
            res["grafica_html"] = generar_grafica_html(modo, datos_graf)
            resultado_data = res

    return render_template_string(
        PLANTILLA_HTML,
        modo=modo,
        expr_simple=expr_simple, diff_simple=diff_simple, a_simple=a_simple, b_simple=b_simple,
        expr_doble=expr_doble, diff_doble=diff_doble,
        l1_inf_doble=l1_inf_doble, l1_sup_doble=l1_sup_doble, l2_inf_doble=l2_inf_doble, l2_sup_doble=l2_sup_doble,
        expr_triple=expr_triple, diff_triple=diff_triple,
        l1_inf_triple=l1_inf_triple, l1_sup_triple=l1_sup_triple,
        l2_inf_triple=l2_inf_triple, l2_sup_triple=l2_sup_triple,
        l3_inf_triple=l3_inf_triple, l3_sup_triple=l3_sup_triple,
        error=error, resultado_data=resultado_data,
    )


if __name__ == "__main__":
    print("Abre en tu navegador: http://127.0.0.1:5000/IntegralesMatematicas")
    app.run(debug=False, port=5000)
'''


def main():
    for folder in ("calculo", "graficos", "interfaz"):
        (BASE / folder).mkdir(parents=True, exist_ok=True)
    (BASE / "calculo" / "__init__.py").write_text("", encoding="utf-8")
    (BASE / "graficos" / "__init__.py").write_text("", encoding="utf-8")
    (BASE / "calculo" / "integrales.py").write_text(INTEGRALES_PY, encoding="utf-8")
    (BASE / "graficos" / "recorrido.py").write_text(RECORRIDO_PY, encoding="utf-8")
    (BASE / "interfaz" / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (BASE / "app.py").write_text(APP_PY, encoding="utf-8")
    print("Todo listo.")


if __name__ == "__main__":
    main()