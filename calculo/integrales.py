import re
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
            return False, "", f"Expresión no soportada ('{token}')."
    limpio = re.sub(r"\bteta\b", "theta", limpio)
    limpio = re.sub(r"\bsen\b", "sin", limpio)
    limpio = re.sub(r"\bln\b", "log", limpio)
    return True, limpio, ""


def _simbolos_base():
    x, y, z, r, rho, phi, theta, t = sp.symbols("x y z r rho phi theta t", real=True)
    return {
        "x": x, "y": y, "z": z, "r": r, "rho": rho, "ro": rho,
        "phi": phi, "fi": phi, "theta": theta, "t": t,
        "pi": sp.pi, "e": sp.E,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
        "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
        "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
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
    texto = (
        str(diff_raw).strip().lower()
        .replace("teta", "theta")
        .replace("ro", "rho")
        .replace("fi", "phi")
    )
    matches = re.findall(r"\bd(theta|rho|phi|[a-z])\b", texto)
    if not matches:
        compacto = re.sub(r"\s+", "", texto)
        matches = re.findall(r"d(theta|rho|phi|[a-z])", compacto)
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

    v_ltx = _ltx(sym_var)
    f_ltx = _ltx(expr_sym)
    primitiva = sp.integrate(expr_sym, sym_var)
    prim_ltx = _ltx(primitiva)
    fn_num = sp.lambdify(sym_var, expr_sym, modules=["numpy", "math"])

    a_limpio = str(a_raw or "").strip()
    b_limpio = str(b_raw or "").strip()

    if not a_limpio and not b_limpio:
        fn_prim = sp.lambdify(sym_var, primitiva, modules=["numpy", "math"])
        latex_integral = rf"\int ({f_ltx})\,d{v_ltx}"
        pasos = [
            {"titulo": "1. Integral indefinida planteada", "latex": latex_integral},
            {
                "titulo": f"2. Antiderivada respecto a {var_name} (+ C)",
                "latex": rf"\int ({f_ltx})\,d{v_ltx} = {prim_ltx} + C",
            },
        ]
        return {
            "tipo_titulo": "Integral Indefinida",
            "latex_integral": latex_integral,
            "resultado_simbolico": rf"{prim_ltx} + C",
            "resultado": "Antiderivada simbólica",
            "detalle_extra": "Integral sin límites (se agrega la constante de integración C)",
            "pasos": pasos,
            "_datos_grafica": {
                "es_indefinida": True,
                "fn": fn_num,
                "fn_prim": fn_prim,
                "var_name": var_name,
                "expr_raw": expr_raw,
            },
        }

    if not a_limpio or not b_limpio:
        return {"error": "Ingresa ambos límites para integral definida, o déjalos vacíos para integral indefinida."}

    a_sym, ea = _parsear_sympy(a_limpio)
    b_sym, eb = _parsear_sympy(b_limpio)
    if ea or eb:
        return {"error": ea or eb}
    if a_sym.free_symbols or b_sym.free_symbols:
        return {"error": "En una integral simple los límites deben ser constantes."}

    a_num, b_num = _a_num_float(a_sym), _a_num_float(b_sym)
    if not (np.isfinite(a_num) and np.isfinite(b_num)) or a_num >= b_num:
        return {"error": "Límites inválidos: se requiere límite inferior < superior."}

    resultado_num, error_num = integrate.quad(fn_num, a_num, b_num)
    eval_sup = sp.expand(primitiva.subs(sym_var, b_sym))
    eval_inf = sp.expand(primitiva.subs(sym_var, a_sym))
    res_simbolico = sp.simplify(eval_sup - eval_inf)

    a_ltx = _fmt_lim_latex(a_limpio, a_sym)
    b_ltx = _fmt_lim_latex(b_limpio, b_sym)
    latex_integral = rf"\int_{{{a_ltx}}}^{{{b_ltx}}} ({f_ltx})\,d{v_ltx}"

    pasos = [
        {"titulo": "1. Integral planteada", "latex": latex_integral},
        {"titulo": f"2. Antiderivada respecto a {var_name}", "latex": rf"F({v_ltx}) = {prim_ltx}"},
        {
            "titulo": f"3. Evaluar desde {a_limpio} hasta {b_limpio}",
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
            "es_indefinida": False,
            "fn": fn_num,
            "var_name": var_name,
            "expr_raw": expr_raw,
            "a": a_num,
            "b": b_num,
            "resultado": resultado_num,
        },
    }


def calcular_doble(expr_raw: str, diff_raw: str, l1_inf: str, l1_sup: str, l2_inf: str, l2_sup: str, coord: str = "cartesianas"):
    ok_d, diffs, err_d = parsear_diferenciales(diff_raw, 2)
    if not ok_d:
        return {"error": err_d}

    v_int, v_ext = diffs[0], diffs[1]
    s_int = sp.Symbol(v_int, real=True)
    s_ext = sp.Symbol(v_ext, real=True)

    expr_sym, err = _parsear_sympy(expr_raw)
    if err:
        return {"error": err}

    if coord == "polares":
        permitidas = {"r", "theta"}
        jacobiano = sp.Symbol("r", real=True)
        jac_texto = "Jacobiano polar aplicado: J = r"
    else:
        permitidas = {"x", "y"}
        jacobiano = sp.Integer(1)
        jac_texto = "Coordenadas cartesianas (J = 1)"

    err_v = _validar_variables_expr(expr_sym, permitidas, f"la integral doble ({coord})")
    if err_v:
        return {"error": err_v}

    integrando_total = sp.expand(expr_sym * jacobiano)
    f_ltx = _ltx(integrando_total)
    vi_ltx = _ltx(s_int)
    ve_ltx = _ltx(s_ext)

    lims = [str(l1_inf or "").strip(), str(l1_sup or "").strip(), str(l2_inf or "").strip(), str(l2_sup or "").strip()]

    # MODO INDEFINIDA: Todos los límites vacíos
    if all(x == "" for x in lims):
        prim_int = sp.integrate(integrando_total, s_int)
        prim_ext = sp.integrate(prim_int, s_ext)
        fn_prim_doble = sp.lambdify((s_ext, s_int), prim_ext, modules=["numpy", "math"])

        latex_integral = rf"\iint ({f_ltx})\,d{vi_ltx}\,d{ve_ltx}"
        pasos = [
            {"titulo": f"1. Integral doble indefinida ({jac_texto})", "latex": latex_integral},
            {
                "titulo": f"2. Primera antiderivada respecto a {v_int}",
                "latex": rf"\int ({f_ltx})\,d{vi_ltx} = {_ltx(prim_int)} + C_1({ve_ltx})",
            },
            {
                "titulo": f"3. Segunda antiderivada respecto a {v_ext}",
                "latex": rf"\int \left({_ltx(prim_int)}\right)d{ve_ltx} = {_ltx(prim_ext)} + C",
            },
        ]
        ext_a, ext_b = (0.0, 2 * np.pi) if v_ext == "theta" else (-3.0, 3.0)
        int_a, int_b = (0.0, 3.0) if v_int == "r" else (-3.0, 3.0)

        return {
            "tipo_titulo": f"Integral Doble Indefinida ({coord.capitalize()})",
            "latex_integral": latex_integral,
            "resultado_simbolico": rf"{_ltx(prim_ext)} + C",
            "resultado": "Antiderivada multivariable",
            "detalle_extra": f"{jac_texto} | Superficie primitiva graficada en dominio estándar",
            "pasos": pasos,
            "_datos_grafica": {
                "fn": fn_prim_doble,
                "v_int": v_int,
                "v_ext": v_ext,
                "coord": coord,
                "diff_raw": diff_raw,
                "expr_raw": f"Antiderivada de {expr_raw}",
                "ext_a": ext_a,
                "ext_b": ext_b,
                "lim_int_a": lambda **kw: int_a,
                "lim_int_b": lambda **kw: int_b,
            },
        }

    if any(x == "" for x in lims):
        return {"error": "Para integral doble definida completa los 4 límites, o déjalos todos vacíos para integral indefinida."}

    l2a_sym, e2a = _parsear_sympy(lims[2])
    l2b_sym, e2b = _parsear_sympy(lims[3])
    if e2a or e2b:
        return {"error": e2a or e2b}
    if l2a_sym.free_symbols or l2b_sym.free_symbols:
        return {"error": f"Los límites de la primera ∫ ({v_ext}) deben ser constantes."}

    l1a_sym, e1a = _parsear_sympy(lims[0])
    l1b_sym, e1b = _parsear_sympy(lims[1])
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

    fn_num = sp.lambdify((s_ext, s_int), integrando_total, modules=["numpy", "math"])

    prim_int = sp.integrate(integrando_total, s_int)
    interior_simplificado = sp.expand(prim_int.subs(s_int, l1b_sym) - prim_int.subs(s_int, l1a_sym))

    prim_ext = sp.integrate(interior_simplificado, s_ext)
    res_simbolico = sp.simplify(prim_ext.subs(s_ext, l2b_sym) - prim_ext.subs(s_ext, l2a_sym))

    try:
        resultado_num = float(sp.N(res_simbolico))
        error_texto = "Cálculo simbólico exacto"
    except Exception:
        resultado_num, error_num = integrate.dblquad(
            lambda u_int, u_ext: fn_num(u_ext, u_int),
            ext_a,
            ext_b,
            lambda u_ext: _eval_lim_int(fn_int_a, u_ext),
            lambda u_ext: _eval_lim_int(fn_int_b, u_ext),
        )
        error_texto = f"Error numérico: {error_num:.2e}"

    l1a_ltx = _fmt_lim_latex(lims[0], l1a_sym)
    l1b_ltx = _fmt_lim_latex(lims[1], l1b_sym)
    l2a_ltx = _fmt_lim_latex(lims[2], l2a_sym)
    l2b_ltx = _fmt_lim_latex(lims[3], l2b_sym)

    prim_int_ltx = _ltx(prim_int)
    int_simp_ltx = _ltx(interior_simplificado)
    prim_ext_ltx = _ltx(prim_ext)

    latex_integral = rf"\int_{{{l2a_ltx}}}^{{{l2b_ltx}}} \int_{{{l1a_ltx}}}^{{{l1b_ltx}}} ({f_ltx})\,d{vi_ltx}\,d{ve_ltx}"

    pasos = [
        {"titulo": f"1. Integral doble ({jac_texto})", "latex": latex_integral},
        {
            "titulo": f"2. Integrar respecto a {v_int} desde {lims[0]} hasta {lims[1]}",
            "latex": rf"\int_{{{l1a_ltx}}}^{{{l1b_ltx}}} ({f_ltx})\,d{vi_ltx} = ({prim_int_ltx})\Big|_{{{l1a_ltx}}}^{{{l1b_ltx}}}",
        },
        {"titulo": f"3. Expandir integral respecto a {v_int}", "latex": int_simp_ltx},
        {
            "titulo": f"4. Integrar respecto a {v_ext} desde {lims[2]} hasta {lims[3]}",
            "latex": rf"\int_{{{l2a_ltx}}}^{{{l2b_ltx}}} ({int_simp_ltx})\,d{ve_ltx} = ({prim_ext_ltx})\Big|_{{{l2a_ltx}}}^{{{l2b_ltx}}}",
        },
        {"titulo": "5. Resultado simplificado", "latex": _ltx(res_simbolico)},
    ]

    return {
        "tipo_titulo": f"Integral Doble ({coord.capitalize()})",
        "latex_integral": latex_integral,
        "resultado_simbolico": _ltx(res_simbolico),
        "resultado": f"{resultado_num:.6f}",
        "detalle_extra": f"{jac_texto} | {error_texto}",
        "pasos": pasos,
        "_datos_grafica": {
            "fn": fn_num,
            "v_int": v_int,
            "v_ext": v_ext,
            "coord": coord,
            "diff_raw": diff_raw,
            "expr_raw": expr_raw,
            "ext_a": ext_a,
            "ext_b": ext_b,
            "lim_int_a": lambda **kw: _eval_lim_int(fn_int_a, kw[v_ext]),
            "lim_int_b": lambda **kw: _eval_lim_int(fn_int_b, kw[v_ext]),
        },
    }


def calcular_triple(expr_raw: str, diff_raw: str, l1_inf: str, l1_sup: str, l2_inf: str, l2_sup: str, l3_inf: str, l3_sup: str, coord: str = "cartesianas"):
    ok_d, diffs, err_d = parsear_diferenciales(diff_raw, 3)
    if not ok_d:
        return {"error": err_d}

    v1, v2, v3 = diffs[0], diffs[1], diffs[2]
    s1, s2, s3 = sp.Symbol(v1, real=True), sp.Symbol(v2, real=True), sp.Symbol(v3, real=True)

    expr_sym, err = _parsear_sympy(expr_raw)
    if err:
        return {"error": err}

    if coord == "cilindricas":
        permitidas = {"r", "theta", "z"}
        jacobiano = sp.Symbol("r", real=True)
        jac_texto = "Jacobiano cilíndrico aplicado: J = r"
    elif coord == "esfericas":
        permitidas = {"rho", "phi", "theta"}
        jacobiano = sp.Symbol("rho", real=True)**2 * sp.sin(sp.Symbol("phi", real=True))
        jac_texto = "Jacobiano esférico aplicado: J = ρ²·sen(φ)"
    else:
        permitidas = {"x", "y", "z"}
        jacobiano = sp.Integer(1)
        jac_texto = "Coordenadas cartesianas (J = 1)"

    err_v = _validar_variables_expr(expr_sym, permitidas, f"la integral triple ({coord})")
    if err_v:
        return {"error": err_v}

    integrando_total = sp.expand(expr_sym * jacobiano)
    f_ltx = _ltx(integrando_total)
    s1_ltx, s2_ltx, s3_ltx = _ltx(s1), _ltx(s2), _ltx(s3)

    lims = [
        str(l1_inf or "").strip(), str(l1_sup or "").strip(),
        str(l2_inf or "").strip(), str(l2_sup or "").strip(),
        str(l3_inf or "").strip(), str(l3_sup or "").strip(),
    ]

    # MODO INDEFINIDA: Todos los límites vacíos
    if all(x == "" for x in lims):
        prim1 = sp.integrate(integrando_total, s1)
        prim2 = sp.integrate(prim1, s2)
        prim3 = sp.integrate(prim2, s3)
        fn_prim3 = sp.lambdify((s1, s2, s3), prim3, modules=["numpy", "math"])

        latex_integral = rf"\iiint ({f_ltx})\,d{s1_ltx}\,d{s2_ltx}\,d{s3_ltx}"
        pasos = [
            {"titulo": f"1. Integral triple indefinida ({jac_texto})", "latex": latex_integral},
            {"titulo": f"2. Integrar respecto a {v1}", "latex": rf"\int ({f_ltx})\,d{s1_ltx} = {_ltx(prim1)}"},
            {"titulo": f"3. Integrar respecto a {v2}", "latex": rf"\int \left({_ltx(prim1)}\right)d{s2_ltx} = {_ltx(prim2)}"},
            {"titulo": f"4. Integrar respecto a {v3} (+ C)", "latex": rf"\int \left({_ltx(prim2)}\right)d{s3_ltx} = {_ltx(prim3)} + C"},
        ]

        def _rango_def(var):
            if var == "theta":
                return 0.0, 2 * np.pi
            if var == "phi":
                return 0.0, np.pi
            if var in ("r", "rho"):
                return 0.0, 2.0
            return -2.0, 2.0

        a3, b3 = _rango_def(v3)
        a2, b2 = _rango_def(v2)
        a1, b1 = _rango_def(v1)

        return {
            "tipo_titulo": f"Integral Triple Indefinida ({coord.capitalize()})",
            "latex_integral": latex_integral,
            "resultado_simbolico": rf"{_ltx(prim3)} + C",
            "resultado": "Antiderivada triple",
            "detalle_extra": f"{jac_texto} | Campo escalar primitivo graficado en dominio estándar",
            "pasos": pasos,
            "_datos_grafica": {
                "eval_fn_orden": fn_prim3,
                "coord": coord,
                "diffs": diffs,
                "diff_raw": diff_raw,
                "expr_raw": f"Primitiva de {expr_raw}",
                "a3": a3, "b3": b3,
                "lim2_a": lambda **kw: a2, "lim2_b": lambda **kw: b2,
                "lim1_a": lambda **kw: a1, "lim1_b": lambda **kw: b1,
                "v1": v1, "v2": v2, "v3": v3,
            },
        }

    if any(x == "" for x in lims):
        return {"error": "Para integral triple definida completa los 6 límites, o déjalos todos vacíos para integral indefinida."}

    l3a_sym, e3a = _parsear_sympy(lims[4])
    l3b_sym, e3b = _parsear_sympy(lims[5])
    if e3a or e3b:
        return {"error": e3a or e3b}
    if l3a_sym.free_symbols or l3b_sym.free_symbols:
        return {"error": f"Los límites de la primera ∫ ({v3}) deben ser constantes."}

    l2a_sym, e2a = _parsear_sympy(lims[2])
    l2b_sym, e2b = _parsear_sympy(lims[3])
    if e2a or e2b:
        return {"error": e2a or e2b}
    if _validar_variables_expr(l2a_sym, {v3}, f"límite inferior de {v2}") or _validar_variables_expr(l2b_sym, {v3}, f"límite superior de {v2}"):
        return {"error": f"Los límites de la segunda ∫ ({v2}) solo pueden depender de {v3}."}

    l1a_sym, e1a = _parsear_sympy(lims[0])
    l1b_sym, e1b = _parsear_sympy(lims[1])
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

    fn_orden = sp.lambdify((s1, s2, s3), integrando_total, modules=["numpy", "math"])

    prim1 = sp.integrate(integrando_total, s1)
    paso1_simp = sp.expand(prim1.subs(s1, l1b_sym) - prim1.subs(s1, l1a_sym))

    prim2 = sp.integrate(paso1_simp, s2)
    paso2_simp = sp.expand(prim2.subs(s2, l2b_sym) - prim2.subs(s2, l2a_sym))

    prim3 = sp.integrate(paso2_simp, s3)
    res_simbolico = sp.expand(prim3.subs(s3, l3b_sym) - prim3.subs(s3, l3a_sym))

    try:
        resultado_num = float(sp.N(res_simbolico))
        if not np.isfinite(resultado_num):
            raise ValueError("No finito")
        error_texto = "Cálculo simbólico exacto"
    except Exception:
        resultado_num, err_n = integrate.tplquad(
            lambda u1, u2, u3: fn_orden(u1, u2, u3),
            a3,
            b3,
            lambda u3: _f1(fn_l2a, u3),
            lambda u3: _f1(fn_l2b, u3),
            lambda u3, u2: _f2(fn_l1a, u3, u2),
            lambda u3, u2: _f2(fn_l1b, u3, u2),
            epsabs=1e-2,
            epsrel=1e-2,
        )
        error_texto = f"Error numérico: {err_n:.2e}"

    l1a_ltx, l1b_ltx = _fmt_lim_latex(lims[0], l1a_sym), _fmt_lim_latex(lims[1], l1b_sym)
    l2a_ltx, l2b_ltx = _fmt_lim_latex(lims[2], l2a_sym), _fmt_lim_latex(lims[3], l2b_sym)
    l3a_ltx, l3b_ltx = _fmt_lim_latex(lims[4], l3a_sym), _fmt_lim_latex(lims[5], l3b_sym)

    latex_integral = rf"\int_{{{l3a_ltx}}}^{{{l3b_ltx}}} \int_{{{l2a_ltx}}}^{{{l2b_ltx}}} \int_{{{l1a_ltx}}}^{{{l1b_ltx}}} ({f_ltx})\,d{s1_ltx}\,d{s2_ltx}\,d{s3_ltx}"

    pasos = [
        {"titulo": f"1. Integral triple ({jac_texto})", "latex": latex_integral},
        {
            "titulo": f"2. Integrar respecto a {v1} desde {lims[0]} hasta {lims[1]}",
            "latex": rf"({_ltx(prim1)})\Big|_{{{l1a_ltx}}}^{{{l1b_ltx}}} = {_ltx(paso1_simp)}",
        },
        {
            "titulo": f"3. Integrar respecto a {v2} desde {lims[2]} hasta {lims[3]}",
            "latex": rf"({_ltx(prim2)})\Big|_{{{l2a_ltx}}}^{{{l2b_ltx}}} = {_ltx(paso2_simp)}",
        },
        {
            "titulo": f"4. Integrar respecto a {v3} desde {lims[4]} hasta {lims[5]}",
            "latex": rf"({_ltx(prim3)})\Big|_{{{l3a_ltx}}}^{{{l3b_ltx}}} = {_ltx(res_simbolico)}",
        },
    ]

    return {
        "tipo_titulo": f"Integral Triple ({coord.capitalize()})",
        "latex_integral": latex_integral,
        "resultado_simbolico": _ltx(res_simbolico),
        "resultado": f"{resultado_num:.6f}",
        "detalle_extra": f"{jac_texto} | {error_texto}",
        "pasos": pasos,
        "_datos_grafica": {
            "eval_fn_orden": fn_orden,
            "coord": coord,
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


def calcular_linea_cerrada(m_raw: str, n_raw: str, xt_raw: str, yt_raw: str, t0_raw: str, t1_raw: str):
    x, y, t = sp.symbols("x y t", real=True)

    M_sym, em = _parsear_sympy(m_raw)
    N_sym, en = _parsear_sympy(n_raw)
    xt_sym, ext = _parsear_sympy(xt_raw)
    yt_sym, eyt = _parsear_sympy(yt_raw)

    for err in (em, en, ext, eyt):
        if err:
            return {"error": err}

    if _validar_variables_expr(M_sym, {"x", "y"}, "M(x,y)") or _validar_variables_expr(N_sym, {"x", "y"}, "N(x,y)"):
        return {"error": "Las componentes M y N del campo deben depender solo de x e y."}
    if _validar_variables_expr(xt_sym, {"t"}, "x(t)") or _validar_variables_expr(yt_sym, {"t"}, "y(t)"):
        return {"error": "La parametrización x(t) e y(t) debe depender únicamente del parámetro t."}

    dxt = sp.diff(xt_sym, t)
    dyt = sp.diff(yt_sym, t)

    M_t = M_sym.subs({x: xt_sym, y: yt_sym})
    N_t = N_sym.subs({x: xt_sym, y: yt_sym})
    integrando_t = sp.simplify(M_t * dxt + N_t * dyt)
    primitiva_t = sp.integrate(integrando_t, t)

    rot_2d = sp.simplify(sp.diff(N_sym, x) - sp.diff(M_sym, y))
    es_conservativo = (rot_2d == 0)

    fn_M = sp.lambdify((x, y), M_sym, modules=["numpy", "math"])
    fn_N = sp.lambdify((x, y), N_sym, modules=["numpy", "math"])
    fn_xt = sp.lambdify(t, xt_sym, modules=["numpy", "math"])
    fn_yt = sp.lambdify(t, yt_sym, modules=["numpy", "math"])

    t0_limpio = str(t0_raw or "").strip()
    t1_limpio = str(t1_raw or "").strip()

    estado_green = (
        rf"\frac{{\partial N}}{{\partial x}} - \frac{{\partial M}}{{\partial y}} = {_ltx(rot_2d)} \quad \Rightarrow \quad "
        + (r"\text{Campo Conservativo}" if es_conservativo else r"\text{Campo No Conservativo}")
    )

    # MODO INDEFINIDA: Si t0 y t1 están vacíos
    if not t0_limpio and not t1_limpio:
        latex_integral = rf"\int_C \left({_ltx(M_sym)}\right)dx + \left({_ltx(N_sym)}\right)dy"
        pasos = [
            {"titulo": "1. Integral de línea indefinida planteada", "latex": latex_integral},
            {"titulo": "2. Vector velocidad de la curva r'(t)", "latex": rf"\mathbf{{r}}'(t) = \left\langle {_ltx(dxt)},\; {_ltx(dyt)} \right\rangle"},
            {"titulo": "3. Primitiva respecto al parámetro t (+ C)", "latex": rf"\int \left({_ltx(integrando_t)}\right)\,dt = {_ltx(primitiva_t)} + C"},
            {"titulo": "4. Diagnóstico por Teorema de Green (∂N/∂x - ∂M/∂y)", "latex": estado_green},
        ]
        return {
            "tipo_titulo": "Integral de Línea Indefinida",
            "latex_integral": latex_integral,
            "resultado_simbolico": rf"{_ltx(primitiva_t)} + C",
            "resultado": "Función trabajo F(t) + C",
            "detalle_extra": f"Sin límites de t (Curva graficada en [0, 2π]) | ∂N/∂x - ∂M/∂y = {rot_2d}",
            "pasos": pasos,
            "_datos_grafica": {
                "fn_M": fn_M, "fn_N": fn_N, "fn_xt": fn_xt, "fn_yt": fn_yt,
                "t0": 0.0, "t1": 2 * np.pi, "es_cerrada": False,
                "m_raw": m_raw, "n_raw": n_raw,
            },
        }

    if not t0_limpio or not t1_limpio:
        return {"error": "Ingresa ambos límites t₀ y t₁, o déjalos vacíos para calcular la integral de línea indefinida."}

    t0_sym, et0 = _parsear_sympy(t0_limpio)
    t1_sym, et1 = _parsear_sympy(t1_limpio)
    if et0 or et1:
        return {"error": et0 or et1}
    if t0_sym.free_symbols or t1_sym.free_symbols:
        return {"error": "Los límites del parámetro t deben ser constantes."}

    t0_num, t1_num = _a_num_float(t0_sym), _a_num_float(t1_sym)
    if not (np.isfinite(t0_num) and np.isfinite(t1_num)) or t0_num >= t1_num:
        return {"error": "Intervalo de parámetro inválido: se requiere t0 < t1."}

    res_simbolico = sp.simplify(primitiva_t.subs(t, t1_sym) - primitiva_t.subs(t, t0_sym))

    fn_int_t = sp.lambdify(t, integrando_t, modules=["numpy", "math"])
    resultado_num, error_num = integrate.quad(fn_int_t, t0_num, t1_num)

    x_ini, y_ini = float(sp.N(xt_sym.subs(t, t0_sym))), float(sp.N(yt_sym.subs(t, t0_sym)))
    x_fin, y_fin = float(sp.N(xt_sym.subs(t, t1_sym))), float(sp.N(yt_sym.subs(t, t1_sym)))
    es_cerrada = np.hypot(x_fin - x_ini, y_fin - y_ini) < 1e-6

    simbolo_int = r"\oint_C" if es_cerrada else r"\int_C"
    latex_integral = rf"{simbolo_int} \left({_ltx(M_sym)}\right)dx + \left({_ltx(N_sym)}\right)dy"

    pasos = [
        {"titulo": "1. Integral de línea planteada (" + ("Curva Cerrada" if es_cerrada else "Curva Abierta") + ")", "latex": latex_integral},
        {"titulo": "2. Vector velocidad de la curva r'(t)", "latex": rf"\mathbf{{r}}'(t) = \left\langle {_ltx(dxt)},\; {_ltx(dyt)} \right\rangle"},
        {"titulo": "3. Producto punto F(r(t)) · r'(t)", "latex": rf"\int_{{{_ltx(t0_sym)}}}^{{{_ltx(t1_sym)}}} \left({_ltx(integrando_t)}\right)\,dt"},
        {"titulo": "4. Diagnóstico por Teorema de Green (∂N/∂x - ∂M/∂y)", "latex": estado_green},
    ]

    return {
        "tipo_titulo": "Integral de Línea Cerrada (Green)" if es_cerrada else "Integral de Línea Paramétrica",
        "latex_integral": latex_integral,
        "resultado_simbolico": _ltx(res_simbolico),
        "resultado": f"{resultado_num:.6f}",
        "detalle_extra": f"Curva {'CERRADA' if es_cerrada else 'ABIERTA'} | ∂N/∂x - ∂M/∂y = {rot_2d} | Error: {error_num:.2e}",
        "pasos": pasos,
        "_datos_grafica": {
            "fn_M": fn_M, "fn_N": fn_N, "fn_xt": fn_xt, "fn_yt": fn_yt,
            "t0": t0_num, "t1": t1_num, "es_cerrada": es_cerrada,
            "m_raw": m_raw, "n_raw": n_raw,
        },
    }


def calcular_campo_vectorial(p_raw: str, q_raw: str, r_raw: str):
    x, y, z = sp.symbols("x y z", real=True)

    P_sym, ep = _parsear_sympy(p_raw)
    Q_sym, eq = _parsear_sympy(q_raw)
    R_sym, er = _parsear_sympy(r_raw)
    for err in (ep, eq, er):
        if err:
            return {"error": err}

    for comp, nombre in [(P_sym, "P"), (Q_sym, "Q"), (R_sym, "R")]:
        err_v = _validar_variables_expr(comp, {"x", "y", "z"}, f"la componente {nombre}")
        if err_v:
            return {"error": err_v}

    div_F = sp.simplify(sp.diff(P_sym, x) + sp.diff(Q_sym, y) + sp.diff(R_sym, z))

    rot_x = sp.simplify(sp.diff(R_sym, y) - sp.diff(Q_sym, z))
    rot_y = sp.simplify(sp.diff(P_sym, z) - sp.diff(R_sym, x))
    rot_z = sp.simplify(sp.diff(Q_sym, x) - sp.diff(P_sym, y))

    es_solenoidal = (div_F == 0)
    es_irrotacional = (rot_x == 0 and rot_y == 0 and rot_z == 0)

    latex_campo = rf"\mathbf{{F}}(x,y,z) = \left({_ltx(P_sym)}\right)\mathbf{{i}} + \left({_ltx(Q_sym)}\right)\mathbf{{j}} + \left({_ltx(R_sym)}\right)\mathbf{{k}}"
    latex_rot = rf"\left\langle {_ltx(rot_x)},\; {_ltx(rot_y)},\; {_ltx(rot_z)} \right\rangle"

    clasif_div = "Campo Solenoidal (div F = 0)" if es_solenoidal else "Campo con fuentes/sumideros (div F ≠ 0)"
    clasif_rot = "Campo Irrotacional / Conservativo (rot F = 0)" if es_irrotacional else "Campo Rotacional (rot F ≠ 0)"

    pasos = [
        {"titulo": "1. Campo vectorial ingresado", "latex": latex_campo},
        {
            "titulo": f"2. Divergencia (∇ · F) — {clasif_div}",
            "latex": rf"\nabla \cdot \mathbf{{F}} = \frac{{\partial P}}{{\partial x}} + \frac{{\partial Q}}{{\partial y}} + \frac{{\partial R}}{{\partial z}} = {_ltx(div_F)}",
        },
        {
            "titulo": f"3. Rotacional (∇ × F) — {clasif_rot}",
            "latex": rf"\nabla \times \mathbf{{F}} = {latex_rot}",
        },
    ]

    fn_P = sp.lambdify((x, y), P_sym.subs(z, 1), modules=["numpy", "math"])
    fn_Q = sp.lambdify((x, y), Q_sym.subs(z, 1), modules=["numpy", "math"])

    return {
        "tipo_titulo": "Análisis de Campo Vectorial (∇)",
        "latex_integral": latex_campo,
        "resultado_simbolico": rf"\text{{div }}\mathbf{{F}} = {_ltx(div_F)} \quad | \quad \text{{rot }}\mathbf{{F}} = {latex_rot}",
        "resultado": f"{'Solenoidal' if es_solenoidal else 'No solenoidal'} · {'Irrotacional' if es_irrotacional else 'Rotacional'}",
        "detalle_extra": f"Clasificación: {clasif_div} y {clasif_rot}",
        "pasos": pasos,
        "_datos_grafica": {
            "fn_P": fn_P, "fn_Q": fn_Q,
            "p_raw": p_raw, "q_raw": q_raw,
        },
    }


def calcular_descenso_gradiente(expr_raw: str, x0_raw: str, y0_raw: str, lr_raw: str, pasos_raw: str):
    x, y = sp.symbols("x y", real=True)

    f_sym, ef = _parsear_sympy(expr_raw)
    x0_sym, ex0 = _parsear_sympy(x0_raw)
    y0_sym, ey0 = _parsear_sympy(y0_raw)
    lr_sym, elr = _parsear_sympy(lr_raw)
    pasos_sym, ep = _parsear_sympy(pasos_raw)

    for err in (ef, ex0, ey0, elr, ep):
        if err:
            return {"error": err}

    err_v = _validar_variables_expr(f_sym, {"x", "y"}, "la función f(x, y)")
    if err_v:
        return {"error": err_v}

    if x0_sym.free_symbols or y0_sym.free_symbols or lr_sym.free_symbols or pasos_sym.free_symbols:
        return {"error": "El punto inicial, la tasa de aprendizaje (α) y los pasos deben ser constantes numéricas."}

    x_curr = _a_num_float(x0_sym)
    y_curr = _a_num_float(y0_sym)
    alpha = _a_num_float(lr_sym)
    n_pasos = int(round(_a_num_float(pasos_sym)))

    if alpha <= 0 or alpha > 2:
        return {"error": "La tasa de aprendizaje α debe estar entre 0 y 2 (ej. 0.15)."}
    if n_pasos < 1 or n_pasos > 150:
        return {"error": "El número de iteraciones debe estar entre 1 y 150."}

    df_dx = sp.simplify(sp.diff(f_sym, x))
    df_dy = sp.simplify(sp.diff(f_sym, y))

    # --- MATRIZ HESSIANA (Criterio de la segunda derivada) ---
    d2f_dx2 = sp.simplify(sp.diff(df_dx, x))
    d2f_dxdy = sp.simplify(sp.diff(df_dx, y))
    d2f_dy2 = sp.simplify(sp.diff(df_dy, y))
    H_sym = sp.Matrix([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])
    # --------------------------------------------------------

    fn_f = sp.lambdify((x, y), f_sym, modules=["numpy", "math"])
    fn_gx = sp.lambdify((x, y), df_dx, modules=["numpy", "math"])
    fn_gy = sp.lambdify((x, y), df_dy, modules=["numpy", "math"])

    tray_x = [x_curr]
    tray_y = [y_curr]
    tray_z = [float(fn_f(x_curr, y_curr))]

    divergio = False
    for _ in range(n_pasos):
        gx = float(fn_gx(x_curr, y_curr))
        gy = float(fn_gy(x_curr, y_curr))
        norma = float(np.hypot(gx, gy))

        if not np.isfinite(norma) or norma > 1e6:
            divergio = True
            break

        x_curr = x_curr - alpha * gx
        y_curr = y_curr - alpha * gy
        z_curr = float(fn_f(x_curr, y_curr))

        if not np.isfinite(z_curr) or abs(x_curr) > 1e5 or abs(y_curr) > 1e5:
            divergio = True
            break

        tray_x.append(x_curr)
        tray_y.append(y_curr)
        tray_z.append(z_curr)

    if divergio:
        return {
            "error": "El algoritmo divergió hacia el infinito. Reduce la tasa de aprendizaje α (ej. 0.05) o acerca el punto inicial."
        }

    x_fin, y_fin, z_fin = tray_x[-1], tray_y[-1], tray_z[-1]
    grad_fin_norm = float(np.hypot(float(fn_gx(x_fin, y_fin)), float(fn_gy(x_fin, y_fin))))

    # Evaluación numérica de la Hessiana en el punto final
    H_eval = H_sym.subs({x: x_fin, y: y_fin})
    det_H = float(sp.N(H_eval.det()))
    f_xx_val = float(sp.N(H_eval[0, 0]))

    if abs(det_H) < 1e-6:
        clasif_hess = "Criterio no concluyente (det(H) ≈ 0)"
        clasif_ltx = r"\text{Criterio no concluyente } (\det(H) \approx 0)"
    elif det_H > 0 and f_xx_val > 0:
        clasif_hess = "Mínimo local (det(H) > 0, f_xx > 0)"
        clasif_ltx = r"\text{Mínimo local } (\det(H) > 0,\; f_{xx} > 0)"
    elif det_H > 0 and f_xx_val < 0:
        clasif_hess = "Máximo local (det(H) > 0, f_xx < 0)"
        clasif_ltx = r"\text{Máximo local } (\det(H) > 0,\; f_{xx} < 0)"
    else:
        clasif_hess = "Punto de silla (det(H) < 0)"
        clasif_ltx = r"\text{Punto de silla } (\det(H) < 0)"

    filas_iter = []
    indices_muestra = sorted(set([0, 1, 2, len(tray_x) // 2, len(tray_x) - 1]))
    for idx in indices_muestra:
        if idx < len(tray_x):
            filas_iter.append(
                rf"k={idx}: \quad (x_{{{idx}}}, y_{{{idx}}}) = ({tray_x[idx]:.4f},\; {tray_y[idx]:.4f}) \;\Rightarrow\; f = {tray_z[idx]:.5f}"
            )
    latex_tabla = r"\begin{aligned}" + r" \\ ".join(filas_iter) + r"\end{aligned}"

    pasos = [
        {
            "titulo": "1. Función objetivo a minimizar",
            "latex": rf"f(x, y) = {_ltx(f_sym)}",
        },
        {
            "titulo": "2. Vector Gradiente ∇f(x, y) (Dirección de máximo ascenso)",
            "latex": rf"\nabla f(x, y) = \left\langle \frac{{\partial f}}{{\partial x}},\; \frac{{\partial f}}{{\partial y}} \right\rangle = \left\langle {_ltx(df_dx)},\; {_ltx(df_dy)} \right\rangle",
        },
        {
            "titulo": f"3. Regla de actualización (Descenso con α = {alpha})",
            "latex": rf"\mathbf{{x}}_{{k+1}} = \mathbf{{x}}_k - {alpha}\,\nabla f(\mathbf{{x}}_k)",
        },
        {
            "titulo": "4. Evolución de la trayectoria hacia el punto crítico",
            "latex": latex_tabla,
        },
        {
            "titulo": "5. Matriz Hessiana y Criterio de la Segunda Derivada",
            "latex": rf"H(x,y) = {_ltx(H_sym)} \quad \Rightarrow \quad \det(H(\mathbf{{x}}^*)) = {det_H:.4f} \;\Rightarrow\; {clasif_ltx}",
        },
    ]

    return {
        "tipo_titulo": "Simulación: Descenso del Gradiente 3D",
        "latex_integral": rf"\min_{{(x,y)\in\mathbb{{R}}^2}} \; f(x, y) = {_ltx(f_sym)}",
        "resultado_simbolico": rf"\mathbf{{x}}^* \approx ({x_fin:.4f},\; {y_fin:.4f}) \quad \Rightarrow \quad f(\mathbf{{x}}^*) \approx {z_fin:.5f}",
        "resultado": f"z = {z_fin:.5f} ({clasif_hess.split(' (')[0]})",
        "detalle_extra": f"Iteraciones: {len(tray_x)-1} | ||∇f|| = {grad_fin_norm:.2e} | {clasif_hess}",
        "pasos": pasos,
        "_datos_grafica": {
            "fn_f": fn_f,
            "expr_raw": expr_raw,
            "tray_x": tray_x,
            "tray_y": tray_y,
            "tray_z": tray_z,
        },
    }