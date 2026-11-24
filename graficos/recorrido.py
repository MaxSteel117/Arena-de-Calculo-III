import numpy as np
import plotly.graph_objects as go

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["sendDataToCloud"],
}


def _layout_pizarra(fig, titulo: str, height: int = 470):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#18221e",
        plot_bgcolor="#1f2e28",
        font=dict(color="#f4f7f5", family="Segoe UI, sans-serif"),
        title=dict(text=titulo, font=dict(size=16, color="#8ce068")),
        height=height,
        margin=dict(l=40, r=25, t=55, b=40),
    )
    return fig


def _eval_vec(fn, *args):
    res = np.asarray(fn(*args), dtype=float)
    if res.ndim == 0 and len(args) > 0:
        res = np.full_like(np.asarray(args[0], dtype=float), float(res))
    return res


def _crear_flechas_2d(fn_u, fn_v, xmin=-3.0, xmax=3.0, ymin=-3.0, ymax=3.0, color="#68b849"):
    gx = np.linspace(xmin, xmax, 13)
    gy = np.linspace(ymin, ymax, 13)
    X, Y = np.meshgrid(gx, gy)

    U = _eval_vec(fn_u, X, Y)
    V = _eval_vec(fn_v, X, Y)

    mag = np.hypot(U, V)
    max_mag = np.nanmax(mag) if np.nanmax(mag) > 1e-9 else 1.0
    escala = (xmax - xmin) / 16.0 / max_mag

    xs, ys = [], []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x0, y0 = X[i, j], Y[i, j]
            dx, dy = U[i, j] * escala, V[i, j] * escala
            if not (np.isfinite(dx) and np.isfinite(dy)):
                continue
            x1, y1 = x0 + dx, y0 + dy
            xs.extend([x0, x1, None])
            ys.extend([y0, y1, None])
            ang = np.arctan2(dy, dx)
            L = np.hypot(dx, dy) * 0.28
            xs.extend([x1, x1 - L * np.cos(ang - 0.45), None, x1, x1 - L * np.cos(ang + 0.45), None])
            ys.extend([y1, y1 - L * np.sin(ang - 0.45), None, y1, y1 - L * np.sin(ang + 0.45), None])

    return go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color=color, width=1.4),
        name="Campo Vectorial F",
        hoverinfo="skip",
    )


def generar_grafica_html(modo: str, datos: dict) -> str:
    if modo == "simple":
        fn = datos["fn"]
        var_name = datos["var_name"]
        expr_raw = datos["expr_raw"]

        if datos.get("es_indefinida"):
            fn_prim = datos["fn_prim"]
            xg = np.linspace(-5.0, 5.0, 300)
            yg = _eval_vec(fn, xg)
            yp = _eval_vec(fn_prim, xg)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=xg, y=yg, mode="lines",
                    name=f"Integrando f({var_name}) = {expr_raw}",
                    line=dict(width=3, color="#8ce068"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=xg, y=yp, mode="lines",
                    name=f"Primitiva F({var_name}) (con C=0)",
                    line=dict(width=3, color="#e2c07d", dash="dash"),
                )
            )
            _layout_pizarra(fig, f"Integral Indefinida: ∫ ({expr_raw}) d{var_name}")
            fig.update_xaxes(title_text=var_name, gridcolor="#2d4038")
            fig.update_yaxes(title_text="Valor", gridcolor="#2d4038")
            return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG)

        a, b = datos["a"], datos["b"]
        resultado = datos["resultado"]

        margen = max(0.25, 0.1 * (b - a))
        xg = np.linspace(a - margen, b + margen, 260)
        yg = _eval_vec(fn, xg)
        xa = np.linspace(a, b, 130)
        ya = _eval_vec(fn, xa)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=xg, y=yg, mode="lines",
                name=f"f({var_name}) = {expr_raw}",
                line=dict(width=3.5, color="#8ce068"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([[xa[0]], xa, [xa[-1]]]),
                y=np.concatenate([[0.0], ya, [0.0]]),
                fill="toself",
                fillcolor="rgba(104, 184, 73, 0.28)",
                line=dict(color="#68b849", width=1.5),
                name=f"Integral = {resultado:.4f}",
            )
        )
        _layout_pizarra(fig, f"Integral Simple: {expr_raw} en [{a:.2f}, {b:.2f}]")
        fig.update_xaxes(title_text=var_name, gridcolor="#2d4038")
        fig.update_yaxes(title_text=f"f({var_name})", gridcolor="#2d4038")
        return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG)

    if modo == "doble":
        fn = datos["fn"]
        v_int, v_ext = datos["v_int"], datos["v_ext"]
        diff_raw = datos["diff_raw"]
        expr_raw = datos["expr_raw"]
        ext_a, ext_b = datos["ext_a"], datos["ext_b"]
        lim_int_a, lim_int_b = datos["lim_int_a"], datos["lim_int_b"]
        coord = datos.get("coord", "cartesianas")

        exts = np.linspace(ext_a, ext_b, 48)
        filas = [np.linspace(lim_int_a(**{v_ext: ve}), lim_int_b(**{v_ext: ve}), 48) for ve in exts]
        G_int = np.array(filas)
        G_ext = np.tile(exts[:, None], (1, 48))
        G_z = _eval_vec(fn, G_ext, G_int)

        if coord == "polares":
            R = G_int if v_int == "r" else G_ext
            TH = G_ext if v_ext == "theta" else G_int
            X_plot, Y_plot = R * np.cos(TH), R * np.sin(TH)
            ejes = dict(xaxis_title="x = r cos(θ)", yaxis_title="y = r sen(θ)", zaxis_title="z")
        else:
            X_plot, Y_plot = G_ext, G_int
            ejes = dict(xaxis_title=v_ext, yaxis_title=v_int, zaxis_title="z")

        fig = go.Figure(
            data=[
                go.Surface(
                    x=X_plot, y=Y_plot, z=G_z,
                    colorscale="Rainbow",
                    opacity=0.95,
                    name=expr_raw,
                    contours=dict(
                        x=dict(show=True, color="#111111", width=1),
                        y=dict(show=True, color="#111111", width=1),
                        z=dict(show=True, color="#111111", width=1),
                    ),
                )
            ]
        )
        _layout_pizarra(fig, f"Superficie 3D ({coord.capitalize()} • {diff_raw}): {expr_raw}", height=500)
        fig.update_layout(scene=ejes)
        return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG)

    if modo == "triple":
        eval_fn_orden = datos["eval_fn_orden"]
        diff_raw = datos["diff_raw"]
        expr_raw = datos["expr_raw"]
        a3, b3 = datos["a3"], datos["b3"]
        lim2_a, lim2_b = datos["lim2_a"], datos["lim2_b"]
        lim1_a, lim1_b = datos["lim1_a"], datos["lim1_b"]
        v1, v2, v3 = datos["v1"], datos["v2"], datos["v3"]
        coord = datos.get("coord", "cartesianas")

        px, py, pz, pw = [], [], [], []
        for c3 in np.linspace(a3, b3, 7):
            a2, b2 = lim2_a(**{v3: c3}), lim2_b(**{v3: c3})
            for c2 in np.linspace(a2, b2, 7):
                a1, b1 = lim1_a(**{v3: c3, v2: c2}), lim1_b(**{v3: c3, v2: c2})
                for c1 in np.linspace(a1, b1, 7):
                    mapa = {v1: c1, v2: c2, v3: c3}
                    if coord == "cilindricas":
                        rv, thv, zv = mapa["r"], mapa["theta"], mapa["z"]
                        px.append(rv * np.cos(thv))
                        py.append(rv * np.sin(thv))
                        pz.append(zv)
                    elif coord == "esfericas":
                        rhov, phiv, thv = mapa["rho"], mapa["phi"], mapa["theta"]
                        px.append(rhov * np.sin(phiv) * np.cos(thv))
                        py.append(rhov * np.sin(phiv) * np.sin(thv))
                        pz.append(rhov * np.cos(phiv))
                    else:
                        px.append(mapa.get("x", c3))
                        py.append(mapa.get("y", c2))
                        pz.append(mapa.get("z", c1))
                    pw.append(float(eval_fn_orden(c1, c2, c3)))

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=px, y=py, z=pz,
                    mode="markers",
                    marker=dict(size=5.5, color=pw, colorscale="Rainbow", showscale=True, colorbar=dict(title="Valor")),
                    name=expr_raw,
                )
            ]
        )
        _layout_pizarra(fig, f"Sólido / Campo 3D ({coord.capitalize()} • {diff_raw})", height=500)
        fig.update_layout(scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
        return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG)

    if modo == "linea":
        fn_M, fn_N = datos["fn_M"], datos["fn_N"]
        fn_xt, fn_yt = datos["fn_xt"], datos["fn_yt"]
        t0, t1 = datos["t0"], datos["t1"]
        es_cerrada = datos["es_cerrada"]

        ts = np.linspace(t0, t1, 200)
        cx = _eval_vec(fn_xt, ts)
        cy = _eval_vec(fn_yt, ts)

        pad = 0.6
        xmin, xmax = float(np.min(cx)) - pad, float(np.max(cx)) + pad
        ymin, ymax = float(np.min(cy)) - pad, float(np.max(cy)) + pad

        fig = go.Figure()
        fig.add_trace(_crear_flechas_2d(fn_M, fn_N, xmin, xmax, ymin, ymax, color="rgba(104, 184, 73, 0.55)"))
        fig.add_trace(
            go.Scatter(
                x=cx, y=cy, mode="lines",
                fill="toself" if es_cerrada else None,
                fillcolor="rgba(226, 192, 125, 0.16)",
                line=dict(color="#e2c07d", width=4),
                name="Curva C (Cerrada)" if es_cerrada else "Trayectoria C",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[cx[0]], y=[cy[0]], mode="markers+text",
                marker=dict(size=10, color="#8ce068"),
                text=["Inicio r(t₀)"], textposition="top right",
                name="Punto Inicial",
            )
        )
        _layout_pizarra(fig, f"Trayectoria C sobre F(x,y) = <{datos['m_raw']}, {datos['n_raw']}>", height=490)
        fig.update_xaxes(title_text="x", gridcolor="#2d4038", zerolinecolor="#4a5d54")
        fig.update_yaxes(title_text="y", gridcolor="#2d4038", zerolinecolor="#4a5d54", scaleanchor="x", scaleratio=1)
        return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG)

    if modo == "gradiente":
        fn_f = datos["fn_f"]
        tx = np.array(datos["tray_x"], dtype=float)
        ty = np.array(datos["tray_y"], dtype=float)
        tz = np.array(datos["tray_z"], dtype=float)

        radio = max(3.5, float(np.max(np.abs(np.concatenate([tx, ty])))) + 1.2)
        grid = np.linspace(-radio, radio, 55)
        GX, GY = np.meshgrid(grid, grid)
        GZ = _eval_vec(fn_f, GX, GY)

        frames = [
            go.Frame(
                data=[
                    go.Surface(x=GX, y=GY, z=GZ, colorscale="Viridis", opacity=0.78, showscale=False),
                    go.Scatter3d(
                        x=tx[: k + 1], y=ty[: k + 1], z=tz[: k + 1],
                        mode="lines+markers",
                        line=dict(color="#e2c07d", width=6),
                        marker=dict(size=4, color="#e2c07d"),
                        name="Trayectoria -∇f",
                    ),
                    go.Scatter3d(
                        x=[tx[k]], y=[ty[k]], z=[tz[k]],
                        mode="markers",
                        marker=dict(size=9, color="#ff595e", symbol="diamond"),
                        name="Posición actual",
                    ),
                ],
                name=f"Paso {k}",
            )
            for k in range(len(tx))
        ]

        fig = go.Figure(
            data=[
                go.Surface(x=GX, y=GY, z=GZ, colorscale="Viridis", opacity=0.78, colorbar=dict(title="f(x,y)")),
                go.Scatter3d(
                    x=tx, y=ty, z=tz,
                    mode="lines+markers",
                    line=dict(color="#e2c07d", width=6),
                    marker=dict(size=4, color="#e2c07d"),
                    name="Ruta del Gradiente",
                ),
                go.Scatter3d(
                    x=[tx[0]], y=[ty[0]], z=[tz[0]],
                    mode="markers",
                    marker=dict(size=8, color="#8ce068"),
                    name="Inicio (x₀, y₀)",
                ),
                go.Scatter3d(
                    x=[tx[-1]], y=[ty[-1]], z=[tz[-1]],
                    mode="markers",
                    marker=dict(size=9, color="#ff595e", symbol="diamond"),
                    name="Mínimo alcanzado",
                ),
            ],
            frames=frames,
        )

        _layout_pizarra(fig, f"Descenso del Gradiente 3D sobre f(x,y) = {datos['expr_raw']}", height=540)
        fig.update_layout(
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="f(x, y)"),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.02, y=0.98,
                    bgcolor="#24352e",
                    bordercolor="#68b849",
                    font=dict(color="#f4f7f5"),
                    buttons=[
                        dict(
                            label="▶ Animar Descenso",
                            method="animate",
                            args=[None, dict(frame=dict(duration=140, redraw=True), fromcurrent=True)],
                        ),
                        dict(
                            label="⏸ Pausar",
                            method="animate",
                            args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                        ),
                    ],
                )
            ],
        )
        return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG)

    # modo == "campo"
    fn_P, fn_Q = datos["fn_P"], datos["fn_Q"]
    fig = go.Figure()
    fig.add_trace(_crear_flechas_2d(fn_P, fn_Q, -3.5, 3.5, -3.5, 3.5, color="#8ce068"))
    _layout_pizarra(fig, f"Campo Vectorial (corte z=1): F = <{datos['p_raw']}, {datos['q_raw']}>", height=490)
    fig.update_xaxes(title_text="x", gridcolor="#2d4038", zerolinecolor="#4a5d54")
    fig.update_yaxes(title_text="y", gridcolor="#2d4038", zerolinecolor="#4a5d54", scaleanchor="x", scaleratio=1)
    return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG)