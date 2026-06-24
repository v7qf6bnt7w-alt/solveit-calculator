from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import sympy as sp
from flask import Flask, jsonify, render_template, request
from sympy import Eq, Interval, Rational, S, symbols
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)
from sympy.solvers.inequalities import solve_univariate_inequality

app = Flask(__name__)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "solveit.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("solveit")

x = symbols("x", real=True)
allowed_symbols = {
    "x": x,
    "pi": sp.pi,
    "e": sp.E,
    "sqrt": sp.sqrt,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "log": sp.log,
    "ln": sp.log,
    "exp": sp.exp,
    "abs": sp.Abs,
    "floor": sp.floor,
    "ceil": sp.ceiling,
    "factorial": sp.factorial,
    "gamma": sp.gamma,
    "sec": sp.sec,
    "csc": sp.csc,
    "cot": sp.cot,
    "asec": sp.asec,
    "acsc": sp.acsc,
    "acot": sp.acot,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "asinh": sp.asinh,
    "acosh": sp.acosh,
    "atanh": sp.atanh,
}

TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def safe_parse(expression: str, allow_x: bool = True) -> sp.Expr:
    expr = expression.strip().replace('^', '**')
    if not expr:
        raise ValueError("Expression is empty")
    local_dict = allowed_symbols.copy()
    if not allow_x:
        local_dict.pop("x", None)
    return parse_expr(expr, local_dict=local_dict, transformations=TRANSFORMATIONS)


def format_result(value: Any) -> str:
    if isinstance(value, sp.Integer) or isinstance(value, sp.Rational):
        return str(value)
    if isinstance(value, sp.Float):
        return f"{float(value):.12g}"
    return sp.sstr(value, full_prec=True)


def numeric_value(value: Any) -> float | None:
    try:
        return float(sp.N(value, 30))
    except Exception:
        return None


def evaluate_expression(expression: str) -> dict[str, Any]:
    parsed = safe_parse(expression)
    if parsed.free_symbols and parsed.free_symbols != {x}:
        raise ValueError("Only x is allowed as a variable for this evaluation")
    result = sp.simplify(parsed)
    if result.has(sp.nan, sp.zoo) or result in (sp.oo, -sp.oo):
        raise ZeroDivisionError("Math ERROR")
    return {
        "expression": expression,
        "result": format_result(result),
        "numeric": numeric_value(result),
        "exact": sp.simplify(result),
    }


def to_fraction(value: Any) -> str:
    numeric = numeric_value(value)
    if numeric is None:
        raise ValueError("Math ERROR")
    frac = sp.Rational(str(numeric)).limit_denominator(10000)
    return str(frac)


def to_decimal(value: Any) -> str:
    numeric = numeric_value(value)
    if numeric is None:
        raise ValueError("Math ERROR")
    return f"{numeric:.15g}"


def compute_domain(expr: sp.Expr) -> str:
    domain_parts = []

    # Denominator restrictions
    denominator = sp.denom(expr)
    if denominator != 1:
        roots = sp.solve(denominator, x)
        if roots:
            domain_parts.append(f"x ≠ {', '.join(format_result(r) for r in roots)}")

    # Log restrictions
    for arg in sp.preorder_traversal(expr):
        if isinstance(arg, sp.log) or getattr(arg, "func", None) == sp.log:
            base_arg = arg.args[0]
            if base_arg.has(x):
                domain_parts.append("log argument must be > 0")

    # Even roots restrictions
    for node in sp.preorder_traversal(expr):
        if getattr(node, "func", None) == sp.sqrt:
            radicand = node.args[0]
            if radicand.has(x):
                domain_parts.append("radicand must be ≥ 0")

    return "; ".join(domain_parts) if domain_parts else "All real numbers"


def compute_roots(expr: sp.Expr) -> list[str]:
    try:
        roots = sp.solve(sp.Eq(expr, 0), x)
        return [format_result(r) for r in roots]
    except Exception:
        return []


def compute_inequalities(expr: sp.Expr) -> dict[str, str]:
    results = {}
    try:
        positive = solve_univariate_inequality(sp.Gt(expr, 0), x, relational=False)
        results["positive"] = str(positive)
    except Exception:
        results["positive"] = "Not determined"
    try:
        negative = solve_univariate_inequality(sp.Lt(expr, 0), x, relational=False)
        results["negative"] = str(negative)
    except Exception:
        results["negative"] = "Not determined"
    return results


def make_graph(expr: sp.Expr, xmin: float = -10, xmax: float = 10) -> dict[str, Any]:
    x_vals = np.linspace(xmin, xmax, 400)
    y_vals = []
    for value in x_vals:
        try:
            y = float(sp.N(expr.subs(x, value), 30))
            y_vals.append(y)
        except Exception:
            y_vals.append(np.nan)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            line=dict(color="#4f46e5", width=2),
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
        xaxis_title="x",
        yaxis_title="f(x)",
    )
    return fig.to_json()


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/api/scientific", methods=["POST"])
def scientific_api():
    payload = request.get_json(silent=True) or {}
    expression = payload.get("expression", "")
    action = payload.get("action", "evaluate")

    try:
        if action == "to_fraction":
            result = evaluate_expression(expression)["exact"]
            return jsonify({"success": True, "result": to_fraction(result)})
        if action == "to_decimal":
            result = evaluate_expression(expression)["exact"]
            return jsonify({"success": True, "result": to_decimal(result)})

        data = evaluate_expression(expression)
        return jsonify({"success": True, "result": data["result"]})
    except Exception as exc:
        logger.exception("Scientific calculator error for expression=%s", expression)
        message = "Syntax ERROR" if "parse" in str(exc).lower() or "empty" in str(exc).lower() else "Math ERROR"
        return jsonify({"success": False, "error": message})


@app.route("/api/polynomial", methods=["POST"])
def polynomial_api():
    payload = request.get_json(silent=True) or {}
    roots_input = payload.get("roots", "")
    coeffs_input = payload.get("coeffs", "")

    try:
        if roots_input.strip():
            raw_roots = [sp.sympify(part.strip()) for part in roots_input.split(",") if part.strip()]
            polynomial = sp.expand(sp.prod([x - r for r in raw_roots]))
            roots_out = [format_result(r) for r in raw_roots]
            return jsonify(
                {
                    "success": True,
                    "polynomial": sp.sstr(polynomial),
                    "roots": roots_out,
                }
            )

        if coeffs_input.strip():
            coeffs = [sp.sympify(part.strip()) for part in coeffs_input.split(",") if part.strip()]
            poly = sp.Poly(coeffs, x)
            roots_out = [format_result(r) for r in poly.all_roots()]
            return jsonify(
                {
                    "success": True,
                    "polynomial": sp.sstr(poly.as_expr()),
                    "roots": roots_out,
                }
            )

        return jsonify({"success": False, "error": "Syntax ERROR"})
    except Exception:
        logger.exception("Polynomial solver error")
        return jsonify({"success": False, "error": "Math ERROR"})


@app.route("/api/function", methods=["POST"])
def function_api():
    payload = request.get_json(silent=True) or {}
    expression = payload.get("expression", "")
    graph = payload.get("graph", False)

    try:
        expr = safe_parse(expression)
        roots = compute_roots(expr)
        domain = compute_domain(expr)
        inequalities = compute_inequalities(expr)
        response = {
            "success": True,
            "roots": roots,
            "domain": domain,
            "positive": inequalities["positive"],
            "negative": inequalities["negative"],
        }
        if graph:
            response["graph"] = make_graph(expr)
        return jsonify(response)
    except Exception:
        logger.exception("Function solver error for expression=%s", expression)
        return jsonify({"success": False, "error": "Syntax ERROR"})


if __name__ == "__main__":
    logger.info("Starting SolveIt v5")
    app.run(debug=True, host="0.0.0.0", port=5000)
