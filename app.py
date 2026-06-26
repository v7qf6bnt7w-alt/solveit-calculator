from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any

import sympy as sp
from flask import Flask, jsonify, request, send_from_directory
from sympy.calculus.util import continuous_domain, function_range
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)
from sympy.solvers.inequalities import solve_univariate_inequality


BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "solveit.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("SolveIt")

app = Flask(__name__)

TRANSFORMATIONS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)

SAFE_FUNCTIONS = {
    "Abs": sp.Abs,
    "abs": sp.Abs,
    "acos": sp.acos,
    "acosh": sp.acosh,
    "acot": sp.acot,
    "acsc": sp.acsc,
    "asec": sp.asec,
    "asin": sp.asin,
    "asinh": sp.asinh,
    "atan": sp.atan,
    "atanh": sp.atanh,
    "ceiling": sp.ceiling,
    "cos": sp.cos,
    "cosh": sp.cosh,
    "cot": sp.cot,
    "csc": sp.csc,
    "E": sp.E,
    "e": sp.E,
    "exp": sp.exp,
    "factorial": sp.factorial,
    "floor": sp.floor,
    "gamma": sp.gamma,
    "I": sp.I,
    "im": sp.im,
    "ln": sp.log,
    "log": sp.log,
    "oo": sp.oo,
    "pi": sp.pi,
    "re": sp.re,
    "sec": sp.sec,
    "sin": sp.sin,
    "sinh": sp.sinh,
    "sqrt": sp.sqrt,
    "tan": sp.tan,
    "tanh": sp.tanh,
}


def response_ok(**payload: Any):
    logger.info("Solved request: %s", payload.get("kind", "unknown"))
    return jsonify({"success": True, **payload})


def response_error(message: str, exc: Exception | None = None):
    if exc:
        logger.exception("%s", message)
    else:
        logger.error("%s", message)
    return jsonify({"success": False, "error": message})


def latex_to_sympy_text(expression: str) -> str:
    """Support common LaTeX input without requiring extra parser packages."""
    text = expression.strip()
    replacements = {
        "\\left": "",
        "\\right": "",
        "\\cdot": "*",
        "\\times": "*",
        "\\pi": "pi",
        "\\infty": "oo",
        "\\ln": "ln",
        "\\log": "log",
        "\\sin": "sin",
        "\\cos": "cos",
        "\\tan": "tan",
        "\\sqrt": "sqrt",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Convert simple \frac{a}{b} and sqrt{a} forms repeatedly.
    frac_pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    while frac_pattern.search(text):
        text = frac_pattern.sub(r"((\1)/(\2))", text)
    text = re.sub(r"sqrt\{([^{}]+)\}", r"sqrt(\1)", text)
    text = text.replace("{", "(").replace("}", ")")
    return text


def symbols_from_text(*parts: str) -> dict[str, sp.Symbol]:
    names = set()
    for part in parts:
        names.update(re.findall(r"\b[a-zA-Z]\w*\b", part))
    reserved = set(SAFE_FUNCTIONS)
    return {name: sp.symbols(name, real=True) for name in names if name not in reserved}


def parse_math(expression: str, extra_symbols: dict[str, sp.Symbol] | None = None) -> sp.Expr:
    if not expression or not expression.strip():
        raise ValueError("Expression is empty")
    text = latex_to_sympy_text(expression)
    local_dict = dict(SAFE_FUNCTIONS)
    local_dict.update(symbols_from_text(text))
    if extra_symbols:
        local_dict.update(extra_symbols)
    return parse_expr(
        text,
        local_dict=local_dict,
        transformations=TRANSFORMATIONS,
        evaluate=True,
    )


def parse_value(value: str, variable: sp.Symbol) -> Any:
    normalized = value.strip().lower()
    if normalized in {"infinity", "+infinity", "inf", "+inf", "oo", "+oo"}:
        return sp.oo
    if normalized in {"-infinity", "-inf", "-oo"}:
        return -sp.oo
    return parse_math(value, {str(variable): variable})


def latex(value: Any) -> str:
    return sp.latex(value)


def text(value: Any) -> str:
    return sp.sstr(value)


def split_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def relation_from_text(raw: str, variable: sp.Symbol) -> sp.Rel:
    for operator, relation in [
        (">=", sp.Ge),
        ("<=", sp.Le),
        ("!=", sp.Ne),
        ("=", sp.Eq),
        (">", sp.Gt),
        ("<", sp.Lt),
    ]:
        if operator in raw:
            left, right = raw.split(operator, 1)
            return relation(
                parse_math(left, {str(variable): variable}),
                parse_math(right, {str(variable): variable}),
            )
    return sp.Eq(parse_math(raw, {str(variable): variable}), 0)


def degree_trig_text(expression: str) -> str:
    text_expr = latex_to_sympy_text(expression)
    for fn in ("sin", "cos", "tan"):
        text_expr = re.sub(rf"\b{fn}\(", f"{fn}(pi/180*", text_expr)
    return text_expr


def scientific_calculate(expression: str, angle_mode: str) -> dict[str, Any]:
    parsed = parse_math(degree_trig_text(expression) if angle_mode == "deg" else expression)
    result = sp.simplify(parsed)
    numeric = sp.N(result, 14)
    return {
        "kind": "scientific",
        "input_latex": latex(parse_math(expression)),
        "result": text(result),
        "result_latex": latex(result),
        "numeric": text(numeric),
        "numeric_latex": latex(numeric),
        "complex": {
            "real_latex": latex(sp.re(result)),
            "imaginary_latex": latex(sp.im(result)),
            "modulus_latex": latex(sp.Abs(result)),
            "argument_latex": latex(sp.arg(result)),
            "conjugate_latex": latex(sp.conjugate(result)),
        },
    }


def polynomial_from_coefficients(coefficients: str, variable: sp.Symbol) -> sp.Expr:
    coeffs = [parse_math(item, {str(variable): variable}) for item in split_csv(coefficients)]
    if not coeffs:
        raise ValueError("No coefficients supplied")
    degree = len(coeffs) - 1
    return sp.expand(sum(coef * variable ** (degree - index) for index, coef in enumerate(coeffs)))


def analyze_polynomial(expr: sp.Expr, variable: sp.Symbol) -> dict[str, Any]:
    poly = sp.Poly(sp.expand(expr), variable)
    return {
        "polynomial_latex": latex(poly.as_expr()),
        "degree": poly.degree(),
        "factor_latex": latex(sp.factor(poly.as_expr())),
        "roots_latex": latex(sp.solve(poly.as_expr(), variable)),
        "real_roots_latex": latex(sp.solve(poly.as_expr(), variable, domain=sp.S.Reals)),
    }


def solve_unknown_coefficients(
    polynomial: str,
    roots: str,
    variables: str,
    variable_name: str,
) -> dict[str, Any]:
    variable = sp.symbols(variable_name or "x", real=True)
    unknowns = [sp.symbols(name.strip(), real=True) for name in split_csv(variables)]
    if not unknowns:
        raise ValueError("Unknown coefficient variables are required")

    local_symbols = {str(variable): variable, **{str(sym): sym for sym in unknowns}}
    expr = parse_math(polynomial, local_symbols)
    root_values = [parse_math(root, local_symbols) for root in split_csv(roots)]
    equations = [sp.Eq(expr.subs(variable, root), 0) for root in root_values]
    solutions = sp.solve(equations, unknowns, dict=True)
    final_expr = sp.expand(expr.subs(solutions[0])) if solutions else expr

    return {
        "kind": "unknown_coefficients",
        "equations_latex": latex(equations),
        "solutions_latex": latex(solutions),
        "final_polynomial_latex": latex(final_expr),
        "final_roots_latex": latex(sp.solve(final_expr, variable)),
    }


def analyze_function(expression: str, function_type: str, variable_name: str) -> dict[str, Any]:
    variable = sp.symbols(variable_name or "x", real=True)
    expr = sp.simplify(parse_math(expression, {str(variable): variable}))
    derivative = sp.diff(expr, variable)
    data: dict[str, Any] = {
        "kind": "function",
        "function_type": function_type,
        "expression_latex": latex(expr),
        "simplified_latex": latex(sp.simplify(expr)),
        "roots_latex": latex(sp.solve(sp.Eq(expr, 0), variable)),
        "derivative_latex": latex(derivative),
    }

    try:
        data["domain_latex"] = latex(continuous_domain(expr, variable, sp.S.Reals))
    except Exception:
        data["domain_latex"] = r"\text{Could not determine exactly}"
    try:
        data["range_latex"] = latex(function_range(expr, variable, sp.S.Reals))
    except Exception:
        data["range_latex"] = r"\text{Could not determine exactly}"
    try:
        data["positive_latex"] = latex(solve_univariate_inequality(expr > 0, variable, relational=False))
        data["negative_latex"] = latex(solve_univariate_inequality(expr < 0, variable, relational=False))
    except Exception:
        data["positive_latex"] = data["negative_latex"] = r"\text{Could not determine exactly}"
    try:
        data["increasing_latex"] = latex(solve_univariate_inequality(derivative > 0, variable, relational=False))
        data["decreasing_latex"] = latex(solve_univariate_inequality(derivative < 0, variable, relational=False))
    except Exception:
        data["increasing_latex"] = data["decreasing_latex"] = r"\text{Could not determine exactly}"

    y = sp.symbols("y", real=True)
    try:
        inverse_solutions = sp.solve(sp.Eq(y, expr), variable)
        data["inverse_latex"] = latex(inverse_solutions)
        data["injective_latex"] = latex(len(inverse_solutions) == 1)
    except Exception:
        data["inverse_latex"] = r"\text{Could not determine exactly}"
        data["injective_latex"] = r"\text{Unknown}"

    data["surjective_latex"] = r"\text{Depends on selected codomain}"
    data["bijective_latex"] = r"\text{Injective and surjective only after codomain/domain restrictions}"
    data["graph"] = graph_points(expr, variable)
    return data


def graph_points(expr: sp.Expr, variable: sp.Symbol, start: float = -10, end: float = 10) -> dict[str, list[float | None]]:
    xs = [start + (end - start) * i / 359 for i in range(360)]
    ys: list[float | None] = []
    for value in xs:
        try:
            y = float(sp.N(expr.subs(variable, value), 15))
            ys.append(y if math.isfinite(y) and abs(y) < 1e6 else None)
        except Exception:
            ys.append(None)
    return {"x": xs, "y": ys}


def detect_indetermination(expr: sp.Expr, variable: sp.Symbol, target: Any) -> str:
    try:
        substituted = sp.simplify(expr.subs(variable, target))
        if substituted in {sp.nan, sp.zoo}:
            return r"\text{Indeterminate or undefined direct substitution}"
        numerator, denominator = sp.fraction(sp.together(expr))
        n_value = sp.simplify(numerator.subs(variable, target))
        d_value = sp.simplify(denominator.subs(variable, target))
        if n_value == 0 and d_value == 0:
            return r"\frac{0}{0}"
        if abs(n_value) is sp.oo and abs(d_value) is sp.oo:
            return r"\frac{\infty}{\infty}"
        return latex(substituted)
    except Exception:
        return r"\text{Direct substitution not available}"


def analyze_asymptotes(expr: sp.Expr, variable: sp.Symbol) -> dict[str, str]:
    result = {
        "vertical_latex": r"\text{None found}",
        "horizontal_latex": r"\text{None found}",
        "oblique_latex": r"\text{None found}",
    }
    try:
        denominator = sp.denom(sp.together(expr))
        vertical = sp.solve(sp.Eq(denominator, 0), variable)
        result["vertical_latex"] = latex(vertical)
    except Exception:
        pass
    try:
        left = sp.limit(expr, variable, -sp.oo)
        right = sp.limit(expr, variable, sp.oo)
        result["horizontal_latex"] = latex({"x_to_-oo": left, "x_to_oo": right})
    except Exception:
        pass
    try:
        m = sp.limit(expr / variable, variable, sp.oo)
        b = sp.limit(expr - m * variable, variable, sp.oo)
        if m not in (0, sp.oo, -sp.oo, sp.nan) and b not in (sp.oo, -sp.oo, sp.nan):
            result["oblique_latex"] = latex(sp.Eq(sp.symbols("y"), m * variable + b))
    except Exception:
        pass
    return result


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "solveit.html")


@app.route("/solveit.html")
def solveit_html():
    return send_from_directory(BASE_DIR, "solveit.html")


@app.route("/api/scientific", methods=["POST"])
def api_scientific():
    payload = request.get_json(silent=True) or {}
    try:
        return response_ok(**scientific_calculate(payload.get("expression", ""), payload.get("angle_mode", "rad")))
    except Exception as exc:
        return response_error("Scientific calculator error", exc)


@app.route("/api/algebra", methods=["POST"])
def api_algebra():
    payload = request.get_json(silent=True) or {}
    variable = sp.symbols(payload.get("variable", "x") or "x", real=True)
    action = payload.get("action", "polynomial_roots")
    try:
        if action == "unknown_coefficients":
            return response_ok(**solve_unknown_coefficients(
                payload.get("polynomial", ""),
                payload.get("roots", ""),
                payload.get("unknowns", ""),
                str(variable),
            ))
        if action == "equation":
            relation = relation_from_text(payload.get("equation", ""), variable)
            solution = sp.solve(relation, variable)
            return response_ok(kind="equation", relation_latex=latex(relation), solution_latex=latex(solution))
        if action == "inequality":
            relation = relation_from_text(payload.get("inequality", ""), variable)
            solution = solve_univariate_inequality(relation, variable, relational=False)
            return response_ok(kind="inequality", relation_latex=latex(relation), solution_latex=latex(solution))

        if payload.get("coefficients", "").strip():
            expr = polynomial_from_coefficients(payload.get("coefficients", ""), variable)
        else:
            expr = parse_math(payload.get("polynomial", ""), {str(variable): variable})
        return response_ok(kind="polynomial", **analyze_polynomial(expr, variable))
    except Exception as exc:
        return response_error("Algebra analysis error", exc)


@app.route("/api/function", methods=["POST"])
def api_function():
    payload = request.get_json(silent=True) or {}
    try:
        return response_ok(**analyze_function(
            payload.get("expression", ""),
            payload.get("function_type", "polynomial"),
            payload.get("variable", "x"),
        ))
    except Exception as exc:
        return response_error("Function analysis error", exc)


@app.route("/api/limit", methods=["POST"])
def api_limit():
    payload = request.get_json(silent=True) or {}
    variable = sp.symbols(payload.get("variable", "x") or "x", real=True)
    try:
        expr = parse_math(payload.get("expression", ""), {str(variable): variable})
        target = parse_value(payload.get("target", "0"), variable)
        side = payload.get("side", "+-")
        limit_value = sp.limit(expr, variable, target, dir=side)
        return response_ok(
            kind="limit",
            expression_latex=latex(expr),
            target_latex=latex(target),
            side=side,
            indetermination_latex=detect_indetermination(expr, variable, target),
            limit_latex=latex(limit_value),
            asymptotes=analyze_asymptotes(expr, variable),
        )
    except Exception as exc:
        return response_error("Limit analysis error", exc)


@app.route("/api/derivative", methods=["POST"])
def api_derivative():
    payload = request.get_json(silent=True) or {}
    variable = sp.symbols(payload.get("variable", "x") or "x", real=True)
    try:
        expr = parse_math(payload.get("expression", ""), {str(variable): variable})
        order = int(payload.get("order", 1) or 1)
        derivative = sp.diff(expr, variable, order)
        return response_ok(
            kind="derivative",
            expression_latex=latex(expr),
            derivative_latex=latex(derivative),
            simplified_latex=latex(sp.simplify(derivative)),
        )
    except Exception as exc:
        return response_error("Derivative analysis error", exc)


if __name__ == "__main__":
    logger.info("Starting SolveIt")
    app.run(debug=True, host="127.0.0.1", port=5000)
