#!/usr/bin/env python3
"""Rewrite the power functions of the generated GOW network (src/networks/GOW/network.hpp) in place.

jaff emits every power as std::pow. In actual_rhs, actual_jac and actual_rhs_and_jac this pass replaces, at the same
double-precision accuracy (no reduced-precision math):

- integer powers x^n, |n| <= 8, by products (x*x, 1/(x*x), ...);
- half-integer powers by sqrt (x^0.5, x^-0.5, x^1.5, x^0.25);
- other literal powers of state.T by exp(c * lnT), with lnT = log(state.T) computed once per function;
- other literal powers of a temporary used as a base at least twice by exp(c * ln_<base>), with the log computed
  once right after the temporary;
- 10^expr by exp(ln10 * expr).

Everything else (powers with an expression exponent of a non-constant base) is left as std::pow. The rewritten file
keeps the generator's layout, so regenerate.sh can run it again after every generation.
"""
import re
import sys
from collections import Counter

LN10 = "2.302585092994046"
FUNCTIONS = ("actual_rhs(const BurnT", "actual_jac(const BurnT", "actual_rhs_and_jac(const BurnT")


def split_pow(body, start):
    """The (base, exponent, end) of the std::pow( call whose arguments start at *start*."""
    depth, j = 1, start
    while depth:
        if body[j] == "(":
            depth += 1
        elif body[j] == ")":
            depth -= 1
        j += 1
    args = body[start : j - 1]
    depth = 0
    for k, ch in enumerate(args):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            return args[:k].strip(), args[k + 1 :].strip(), j
    raise ValueError(f"std::pow without a second argument: {args}")


def literal(expo):
    """The value of a literal exponent such as 2, -0.5, 3.0 / 2.0, else None."""
    if re.fullmatch(r"-?\d+(\.\d*)?([eE][-+]?\d+)?", expo):
        return float(expo)
    m = re.fullmatch(r"(-?\d+\.\d*) / (\d+\.\d*)", expo)
    if m:
        return float(m.group(1)) / float(m.group(2))
    return None


def paren(x):
    return x if re.fullmatch(r"[\w.]+", x) else f"({x})"


def product(x, n):
    """x^n for an integer n != 0 as products, parenthesised so that it binds like the call it replaces."""
    p = paren(x)
    pos = " * ".join([p] * abs(n))
    if n == 1:
        return p
    return f"({pos})" if n > 0 else f"(1.0 / ({pos}))"


def rewrite_function(body):
    """The function body with its powers rewritten."""
    calls = []
    pos = 0
    while True:
        i = body.find("std::pow(", pos)
        if i < 0:
            break
        base, expo, end = split_pow(body, i + len("std::pow("))
        calls.append((i, end, base, expo))
        pos = end
    # bases that get a hoisted log: state.T always, a temporary when it is the base of >= 2 powers that are not
    # integer powers (literal or expression exponents alike)
    def needs_log(expo):
        c = literal(expo)
        return c is None or not (c.is_integer() and 1 <= abs(int(c)) <= 8) and c not in (0.5, -0.5, 1.5, 0.25)

    base_use = Counter(base for _, _, base, expo in calls if needs_log(expo))
    hoisted = {"state.T": "lnT"}
    for base, n in base_use.items():
        if n >= 2 and re.fullmatch(r"cse\d+", base):
            hoisted[base] = f"ln_{base}"
    out = []
    pos = 0
    kept = 0
    for i, end, base, expo in calls:
        out.append(body[pos:i])
        c = literal(expo)
        if c is not None and c.is_integer() and 1 <= abs(int(c)) <= 8:
            out.append(product(base, int(c)))
        elif c == 0.5:
            out.append(f"std::sqrt({base})")
        elif c == -0.5:
            out.append(f"(1.0 / std::sqrt({base}))")
        elif c == 1.5:
            out.append(f"({paren(base)} * std::sqrt({base}))")
        elif c == 0.25:
            out.append(f"std::sqrt(std::sqrt({base}))")
        elif base in hoisted:
            out.append(f"std::exp(({expo}) * {hoisted[base]})")
        elif base in ("10", "10.0"):
            out.append(f"std::exp({LN10} * ({expo}))")
        else:
            out.append(body[i:end])
            kept += 1
        pos = end
    out.append(body[pos:])
    text = "".join(out)
    # hoist the logs: lnT after the static_asserts (the first statement of the function), ln_<cse> after the temporary
    if "lnT" in text:
        m = re.search(r"^(\t+)const amrex::Real cse", text, re.M)
        indent = m.group(1)
        text = text[: m.start()] + f"{indent}const amrex::Real lnT = std::log(state.T);\n" + text[m.start() :]
    for base, name in hoisted.items():
        if base == "state.T" or name not in text:
            continue
        m = re.search(r"^(\t+)const amrex::Real " + base + r" = [^;]*;\n", text, re.M)
        text = text[: m.end()] + f"{m.group(1)}const amrex::Real {name} = std::log({base});\n" + text[m.end() :]
    return text, len(calls), kept


def main(path):
    src = open(path).read()
    starts = sorted(src.find(f) for f in FUNCTIONS)
    assert all(s >= 0 for s in starts), "a network function is missing"
    ends = starts[1:] + [src.rfind("\n};")]
    pieces = [src[: starts[0]]]
    for s, e in zip(starts, ends):
        text, n, kept = rewrite_function(src[s:e])
        pieces.append(text)
        print(f"{src[s:s + 40].split('(')[0]}: {n} pow calls, {kept} kept")
    pieces.append(src[ends[-1] :])
    open(path, "w").write("".join(pieces))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "src/networks/GOW/network.hpp")
