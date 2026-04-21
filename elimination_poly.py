"""Polynomial elimination via Gröbner-basis reduction on Q[x1,...,xn]."""

from __future__ import annotations

import dataclasses
import fractions
import heapq

Fraction = fractions.Fraction


@dataclasses.dataclass(frozen=True)
class PolyVar:
  name: str

  def __str__(self):
    return self.name


@dataclasses.dataclass(frozen=True, order=True)
class Monom:
  exps: tuple[int, ...]

  def __mul__(self, other: Monom) -> Monom:
    return Monom(tuple(x + y for x, y in zip(self.exps, other.exps)))

  def div(self, other: Monom) -> Monom | None:
    out = []
    for x, y in zip(self.exps, other.exps):
      if x < y:
        return None
      out.append(x - y)
    return Monom(tuple(out))

  def lcm(self, other: Monom) -> Monom:
    return Monom(tuple(max(x, y) for x, y in zip(self.exps, other.exps)))

  def is_zero(self) -> bool:
    return all(e == 0 for e in self.exps)


TermDict = dict[Monom, Fraction]


class Poly:
  """Multivariate polynomial over Q."""

  def __init__(self, terms: TermDict, nvars: int):
    self.terms = {m: Fraction(c) for m, c in terms.items() if c != 0}
    self.nvars = nvars
    self._hash = None
    self._cleanup()

  def _cleanup(self):
    self.terms = {m: c for m, c in self.terms.items() if c != 0}
    self._hash = None

  @classmethod
  def zero(cls, nvars: int) -> Poly:
    return cls({}, nvars)

  @classmethod
  def const(cls, c: int | Fraction, nvars: int) -> Poly:
    c = Fraction(c)
    if c == 0:
      return cls.zero(nvars)
    return cls({Monom((0,) * nvars): c}, nvars)

  @classmethod
  def var(cls, idx: int, nvars: int) -> Poly:
    m = [0] * nvars
    m[idx] = 1
    return cls({Monom(tuple(m)): Fraction(1)}, nvars)

  def copy(self) -> Poly:
    return Poly(dict(self.terms), self.nvars)

  def is_zero(self) -> bool:
    return not self.terms

  def lead_monom(self) -> Monom | None:
    if self.is_zero():
      return None
    return max(self.terms.keys())

  def lead_coef(self) -> Fraction:
    lm = self.lead_monom()
    if lm is None:
      return Fraction(0)
    return self.terms[lm]

  def monic(self) -> Poly:
    if self.is_zero():
      return self
    lc = self.lead_coef()
    return self.scale(Fraction(1, 1) / lc)

  def normalize_sign(self) -> Poly:
    if self.is_zero():
      return self
    lc = self.lead_coef()
    if lc < 0:
      return self.scale(-1)
    return self

  # arithmetic
  def __add__(self, other: Poly | int | Fraction) -> Poly:
    if isinstance(other, (int, Fraction)):
      other = Poly.const(other, self.nvars)
    assert self.nvars == other.nvars
    out = dict(self.terms)
    for m, c in other.terms.items():
      out[m] = out.get(m, Fraction(0)) + c
      if out[m] == 0:
        del out[m]
    return Poly(out, self.nvars)

  def __radd__(self, other: int | Fraction) -> Poly:
    return self + other

  def __sub__(self, other: Poly | int | Fraction) -> Poly:
    if isinstance(other, (int, Fraction)):
      return self + (-other)
    return self + other.scale(-1)

  def __rsub__(self, other: int | Fraction) -> Poly:
    return (-self) + other

  def __neg__(self) -> Poly:
    return self.scale(-1)

  def scale(self, k: int | Fraction) -> Poly:
    k = Fraction(k)
    if k == 0:
      return Poly.zero(self.nvars)
    return Poly({m: c * k for m, c in self.terms.items()}, self.nvars)

  def mul_monom(self, m: Monom, c: int | Fraction = 1) -> Poly:
    c = Fraction(c)
    if c == 0 or self.is_zero():
      return Poly.zero(self.nvars)
    out = {}
    for m0, c0 in self.terms.items():
      out[m0 * m] = c0 * c
    return Poly(out, self.nvars)

  def __mul__(self, other: Poly | int | Fraction) -> Poly:
    if isinstance(other, (int, Fraction)):
      return self.scale(other)
    assert self.nvars == other.nvars
    out: TermDict = {}
    for m1, c1 in self.terms.items():
      for m2, c2 in other.terms.items():
        m = m1 * m2
        out[m] = out.get(m, Fraction(0)) + c1 * c2
        if out[m] == 0:
          del out[m]
    return Poly(out, self.nvars)

  def __rmul__(self, other: int | Fraction) -> Poly:
    return self.scale(other)

  def __hash__(self):
    if self._hash is None:
      self._hash = hash((self.nvars, frozenset(self.terms.items())))
    return self._hash

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, Poly):
      return False
    return (
        self.nvars == other.nvars
        and self.terms == other.terms
    )

  def __str__(self):
    if self.is_zero():
      return "0"
    parts = []
    mons = sorted(self.terms.keys(), reverse=True)
    for m in mons:
      c = self.terms[m]
      mon_s = []
      for i, e in enumerate(m.exps):
        if e == 0:
          continue
        if e == 1:
          mon_s.append(f"x{i}")
        else:
          mon_s.append(f"x{i}^{e}")
      if mon_s:
        ms = "*".join(mon_s)
        if c == 1:
          parts.append(ms)
        elif c == -1:
          parts.append(f"-{ms}")
        else:
          parts.append(f"{c}*{ms}")
      else:
        parts.append(f"{c}")
    return " + ".join(parts)


class PolyEnv:
  """Variable registry and constructors."""

  def __init__(self):
    self.vars: list[PolyVar] = []
    self.name_to_idx: dict[str, int] = {}

  @property
  def nvars(self) -> int:
    return len(self.vars)

  def new_var(self, name: str) -> int:
    if name in self.name_to_idx:
      return self.name_to_idx[name]
    idx = len(self.vars)
    self.vars.append(PolyVar(name=name))
    self.name_to_idx[name] = idx
    return idx

  def poly_var(self, name: str) -> Poly:
    idx = self.name_to_idx[name]
    return Poly.var(idx, self.nvars)

  def const(self, c: int | Fraction) -> Poly:
    return Poly.const(c, self.nvars)


class ElimPoly:
  """Gröbner-basis elimination engine."""

  def __init__(self, env: PolyEnv):
    self.env = env
    self.gb: list[Poly] = []
    self._pair_heap: list[tuple[Monom, int, int]] = []
    self._pairs_seen: set[tuple[int, int]] = set()

  def _reduce_poly(self, f: Poly, reducers: list[Poly]) -> Poly:
    r = f.copy()
    out = Poly.zero(f.nvars)
    while not r.is_zero():
      lm_r = r.lead_monom()
      lc_r = r.lead_coef()
      reduced = False
      for g in reducers:
        lm_g = g.lead_monom()
        if lm_g is None:
          continue
        q = lm_r.div(lm_g)
        if q is not None:
          c = lc_r / g.lead_coef()
          r = r - g.mul_monom(q, c)
          reduced = True
          break
      if not reduced:
        # move lead term to output
        lt = Poly({lm_r: lc_r}, r.nvars)
        out = out + lt
        r = r - lt
    return out

  def _normal_form(self, f: Poly) -> Poly:
    if f.is_zero():
      return f
    red = self._reduce_poly(f, self.gb)
    if red.is_zero():
      return red
    return red.monic().normalize_sign()

  def simplify(self, f: Poly) -> Poly:
    return self._normal_form(f)

  def check_zero(self, f: Poly) -> bool:
    return self.simplify(f).is_zero()

  def _s_poly(self, f: Poly, g: Poly) -> Poly:
    lm_f = f.lead_monom()
    lm_g = g.lead_monom()
    assert lm_f is not None and lm_g is not None
    l = lm_f.lcm(lm_g)
    tf = l.div(lm_f)
    tg = l.div(lm_g)
    assert tf is not None and tg is not None
    return f.mul_monom(tf, Fraction(1, 1) / f.lead_coef()) - g.mul_monom(
        tg, Fraction(1, 1) / g.lead_coef()
    )

  def _add_pairs_for_new_basis_poly(self, idx_new: int):
    for i in range(idx_new):
      a, b = min(i, idx_new), max(i, idx_new)
      if (a, b) in self._pairs_seen:
        continue
      lm_a = self.gb[a].lead_monom()
      lm_b = self.gb[b].lead_monom()
      if lm_a is None or lm_b is None:
        continue  # shouldn't happen
      self._pairs_seen.add((a, b))
      lcm = lm_a.lcm(lm_b)
      heapq.heappush(self._pair_heap, (lcm, a, b))

  def _buchberger_closure(self):
    while self._pair_heap:
      _, i, j = heapq.heappop(self._pair_heap)
      s = self._s_poly(self.gb[i], self.gb[j])
      h = self._normal_form(s)
      if h.is_zero():
        continue
      self.gb.append(h)
      self._add_pairs_for_new_basis_poly(len(self.gb) - 1)

  def _batch_same_lm_reduce(self):
    """F4 row-batch reduction."""
    #TODO: Add F4 batch reduction
    pass

  def force_zero(self, f: Poly) -> bool:
    h = self._normal_form(f)
    if h.is_zero():
      return False
    self.gb.append(h)
    self._add_pairs_for_new_basis_poly(len(self.gb) - 1)
    self._batch_same_lm_reduce()
    self._buchberger_closure()
    return True

  def clone(self) -> ElimPoly:
    res = ElimPoly(self.env)
    res.gb = [g.copy() for g in self.gb]
    res._pair_heap = list(self._pair_heap)
    res._pairs_seen = set(self._pairs_seen)
    return res
