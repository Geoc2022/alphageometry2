from ddar import DDAR
from parse import AGProblem


def solve(problem_str):
  p = AGProblem.parse(problem_str)
  d = DDAR(p.points)
  for pred in p.preds:
    d.force_pred(pred)
  d.deduction_closure(progress_dot=False)
  return d, p


def _pt(problem, name):
  return next(x for x in problem.points if x.name == name)


def test_bridge_cong():
  s = "a@0_0 = ; b@1_0 = ; c@2_0 = ; d@3_0 = cong a b c d ? cong a b c d"
  d, p = solve(s)
  assert d.check_pred(p.goal)

  a, b, c, dd = (_pt(p, x) for x in ("a", "b", "c", "d"))
  poly = d.get_len_poly(a, b) - d.get_len_poly(c, dd)
  assert d.check_poly_zero(poly)


def test_bridge_cong_trans():
  s = (
      "a@0_0 = ; b@1_0 = ; c@2_0 = ; d@3_0 = ; e@4_0 = ; f@5_0 = "
      "cong a b c d, cong c d e f ? cong a b e f"
  )
  d, p = solve(s)
  assert d.check_pred(p.goal)

  a, b, e, f = (_pt(p, x) for x in ("a", "b", "e", "f"))
  poly = d.get_len_poly(a, b) - d.get_len_poly(e, f)
  assert d.check_poly_zero(poly)


def test_manual_force():
  s = "a@0_0 = ; b@2_0 = ; c@10_0 = ; d@12_0 = ? cong a b c d"
  d, p = solve(s)

  a, b, c, dd = (_pt(p, x) for x in ("a", "b", "c", "d"))
  x = d.get_len_poly(a, b)
  y = d.get_len_poly(c, dd)

  poly = x * x - y * y
  changed = d.force_poly_zero(poly)
  assert changed
  assert d.check_poly_zero(poly)


def test_no_numerics():
  s = "a@0_0 = ; b@1_0 = ; c@0_1 = ; d@1_1 = ? cong a b c d"
  d, p = solve(s)

  a, b, c, dd = (_pt(p, x) for x in ("a", "b", "c", "d"))
  x = d.get_len_poly(a, b)
  y = d.get_len_poly(c, dd)

  poly = x * x - y * y
  assert not d.check_poly_zero(poly)


def test_by_trivial():
  s = "a@0_0 = ; b@1_0 = ? cong a b a b"
  d, p = solve(s)

  a, b = (_pt(p, x) for x in ("a", "b"))
  poly = d.get_len_poly(a, b) - d.get_len_poly(a, b)
  assert d.check_poly_zero(poly)


def test_scaling_linear():
  s = "a@0_0 = ; b@2_0 = ; c@10_0 = ; d@12_0 = cong a b c d ? cong a b c d"
  d, p = solve(s)
  assert d.check_pred(p.goal)

  a, b, c, dd = (_pt(p, x) for x in ("a", "b", "c", "d"))
  x = d.get_len_poly(a, b)
  y = d.get_len_poly(c, dd)

  poly = 2 * x - 2 * y
  assert d.check_poly_zero(poly)


def test_scaling_quadratic():
  s = "a@0_0 = ; b@3_0 = ; c@10_0 = ; d@13_0 = cong a b c d ? cong a b c d"
  d, p = solve(s)
  assert d.check_pred(p.goal)

  a, b, c, dd = (_pt(p, x) for x in ("a", "b", "c", "d"))
  x = d.get_len_poly(a, b)
  y = d.get_len_poly(c, dd)

  poly = (x - y) * (x - y)
  assert d.check_poly_zero(poly)


def test_cong_complex():
  s = (
      "a@0_0 = ; b@1_0 = ; c@2_0 = ; d@3_0 = ; e@4_0 = ; f@5_0 = ; g@6_0 = ; h@7_0 = "
      "cong a b c d, cong c d e f, cong e f g h ? cong a b g h"
  )
  d, p = solve(s)
  assert d.check_pred(p.goal)

  a, b, c, dd, e, f, g, h = (
      _pt(p, x) for x in ("a", "b", "c", "d", "e", "f", "g", "h")
  )
  lab = d.get_len_poly(a, b)
  lcd = d.get_len_poly(c, dd)
  lef = d.get_len_poly(e, f)
  lgh = d.get_len_poly(g, h)

  poly = 3 * lab - 2 * lcd + 5 * lef - 6 * lgh
  assert d.check_poly_zero(poly)


def test_force_complex():
  s = (
      "a@0_0 = ; b@2_0 = ; c@10_0 = ; d@13_0 = ; e@20_0 = ; f@22_0 = ; g@30_0 = ; h@33_0 = ? cong a b c d"
  )
  d, p = solve(s)

  a, b, c, dd, e, f, g, h = (
      _pt(p, x) for x in ("a", "b", "c", "d", "e", "f", "g", "h")
  )
  lab = d.get_len_poly(a, b)
  lcd = d.get_len_poly(c, dd)
  lef = d.get_len_poly(e, f)
  lgh = d.get_len_poly(g, h)

  # Force ab*cd = ef*gh
  p0 = lab * lcd - lef * lgh
  changed = d.force_poly_zero(p0)
  assert changed
  assert d.check_poly_zero(p0)

  # 2*(ab*cd - ef*gh) + (ab*cd - ef*gh)^2 = 0
  q = 2 * p0 + p0 * p0
  assert d.check_poly_zero(q)


def test_mixed_linear_quad():
  s = "a@0_0 = ; b@4_0 = ; c@10_0 = ; d@14_0 = cong a b c d ? cong a b c d"
  d, p = solve(s)
  assert d.check_pred(p.goal)

  a, b, c, dd = (_pt(p, x) for x in ("a", "b", "c", "d"))
  x = d.get_len_poly(a, b)
  y = d.get_len_poly(c, dd)

  # x^2 - x*y + 7x - (y^2 - x*y + 7y) = (x-y)(x+y+7) = 0
  poly = (x * x - x * y + 7 * x) - (y * y - x * y + 7 * y)
  assert d.check_poly_zero(poly)


def test_basic_linear_system():
  s = (
      "a@0_0 = ; b@5_0 = ; c@10_0 = ; d@15_0 = ; e@20_0 = ; f@25_0 = "
      "cong a b c d, cong c d e f ? cong a b e f"
  )
  d, p = solve(s)
  assert d.check_pred(p.goal)

  a, b, e, f = (_pt(p, x) for x in ("a", "b", "e", "f"))
  x = d.get_len_poly(a, b)
  z = d.get_len_poly(e, f)

  poly = x - z
  assert d.check_poly_zero(poly)


def test_complex_negative():
  s = "a@0_0 = ; b@2_0 = ; c@0_0 = ; d@5_0 = ; e@0_0 = ; f@7_0 = ? cong a b c d"
  d, p = solve(s)

  a, b, c, dd, e, f = (_pt(p, x) for x in ("a", "b", "c", "d", "e", "f"))
  x = d.get_len_poly(a, b)
  y = d.get_len_poly(c, dd)
  z = d.get_len_poly(e, f)

  poly = x * x + 3 * y - z * z
  assert not d.check_poly_zero(poly)


def test_eqratio_bridge():
  s = (
      "a@0_0 = ; b@2_0 = ; c@0_0 = ; d@3_0 = ; "
      "e@10_0 = ; f@14_0 = ; g@10_0 = ; h@16_0 = eqratio a b c d e f g h ? eqratio a b c d e f g h"
  )
  d, p = solve(s)
  assert d.check_pred(p.goal)
  a, b, c, dd, e, f, g, h = (
      _pt(p, x) for x in ("a", "b", "c", "d", "e", "f", "g", "h")
  )
  lab = d.get_len_poly(a, b)
  lcd = d.get_len_poly(c, dd)
  lef = d.get_len_poly(e, f)
  lgh = d.get_len_poly(g, h)

  poly = lab * lgh - lcd * lef
  assert d.check_poly_zero(poly)


def main():
  # Basic
  test_bridge_cong()
  test_bridge_cong_trans()
  test_manual_force()
  test_no_numerics()
  test_by_trivial()

  # Complex
  test_scaling_linear()
  test_scaling_quadratic()
  test_cong_complex()
  test_force_complex()
  test_mixed_linear_quad()
  test_basic_linear_system()
  test_complex_negative()

  # Eqratio
  test_eqratio_bridge()

  print("All poly tests passed!")


if __name__ == "__main__":
  main()
