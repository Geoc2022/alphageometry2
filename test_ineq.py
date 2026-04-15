from ddar import DDAR
from parse import AGProblem


def solve(problem_str):
  p = AGProblem.parse(problem_str)
  d = DDAR(p.points)
  for pred in p.preds:
    d.force_pred(pred)
  d.deduction_closure(progress_dot=False)
  return d, p


def test_ltdist_basic():
  s = (
    "a@0_0 = ; b@1_0 = ; c@2_0 = ltdist a b a c ? ltdist a b a c"
  )
  d, p = solve(s)
  assert d.check_pred(p.goal)


def test_ledist_basic():
  s = (
    "a@0_0 = ; b@1_0 = ; c@2_0 = ledist a b a c ? ledist a b a c"
  )
  d, p = solve(s)
  assert d.check_pred(p.goal)


def test_ltdist_via_eq_subst():
  s = (
    "a@0_0 = ; b@1_0 = ; c@10_0 = ; d@13_0 = ; e@20_0 = ; f@23_0 = ltdist a b c d, cong c d e f ? ltdist a b e f"
  )
  d, p = solve(s)
  assert d.check_pred(p.goal)


def test_distslt_linear_combo():
  s = (
    "a@0_0 = ; b@1_0 = ; c@3_0 = distslt a b a c 1 -1 ? ltdist a b a c"
  )
  d, p = solve(s)
  assert d.check_pred(p.goal)


def test_distsle_linear_combo():
  s = (
    "a@0_0 = ; b@1_0 = ; c@3_0 = coll a b c, distsle a b b c a c 1 1 -1 ? distsle a b b c a c 1 1 -1"
  )
  d, p = solve(s)
  assert d.check_pred(p.goal)


def test_ratio_rlt():
  s = (
    "a@0_0 = ; b@1_0 = ; c@3_0 = rlt a b a c 1 ? rlt a b a c 1"
  )
  d, p = solve(s)
  assert d.check_pred(p.goal)


def test_ratio_chain_with_cong():
  s = (
    "a@0_0 = ; b@1_0 = ; c@10_0 = ; d@13_0 = ; e@20_0 = ; f@23_0 = rlt a b c d 1, cong c d e f ? rlt a b e f 1"
  )
  d, p = solve(s)
  assert d.check_pred(p.goal)


def test_strict_implies_nonstrict():
  s = (
    "a@0_0 = ; b@1_0 = ; c@3_0 = ltdist a b a c ? ledist a b a c"
  )
  d, p = solve(s)
  assert d.check_pred(p.goal)


def test_ratio_rlt_const():
  s = (
    "a@0_0 = ; b@1_0 = ; c@3_0 = rlt a b a c 1/2 ? rlt a b a c 1/2"
  )
  d, p = solve(s)
  assert d.check_pred(p.goal)


def test_ratio_rlt_weaker():
  s = (
    "a@0_0 = ; b@1_0 = ; c@3_0 = rlt a b a c 1/3 ? rlt a b a c 1/2"
  )
  d, p = solve(s)
  assert d.check_pred(p.goal)


def test_ratio_rlt_const_false():
  s = (
    "a@0_0 = ; b@1_0 = ; c@3_0 = rlt a b a c 1/2 ? rlt a b a c 1/3"
  )
  d, p = solve(s)
  assert not d.check_pred(p.goal)


def test_transitivity_lt():
  s = (
    "a@0_0 = ; b@1_0 = ; c@10_0 = ; d@20_0 = ltdist a b b c, ltdist b c c d ? ltdist a b c d"
  )
  d, p = solve(s)
  assert d.check_pred(p.goal)


def test_transitivity_le():
  s = (
    "a@0_0 = ; b@1_0 = ; c@10_0 = ; d@20_0 = ledist a b b c, ledist b c c d ? ledist a b c d"
  )
  d, p = solve(s)
  assert d.check_pred(p.goal)


def test_triangle_inequality():
  s = (
    "a@0_0 = ; b@1_0 = ; c@2_0 = coll a b c ? distsle a b b c a c -1 -1 1"
  )
  d, p = solve(s)
  assert d.check_pred(p.goal)


def test_triangle_inequality_strict():
  s = (
    "a@0_0 = ; b@1_0 = ; c@0_1 = ? distslt a b b c a c -1 -1 1"
  )
  d, p = solve(s)
  assert d.check_pred(p.goal)


def test_triangle_inequality_with_cong():
  s = (
    "a@0_0 = ; b@1_0 = ; c@2_0 = ; d@1_1 = coll a b c, cong b c b d ? distsle a b b d a c -1 -1 1"
  )
  d, p = solve(s)
  assert d.check_pred(p.goal)


def test_triangle_inequality_weaker():
  s = (
    "a@0_0 = ; b@1_0 = ; c@2_0 = ; d@1_1 = coll a b c, cong b c b d ? distsle a b b c a c b c -1 -1 1 -1"
  )
  d, p = solve(s)
  assert d.check_pred(p.goal)


def test_triangle_inequality_weaker2():
  s = (
    "a@0_0 = ; b@1_0 = ; c@2_0 = coll a b c ? distsle a b b c a c -2 -1 1"
  )
  d, p = solve(s)
  assert d.check_pred(p.goal)


def main():
  test_ltdist_basic()
  test_ledist_basic()
  test_ltdist_via_eq_subst()
  test_distslt_linear_combo()
  test_distsle_linear_combo()
  test_ratio_rlt()
  test_ratio_chain_with_cong()
  test_strict_implies_nonstrict()
  test_ratio_rlt_const()
  test_ratio_rlt_const_false()
  test_transitivity_lt()
  test_transitivity_le()

  print("All passed!")

if __name__ == "__main__":
  main()
