"""Serialization and deserialization of DDAR state for database storage."""

import collections
import fractions
import itertools
import numpy as np

import elimination as el
import numericals as ng
from parse import AGPoint
from ddar import DDAR, FormalLine, FormalCircle


class DDARSerializer:
    """Converts DDAR internal state to flat row-lists for database insertion."""

    @staticmethod
    def serialize_elim_core(
        core: el.ElimCore, elim_type: str, snapshot_id: str
    ) -> list[dict]:
        rows = []
        for pivot_var, lc in core.instantiated.items():
            for free_var, coef in lc.d.items():
                rows.append(
                    {
                        "snapshot_id": snapshot_id,
                        "elim_type": elim_type,
                        "pivot_var_name": pivot_var.name,
                        "pivot_var_value": float(pivot_var.value),
                        "free_var_name": free_var.name,
                        "free_var_value": float(free_var.value),
                        "coef_num": int(coef.numerator),
                        "coef_den": int(coef.denominator),
                    }
                )
        return rows

    @staticmethod
    def serialize_ddar(ddar: DDAR, snapshot_id: str) -> dict:
        result = {
            "points": [],
            "elim_rows": [],
            "lines": [],
            "line_points": [],
            "circles": [],
            "circle_points": [],
            "point_subst": [],
            "known_similar": [],
        }

        for p in ddar.points:
            result["points"].append(
                {
                    "snapshot_id": snapshot_id,
                    "point_name": p.name,
                    "point_x": float(p.value[0]),
                    "point_y": float(p.value[1]),
                }
            )

        for elim_type, elim_obj in [
            ("angle", ddar.elim_angle),
            ("dist_mul", ddar.elim_dist_mul),
            ("dist_add", ddar.elim_dist_add),
        ]:
            result["elim_rows"].extend(
                DDARSerializer.serialize_elim_core(
                    elim_obj.core, elim_type, snapshot_id
                )
            )

        for line_id, line in enumerate(ddar.lines):
            dir_lhs_name = next(iter(line.direction.comb.d)).name
            result["lines"].append(
                {
                    "snapshot_id": snapshot_id,
                    "line_id": line_id,
                    "main_pair_a": line.main_pair[0].name,
                    "main_pair_b": line.main_pair[1].name,
                    "dir_var_name": dir_lhs_name,
                    "dir_var_value": float(line.value.direction()),
                    "line_nx": float(line.value.n[0]),
                    "line_ny": float(line.value.n[1]),
                    "line_c": float(line.value.c),
                }
            )
            for point in line.points:
                result["line_points"].append(
                    {
                        "snapshot_id": snapshot_id,
                        "line_id": line_id,
                        "point_name": point.name,
                        "position": float(line.value.position(point.value)),
                    }
                )

        for circle_id, circle in enumerate(ddar.circles):
            result["circles"].append(
                {
                    "snapshot_id": snapshot_id,
                    "circle_id": circle_id,
                    "center_x": float(circle.value.center[0]),
                    "center_y": float(circle.value.center[1]),
                    "radius": float(circle.value.r),
                }
            )
            defining_names = {p.name for p in circle.defining_points}
            for point in circle.points:
                result["circle_points"].append(
                    {
                        "snapshot_id": snapshot_id,
                        "circle_id": circle_id,
                        "point_name": point.name,
                        "is_center": False,
                        "is_defining": point.name in defining_names,
                    }
                )
            for center in circle.centers:
                result["circle_points"].append(
                    {
                        "snapshot_id": snapshot_id,
                        "circle_id": circle_id,
                        "point_name": center.name,
                        "is_center": True,
                        "is_defining": False,
                    }
                )

        for from_p, to_p in ddar.point_subst.items():
            result["point_subst"].append(
                {
                    "snapshot_id": snapshot_id,
                    "from_point": from_p.name,
                    "to_point": to_p.name,
                }
            )

        point_names = {p.name for p in ddar.points}
        seen = set()

        def _canon(p: AGPoint) -> str:
            return ddar.point_subst.get(p, p).name

        for t1, t2 in ddar.known_similar:
            t1_names = (_canon(t1[0]), _canon(t1[1]), _canon(t1[2]))
            t2_names = (_canon(t2[0]), _canon(t2[1]), _canon(t2[2]))

            if not all(n in point_names for n in t1_names + t2_names):
                continue

            key = min((t1_names, t2_names), (t2_names, t1_names))
            if key in seen:
                continue
            seen.add(key)
            (a1, b1, c1), (a2, b2, c2) = key
            result["known_similar"].append(
                {
                    "snapshot_id": snapshot_id,
                    "t1_a": a1,
                    "t1_b": b1,
                    "t1_c": c1,
                    "t2_a": a2,
                    "t2_b": b2,
                    "t2_c": c2,
                }
            )

        return result


class DDARDeserializer:
    """Reconstructs a DDAR instance from serialized row-lists."""

    @staticmethod
    def _reconstruct_free_var(name: str, value: float, elim_type: str) -> el.ElimVar:
        if elim_type == "angle" and name == "pi":
            return el.angle_unit
        if elim_type == "dist_mul" and name.startswith("log(") and name.endswith(")"):
            try:
                return el.DistMulConst.prime_value(int(name[4:-1]))
            except ValueError:
                pass
        return el.ElimLHS(value, name)

    @staticmethod
    def _deserialize_elim_core(rows: list[dict], elim_type: str) -> el.ElimCore:
        core = el.ElimCore()
        pivot_to_rows = collections.defaultdict(list)
        for row in rows:
            if row["elim_type"] == elim_type:
                pivot_to_rows[(row["pivot_var_name"], row["pivot_var_value"])].append(
                    row
                )

        dst_free_by_name = {}
        for (pivot_name, pivot_value), group in pivot_to_rows.items():
            pivot_var = el.ElimLHS(pivot_value, pivot_name)
            dst_free_by_name[pivot_name] = pivot_var

            lc_d = {}
            for row in group:
                free_var = DDARDeserializer._reconstruct_free_var(
                    row["free_var_name"], row["free_var_value"], elim_type
                )
                canonical = dst_free_by_name.setdefault(free_var.name, free_var)
                lc_d[canonical] = fractions.Fraction(row["coef_num"], row["coef_den"])

            lc = el.LinComb(lc_d)
            core.instantiated[pivot_var] = lc
            for free_var in lc_d:
                core.free_to_usage[free_var].add(pivot_var)

        return core

    @staticmethod
    def _collect_lhs_vars(core: el.ElimCore) -> dict[str, el.ElimLHS]:
        result = {}
        for pivot, lc in core.instantiated.items():
            result[pivot.name] = pivot
            for var in lc.d:
                if isinstance(var, el.ElimLHS):
                    result[var.name] = var
        return result

    @staticmethod
    def reconstruct(
        point_rows: list[dict],
        elim_rows: list[dict],
        line_rows: list[dict],
        line_point_rows: list[dict],
        circle_rows: list[dict],
        circle_point_rows: list[dict],
        subst_rows: list[dict],
        similar_rows: list[dict] | None = None,
    ) -> DDAR:
        name_to_point = {
            row["point_name"]: AGPoint(
                row["point_name"], np.array([row["point_x"], row["point_y"]])
            )
            for row in point_rows
        }
        points = list(name_to_point.values())

        ddar = DDAR.__new__(DDAR)
        ddar.points = points
        ddar.lines = set()
        ddar.circles = set()
        ddar.point_subst = {p: p for p in points}
        ddar.pair_to_line = {}
        ddar.pair_to_dist_mul = {}
        ddar.pair_to_dist_add = {}
        ddar.pair_to_dir = {}
        ddar.triple_to_circle = {}
        ddar.known_similar = set()
        ddar.last_small_circles = []

        ddar.elim_angle = el.ElimAngle()
        ddar.elim_angle.core = DDARDeserializer._deserialize_elim_core(
            elim_rows, "angle"
        )
        ddar.elim_dist_mul = el.ElimDistMul()
        ddar.elim_dist_mul.core = DDARDeserializer._deserialize_elim_core(
            elim_rows, "dist_mul"
        )
        ddar.elim_dist_add = el.ElimDistAdd()
        ddar.elim_dist_add.core = DDARDeserializer._deserialize_elim_core(
            elim_rows, "dist_add"
        )

        all_lhs_angle = DDARDeserializer._collect_lhs_vars(ddar.elim_angle.core)
        all_lhs_mul = DDARDeserializer._collect_lhs_vars(ddar.elim_dist_mul.core)
        all_lhs_add = DDARDeserializer._collect_lhs_vars(ddar.elim_dist_add.core)

        for a, b in itertools.combinations(points, 2):
            if ng.distance(a.value, b.value) < ng.ATOM:
                continue

            dist = ng.distance(a.value, b.value)

            # Direction
            dn, dna = f"d({a} {b})", f"d({b} {a})"
            dir_lhs = (
                all_lhs_angle.get(dn)
                or all_lhs_angle.get(dna)
                or el.ElimLHS(
                    ng.NumLine.through(a.value, b.value).direction(),
                    dn if a.name <= b.name else dna,
                )
            )
            direction = el.FormalAngle(el.LinComb.singleton(dir_lhs))
            ddar.pair_to_dir[a, b] = ddar.pair_to_dir[b, a] = direction

            # Multiplicative distance
            mn, mna = f"log(|{a} {b}|)", f"log(|{b} {a}|)"
            mul_lhs = (
                all_lhs_mul.get(mn) or all_lhs_mul.get(mna) or el.ElimLHS(dist, mn)
            )
            dist_mul = el.DistMul(el.LinComb.singleton(mul_lhs))
            ddar.pair_to_dist_mul[a, b] = ddar.pair_to_dist_mul[b, a] = dist_mul

            # Additive distance
            an, ana = f"|{a} {b}|", f"|{b} {a}|"
            add_lhs = (
                all_lhs_add.get(an) or all_lhs_add.get(ana) or el.ElimLHS(dist, an)
            )
            dist_add = el.DistAdd(el.LinComb.singleton(add_lhs))
            ddar.pair_to_dist_add[a, b] = ddar.pair_to_dist_add[b, a] = dist_add

        line_point_by_id = collections.defaultdict(list)
        for row in line_point_rows:
            line_point_by_id[row["line_id"]].append(row)

        for line_row in line_rows:
            lid = line_row["line_id"]
            lp_rows = sorted(line_point_by_id[lid], key=lambda r: r["position"])
            line_points = [name_to_point[r["point_name"]] for r in lp_rows]

            num_line = ng.NumLine(
                n=np.array([line_row["line_nx"], line_row["line_ny"]]),
                c=line_row["line_c"],
            )

            dir_name = line_row["dir_var_name"]
            dir_lhs = all_lhs_angle.get(dir_name) or el.ElimLHS(
                line_row["dir_var_value"], dir_name
            )

            formal_line = FormalLine(
                points=line_points,
                main_pair=(
                    name_to_point[line_row["main_pair_a"]],
                    name_to_point[line_row["main_pair_b"]],
                ),
                direction=el.FormalAngle(el.LinComb.singleton(dir_lhs)),
                value=num_line,
            )
            ddar.lines.add(formal_line)
            for x, y in itertools.combinations(line_points, 2):
                if not ddar.num_identical(x, y):
                    ddar.pair_to_line[x, y] = ddar.pair_to_line[y, x] = formal_line

        for a, b in itertools.combinations(ddar.points, 2):
            if not ddar.num_identical(a, b) and (a, b) not in ddar.pair_to_line:
                stub = FormalLine(
                    points=[a, b],
                    main_pair=(a, b),
                    direction=ddar.pair_to_dir[a, b],
                    value=ng.NumLine.through(a.value, b.value),
                )
                ddar.lines.add(stub)
                ddar.pair_to_line[a, b] = ddar.pair_to_line[b, a] = stub

        cp_by_id = collections.defaultdict(list)
        for row in circle_point_rows:
            cp_by_id[row["circle_id"]].append(row)

        for circle_row in circle_rows:
            cp_rows = cp_by_id[circle_row["circle_id"]]
            circle_pts = [
                name_to_point[r["point_name"]] for r in cp_rows if not r["is_center"]
            ]

            formal_circle = FormalCircle(
                defining_points=[
                    name_to_point[r["point_name"]] for r in cp_rows if r["is_defining"]
                ],
                points=circle_pts,
                centers=[
                    name_to_point[r["point_name"]] for r in cp_rows if r["is_center"]
                ],
                value=ng.NumCircle(
                    center=np.array([circle_row["center_x"], circle_row["center_y"]]),
                    r=circle_row["radius"],
                ),
            )
            ddar.circles.add(formal_circle)
            for a, b, c in itertools.permutations(circle_pts, 3):
                if not (
                    ddar.num_identical(a, b)
                    or ddar.num_identical(b, c)
                    or ddar.num_identical(a, c)
                ):
                    ddar.triple_to_circle[a, b, c] = formal_circle

        for row in subst_rows:
            from_p = name_to_point.get(row["from_point"])
            to_p = name_to_point.get(row["to_point"])
            if from_p and to_p:
                ddar.point_subst[from_p] = to_p

        ddar.dist_mul_cache = dict(ddar.pair_to_dist_mul)
        ddar.direction_cache = dict(ddar.pair_to_dir)
        ddar.update_cache()

        if similar_rows:
            for row in similar_rows:
                names = (
                    row["t1_a"],
                    row["t1_b"],
                    row["t1_c"],
                    row["t2_a"],
                    row["t2_b"],
                    row["t2_c"],
                )
                if any(n not in name_to_point for n in names):
                    continue
                t1 = tuple(name_to_point[n] for n in names[:3])
                t2 = tuple(name_to_point[n] for n in names[3:])
                a, b, c = t1
                x, y, z = t2
                ddar.known_similar.update(
                    [
                        ((a, b, c), (x, y, z)),
                        ((a, c, b), (x, z, y)),
                        ((b, a, c), (y, x, z)),
                        ((c, a, b), (z, x, y)),
                        ((b, c, a), (y, z, x)),
                        ((c, b, a), (z, y, x)),
                        ((x, y, z), (a, b, c)),
                        ((x, z, y), (a, c, b)),
                        ((y, x, z), (b, a, c)),
                        ((z, x, y), (c, a, b)),
                        ((y, z, x), (b, c, a)),
                        ((z, y, x), (c, b, a)),
                    ]
                )

        return ddar


def _clone_ddar(source: DDAR) -> DDAR:
    """Return a shallow-structure copy of a DDAR safe for independent mutation."""
    cloned = DDAR.__new__(DDAR)
    cloned.points = list(source.points)
    cloned.lines = set(source.lines)
    cloned.circles = set(source.circles)
    cloned.elim_dist_mul = source.elim_dist_mul.clone()
    cloned.elim_dist_add = source.elim_dist_add.clone()
    cloned.elim_angle = source.elim_angle.clone()
    cloned.point_subst = dict(source.point_subst)
    cloned.pair_to_line = dict(source.pair_to_line)
    cloned.pair_to_dist_mul = dict(source.pair_to_dist_mul)
    cloned.pair_to_dist_add = dict(source.pair_to_dist_add)
    cloned.pair_to_dir = dict(source.pair_to_dir)
    cloned.triple_to_circle = dict(source.triple_to_circle)
    cloned.known_similar = set(source.known_similar)
    cloned.last_small_circles = list(source.last_small_circles)
    cloned.dist_mul_cache = dict(source.dist_mul_cache)
    cloned.direction_cache = dict(source.direction_cache)
    return cloned
