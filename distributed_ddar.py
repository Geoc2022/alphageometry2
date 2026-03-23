"""Distributed DDAR engine for multi-worker geometry proof search."""

import json
import threading
import time
import uuid
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
import numpy as np

from ddar import DDAR, FormalLine
import numericals as ng
import elimination as el
from parse import AGProblem, AGPoint, AGPredicate
from db_interface import DDARSerializer, DDARDeserializer, _clone_ddar


_SNAPSHOT_RELATIONS: list[tuple[str, str, tuple[str, ...]]] = [
    (
        "points",
        "snapshot_points",
        ("snapshot_id", "point_name", "point_x", "point_y"),
    ),
    (
        "elim_rows",
        "elim_rows",
        (
            "snapshot_id",
            "elim_type",
            "pivot_var_name",
            "pivot_var_value",
            "free_var_name",
            "free_var_value",
            "coef_num",
            "coef_den",
        ),
    ),
    (
        "lines",
        "snapshot_lines",
        (
            "snapshot_id",
            "line_id",
            "main_pair_a",
            "main_pair_b",
            "dir_var_name",
            "dir_var_value",
            "line_nx",
            "line_ny",
            "line_c",
        ),
    ),
    (
        "line_points",
        "line_points",
        ("snapshot_id", "line_id", "point_name", "position"),
    ),
    (
        "circles",
        "snapshot_circles",
        ("snapshot_id", "circle_id", "center_x", "center_y", "radius"),
    ),
    (
        "circle_points",
        "circle_points",
        ("snapshot_id", "circle_id", "point_name", "is_center", "is_defining"),
    ),
    (
        "point_subst",
        "point_subst",
        ("snapshot_id", "from_point", "to_point"),
    ),
    (
        "known_similar",
        "known_similar",
        ("snapshot_id", "t1_a", "t1_b", "t1_c", "t2_a", "t2_b", "t2_c"),
    ),
]


class DDARStateCache:
    """LRU cache of DDAR instances on a worker."""

    def __init__(self, max_size: int = 32):
        self.max_size = max_size
        self._cache: dict[str, DDAR] = {}
        self._access_times: dict[str, float] = {}
        self._lock = threading.Lock()

    def get(self, snapshot_id: str) -> DDAR | None:
        with self._lock:
            if snapshot_id in self._cache:
                self._access_times[snapshot_id] = time.monotonic()
                return self._cache[snapshot_id]
        return None

    def put(self, snapshot_id: str, ddar: DDAR) -> None:
        with self._lock:
            if len(self._cache) >= self.max_size:
                lru = min(self._access_times, key=lambda k: self._access_times[k])
                del self._cache[lru]
                del self._access_times[lru]
            self._cache[snapshot_id] = ddar
            self._access_times[snapshot_id] = time.monotonic()


class DistributedDDAR:
    """DDAR wrapper for distributed work over a shared database."""

    def __init__(self, db_config: dict, worker_id: str):
        self.db_config = db_config
        self.worker_id = worker_id
        self.state_cache = DDARStateCache(max_size=64)

    @contextmanager
    def _db(self):
        conn = psycopg2.connect(**self.db_config)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def submit_problem(
        self,
        problem: AGProblem,
        parent_problem_id: str | None = None,
        added_preds: list[AGPredicate | str] | None = None,
        problem_name: str | None = None,
    ) -> str:
        problem_id = str(uuid.uuid4())
        with self._db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO problems
                        (problem_id, parent_id, problem_name, base_pstring,
                         added_preds, depth, status, task_type,
                         join_ready, created_at)
                    VALUES (%s, %s, %s, %s, %s,
                        COALESCE(
                            (SELECT depth+1 FROM problems
                             WHERE problem_id=%s), 0
                        ),
                        'pending', 'regular', TRUE, NOW())
                """,
                    (
                        problem_id,
                        parent_problem_id,
                        problem_name,
                        problem.pstring(),
                        json.dumps([str(p) for p in (added_preds or [])]),
                        parent_problem_id,
                    ),
                )
        return problem_id

    def submit_join_problem(
        self,
        problem: AGProblem,
        dep_problem_ids: list[str],
        base_snapshot_id: str,
        added_preds: list[AGPredicate | str] | None = None,
        parent_problem_id: str | None = None,
        problem_name: str | None = None,
    ) -> str:
        problem_id = str(uuid.uuid4())
        meta = {
            "__join__": True,
            "base_snapshot_id": base_snapshot_id,
            "dep_problem_ids": dep_problem_ids,
        }
        preds_payload = json.dumps(
            [json.dumps(meta)] + [str(p) for p in (added_preds or [])]
        )
        join_ready = len(dep_problem_ids) == 0

        with self._db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO problems
                        (problem_id, parent_id, problem_name, base_pstring,
                         added_preds, depth, status, task_type, join_ready)
                    VALUES (%s, %s, %s, %s, %s,
                            COALESCE((SELECT MAX(depth) + 1 FROM problems WHERE problem_id = ANY(%s::text[])), 0),
                            'pending', 'join', %s)
                """,
                    (
                        problem_id,
                        parent_problem_id,
                        problem_name,
                        problem.pstring(),
                        preds_payload,
                        dep_problem_ids,
                        join_ready,
                    ),
                )
                if dep_problem_ids:
                    psycopg2.extras.execute_values(
                        cur,
                        "INSERT INTO problem_dependencies (join_problem_id, dep_problem_id) VALUES %s",
                        [(problem_id, dep_id) for dep_id in dep_problem_ids],
                    )
        return problem_id

    def claim_task(self) -> dict | None:
        with self._db() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    UPDATE problems SET status = 'running', worker_id = %s
                    WHERE problem_id = (
                        SELECT p.problem_id FROM problems p
                        WHERE p.status = 'pending' AND p.join_ready = TRUE
                          AND (
                            (p.task_type = 'regular' AND (p.parent_id IS NULL OR EXISTS (SELECT 1 FROM state_snapshots s WHERE s.problem_id = p.parent_id)))
                            OR
                            (p.task_type = 'join' AND NOT EXISTS (
                                SELECT 1 FROM problem_dependencies pd WHERE pd.join_problem_id = p.problem_id AND NOT EXISTS (SELECT 1 FROM state_snapshots s WHERE s.problem_id = pd.dep_problem_id)
                            ))
                          )
                        ORDER BY p.depth ASC, p.created_at ASC FOR UPDATE SKIP LOCKED LIMIT 1
                    ) RETURNING *
                """,
                    (self.worker_id,),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def _write_snapshot_rows(
        self, cur, snapshot_id: str, problem_id: str, serialized: dict
    ) -> None:
        cur.execute(
            "INSERT INTO state_snapshots (snapshot_id, problem_id) VALUES (%s, %s)",
            (snapshot_id, problem_id),
        )
        for key, table, columns in _SNAPSHOT_RELATIONS:
            if rows := serialized[key]:
                psycopg2.extras.execute_values(
                    cur,
                    f"INSERT INTO {table} ({', '.join(columns)}) VALUES %s",
                    [tuple(r[col] for col in columns) for r in rows],
                )

    def _write_result(self, cur, problem_id: str, proved: bool) -> None:
        cur.execute(
            "UPDATE problems SET status = %s, completed_at = NOW() WHERE problem_id = %s",
            ("proved" if proved else "failed", problem_id),
        )

    def report_result(self, problem_id: str, proved: bool) -> None:
        with self._db() as conn, conn.cursor() as cur:
            self._write_result(cur, problem_id, proved)

    def save_snapshot_and_report(
        self, ddar: DDAR, problem_id: str, proved: bool
    ) -> str:
        snapshot_id = str(uuid.uuid4())
        serialized = DDARSerializer.serialize_ddar(ddar, snapshot_id)
        with self._db() as conn, conn.cursor() as cur:
            self._write_snapshot_rows(cur, snapshot_id, problem_id, serialized)
            self._write_result(cur, problem_id, proved)
        self.state_cache.put(snapshot_id, ddar)
        return snapshot_id

    def load_snapshot(self, snapshot_id: str) -> DDAR:
        if cached := self.state_cache.get(snapshot_id):
            return cached

        fetched = {}
        with (
            self._db() as conn,
            conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur,
        ):
            for key, table, _ in _SNAPSHOT_RELATIONS:
                cur.execute(
                    f"SELECT * FROM {table} WHERE snapshot_id = %s {'ORDER BY line_id, position' if table == 'line_points' else ''}",
                    (snapshot_id,),
                )
                fetched[key] = cur.fetchall()

        ddar = DDARDeserializer.reconstruct(
            fetched["points"],
            fetched["elim_rows"],
            fetched["lines"],
            fetched["line_points"],
            fetched["circles"],
            fetched["circle_points"],
            fetched["point_subst"],
            fetched["known_similar"],
        )
        self.state_cache.put(snapshot_id, ddar)
        return ddar

    def _build_fresh(self, problem: AGProblem) -> DDAR:
        ddar = DDAR(problem.points)
        for pred in problem.preds:
            ddar.force_pred(pred)
        return ddar

    def _register_new_point(self, ddar: DDAR, point: AGPoint) -> None:
        ddar.points.append(point)
        ddar.point_subst[point] = point
        for other in ddar.points[:-1]:
            dist = ng.distance(point.value, other.value)
            if dist < ng.ATOM:
                ddar.point_subst[point] = ddar.point_subst.get(other, other)
                continue

            num_line = ng.NumLine.through(point.value, other.value)
            a_name, b_name = sorted([point.name, other.name])

            dir_name = f"d({a_name} {b_name})"
            all_lhs_angle = DDARDeserializer._collect_lhs_vars(ddar.elim_angle.core)
            direction = (
                el.FormalAngle(el.LinComb.singleton(all_lhs_angle[dir_name]))
                if dir_name in all_lhs_angle
                else ddar.elim_angle.new_var(num_line.direction(), dir_name)
            )

            mul_name = f"log(|{a_name} {b_name}|)"
            all_lhs_mul = DDARDeserializer._collect_lhs_vars(ddar.elim_dist_mul.core)
            dist_mul = (
                el.DistMul(el.LinComb.singleton(all_lhs_mul[mul_name]))
                if mul_name in all_lhs_mul
                else ddar.elim_dist_mul.new_var(dist, mul_name)
            )

            add_name = f"|{a_name} {b_name}|"
            all_lhs_add = DDARDeserializer._collect_lhs_vars(ddar.elim_dist_add.core)
            dist_add = (
                el.DistAdd(el.LinComb.singleton(all_lhs_add[add_name]))
                if add_name in all_lhs_add
                else ddar.elim_dist_add.new_var(dist, add_name)
            )

            ddar.pair_to_dir[point, other] = ddar.pair_to_dir[other, point] = direction
            ddar.pair_to_dist_mul[point, other] = ddar.pair_to_dist_mul[
                other, point
            ] = dist_mul
            ddar.pair_to_dist_add[point, other] = ddar.pair_to_dist_add[
                other, point
            ] = dist_add

            formal_line = FormalLine(
                [point, other], (point, other), direction, num_line
            )
            ddar.lines.add(formal_line)
            ddar.pair_to_line[point, other] = ddar.pair_to_line[other, point] = (
                formal_line
            )

    def load_and_extend(
        self,
        parent_snapshot_id: str | None,
        problem: AGProblem,
        new_preds: list[AGPredicate],
    ) -> DDAR:
        ddar = (
            _clone_ddar(self.load_snapshot(parent_snapshot_id))
            if parent_snapshot_id
            else self._build_fresh(problem)
        )
        name_to_point = {p.name: p for p in ddar.points}

        for pred in new_preds:
            points = []
            for point in pred.points:
                assert isinstance(point, AGPoint), (
                    f"load_and_extend expects resolved AGPoint objects, "
                    f"got {type(point)}: {point!r}"
                )
                if point.name not in name_to_point:
                    self._register_new_point(ddar, point)
                    name_to_point[point.name] = point
                points.append(point)
            ddar.force_pred(
                pred.replace_points({p: name_to_point[p.name] for p in points})
            )

        ddar.deduction_closure(verbose=False, progress_dot=False)
        return ddar

    def load_and_merge_extend(
        self,
        base_snapshot_id: str,
        dep_problem_ids: list[str],
        problem: AGProblem,
        new_preds: list[AGPredicate],
    ) -> DDAR:
        base_ddar = _clone_ddar(self.load_snapshot(base_snapshot_id))
        name_to_point = {p.name: p for p in base_ddar.points}

        for p in problem.points:
            if p.name not in name_to_point:
                self._register_new_point(base_ddar, p)
                name_to_point[p.name] = p

        for s in self._collect_dep_pred_strings(dep_problem_ids) + [
            str(p) for p in new_preds
        ]:
            if "@" in s and "=" in s:
                nm, coord = s.split("=", 1)[0].strip().split("@")
                nm = nm.strip()
                if nm not in name_to_point:
                    xs, ys = coord.strip().split("_")
                    pt = AGPoint(name=nm, value=np.array([float(xs), float(ys)]))
                    self._register_new_point(base_ddar, pt)
                    name_to_point[nm] = pt
                pred_parts = [
                    x.strip() for x in s.split("=", 1)[1].split(",") if x.strip()
                ]
            else:
                pred_parts = [s]

            for ps in pred_parts:
                pred = AGPredicate.parse(ps)
                base_ddar.force_pred(
                    pred.replace_points(
                        {
                            tok: name_to_point[
                                tok if isinstance(tok, str) else tok.name
                            ]
                            for tok in pred.points
                            if (tok if isinstance(tok, str) else tok.name)
                            in name_to_point
                        }
                    )
                )

        base_ddar.deduction_closure(verbose=False, progress_dot=False)
        return base_ddar

    def _collect_dep_pred_strings(self, dep_problem_ids: list[str]) -> list[str]:
        with (
            self._db() as conn,
            conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur,
        ):
            cur.execute(
                """
                WITH RECURSIVE anc AS (
                  SELECT p.problem_id, p.parent_id, p.depth, p.added_preds FROM problems p WHERE p.problem_id = ANY(%s::text[])
                  UNION SELECT pp.problem_id, pp.parent_id, pp.depth, pp.added_preds FROM problems pp JOIN anc ON anc.parent_id = pp.problem_id
                ) SELECT DISTINCT problem_id, depth, added_preds FROM anc ORDER BY depth ASC
            """,
                (dep_problem_ids,),
            )
            rows = cur.fetchall()

        out = []
        for r in rows:
            added = (
                json.loads(r["added_preds"])
                if isinstance(r["added_preds"], str)
                else (r["added_preds"] or [])
            )
            out.extend(
                str(s).strip()
                for s in added
                if str(s).strip() and not str(s).startswith("{")
            )
        return out
