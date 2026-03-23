"""Distributed DDAR demo"""

import json
import logging
import os
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import psycopg2

from db.setup_db import DB_CONFIG
from distributed_ddar import DistributedDDAR
from parse import AGPoint, AGPredicate, AGProblem
from test import problems_with_aux, problems_without_aux

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BATCH_EXPLANATION = """
Batch distributed test
----------------------
We run the same group of IMO DDAR problems across multiple workers to make sure
the distributed setup still behaves like the original logic-core tests.
"""

DAG_EXPLANATION = """
DAG distributed test (IMO 2021 P3 rabbit branches)
---------------------------------------------------
We run several rabbit branches in parallel, each on different workers, then join
their results. This shows a workflow where we can try different rabbits at once.
"""

BASE_PSTRING = (
    "a@0.35158228874560216_0.6253011491766167 = ;"
    " b@-0.32787335987583544_-0.11434443729989269 = ;"
    " c@0.593611241171965_-0.13630297414077203 = ;"
    " d@0.25541529417090475_0.19047220899818684 = ;"
    " e@0.4472680874521529_0.32420205294349663 = ;"
    " f@0.18152091649019214_0.4401748003500804 = ;"
    " x@0.1647645897963786_1.2131693707344662 = ;"
    " o1@0.7063662227386552_0.3187883084445305 = ;"
    " o2@0.0513262541690787_0.687748158526416 = ;"
    " y@1.5549937258179136_-0.15921225844489367 = "
    "coll y e f, coll y b c, eqangle a b a d a d a c,"
    " eqangle d e d a c d c b, coll e a c, eqangle d f d a b d b c, coll f"
    " a b, cong x b x c, eqangle b x b c c b c x, coll x a c, cong o1 a o1"
    " d, cong o1 d o1 c, cong o2 e o2 x, cong o2 x o2 d"
    " ? coll o1 o2 y"
)

AUXILIARY_STEPS: list[tuple[str, str]] = [
    ("da", "da@0.24023420886499053_-0.44659830430440767 = cong b d b da, cong c d c da, perp b c d da"),
    ("db", "db@0.6811243375628278_0.3257576028873138 = cong c d c db, cong a d a db, perp c a d db"),
    ("dc", "dc@-0.07354195379055742_0.49265999240284974 = cong a d a dc, cong b d b dc, perp a b d dc"),
    ("q", "q@0.22915675122419496_0.07174151089699222 = cong q da q db, cong q db q dc"),
    ("m", "m@0.49881279704621323_0.5293204195487172 = coll m a y, cyclic a b c m"),
    ("k", "k@0.5823757264495235_0.7709477031088625 = cong y k y d, cyclic a d c k"),
    ("yp", "yp@1.5549937258179136_-0.15921225844489367 = coll yp b c, eqangle d yp d c b d b c"),
    ("ep", "ep@0.4472680874521529_0.32420205294349663 = coll ep f yp, cyclic f b c ep"),
    ("p", "p@0.4052374543908715_0.4564618636983748 = coll p a c, coll p d k"),
    ("mp", "mp@0.49881279704621323_0.5293204195487172 = coll mp p b, cyclic b d k mp"),
]
_STEP_MAP = dict(AUXILIARY_STEPS)


def _truncate_db(db_config: dict) -> None:
    conn = psycopg2.connect(**db_config)
    with conn.cursor() as cur:
        cur.execute("""
            TRUNCATE delta_log, known_similar, point_subst,
                     circle_points, snapshot_circles,
                     line_points, snapshot_lines,
                     elim_rows, snapshot_points,
                     state_snapshots, problem_dependencies,
                     problems CASCADE
        """)
    conn.commit()
    conn.close()
    print("Database cleared.\n")


def _all_done(db_config: dict) -> bool:
    conn = psycopg2.connect(**db_config)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM problems WHERE status IN ('pending', 'running')")
        (remaining,) = cur.fetchone()
    conn.close()
    return remaining == 0


def _inspect(cmd: str = "all") -> None:
    print(f"\n[inspect] python -m db.inspect_snapshot {cmd}\n")
    subprocess.run([sys.executable, "-m", "db.inspect_snapshot", cmd], check=False)


def _start_watch(interval: float = 1.0):
    print(f"\n[inspect-watch] starting (interval={interval}s)\n")
    return subprocess.Popen([sys.executable, "-m", "db.inspect_snapshot", "watch", str(interval)])


def _stop_watch(proc) -> None:
    if not proc:
        return
    print("\n[inspect-watch] stopping\n")
    proc.terminate()
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        proc.kill()


def _parse_point_decl(step_pstring: str, name_to_point: dict[str, AGPoint]) -> tuple[AGPoint, list[AGPredicate]]:
    point_decl, pred_list_str = step_pstring.split("=", 1)
    name, coord = point_decl.strip().split("@")
    x_str, y_str = coord.strip().split("_")
    new_point = AGPoint(name=name.strip(), value=np.array([float(x_str), float(y_str)]))
    name_to_point[name.strip()] = new_point

    preds = []
    for pred_str in pred_list_str.strip().split(","):
        if not (pred_str := pred_str.strip()):
            continue
        pred = AGPredicate.parse(pred_str)
        preds.append(pred.replace_points({t: name_to_point[t] for t in pred.points if t in name_to_point}))
    return new_point, preds


def _decode_added_preds(raw: list[str], name_to_point: dict[str, AGPoint]) -> tuple[list[AGPredicate], dict | None]:
    if not raw:
        return [], None

    join_meta = None
    preds = []
    start = 0

    if raw[0].strip().startswith("{"):
        try:
            meta = json.loads(raw[0])
            if meta.get("__join__"):
                join_meta = meta
                start = 1
        except json.JSONDecodeError:
            pass

    for s in raw[start:]:
        if not (s := s.strip()):
            continue
        if "@" in s and "=" in s:
            _, step_preds = _parse_point_decl(s, name_to_point)
            preds.extend(step_preds)
        else:
            pred = AGPredicate.parse(s)
            preds.append(pred.replace_points({t: name_to_point[t] for t in pred.points if t in name_to_point}))

    return preds, join_meta


def _get_parent_snapshot(engine: DistributedDDAR, parent_id: str | None) -> str | None:
    if not parent_id:
        return None
    with engine._db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT snapshot_id FROM state_snapshots WHERE problem_id = %s ORDER BY created_at DESC LIMIT 1",
            (parent_id,),
        )
        row = cur.fetchone()
    return row[0] if row else None


def execute_task(engine: DistributedDDAR, task: dict) -> bool:
    problem_id = task["problem_id"]
    task_type = task.get("task_type", "regular")

    problem = AGProblem.parse(task["base_pstring"])
    name_to_point = {p.name: p for p in problem.points}
    added_preds, join_meta = _decode_added_preds(task.get("added_preds") or [], name_to_point)

    if task_type == "regular":
        ddar = engine.load_and_extend(
            parent_snapshot_id=_get_parent_snapshot(engine, task.get("parent_id")),
            problem=problem,
            new_preds=added_preds,
        )
    elif task_type == "join":
        if not join_meta:
            raise ValueError(f"Join task {problem_id} has no __join__ metadata")
        base_snapshot_id = _get_parent_snapshot(engine, join_meta["base_snapshot_id"])
        if not base_snapshot_id:
            raise RuntimeError(f"Base problem {join_meta['base_snapshot_id']} has no snapshot yet")
        ddar = engine.load_and_merge_extend(
            base_snapshot_id=base_snapshot_id,
            dep_problem_ids=join_meta["dep_problem_ids"],
            problem=problem,
            new_preds=added_preds,
        )
    else:
        raise ValueError(f"Unknown task_type {task_type!r}")

    proved = bool(problem.goal and ddar.check_pred(problem.goal))
    engine.save_snapshot_and_report(ddar, problem_id, proved)
    return proved


def run_worker_loop(
    worker_id: str,
    db_config: dict,
    *,
    poll_interval: float = 0.1,
    max_idle_rounds: int | None = None,
    stop_on_proved: bool = False,
) -> None:
    engine = DistributedDDAR(db_config, worker_id)
    idle_count = 0
    logging.info(f"[{worker_id}] Worker started.")

    while True:
        task = engine.claim_task()
        if not task:
            idle_count += 1
            if max_idle_rounds and idle_count >= max_idle_rounds:
                logging.info(f"[{worker_id}] No tasks after {idle_count} polls - exiting.")
                break
            time.sleep(poll_interval)
            continue

        idle_count = 0
        problem_id = task["problem_id"]
        task_type = task.get("task_type", "regular")

        try:
            t0 = time.monotonic()
            proved = execute_task(engine, task)
            elapsed = time.monotonic() - t0
            logging.info(
                f"[{worker_id}] depth={task.get('depth', '?')} type={task_type} {problem_id[:8]}… "
                f"{elapsed:.2f}s {'PROVED' if proved else ''}"
            )
            if proved and stop_on_proved:
                logging.info(f"[{worker_id}] Proof found - stopping.")
                break
        except Exception as e:
            logging.error(f"[{worker_id}] ERROR on {problem_id}: {e}\n{traceback.format_exc()}")
            try:
                engine.report_result(problem_id, proved=False)
            except Exception:
                pass


def submit_dag(db_config: dict) -> dict[str, str]:
    engine = DistributedDDAR(db_config, "submitter")
    base_problem = AGProblem.parse(BASE_PSTRING)
    base_no_goal = AGProblem(points=base_problem.points, preds=base_problem.preds, goal=base_problem.goal)
    ids: dict[str, str] = {}

    def _problem_for_steps(step_names: list[str]) -> AGProblem:
        ntp = {p.name: p for p in base_problem.points}
        for sn in step_names:
            name, coord = _STEP_MAP[sn].split("=", 1)[0].strip().split("@")
            ntp[name.strip()] = AGPoint(name=name.strip(), value=np.array([float(x) for x in coord.strip().split("_")]))
        return AGProblem(points=list(ntp.values()), preds=[], goal=base_problem.goal)

    ids["base"] = engine.submit_problem(base_no_goal, problem_name="imo2021p3/base")

    ids["q_da"] = engine.submit_problem(_problem_for_steps(["da"]),
        parent_problem_id=ids["base"],
        added_preds=[_STEP_MAP["da"]],
        problem_name="imo2021p3/q_da")
    ids["q_db"] = engine.submit_problem(_problem_for_steps(["da", "db"]),
        parent_problem_id=ids["q_da"],
        added_preds=[_STEP_MAP["db"]],
        problem_name="imo2021p3/q_db")
    ids["q_dc"] = engine.submit_problem(_problem_for_steps(["da", "db", "dc"]),
        parent_problem_id=ids["q_db"],
        added_preds=[_STEP_MAP["dc"]],
        problem_name="imo2021p3/q_dc")
    ids["q_q"] = engine.submit_problem(_problem_for_steps(["da", "db", "dc", "q"]),
        parent_problem_id=ids["q_dc"],
        added_preds=[_STEP_MAP["q"]],
        problem_name="imo2021p3/q_q")

    ids["m_m"] = engine.submit_problem(_problem_for_steps(["m"]),
        parent_problem_id=ids["base"],
        added_preds=[_STEP_MAP["m"]],
        problem_name="imo2021p3/m_m")

    ids["mk_k"] = engine.submit_problem(_problem_for_steps(["k"]),
        parent_problem_id=ids["base"],
        added_preds=[_STEP_MAP["k"]],
        problem_name="imo2021p3/mk_k")
    ids["mk_p"] = engine.submit_problem(_problem_for_steps(["k", "p"]),
        parent_problem_id=ids["mk_k"],
        added_preds=[_STEP_MAP["p"]],
        problem_name="imo2021p3/mk_p")
    ids["mk_mp"] = engine.submit_problem(_problem_for_steps(["k", "p", "mp"]),
        parent_problem_id=ids["mk_p"],
        added_preds=[_STEP_MAP["mp"]],
        problem_name="imo2021p3/mk_mp")

    ids["yp_yp"] = engine.submit_problem(_problem_for_steps(["yp"]),
        parent_problem_id=ids["base"],
        added_preds=[_STEP_MAP["yp"]],
        problem_name="imo2021p3/yp_yp")
    ids["yp_ep"] = engine.submit_problem(_problem_for_steps(["yp", "ep"]),
        parent_problem_id=ids["yp_yp"],
        added_preds=[_STEP_MAP["ep"]],
        problem_name="imo2021p3/yp_ep")

    all_points_problem = AGProblem(
        points=list({
            **{p.name: p for p in base_problem.points},
            **{
                n: AGPoint(name=n, value=np.array([float(x) for x in c.split("_")]))
                for step in AUXILIARY_STEPS
                for n, c in [step[1].split("=")[0].strip().split("@")]
            },
        }.values()),
        preds=[],
        goal=base_problem.goal,
    )

    ids["final"] = engine.submit_join_problem(
        problem=all_points_problem,
        dep_problem_ids=[ids["m_m"], ids["q_q"], ids["mk_mp"], ids["yp_ep"]],
        base_snapshot_id=ids["base"],
        parent_problem_id=ids["base"],
        problem_name="imo2021p3/final_join",
    )

    print(f"Submitted {len(ids)} DAG tasks.")
    return ids


def run_dag_demo(num_workers: int = 2, timeout: float = 30.0) -> None:
    print(DAG_EXPLANATION)
    time.sleep(5.0)
    print(f"\nDAG distributed solve (workers={num_workers})\n")
    _truncate_db(DB_CONFIG)
    submit_dag(DB_CONFIG)

    watch = _start_watch(1.0)
    try:
        threads = [
            threading.Thread(
                target=run_worker_loop,
                args=(f"worker-{i}", DB_CONFIG),
                kwargs={"poll_interval": 0.05, "max_idle_rounds": 200},
                daemon=True,
            )
            for i in range(num_workers)
        ]
        for t in threads:
            t.start()

        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if _all_done(DB_CONFIG):
                print(f"\nAll tasks finished in {time.monotonic() - start:.1f}s.")
                break
            time.sleep(0.25)
        else:
            print(f"\nTimeout after {timeout}s.")

        for t in threads:
            t.join(timeout=5.0)
    finally:
        _stop_watch(watch)

    _inspect("all")


def prove_all_problems(num_workers: int = 2, timeout: float = 60.0) -> None:
    print(BATCH_EXPLANATION)
    time.sleep(5.0)
    print(f"\nprove_all_problems (workers={num_workers})")
    _truncate_db(DB_CONFIG)

    problems = {}
    problems.update(problems_without_aux)
    problems.update(problems_with_aux)

    submitter = DistributedDDAR(DB_CONFIG, "submitter")
    print(f"\nSubmitting {len(problems)} problems …")
    for name, pstring in problems.items():
        pid = submitter.submit_problem(AGProblem.parse(pstring), problem_name=name)
        print(f"  submitted {name!r:30s} -> {pid}")

    watch = _start_watch(1.0)
    try:
        threads = [
            threading.Thread(
                target=run_worker_loop,
                args=(f"worker-{i}", DB_CONFIG),
                kwargs={"poll_interval": 0.05, "max_idle_rounds": None},
                daemon=True,
            )
            for i in range(num_workers)
        ]
        for t in threads:
            t.start()

        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if _all_done(DB_CONFIG):
                break
            time.sleep(0.5)
        else:
            print(f"\nTimeout after {timeout}s.")
    finally:
        _stop_watch(watch)

    _inspect("all")


def main() -> None:
    argv = sys.argv[1:]
    cmd = argv[0] if argv else ""

    if cmd == "worker":
        wid = argv[1] if len(argv) > 1 else f"worker-{os.getpid()}"
        run_worker_loop(wid, DB_CONFIG)

    elif cmd == "dag":
        n = int(argv[1]) if len(argv) > 1 else 2
        timeout = float(argv[2]) if len(argv) > 2 else 30.0
        run_dag_demo(n, timeout)

    elif cmd == "batch":
        n = int(argv[1]) if len(argv) > 1 else 2
        timeout = float(argv[2]) if len(argv) > 2 else 60.0
        prove_all_problems(n, timeout)

    elif cmd == "inspect":
        sub = argv[1] if len(argv) > 1 else "all"
        _inspect(sub)

    elif cmd == "all":
        n = int(argv[1]) if len(argv) > 1 else 2
        prove_all_problems(n, timeout=60.0)
        time.sleep(2)
        os.system('clear')
        run_dag_demo(n, timeout=30.0)

    else:
        print("Unknown command.")
        print("Use: worker [id] | dag [n] [timeout] | batch [n] [timeout] | inspect [cmd] | all [n]")
        print("""Recomended Use: Run `python distributed_test.py worker` in several terminals (2-3 is fine), then run `python distributed_test.py all 0` in another terminal to submit the batch test and then the DAG test.""")
        sys.exit(1)


if __name__ == "__main__":
    main()
