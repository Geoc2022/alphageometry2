"""Interactive inspector for the distributed DDAR proof-search database.

Usage:
    python -m db.inspect_snapshot            # summary of all problems
    python -m db.inspect_snapshot tree       # proof tree with timing
    python -m db.inspect_snapshot proved     # only proved problems
    python -m db.inspect_snapshot <problem_id_prefix>  # detail for one problem
    python -m db.inspect_snapshot deps       # dependency graph (join nodes)
    python -m db.inspect_snapshot watch      # live-refresh every 2s (Ctrl-C to stop)
"""

import json
import os
import sys
import time

import psycopg2
import psycopg2.extras

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from db.setup_db import DB_CONFIG


_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_BLUE   = "\033[94m"
_GREY   = "\033[90m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

def _color(text: str, *codes: str) -> str:
    return "".join(codes) + text + _RESET

def _status_color(status: str) -> str:
    return {
        'proved':  _color(f"{status:8s}", _GREEN,  _BOLD),
        'failed':  _color(f"{status:8s}", _RED),
        'running': _color(f"{status:8s}", _YELLOW),
        'pending': _color(f"{status:8s}", _GREY),
    }.get(status, status)


def _connect() -> psycopg2.extensions.connection:
    return psycopg2.connect(**DB_CONFIG)


def _fetch_all_problems(conn) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                p.problem_id,
                p.parent_id,
                p.problem_name,
                p.depth,
                p.status,
                p.task_type,
                p.worker_id,
                p.join_ready,
                p.created_at,
                p.completed_at,
                EXTRACT(EPOCH FROM (p.completed_at - p.created_at))
                    AS duration_seconds,
                EXTRACT(EPOCH FROM (NOW() - p.created_at))
                    AS age_seconds,
                s.snapshot_id,
                p.added_preds
            FROM   problems p
            LEFT JOIN state_snapshots s USING (problem_id)
            ORDER  BY p.depth ASC, p.created_at ASC
        """)
        return [dict(r) for r in cur.fetchall()]


def _fetch_deps(conn) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT join_problem_id, dep_problem_id
            FROM   problem_dependencies
            ORDER  BY join_problem_id, dep_problem_id
        """)
        return [dict(r) for r in cur.fetchall()]


def _fetch_snapshot_sizes(conn) -> dict[str, dict]:
    """Return per-snapshot row counts for each relation."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                s.snapshot_id,
                s.problem_id,
                (SELECT COUNT(*) FROM snapshot_points  sp WHERE sp.snapshot_id = s.snapshot_id) AS n_points,
                (SELECT COUNT(*) FROM elim_rows        er WHERE er.snapshot_id = s.snapshot_id) AS n_elim,
                (SELECT COUNT(*) FROM snapshot_lines   sl WHERE sl.snapshot_id = s.snapshot_id) AS n_lines,
                (SELECT COUNT(*) FROM snapshot_circles sc WHERE sc.snapshot_id = s.snapshot_id) AS n_circles,
                (SELECT COUNT(*) FROM known_similar    ks WHERE ks.snapshot_id = s.snapshot_id) AS n_similar,
                (SELECT COUNT(*) FROM point_subst      ps WHERE ps.snapshot_id = s.snapshot_id) AS n_subst
            FROM state_snapshots s
        """)
        return {r['snapshot_id']: dict(r) for r in cur.fetchall()}


def _fetch_one_problem(conn, prefix: str) -> dict | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                p.*,
                p.problem_name,
                s.snapshot_id,
                EXTRACT(EPOCH FROM (p.completed_at - p.created_at))
                    AS duration_seconds
            FROM   problems p
            LEFT JOIN state_snapshots s USING (problem_id)
            WHERE  p.problem_id LIKE %s
            ORDER  BY p.created_at ASC
            LIMIT  1
        """, (prefix + '%',))
        row = cur.fetchone()
    return dict(row) if row else None

def _snap_detail(prefix: str) -> None:
    """Show every point, line, circle in a snapshot."""
    conn = _connect()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT * FROM state_snapshots WHERE snapshot_id LIKE %s LIMIT 1",
            (prefix + '%',)
        )
        snap = cur.fetchone()
    if not snap:
        print(f"No snapshot with prefix {prefix!r}")
        conn.close()
        return

    sid = snap['snapshot_id']
    print(_color(f"\nSnapshot {sid}", _BOLD))
    print(f"  problem_id: {snap['problem_id'][:8]}")
    print(f"  created:    {snap['created_at']}\n")

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT point_name, point_x, point_y "
                    "FROM snapshot_points WHERE snapshot_id=%s "
                    "ORDER BY point_name", (sid,))
        pts = cur.fetchall()
        print(_color(f"  Points ({len(pts)}):", _BOLD))
        for p in pts:
            print(f"    {p['point_name']:6s}  ({p['point_x']:.4f}, {p['point_y']:.4f})")

        cur.execute("SELECT elim_type, COUNT(*) as n "
                    "FROM elim_rows WHERE snapshot_id=%s "
                    "GROUP BY elim_type ORDER BY elim_type", (sid,))
        elims = cur.fetchall()
        print(_color(f"\n  Elim rows:", _BOLD))
        for e in elims:
            print(f"    {e['elim_type']:10s}  {e['n']} rows")

        cur.execute("""
            SELECT sl.line_id, sl.main_pair_a, sl.main_pair_b,
                   STRING_AGG(lp.point_name, ',' ORDER BY lp.position) AS pts
            FROM   snapshot_lines sl
            JOIN   line_points lp USING (snapshot_id, line_id)
            WHERE  sl.snapshot_id = %s
            GROUP  BY sl.line_id, sl.main_pair_a, sl.main_pair_b
            HAVING COUNT(lp.point_name) >= 3
            ORDER  BY sl.line_id
        """, (sid,))
        lines = cur.fetchall()
        print(_color(f"\n  Multi-point lines (≥3 pts) ({len(lines)}):", _BOLD))
        for l in lines:
            print(f"    [{l['line_id']}] {l['main_pair_a']}-{l['main_pair_b']}: {l['pts']}")

        cur.execute("""
            SELECT sc.circle_id, sc.center_x, sc.center_y, sc.radius,
                   STRING_AGG(cp.point_name, ',' ORDER BY cp.point_name)
                       FILTER (WHERE NOT cp.is_center) AS pts,
                   STRING_AGG(cp.point_name, ',')
                       FILTER (WHERE cp.is_center) AS centers
            FROM   snapshot_circles sc
            JOIN   circle_points cp USING (snapshot_id, circle_id)
            WHERE  sc.snapshot_id = %s
            GROUP  BY sc.circle_id, sc.center_x, sc.center_y, sc.radius
            ORDER  BY sc.circle_id
        """, (sid,))
        circs = cur.fetchall()
        print(_color(f"\n  Circles ({len(circs)}):", _BOLD))
        for c in circs:
            print(f"    [{c['circle_id']}] center=({c['center_x']:.3f},{c['center_y']:.3f}) "
                  f"r={c['radius']:.3f}  pts={c['pts']}  centers={c['centers']}")

        cur.execute("SELECT t1_a,t1_b,t1_c,t2_a,t2_b,t2_c "
                    "FROM known_similar WHERE snapshot_id=%s", (sid,))
        sims = cur.fetchall()
        print(_color(f"\n  Known similar ({len(sims)}):", _BOLD))
        for s in sims:
            print(f"    ({s['t1_a']},{s['t1_b']},{s['t1_c']}) ~ "
                  f"({s['t2_a']},{s['t2_b']},{s['t2_c']})")

        cur.execute("SELECT from_point, to_point FROM point_subst "
                    "WHERE snapshot_id=%s AND from_point != to_point", (sid,))
        substs = cur.fetchall()
        if substs:
            print(_color(f"\n  Non-identity substitutions ({len(substs)}):", _BOLD))
            for s in substs:
                print(f"    {s['from_point']} → {s['to_point']}")

    conn.close()


def _summary(problems: list[dict]) -> None:
    """One-line-per-problem table."""
    counts = {'proved': 0, 'failed': 0, 'running': 0, 'pending': 0}
    for p in problems:
        counts[p['status']] = counts.get(p['status'], 0) + 1

    total = len(problems)
    print(_color(f"\n{'─'*78}", _GREY))
    print(_color(" DDAR Distributed Proof Search — Problem Summary", _BOLD))
    print(_color(f"{'─'*78}", _GREY))
    print(
        f"  Total: {total}   "
        + _color(f"proved: {counts['proved']}", _GREEN) + "   "
        + _color(f"failed: {counts['failed']}", _RED) + "   "
        + _color(f"running: {counts['running']}", _YELLOW) + "   "
        + _color(f"pending: {counts['pending']}", _GREY)
    )
    print(_color(f"{'─'*78}\n", _GREY))

    header = (
        f"  {'id':8s}  {'name':22s}  {'d':2s}  {'type':8s}  {'status':8s}  "
        f"{'worker':18s}  {'dur(s)':>7s}  {'snap':8s}  ready"
    )
    print(_color(header, _BOLD))
    print(_color(f"  {'─'*105}", _GREY))

    for p in problems:
        dur   = p['duration_seconds']
        dur_s = f"{dur:7.2f}" if dur is not None else f"{'—':>7s}"
        name  = (p.get('problem_name') or '—')[:22]
        snap  = (p['snapshot_id'] or '')[:8] or '—'
        ready = '✓' if p['join_ready'] else '·'
        wid   = (p['worker_id'] or '')[:18]
        print(
            f"  {p['problem_id'][:8]}  {name:22s}  {p['depth']:2d}  "
            f"{p['task_type']:8s}  {_status_color(p['status'])}  "
            f"{wid:18s}  {dur_s}  {snap:8s}  {ready}"
        )

    print()


def _tree(problems: list[dict]) -> None:
    """Parent→child tree with branch labels inferred from added_preds."""
    id_to_p = {p['problem_id']: p for p in problems}
    children: dict[str | None, list[dict]] = {}
    for p in problems:
        children.setdefault(p['parent_id'], []).append(p)

    print(_color(f"\n{'─'*78}", _GREY))
    print(_color(" Proof Tree", _BOLD))
    print(_color(f"{'─'*78}\n", _GREY))

    def _label(p: dict) -> str:
        pname = (p.get('problem_name') or '').strip()
        if pname:
            return pname
        raw = p.get('added_preds') or []
        if not raw:
            return 'BASE'
        first = raw[0] if isinstance(raw, list) else str(raw)
        if first.strip().startswith('{'):
            return 'JOIN'
        if '@' in first and '=' in first:
            return first.split('@')[0].strip()
        return first[:20]

    def _print_node(pid: str | None, prefix: str, is_last: bool) -> None:
        if pid is not None:
            p    = id_to_p[pid]
            conn = '└─' if is_last else '├─'
            dur  = p['duration_seconds']
            dur_s = f"{dur:.2f}s" if dur is not None else "…"
            snap  = (p['snapshot_id'] or '')[:8] or '—'
            label = _label(p)
            line = (
                f"{prefix}{conn} [{p['problem_id'][:8]}] "
                f"{_color(label, _BOLD):30s} "
                f"{_status_color(p['status'])}  "
                f"{dur_s:>7s}  snap={snap}"
            )
            print(line)
            child_prefix = prefix + ('   ' if is_last else '│  ')
        else:
            child_prefix = ''

        kids = sorted(children.get(pid, []), key=lambda x: x['created_at'])
        for i, child in enumerate(kids):
            _print_node(child['problem_id'], child_prefix, i == len(kids) - 1)

    _print_node(None, '', True)
    print()


def _deps(problems: list[dict], dep_rows: list[dict]) -> None:
    """Show join-node dependency edges."""
    id_to_p = {p['problem_id']: p for p in problems}

    join_nodes = [p for p in problems if p['task_type'] == 'join']
    if not join_nodes:
        print("  No join nodes found.\n")
        return

    print(_color(f"\n{'─'*78}", _GREY))
    print(_color(" Join-Node Dependencies", _BOLD))
    print(_color(f"{'─'*78}\n", _GREY))

    by_join: dict[str, list[str]] = {}
    for row in dep_rows:
        by_join.setdefault(row['join_problem_id'], []).append(row['dep_problem_id'])

    for jn in join_nodes:
        jid  = jn['problem_id']
        deps = by_join.get(jid, [])
        print(
            f"  JOIN {jid[:8]} ({(jn.get('problem_name') or '—')[:30]})  {_status_color(jn['status'])}  "
            f"ready={jn['join_ready']}"
        )
        for dep_id in deps:
            dep = id_to_p.get(dep_id)
            if dep:
                snap = (dep['snapshot_id'] or '')[:8] or '—'
                has_snap = '✓' if dep['snapshot_id'] else '✗'
                print(
                    f"    └─ dep {dep_id[:8]} ({(dep.get('problem_name') or '—')[:28]})  "
                    f"{_status_color(dep['status'])}  "
                    f"snap={snap} {has_snap}"
                )
            else:
                print(f"    └─ dep {dep_id[:8]}  (not found)")
        print()


def _detail(conn, prefix: str, snap_sizes: dict[str, dict]) -> None:
    """Full detail for one problem."""
    p = _fetch_one_problem(conn, prefix)
    if p is None:
        print(f"  No problem found with id prefix {prefix!r}")
        return

    print(_color(f"\n{'─'*78}", _GREY))
    print(_color(f" Problem Detail: {p['problem_id']}", _BOLD))
    print(_color(f"{'─'*78}\n", _GREY))

    dur = p['duration_seconds']
    print(f"  name:       {p.get('problem_name') or '—'}")
    print(f"  status:     {_status_color(p['status'])}")
    print(f"  task_type:  {p['task_type']}")
    print(f"  depth:      {p['depth']}")
    print(f"  join_ready: {p['join_ready']}")
    print(f"  worker:     {p['worker_id'] or '—'}")
    print(f"  created:    {p['created_at']}")
    print(f"  completed:  {p['completed_at'] or '—'}")
    print(f"  duration:   {f'{dur:.3f}s' if dur is not None else '—'}")
    print(f"  parent_id:  {(p['parent_id'] or '—')[:8]}")

    sid = p.get('snapshot_id')
    if sid:
        print(f"\n  snapshot:   {sid}")
        sz = snap_sizes.get(sid, {})
        if sz:
            print(f"    points:   {sz['n_points']}")
            print(f"    elim rows:{sz['n_elim']}")
            print(f"    lines:    {sz['n_lines']}")
            print(f"    circles:  {sz['n_circles']}")
            print(f"    similar:  {sz['n_similar']}")
            print(f"    subst:    {sz['n_subst']}")
    else:
        print(f"\n  snapshot:   (none)")

    raw = p.get('added_preds') or []
    print(f"\n  added_preds ({len(raw)} entries):")
    for i, entry in enumerate(raw):
        if isinstance(entry, str) and entry.strip().startswith('{'):
            try:
                meta = json.loads(entry)
                print(f"    [{i}] JOIN META:")
                for k, v in meta.items():
                    if isinstance(v, list):
                        print(f"          {k}:")
                        for item in v:
                            print(f"            {str(item)[:60]}")
                    else:
                        print(f"          {k}: {str(v)[:60]}")
                continue
            except json.JSONDecodeError:
                pass
        preview = str(entry)[:72]
        print(f"    [{i}] {preview}")

    pstr = p.get('base_pstring') or ''
    print(f"\n  base_pstring ({len(pstr)} chars):")
    print(f"    {pstr[:120]}{'…' if len(pstr) > 120 else ''}")
    print()


def _proved_only(problems: list[dict]) -> None:
    proved = [p for p in problems if p['status'] == 'proved']
    if not proved:
        print(_color("\n  No proved problems.\n", _RED))
        return
    print(_color(f"\n  {len(proved)} proved problem(s):\n", _GREEN + _BOLD))
    for p in proved:
        dur = p['duration_seconds']
        dur_s = f"{dur:.3f}s" if dur is not None else "—"
        name  = p.get('problem_name') or '—'
        snap  = (p['snapshot_id'] or '')[:8] or '—'
        print(
            f"  {p['problem_id'][:8]}  name={name[:28]:28s}  depth={p['depth']}  "
            f"type={p['task_type']}  "
            f"worker={p['worker_id'] or '—'}  "
            f"dur={dur_s}  snap={snap}"
        )
    print()


def main() -> None:
    args = sys.argv[1:]
    cmd  = args[0] if args else 'summary'

    if cmd == 'watch':
        interval = float(args[1]) if len(args) > 1 else 2.0
        try:
            while True:
                os.system('clear')
                with _connect() as conn:
                    problems   = _fetch_all_problems(conn)
                    dep_rows   = _fetch_deps(conn)
                    snap_sizes = _fetch_snapshot_sizes(conn)
                _summary(problems)
                _tree(problems)
                _deps(problems, dep_rows)
                n_proved = sum(1 for p in problems if p['status'] == 'proved')
                if n_proved:
                    _proved_only(problems)
                print(_color(f"  Refreshing every {interval}s — Ctrl-C to stop", _GREY))
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")
        return

    with _connect() as conn:
        problems   = _fetch_all_problems(conn)
        dep_rows   = _fetch_deps(conn)
        snap_sizes = _fetch_snapshot_sizes(conn)

    if cmd == 'summary':
        _summary(problems)

    elif cmd == 'tree':
        _summary(problems)
        _tree(problems)

    elif cmd == 'proved':
        _proved_only(problems)

    elif cmd == 'deps':
        _deps(problems, dep_rows)

    elif cmd == 'all':
        _summary(problems)
        _tree(problems)
        _deps(problems, dep_rows)
        _proved_only(problems)

    elif cmd == 'snap':
        prefix = args[1] if len(args) > 1 else ''
        _snap_detail(prefix)

    else:
        with _connect() as conn:
            _detail(conn, cmd, snap_sizes)


if __name__ == '__main__':
    main()
