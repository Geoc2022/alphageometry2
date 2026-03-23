CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS problems (
    problem_id      TEXT PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    problem_name    TEXT DEFAULT '',
    parent_id       TEXT REFERENCES problems(problem_id),
    base_pstring    TEXT NOT NULL,
    added_preds     JSONB DEFAULT '[]',
    depth           INTEGER DEFAULT 0,
    status          TEXT DEFAULT 'pending' CHECK (status IN ('pending','running','proved','failed')),
    worker_id       TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    task_type       TEXT DEFAULT 'regular' CHECK (task_type IN ('regular', 'join')),
    join_ready      BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_problems_status_depth ON problems(status, depth) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_problems_parent ON problems(parent_id);
CREATE INDEX IF NOT EXISTS idx_problems_join_ready ON problems(status, join_ready) WHERE status = 'pending';

CREATE TABLE IF NOT EXISTS problem_dependencies (
    join_problem_id     TEXT REFERENCES problems(problem_id),
    dep_problem_id      TEXT REFERENCES problems(problem_id),
    PRIMARY KEY (join_problem_id, dep_problem_id)
);

CREATE INDEX IF NOT EXISTS idx_deps_join ON problem_dependencies(join_problem_id);
CREATE INDEX IF NOT EXISTS idx_deps_dep ON problem_dependencies(dep_problem_id);

CREATE TABLE IF NOT EXISTS state_snapshots (
    snapshot_id     TEXT PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    problem_id      TEXT REFERENCES problems(problem_id),
    parent_snapshot TEXT REFERENCES state_snapshots(snapshot_id),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_snapshots_problem ON state_snapshots(problem_id);

CREATE TABLE IF NOT EXISTS snapshot_points (
    snapshot_id     TEXT REFERENCES state_snapshots(snapshot_id),
    point_name      TEXT,
    point_x         DOUBLE PRECISION,
    point_y         DOUBLE PRECISION,
    PRIMARY KEY (snapshot_id, point_name)
);

CREATE TABLE IF NOT EXISTS elim_rows (
    snapshot_id     TEXT REFERENCES state_snapshots(snapshot_id),
    elim_type       TEXT CHECK (elim_type IN ('angle','dist_mul','dist_add')),
    pivot_var_name  TEXT,
    pivot_var_value DOUBLE PRECISION,
    free_var_name   TEXT,
    free_var_value  DOUBLE PRECISION,
    coef_num        BIGINT,
    coef_den        BIGINT,
    PRIMARY KEY (snapshot_id, elim_type, pivot_var_name, free_var_name)
);

CREATE INDEX IF NOT EXISTS idx_elim_rows_free ON elim_rows(snapshot_id, elim_type, free_var_name);

CREATE TABLE IF NOT EXISTS snapshot_lines (
    snapshot_id     TEXT REFERENCES state_snapshots(snapshot_id),
    line_id         INTEGER,
    main_pair_a     TEXT,
    main_pair_b     TEXT,
    dir_var_name    TEXT,
    dir_var_value   DOUBLE PRECISION,
    line_nx         DOUBLE PRECISION,
    line_ny         DOUBLE PRECISION,
    line_c          DOUBLE PRECISION,
    PRIMARY KEY (snapshot_id, line_id)
);

CREATE TABLE IF NOT EXISTS line_points (
    snapshot_id     TEXT,
    line_id         INTEGER,
    point_name      TEXT,
    position        DOUBLE PRECISION,
    PRIMARY KEY (snapshot_id, line_id, point_name),
    FOREIGN KEY (snapshot_id, line_id) REFERENCES snapshot_lines(snapshot_id, line_id)
);

CREATE INDEX IF NOT EXISTS idx_line_points_point ON line_points(snapshot_id, point_name);

CREATE TABLE IF NOT EXISTS snapshot_circles (
    snapshot_id     TEXT REFERENCES state_snapshots(snapshot_id),
    circle_id       INTEGER,
    center_x        DOUBLE PRECISION,
    center_y        DOUBLE PRECISION,
    radius          DOUBLE PRECISION,
    PRIMARY KEY (snapshot_id, circle_id)
);

CREATE TABLE IF NOT EXISTS circle_points (
    snapshot_id     TEXT,
    circle_id       INTEGER,
    point_name      TEXT,
    is_center       BOOLEAN DEFAULT FALSE,
    is_defining     BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (snapshot_id, circle_id, point_name),
    FOREIGN KEY (snapshot_id, circle_id) REFERENCES snapshot_circles(snapshot_id, circle_id)
);

CREATE INDEX IF NOT EXISTS idx_circle_points_point ON circle_points(snapshot_id, point_name) WHERE NOT is_center;

CREATE TABLE IF NOT EXISTS point_subst (
    snapshot_id     TEXT REFERENCES state_snapshots(snapshot_id),
    from_point      TEXT,
    to_point        TEXT,
    PRIMARY KEY (snapshot_id, from_point)
);

CREATE TABLE IF NOT EXISTS known_similar (
    snapshot_id     TEXT REFERENCES state_snapshots(snapshot_id),
    t1_a TEXT, t1_b TEXT, t1_c TEXT,
    t2_a TEXT, t2_b TEXT, t2_c TEXT,
    PRIMARY KEY (snapshot_id, t1_a, t1_b, t1_c, t2_a, t2_b, t2_c)
);


CREATE OR REPLACE VIEW proof_tree AS
SELECT
    p.problem_id, p.parent_id, p.depth, p.status, p.task_type, p.worker_id,
    p.created_at, p.completed_at,
    EXTRACT(EPOCH FROM (p.completed_at - p.created_at)) as duration_seconds,
    s.snapshot_id
FROM problems p
LEFT JOIN state_snapshots s USING (problem_id);

CREATE OR REPLACE FUNCTION _check_join_readiness() RETURNS TRIGGER AS $$
BEGIN
    UPDATE problems p
    SET    join_ready = TRUE
    WHERE  p.task_type = 'join' AND p.status = 'pending' AND p.join_ready = FALSE
      AND  p.problem_id IN (SELECT pd.join_problem_id FROM problem_dependencies pd WHERE pd.dep_problem_id = NEW.problem_id)
      AND  NOT EXISTS (
               SELECT 1 FROM problem_dependencies pd2
               WHERE  pd2.join_problem_id = p.problem_id
                 AND  NOT EXISTS (SELECT 1 FROM state_snapshots s WHERE s.problem_id = pd2.dep_problem_id)
           );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_check_join_readiness ON state_snapshots;
CREATE TRIGGER trg_check_join_readiness
    AFTER INSERT ON state_snapshots
    FOR EACH ROW EXECUTE FUNCTION _check_join_readiness();
