import json
import sqlite3
import threading
import time
from .config import DATA

LOCK = threading.RLock()


def connection():
    conn = sqlite3.connect(DATA / 'studio.db', timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init():
    with connection() as conn:
        conn.executescript('''
        CREATE TABLE IF NOT EXISTS tasks (
          id TEXT PRIMARY KEY, status TEXT NOT NULL, body TEXT NOT NULL,
          created REAL NOT NULL, updated REAL NOT NULL, request_key TEXT UNIQUE
        );
        CREATE TABLE IF NOT EXISTS events (
          seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
          time REAL NOT NULL, stage TEXT NOT NULL, message TEXT NOT NULL, detail TEXT
        );
        ''')


def get(task_id):
    with connection() as conn:
        row = conn.execute('SELECT body FROM tasks WHERE id=?', (task_id,)).fetchone()
    return json.loads(row['body']) if row else None


def by_key(key):
    with connection() as conn:
        row = conn.execute('SELECT body FROM tasks WHERE request_key=?', (key,)).fetchone()
    return json.loads(row['body']) if row else None


def create(task, request_key=None):
    now = time.time()
    task.update(created_at=now, updated_at=now)
    with LOCK, connection() as conn:
        conn.execute('INSERT INTO tasks VALUES (?,?,?,?,?,?)', (task['id'], task['status'], json.dumps(task, ensure_ascii=False), now, now, request_key))
    return task


def update(task_id, **values):
    with LOCK, connection() as conn:
        row = conn.execute('SELECT body FROM tasks WHERE id=?', (task_id,)).fetchone()
        if not row:
            raise KeyError(task_id)
        task = json.loads(row['body'])
        task.update(values, updated_at=time.time())
        conn.execute('UPDATE tasks SET status=?,body=?,updated=? WHERE id=?', (task['status'], json.dumps(task, ensure_ascii=False), task['updated_at'], task_id))
    return task


def list_tasks():
    with connection() as conn:
        rows = conn.execute('SELECT body FROM tasks ORDER BY created DESC LIMIT 100').fetchall()
    return [json.loads(row['body']) for row in rows]


def event(task_id, stage, message, detail=None):
    with LOCK, connection() as conn:
        conn.execute('INSERT INTO events(task_id,time,stage,message,detail) VALUES (?,?,?,?,?)', (task_id, time.time(), stage, message, json.dumps(detail, ensure_ascii=False) if detail else None))


def events(task_id, after=0):
    with connection() as conn:
        rows = conn.execute('SELECT * FROM events WHERE task_id=? AND seq>? ORDER BY seq', (task_id, after)).fetchall()
    return [{**dict(row), 'detail': json.loads(row['detail']) if row['detail'] else None} for row in rows]


def delete(task_id):
    """Remove only this task's database records, including LangGraph state."""
    with LOCK, connection() as conn:
        checkpoint_path = DATA / 'checkpoints.db'
        if checkpoint_path.exists():
            conn.execute('ATTACH DATABASE ? AS graph', (str(checkpoint_path),))
            tables = {row[0] for row in conn.execute("SELECT name FROM graph.sqlite_master WHERE type='table'")}
            for table in ('checkpoints', 'writes'):
                if table in tables:
                    conn.execute(f'DELETE FROM graph.{table} WHERE thread_id=?', (task_id,))
        conn.execute('DELETE FROM events WHERE task_id=?', (task_id,))
        conn.execute('DELETE FROM tasks WHERE id=?', (task_id,))
