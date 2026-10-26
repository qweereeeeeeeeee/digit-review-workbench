"""Small local workflow store. Predictions are immutable; reviews are versioned."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid


def now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


class Conflict(Exception):
    pass


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS batches(
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, source TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS items(
                    id TEXT PRIMARY KEY, batch_id TEXT NOT NULL REFERENCES batches(id),
                    name TEXT NOT NULL, digest TEXT NOT NULL, pixels TEXT NOT NULL,
                    prediction INTEGER NOT NULL, confidence REAL NOT NULL, probabilities TEXT NOT NULL,
                    model_sha TEXT NOT NULL, preprocess TEXT NOT NULL,
                    final_label INTEGER CHECK(final_label BETWEEN 0 AND 9),
                    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','confirmed','corrected','unreadable')),
                    version INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
                    UNIQUE(batch_id,digest));
                CREATE TABLE IF NOT EXISTS events(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, item_id TEXT NOT NULL REFERENCES items(id),
                    action TEXT NOT NULL, before_state TEXT NOT NULL, after_state TEXT NOT NULL,
                    note TEXT NOT NULL, actor TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS requests(
                    key TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, response TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS queue ON items(batch_id,status,confidence,id);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def decode(row):
        if row is None:
            raise KeyError('记录不存在')
        obj = dict(row)
        for key in ['pixels','probabilities']:
            if key in obj: obj[key] = json.loads(obj[key])
        return obj

    @staticmethod
    def cached(db, key, fingerprint):
        previous = db.execute('SELECT * FROM requests WHERE key=?',(key,)).fetchone()
        if previous:
            if previous['fingerprint'] != fingerprint:
                raise Conflict('同一请求编号不能用于不同内容')
            return json.loads(previous['response'])

    def create_batch(self, name, source, items, key):
        payload = json.dumps([name,source,items],sort_keys=True,ensure_ascii=False)
        fingerprint = hashlib.sha256(payload.encode()).hexdigest()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            cached = self.cached(db,key,fingerprint)
            if cached: return cached
            batch_id = str(uuid.uuid4())
            db.execute('INSERT INTO batches VALUES(?,?,?,?)',(batch_id,name,source,now()))
            inserted = 0
            for item in items:
                values = (str(uuid.uuid4()),batch_id,item['name'],item['digest'],json.dumps(item['pixels']),
                          item['prediction'],item['confidence'],json.dumps(item['probabilities']),
                          item['model_sha'],item['preprocess'],now())
                cur = db.execute('''INSERT OR IGNORE INTO items
                    (id,batch_id,name,digest,pixels,prediction,confidence,probabilities,model_sha,preprocess,created_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)''',values)
                inserted += cur.rowcount
            result = dict(id=batch_id,inserted=inserted,duplicates=len(items)-inserted)
            db.execute('INSERT INTO requests VALUES(?,?,?)',(key,fingerprint,json.dumps(result)))
            return result

    def batches(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute('''SELECT b.*,COUNT(i.id) AS total,
                SUM(CASE WHEN i.status='pending' THEN 1 ELSE 0 END) AS pending,
                SUM(CASE WHEN i.status='corrected' THEN 1 ELSE 0 END) AS corrected
                FROM batches b LEFT JOIN items i ON b.id=i.batch_id GROUP BY b.id ORDER BY b.created_at DESC,b.id''')]

    def items(self, batch_id, status='all', order='uncertain', limit=100, offset=0):
        where = 'batch_id=?'; args = [batch_id]
        if status != 'all': where += ' AND status=?'; args.append(status)
        ordering = 'confidence ASC,id' if order == 'uncertain' else 'created_at ASC,name,id'
        with self.connect() as db:
            if not db.execute('SELECT id FROM batches WHERE id=?',(batch_id,)).fetchone(): raise KeyError('批次不存在')
            total = db.execute(f'SELECT COUNT(*) FROM items WHERE {where}',args).fetchone()[0]
            rows = db.execute(f'SELECT * FROM items WHERE {where} ORDER BY {ordering} LIMIT ? OFFSET ?',args+[limit,offset])
            return dict(total=total,items=[self.decode(r) for r in rows])

    def get(self, item_id):
        with self.connect() as db:
            obj = self.decode(db.execute('SELECT * FROM items WHERE id=?',(item_id,)).fetchone())
            obj['events'] = [dict(r) for r in db.execute('SELECT * FROM events WHERE item_id=? ORDER BY id',(item_id,))]
            return obj

    def review(self, item_id, version, label, action, note, actor, key):
        payload = json.dumps([item_id,version,label,action,note,actor],ensure_ascii=False)
        fingerprint = hashlib.sha256(payload.encode()).hexdigest()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            cached = self.cached(db,key,fingerprint)
            if cached: return cached
            row = self.decode(db.execute('SELECT * FROM items WHERE id=?',(item_id,)).fetchone())
            if row['version'] != version: raise Conflict('记录已被修改，请刷新后再确认，当前输入没有覆盖已保存结果')
            if action == 'reopen': status, label = 'pending', None
            elif action == 'unreadable': status, label = 'unreadable', None
            else: status = 'confirmed' if label == row['prediction'] else 'corrected'
            before = {k:row[k] for k in ['status','final_label','version']}
            after = dict(status=status,final_label=label,version=version+1)
            db.execute('UPDATE items SET status=?,final_label=?,version=? WHERE id=?',(status,label,version+1,item_id))
            db.execute('''INSERT INTO events(item_id,action,before_state,after_state,note,actor,created_at)
                VALUES(?,?,?,?,?,?,?)''',(item_id,action,json.dumps(before),json.dumps(after),note,actor,now()))
            result = dict(id=item_id,**after)
            db.execute('INSERT INTO requests VALUES(?,?,?)',(key,fingerprint,json.dumps(result)))
            return result

    def export(self,batch_id):
        with self.connect() as db:
            if not db.execute('SELECT id FROM batches WHERE id=?',(batch_id,)).fetchone():raise KeyError('批次不存在')
            return [dict(r) for r in db.execute('''SELECT name,prediction,confidence,status,final_label,version,
                model_sha,preprocess,digest FROM items WHERE batch_id=? ORDER BY name,id''',(batch_id,))]
