"""Immutable inference records with explicit, versioned review decisions."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid


class Conflict(Exception):
    pass


class Store:
    def __init__(self,path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS records(
                    id TEXT PRIMARY KEY,name TEXT NOT NULL,created_at TEXT NOT NULL,
                    digest TEXT NOT NULL,original BLOB NOT NULL,result TEXT NOT NULL,
                    final_label TEXT,status TEXT NOT NULL DEFAULT 'pending',version INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS receipts(key TEXT PRIMARY KEY,fingerprint TEXT NOT NULL,response TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS reviews(id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL REFERENCES records(id),created_at TEXT NOT NULL,
                    before_state TEXT NOT NULL,after_state TEXT NOT NULL,note TEXT NOT NULL);
            ''')

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=10)
        db.row_factory=sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback();raise
        finally:
            db.close()

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat(timespec='milliseconds')

    @staticmethod
    def unpack(row):
        if row is None:raise KeyError('记录不存在')
        result=dict(row)
        result.pop('original',None)
        result['result']=json.loads(result['result'])
        return result

    def cached(self,key,fingerprint,db=None):
        if db is None:
            with self.connect() as conn:return self.cached(key,fingerprint,conn)
        row=db.execute('SELECT * FROM receipts WHERE key=?',(key,)).fetchone()
        if row:
            if row['fingerprint']!=fingerprint:raise Conflict('这个请求编号已经用于其他内容')
            return json.loads(row['response'])

    def save(self,name,raw,result,key,fingerprint):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            previous=self.cached(key,fingerprint,db)
            if previous:return previous
            record_id=str(uuid.uuid4())
            db.execute('INSERT INTO records(id,name,created_at,digest,original,result) VALUES(?,?,?,?,?,?)',
                       (record_id,name,self.now(),hashlib.sha256(raw).hexdigest(),raw,json.dumps(result)))
            response=self.unpack(db.execute('SELECT * FROM records WHERE id=?',(record_id,)).fetchone())
            db.execute('INSERT INTO receipts VALUES(?,?,?)',(key,fingerprint,json.dumps(response)))
            return response

    def get(self,record_id):
        with self.connect() as db:
            obj=self.unpack(db.execute('SELECT * FROM records WHERE id=?',(record_id,)).fetchone())
            obj['reviews']=[dict(r) for r in db.execute('SELECT * FROM reviews WHERE record_id=? ORDER BY id',(record_id,))]
            return obj

    def original(self,record_id):
        with self.connect() as db:
            row=db.execute('SELECT original FROM records WHERE id=?',(record_id,)).fetchone()
            if row is None:raise KeyError('记录不存在')
            return row['original']

    def list(self,limit,offset):
        with self.connect() as db:
            total=db.execute('SELECT count(*) FROM records').fetchone()[0]
            rows=db.execute('SELECT id,name,created_at,digest,result,final_label,status,version FROM records ORDER BY created_at DESC,rowid DESC LIMIT ? OFFSET ?',(limit,offset))
            return dict(total=total,items=[self.unpack(r) for r in rows])

    def export_rows(self):
        # One SELECT gives a consistent snapshot even when another request saves a row.
        with self.connect() as db:
            return [self.unpack(r) for r in db.execute(
                'SELECT id,name,created_at,digest,result,final_label,status,version FROM records ORDER BY created_at DESC,rowid DESC')]

    def review(self,record_id,version,label,status,note,key):
        fingerprint=hashlib.sha256(json.dumps([record_id,version,label,status,note]).encode()).hexdigest()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            previous=self.cached(key,fingerprint,db)
            if previous:return previous
            obj=self.unpack(db.execute('SELECT * FROM records WHERE id=?',(record_id,)).fetchone())
            if obj['version']!=version:raise Conflict('记录已经被修改，请重新打开后再保存')
            before={k:obj[k] for k in ['final_label','status','version']}
            after=dict(final_label=label,status=status,version=version+1)
            db.execute('UPDATE records SET final_label=?,status=?,version=? WHERE id=?',(label,status,version+1,record_id))
            db.execute('INSERT INTO reviews(record_id,created_at,before_state,after_state,note) VALUES(?,?,?,?,?)',
                       (record_id,self.now(),json.dumps(before),json.dumps(after),note))
            response=dict(id=record_id,**after)
            db.execute('INSERT INTO receipts VALUES(?,?,?)',(key,fingerprint,json.dumps(response)))
            return response
