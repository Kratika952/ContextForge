-- ContextForge SQLite Schema
-- ============================
-- Section 7 of IMPLEMENTATION_PLAN.md

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT,
    source_path TEXT NOT NULL,
    checksum TEXT UNIQUE NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'pdf',
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_profile TEXT NOT NULL DEFAULT 'medium',
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    section_heading TEXT,
    embedding_model TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_documents_checksum ON documents(checksum);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_profile ON chunks(chunk_profile);
