PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS schema_meta(version INTEGER PRIMARY KEY, migrated_at_utc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions(
  session_id TEXT PRIMARY KEY,
  started_at_utc TEXT NOT NULL,
  ended_at_utc TEXT,
  source_mode TEXT NOT NULL,
  config_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS object_tracks(
  track_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(session_id),
  first_seen_utc TEXT NOT NULL,
  last_seen_utc TEXT NOT NULL,
  current_label TEXT NOT NULL,
  current_label_confidence REAL NOT NULL,
  active INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS observations(
  observation_id TEXT PRIMARY KEY,
  track_id TEXT NOT NULL REFERENCES object_tracks(track_id),
  session_id TEXT NOT NULL REFERENCES sessions(session_id),
  observed_at_utc TEXT NOT NULL,
  observed_at_mono_ns INTEGER NOT NULL,
  label TEXT NOT NULL,
  label_source TEXT NOT NULL,
  detection_confidence REAL NOT NULL,
  bbox_json TEXT NOT NULL,
  horizontal_region TEXT,
  depth_band TEXT,
  anchor_name TEXT,
  location_confidence REAL NOT NULL,
  frame_ref TEXT,
  snapshot_path TEXT,
  correlation_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS observation_aliases(
  alias TEXT NOT NULL,
  canonical_label TEXT NOT NULL,
  observation_id TEXT NOT NULL REFERENCES observations(observation_id),
  PRIMARY KEY(alias, observation_id)
);
CREATE TABLE IF NOT EXISTS last_known_objects(
  canonical_label TEXT PRIMARY KEY,
  observation_id TEXT NOT NULL REFERENCES observations(observation_id),
  updated_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS behavior_preferences(
  context_key TEXT NOT NULL,
  behavior_key TEXT NOT NULL,
  score REAL NOT NULL,
  evidence_count INTEGER NOT NULL,
  updated_at_utc TEXT NOT NULL,
  PRIMARY KEY(context_key, behavior_key)
);
CREATE TABLE IF NOT EXISTS preference_audit(
  audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
  context_key TEXT NOT NULL,
  behavior_key TEXT NOT NULL,
  outcome TEXT NOT NULL,
  previous_score REAL NOT NULL,
  new_score REAL NOT NULL,
  correlation_id TEXT NOT NULL,
  created_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS observations_label_time ON observations(label, observed_at_mono_ns DESC);
CREATE INDEX IF NOT EXISTS observations_session_time ON observations(session_id, observed_at_mono_ns DESC);
CREATE INDEX IF NOT EXISTS observations_track_time ON observations(track_id, observed_at_mono_ns DESC);
CREATE INDEX IF NOT EXISTS observation_aliases_alias ON observation_aliases(alias);
