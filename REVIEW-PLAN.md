# Proteus Review Remediation Plan

Expert review identified seven issues. This plan addresses them in order of
effort-to-impact ratio, starting with small high-confidence fixes and building
toward the larger structural change.

## 1. Durable and bounded conversation memory

`InMemorySaver` is process-local and unbounded. Under sustained load with many
thread IDs memory grows without limit and all state is lost on restart.

Fix: swap `InMemorySaver` for `SqliteSaver` with a configurable path. Add a
TTL-based or LRU eviction policy to bound memory. This is a deployment concern
and lowest urgency.

Files: `images/proteus/graph.py`, deployment configs
