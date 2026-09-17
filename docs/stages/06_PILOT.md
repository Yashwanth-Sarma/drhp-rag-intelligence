# Stage 06 — private professional pilot

Prerequisites: held-out quality evidence and functioning report workflow. Current localhost server is not a deployment architecture.

Add authentication and document-level authorization before private external use; tenant-aware retrieval/cache keys; upload isolation and resource quotas; object storage and transactional metadata; versioned schema migrations; explicit tracing/redaction; scoped ingestion URL controls; queue/retry/crash recovery; cost limits and provider circuit breakers.

Build clean-environment locked installation, CI unit/integration/browser gates, load tests on declared hardware, backup/restore test, source/index integrity audit and rollback. Review dependency licensing, privacy requirements and retention with deployment context. Never disable TLS verification to fix connectivity.

Acceptance: documented pilot scope; all critical boundary cases pass; p95 latency measured rather than asserted; monitored failure paths; report quality reviewed by target analysts; limitations and unanswered questions visible. Only then expand corpus discovery, visual retrieval or graph queries when evaluated use cases justify them.
