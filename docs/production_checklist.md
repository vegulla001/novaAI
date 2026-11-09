# Production readiness checklist — not yet implemented

This repository is a **reference implementation**, not a certified industry-grade hosted service.

- [ ] Versioned Alembic migrations and production schema lifecycle, zero-downtime rollout.
- [ ] Explicit row-level DB tenant policies and security review; current enforcement lives in API queries.
- [ ] OAuth/OIDC browser PKCE flow, real user sessions, key rotation, policy-based entitlements.
- [ ] Per-tenant request throttles, token limits, spend ceilings, daily jobs and GPU quota limits.
- [ ] Dedicated ingress with TLS, CORS allowlist, CSRF review, private model network and WAF.
- [ ] Transactional outbox for queue dispatch and cancellation-safe lease heartbeats.
- [ ] Lease reaper and max-job wall time / watchdog; broker/worker failure drills.
- [ ] Separate task queues for planning vs drafting vs verification; priority classes.
- [ ] Human-reviewed content policies, abuse controls and copyrighted-text similarity detection.
- [ ] Usage metering / cost accounting, billing, export-retention and deletion controls.
- [ ] OpenTelemetry distributed traces, metrics dashboard, logging and alerting (latency, retries, model errors, GPU saturation).
- [ ] Rolling/deployment strategy, HPA/KEDA, GPU autoscaling and load tests.
- [ ] Backups/PITR, disaster recovery, secrets manager, SBOM, dependency and container scans.
- [ ] Data privacy policy, training opt-in, deletion, retention, GDPR/DPDP assessment as relevant.
- [ ] Model and LoRA licensing and provenance; security scanning for uploaded training content.
- [ ] Full eval gate for continuity, instruction following, narrative arc, dialogue and quality.
