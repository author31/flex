<!--
Sync Impact Report
==================
Version change: (template/unversioned) → 1.0.0
Rationale: Initial ratification of concrete project constitution from template placeholders.

Modified principles:
  [PRINCIPLE_1_NAME] → I. Monorepo Boundaries & Structure
  [PRINCIPLE_2_NAME] → II. Backend Layered Architecture (NON-NEGOTIABLE)
  [PRINCIPLE_3_NAME] → III. Test-First & Coverage Standards (NON-NEGOTIABLE)
  [PRINCIPLE_4_NAME] → IV. Code Quality & Consistency
  [PRINCIPLE_5_NAME] → V. Containerized Deployment Parity

Added sections:
  - Technology Constraints (SECTION_2)
  - Development Workflow & Quality Gates (SECTION_3)

Removed sections: none

Templates requiring updates:
  ✅ .specify/memory/constitution.md (this file)
  ⚠ .specify/templates/plan-template.md (verify "Constitution Check" gate references new principles — not readable this session)
  ⚠ .specify/templates/spec-template.md (verify alignment — not readable this session)
  ⚠ .specify/templates/tasks-template.md (verify task categories cover testing/layer discipline — not readable this session)

Follow-up TODOs:
  - RATIFICATION_DATE set to 2026-06-01 (today) as original adoption; revise if earlier adoption date applies.
-->

# Flex Constitution

## Core Principles

### I. Monorepo Boundaries & Structure

The repository is a single monorepo containing exactly two deployable packages:
`backend/` and `frontend/`. Each package MUST be self-contained, independently
buildable, and independently testable. Cross-package coupling MUST occur only
through defined network contracts (HTTP/API), never through shared in-language
imports across the package boundary. Code, config, and dependencies for a package
MUST live inside that package's directory.

Rationale: Clear boundaries keep the two stacks (Python backend, React frontend)
loosely coupled, enabling independent testing, builds, and reasoning.

### II. Backend Layered Architecture (NON-NEGOTIABLE)

The `backend/` package MUST follow domain-driven design organized into EXACTLY
four layers, no more and no less:

- `infrastructure.py` — external concerns: DB clients, HTTP clients, framework wiring, I/O.
- `domain.py` — entities, value objects, domain rules; pure, framework-free, no I/O.
- `application.py` — use cases / orchestration that coordinate domain and repository.
- `repository.py` — persistence contracts and data access for domain objects.

Dependency direction MUST flow inward: `infrastructure` → `application` →
`domain`; `repository` depends on `domain` only. `domain.py` MUST NOT import the
other three layers. Adding, removing, renaming, or splitting these layer files is
prohibited.

Rationale: A fixed four-layer contract enforces DDD discipline, keeps the domain
pure and testable, and makes every backend feature land in a predictable place.

### III. Test-First & Coverage Standards (NON-NEGOTIABLE)

Tests MUST be written before implementation. The cycle is strict: write tests →
confirm they fail → implement → make them pass → refactor (Red-Green-Refactor).
Every backend layer with logic (`domain`, `application`, `repository`) MUST have
unit tests; `domain.py` MUST be tested with no I/O or mocks of external systems.
Frontend components and hooks MUST have tests for behavior and rendering.
Inter-package contracts (backend API consumed by frontend) MUST have integration
tests. A change MUST NOT merge with failing or skipped tests.

Rationale: Test-first locks intent before code and guards the layered design and
the API contract between packages from regression.

### IV. Code Quality & Consistency

All code MUST pass automated linting and formatting before merge. Backend:
Python 3.12 with type hints on public functions, managed by `uv`; code MUST pass
the project linter/formatter and a static type check. Frontend: React + Vite
with the project's ESLint config; builds MUST be warning-clean. No commented-out
dead code, no unused exports, and public functions/modules MUST be documented
where intent is non-obvious. Complexity MUST be justified, not assumed.

Rationale: Consistent, typed, linted code lowers review cost and defect rate
across both stacks.

### V. Containerized Deployment Parity

Both packages MUST be deployable via Docker Compose. Each package MUST ship a
Dockerfile, and the root MUST provide a `docker-compose` definition that runs the
full system. Configuration MUST come from environment variables, never hardcoded
secrets. The Compose stack MUST be the single source of truth for how services
wire together, so local and deployed topologies match.

Rationale: One Compose definition gives reproducible local/prod parity and keeps
service wiring explicit and auditable.

## Technology Constraints

- Backend: Python 3.12; `uv` is the primary package and environment manager.
- Backend architecture: exactly the four DDD layer files defined in Principle II.
- Frontend: React with Vite as the build tool.
- Deployment: Docker Compose orchestrating `backend/` and `frontend/` services.
- Secrets and environment-specific values MUST be supplied via environment, not committed.

## Development Workflow & Quality Gates

- Every change is gated by: passing tests (Principle III), passing lint/format/type
  checks (Principle IV), and a successful Docker Compose build (Principle V).
- Pull requests MUST verify compliance with all principles above before merge.
- Backend reviews MUST confirm the four-layer structure and inward dependency flow.
- Any deviation from a principle MUST be documented with explicit justification in
  the PR; unjustified complexity is rejected.

## Governance

This constitution supersedes all other development practices. Amendments require a
documented proposal, review approval, and a migration note when they change
existing structure. Versioning follows semantic rules: MAJOR for backward-
incompatible governance/principle removals or redefinitions, MINOR for new or
materially expanded principles/sections, PATCH for clarifications and wording.
All PRs and reviews MUST verify compliance; complexity MUST be justified. Use the
plan, spec, and tasks templates under `.specify/` for runtime development guidance.

**Version**: 1.0.0 | **Ratified**: 2026-06-01 | **Last Amended**: 2026-06-01
