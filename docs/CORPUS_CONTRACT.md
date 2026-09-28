# Corpus Contract

> **Purpose:** Define which documents may enter the ContextForge corpus, what metadata every document must carry, and what content is included or excluded.
>
> **Scope:** Phase 1 — Domain and Dataset Contract. No PDF extraction or chunking is performed here.
>
> **Rule:** This contract is the authoritative gate for corpus admission. No document may be ingested in Phase 2 unless it satisfies the rules below.

---

## 1. Domain

**Cybersecurity knowledge and defensive security guidance.**

ContextForge is a defensive knowledge assistant. It answers questions grounded in authoritative cybersecurity documentation intended to help practitioners understand risks, apply controls, and follow established frameworks.

---

## 2. Allowed Source Types

Only primary/official sources from recognized standards bodies, government agencies, or established open-source security organizations are permitted as authoritative corpus content.

### Approved source families for Phase 1

| # | Organization | Document family |
|---|---|---|
| 1 | OWASP | OWASP Top 10 (all approved editions) |
| 2 | NIST | NIST Cybersecurity Framework (all approved editions) |
| 3 | CISA | Cross-Sector Cybersecurity Performance Goals (all approved editions) |

Additional source families may be added only after explicit human approval.

### Prohibited source types (must never enter the authoritative corpus)

| Type | Reason |
|---|---|
| Random blogs or editorials | Unverified quality, potential misinformation |
| SEO-optimized security articles | Unreliable accuracy |
| Forum posts (Reddit, Stack Overflow, etc.) | Not authoritative |
| Unofficial vendor summaries | May promote vendor interest over accuracy |
| AI-generated cybersecurity text | No authoritative ground truth |
| Paywalled or unlicensed commercial content | Legal and provenance issues |
| Offensive-tool documentation | Out of scope (see §6) |

---

## 3. Source Quality Rule

> **Prefer primary, official, and directly downloadable sources.**

When multiple versions of a document exist, use the most recent approved version unless a prior version is explicitly required for a controlled experiment.

Record the exact URL and version. Do not summarize or rewrite official text.

---

## 4. Document Metadata Requirements

Every corpus document **must** track the following fields before it may be ingested. Fields that are genuinely unknown remain explicitly `null` or `PENDING_VERIFICATION` — never fabricated.

| Field | Description | Required |
|---|---|---|
| `document_id` | Stable unique identifier (e.g., `owasp-top10-2025`) | YES |
| `title` | Official document title | YES |
| `organization` | Issuing organization (OWASP, NIST, CISA, …) | YES |
| `source_url` | Authoritative download or landing page URL | YES |
| `source_version` | Version string or year edition | YES |
| `published_date` | Official publication/update date (ISO 8601) | YES if known; else `PENDING_VERIFICATION` |
| `retrieved_at` | Date the file was actually downloaded (ISO 8601) | Set at ingestion time |
| `local_filename` | Relative path under `data/raw/` | Set at ingestion time |
| `content_type` | `pdf`, `html`, `txt`, `csv`, … | YES |
| `license_or_usage_notes` | Usage rights, license, or attribution requirement | YES if known; else `PENDING_VERIFICATION` |
| `checksum` | SHA-256 of the downloaded file | Set at ingestion time; NEVER fabricated |
| `status` | `PLANNED` / `PENDING_DOWNLOAD` / `INGESTED` / `EXCLUDED` | YES |
| `notes` | Any caveats, scope limits, or version notes | Optional |

### Checksum rule

**Do not compute or record a checksum until the file has actually been downloaded.** The `checksum` column must remain `PENDING_DOWNLOAD` until Phase 2 ingestion. Recording a fabricated or guessed checksum is a data-integrity violation.

---

## 5. Content Inclusion Rules

### Include

- Definitions of security concepts, vulnerabilities, and risk categories
- Control objectives and security control guidance
- Risk-management principles and processes
- Mitigation and remediation guidance
- Secure development and secure-by-design recommendations
- Framework terminology, functions, categories, and subcategories
- Procedural guidance for incident response, recovery, and resilience
- Mapping tables between frameworks (e.g., NIST CSF ↔ CISA CPGs)

### Exclude

Content is excluded if its **primary purpose** is any of the following:

- Exploit generation or execution
- Malware development or distribution
- Credential theft, password cracking, or phishing automation
- Persistence, lateral movement, or evasion techniques described for offensive use
- Destructive or unauthorized operations against systems or infrastructure
- Autonomous offensive pentesting or attack automation

> **Important distinction:** Defensive documentation frequently *describes* attack patterns in order to explain how to defend against them (e.g., OWASP describes SQL injection to guide developers away from it). Such content is **included** — the exclusion test is the *purpose* of the content, not the mere presence of attack terminology.

---

## 6. Scope Boundaries

### In scope

- Students learning cybersecurity fundamentals
- Developers seeking secure-coding guidance
- Junior security professionals applying defensive frameworks
- Teams searching for authoritative defensive cybersecurity answers

### Out of scope — will not be supported

The system will not assist with:

- Malware development
- Exploit generation
- Credential attacks
- Phishing automation
- Offensive attack automation
- Autonomous pentesting agents
- Destructive or unauthorized security operations

---

## 7. Version and Update Policy

- When a new official version of an approved document is released, add a new manifest entry with a new `document_id` rather than overwriting the existing entry.
- The active version used for a given experiment must be recorded in `docs/EXPERIMENT_LOG.md`.
- Backward-incompatible corpus changes require a new experiment entry so results remain reproducible.

---

## 8. Provenance and Attribution

All corpus documents remain the intellectual property of their issuing organizations. This project uses them under their respective open-access or public-domain terms. Attribution must be preserved in:

- the corpus manifest
- SQLite document metadata
- any publication or report derived from this project

Do not strip authorship, version, or URL attribution from extracted text.
