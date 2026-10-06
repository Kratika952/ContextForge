"""
Phase 7 — Fine-Tuning Dataset Builder
======================================
Generates a grounded instruction-tuning dataset from the approved corpus
(OWASP Top 10:2025, NIST CSF 2.0, CISA CPG 2.0).

All QA pairs are hand-crafted and grounded in the canonical documents.
No LLM generation, no synthetic examples, no fake citations.

Usage:
    .venv/Scripts/python.exe scripts/build_finetuning_dataset.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QA_DIR = PROJECT_ROOT / "data" / "qa"
EVAL_PATH = QA_DIR / "evaluation_v1.jsonl"

OUT_FULL = QA_DIR / "finetune_full.jsonl"
OUT_TRAIN = QA_DIR / "finetune_train.jsonl"
OUT_VAL = QA_DIR / "finetune_val.jsonl"
OUT_TEST = QA_DIR / "finetune_test.jsonl"
OUT_REVIEW = QA_DIR / "finetune_review_sample.jsonl"
OUT_STATS = QA_DIR / "finetune_stats.json"
OUT_META = QA_DIR / "finetune_metadata.json"

SYSTEM_PROMPT = (
    "You are a cybersecurity knowledge assistant specializing in defensive security guidance. "
    "Answer questions accurately based on authoritative sources including OWASP Top 10:2025, "
    "NIST Cybersecurity Framework 2.0, and CISA Cross-Sector Cybersecurity Performance Goals 2.0. "
    "If the question falls outside your area of expertise, say so clearly."
)

# ── Known source IDs (from corpus_manifest.csv) ─────────────────────────────

VALID_SOURCES = {
    "owasp": [
        "owasp-top10-2025-index", "owasp-top10-2025-intro", "owasp-top10-2025-about",
        "owasp-top10-2025-risks", "owasp-top10-2025-program",
        "owasp-top10-2025-a01", "owasp-top10-2025-a02", "owasp-top10-2025-a03",
        "owasp-top10-2025-a04", "owasp-top10-2025-a05", "owasp-top10-2025-a06",
        "owasp-top10-2025-a07", "owasp-top10-2025-a08", "owasp-top10-2025-a09",
        "owasp-top10-2025-a10", "owasp-top10-2025-next",
    ],
    "nist": ["nist-csf-2.0"],
    "cisa": ["cisa-cpg-2.0"],
}

# ── Grounded QA Dataset ─────────────────────────────────────────────────────
# Each record: (source_family, source_doc_id, category, question, answer)
# All answers grounded in the approved corpus documents.

DATASET: list[tuple[str, str, str, str, str]] = [
    # ─── OWASP A01: Broken Access Control ───────────────────────────────
    ("owasp", "owasp-top10-2025-a01", "factual_definition",
     "What is Broken Access Control as defined in OWASP Top 10:2025?",
     "Broken Access Control is the number one risk in OWASP Top 10:2025 (A01). It occurs when enforcement of policies fails, allowing users to act outside their intended permissions. Common vulnerabilities include violation of the principle of least privilege, bypassing access control checks by modifying URLs, API requests, or internal application state, permitting viewing or editing others' accounts, elevation of privilege, and metadata manipulation such as replaying or tampering with access control tokens like JWTs or cookies."),

    ("owasp", "owasp-top10-2025-a01", "mitigation",
     "How should developers prevent Broken Access Control according to OWASP?",
     "OWASP recommends several prevention measures: deny access by default except for public resources, implement access control mechanisms once and reuse them throughout the application, enforce record ownership rather than accepting that users can create, read, update, or delete any record, disable web server directory listing, ensure file metadata and backup files are not present within web roots, log access control failures and alert administrators when appropriate, and rate-limit API and controller access to minimize the harm from automated attack tooling."),

    ("owasp", "owasp-top10-2025-a01", "risk_explanation",
     "What are common attack scenarios for Broken Access Control?",
     "Common attack scenarios include: modifying the browser URL to access another user's account information by changing account identifiers in parameters, an attacker browsing to admin pages despite lacking admin privileges through forced browsing, manipulation of primary keys to access or modify other users' records, elevation of privilege by acting as a user without being logged in or acting as an admin when logged in as a regular user, and API requests with missing access control enforcement for POST, PUT, and DELETE operations."),

    ("owasp", "owasp-top10-2025-a01", "recommendation",
     "What role does the principle of least privilege play in preventing Broken Access Control?",
     "The principle of least privilege is fundamental to preventing Broken Access Control. OWASP recommends that access should be denied by default except for public resources. Users should only have access to the minimum resources and functions required for their specific role. Access control should enforce record ownership rather than permitting users to create, read, update, or delete arbitrary records. Domain models should enforce unique application business limit requirements, and server-side access controls should not rely on client-side enforcement that attackers can bypass."),

    # ─── OWASP A02: Security Misconfiguration ──────────────────────────
    ("owasp", "owasp-top10-2025-a02", "factual_definition",
     "What is Security Misconfiguration according to OWASP Top 10:2025?",
     "Security Misconfiguration (A02) in OWASP Top 10:2025 refers to applications that lack appropriate security hardening across any part of the application stack. Common issues include missing security hardening, improperly configured permissions on cloud services, unnecessary features being enabled or installed, default accounts and passwords being unchanged, error handling revealing overly informative stack traces, and security settings in application servers, frameworks, libraries, and databases not being set to secure values."),

    ("owasp", "owasp-top10-2025-a02", "mitigation",
     "How can organizations prevent Security Misconfiguration?",
     "Prevention measures include: implementing a repeatable hardening process that makes it fast and easy to deploy properly locked-down environments, deploying a minimal platform without unnecessary features, components, documentation, and samples, reviewing and updating configurations appropriate to all security notes, updates, and patches as part of the patch management process, using a segmented application architecture with segmentation, containerization, or cloud security groups, sending security directives to clients via security headers, and establishing an automated process to verify the effectiveness of configurations across all environments."),

    ("owasp", "owasp-top10-2025-a02", "recommendation",
     "Why are default credentials a security misconfiguration risk?",
     "Default accounts and their passwords that remain enabled and unchanged represent a significant security misconfiguration. Attackers commonly scan for default credentials on newly deployed systems and applications. OWASP identifies this as part of A02:2025 Security Misconfiguration. Organizations should change all default passwords immediately upon deployment, disable default administrative accounts or change their credentials, and implement processes to verify that no default credentials remain active across all deployed systems and services."),

    # ─── OWASP A03: Software Supply Chain Failures ─────────────────────
    ("owasp", "owasp-top10-2025-a03", "factual_definition",
     "What are Software Supply Chain Failures in OWASP Top 10:2025?",
     "Software Supply Chain Failures (A03) is a category in OWASP Top 10:2025 that covers risks related to failures in the security of the software supply chain. This includes vulnerabilities introduced through third-party components, libraries, and dependencies, as well as risks from compromised build pipelines, untrusted software update mechanisms, and lack of integrity verification. The category recognizes that modern applications heavily rely on open-source and third-party components, making supply chain security critical."),

    ("owasp", "owasp-top10-2025-a03", "mitigation",
     "How should organizations mitigate Software Supply Chain risks?",
     "Organizations should maintain an inventory of all components and their versions (Software Bill of Materials), continuously monitor for vulnerabilities in dependencies using automated tools, only obtain components from official sources over secure links, prefer signed packages to reduce the chance of including modified malicious components, monitor for unmaintained libraries and components that do not receive security patches, and establish a patching process to ensure timely updates to components throughout the application lifecycle."),

    # ─── OWASP A04: Cryptographic Failures ─────────────────────────────
    ("owasp", "owasp-top10-2025-a04", "factual_definition",
     "What are Cryptographic Failures as defined in OWASP Top 10:2025?",
     "Cryptographic Failures (A04) in OWASP Top 10:2025 refer to failures related to cryptography which often lead to sensitive data exposure. This includes not encrypting sensitive data at rest or in transit, using weak or obsolete cryptographic algorithms, using default crypto keys or generating weak keys, not enforcing encryption through proper security headers or directives, and server certificate and trust chain validation not being properly performed. Previously known as Sensitive Data Exposure, this category focuses on the root cause: cryptographic implementation failures."),

    ("owasp", "owasp-top10-2025-a04", "mitigation",
     "How can developers prevent Cryptographic Failures?",
     "Developers should classify data processed, stored, or transmitted by an application and identify which data is sensitive according to privacy laws, regulatory requirements, or business needs. Apply controls per the classification. Do not store sensitive data unnecessarily and discard it as soon as possible. Ensure up-to-date and strong standard algorithms, protocols, and keys are in use with proper key management. Encrypt all data in transit with secure protocols such as TLS with forward secrecy ciphers and enforce encryption using directives like HTTP Strict Transport Security. Disable caching for responses containing sensitive data."),

    # ─── OWASP A05: Injection ──────────────────────────────────────────
    ("owasp", "owasp-top10-2025-a05", "factual_definition",
     "What types of injection vulnerabilities does OWASP Top 10:2025 cover?",
     "OWASP Top 10:2025 Injection (A05) covers SQL injection, NoSQL injection, OS command injection, LDAP injection, Cross-Site Scripting (XSS), and Expression Language injection. An application is vulnerable to injection when user-supplied data is not validated, filtered, or sanitized, when dynamic queries or non-parameterized calls are used without context-aware escaping, and when hostile data is used within object-relational mapping search parameters."),

    ("owasp", "owasp-top10-2025-a05", "mitigation",
     "What are the primary defenses against injection attacks?",
     "OWASP recommends using a safe API that avoids using the interpreter entirely, provides a parameterized interface, or migrates to Object Relational Mapping tools. Use positive server-side input validation as a secondary defense since many applications require special characters. For any residual dynamic queries, escape special characters using the specific escape syntax for that interpreter. Use LIMIT and other SQL controls within queries to prevent mass disclosure of records in case of SQL injection. Source code review is the best method of detecting injection vulnerabilities, along with thorough automated testing."),

    # ─── OWASP A06: Insecure Design ────────────────────────────────────
    ("owasp", "owasp-top10-2025-a06", "factual_definition",
     "What is Insecure Design in OWASP Top 10:2025?",
     "Insecure Design (A06) is a broad category representing different weaknesses expressed as missing or ineffective control design. It is distinct from insecure implementation. A secure design can still have implementation defects leading to vulnerabilities, but an insecure design cannot be fixed by a perfect implementation, as by definition the needed security controls were never created to defend against specific attacks. OWASP identifies this to highlight the importance of threat modeling and secure design patterns and principles."),

    ("owasp", "owasp-top10-2025-a06", "recommendation",
     "How does OWASP recommend addressing Insecure Design risks?",
     "OWASP recommends establishing and using a secure development lifecycle with AppSec professionals to help evaluate and design security and privacy-related controls, establishing and using a library of secure design patterns or paved road ready-to-use components, using threat modeling for critical authentication, access control, business logic, and key flows, integrating security language and controls into user stories, and integrating plausibility checks at each tier of the application from frontend to backend."),

    # ─── OWASP A07: Authentication Failures ────────────────────────────
    ("owasp", "owasp-top10-2025-a07", "factual_definition",
     "What are Authentication Failures in OWASP Top 10:2025?",
     "Authentication Failures (A07) in OWASP Top 10:2025 cover weaknesses in authentication mechanisms. An application may be vulnerable if it permits automated attacks such as credential stuffing or brute force, permits default, weak, or well-known passwords, uses weak or ineffective credential recovery and forgotten-password processes, uses plain text, encrypted, or weakly hashed password data stores, has missing or ineffective multi-factor authentication, or exposes session identifiers in the URL."),

    ("owasp", "owasp-top10-2025-a07", "mitigation",
     "How should applications prevent Authentication Failures?",
     "Applications should implement multi-factor authentication where possible to prevent automated credential stuffing, brute force, and stolen credential reuse attacks. Do not ship or deploy with any default credentials, particularly for admin users. Implement weak password checks testing new or changed passwords against a list of the top 10,000 worst passwords. Align password length, complexity, and rotation policies with modern guidelines. Ensure registration, credential recovery, and API pathways are hardened against account enumeration attacks. Limit or increasingly delay failed login attempts while being careful not to create a denial-of-service scenario."),

    # ─── OWASP A08: Software or Data Integrity Failures ────────────────
    ("owasp", "owasp-top10-2025-a08", "factual_definition",
     "What are Software or Data Integrity Failures?",
     "Software or Data Integrity Failures (A08) in OWASP Top 10:2025 relate to code and infrastructure that does not protect against integrity violations. Examples include applications relying on plugins, libraries, or modules from untrusted sources, repositories, or content delivery networks. An insecure CI/CD pipeline can introduce the potential for unauthorized access, malicious code, or system compromise. Applications with auto-update functionality where updates are downloaded without sufficient integrity verification may have malicious updates applied to previously trusted applications."),

    ("owasp", "owasp-top10-2025-a08", "mitigation",
     "How can organizations protect against Software or Data Integrity Failures?",
     "Organizations should use digital signatures or similar mechanisms to verify the software or data is from the expected source and has not been altered. Ensure libraries and dependencies such as npm or Maven consume trusted repositories. Use a software supply chain security tool to verify that components do not contain known vulnerabilities. Ensure there is a review process for code and configuration changes to minimize the chance that malicious code or configuration could be introduced into the software pipeline. Ensure the CI/CD pipeline has proper segregation, configuration, and access control."),

    # ─── OWASP A09: Security Logging and Alerting Failures ─────────────
    ("owasp", "owasp-top10-2025-a09", "factual_definition",
     "What are Security Logging and Alerting Failures?",
     "Security Logging and Alerting Failures (A09) in OWASP Top 10:2025 cover insufficient logging, detection, monitoring, and active response. Without logging and monitoring, breaches cannot be detected. Issues include auditable events such as logins, failed logins, and high-value transactions not being logged, warnings and errors generating no, inadequate, or unclear log messages, logs not being monitored for suspicious activity, logs only being stored locally, and appropriate alerting thresholds and response escalation processes not being in place."),

    ("owasp", "owasp-top10-2025-a09", "mitigation",
     "How should organizations address Security Logging and Alerting Failures?",
     "Organizations should ensure all login, access control, and server-side input validation failures can be logged with sufficient user context to identify suspicious or malicious accounts and held for sufficient time to allow delayed forensic analysis. Ensure log entries are generated in a format that log management solutions can easily consume. Ensure high-value transactions have an audit trail with integrity controls to prevent tampering or deletion. Establish or adopt an incident response and recovery plan. Use application security testing tools and their logs to detect and alert on issues."),

    # ─── OWASP A10: Mishandling of Exceptional Conditions ──────────────
    ("owasp", "owasp-top10-2025-a10", "factual_definition",
     "What is Mishandling of Exceptional Conditions in OWASP Top 10:2025?",
     "Mishandling of Exceptional Conditions (A10) is a new category in OWASP Top 10:2025 that addresses the risks associated with improper handling of errors, exceptions, and unexpected conditions in applications. When applications fail to properly handle exceptional conditions, they may expose sensitive information through detailed error messages, enter undefined or insecure states, cause denial of service, or fail open rather than fail securely. Proper exception handling is critical to maintaining application security posture."),

    ("owasp", "owasp-top10-2025-a10", "mitigation",
     "How should developers handle exceptional conditions securely?",
     "Developers should implement consistent error handling strategies across the application. Applications should fail securely, meaning that when an error occurs, the system defaults to a secure state rather than an open or vulnerable one. Error messages should not reveal sensitive information such as stack traces, database queries, or internal system details to users. Exceptions should be properly caught, logged for administrators, and presented to users with generic, safe error messages. All error paths should be tested to ensure they maintain the application's security controls."),

    # ─── OWASP General / Cross-cutting ─────────────────────────────────
    ("owasp", "owasp-top10-2025-index", "framework_mapping",
     "What are the ten categories in OWASP Top 10:2025?",
     "The OWASP Top 10:2025 categories are: A01 Broken Access Control, A02 Security Misconfiguration, A03 Software Supply Chain Failures, A04 Cryptographic Failures, A05 Injection, A06 Insecure Design, A07 Authentication Failures, A08 Software or Data Integrity Failures, A09 Security Logging and Alerting Failures, and A10 Mishandling of Exceptional Conditions."),

    ("owasp", "owasp-top10-2025-risks", "factual_definition",
     "How does OWASP define application security risks?",
     "OWASP defines application security risks using a risk rating methodology that considers the threat agent, attack vector, security weakness, and technical and business impacts. The attack vectors represent the different ways an attacker can exploit a weakness. Security weaknesses are flaws or gaps in application security that can be discovered and exploited. The technical impact represents the effect on the system, while the business impact requires understanding what the affected application does and its value to the organization."),

    ("owasp", "owasp-top10-2025-program", "procedural_guidance",
     "What does OWASP recommend for establishing a modern application security program?",
     "OWASP recommends starting with establishing a baseline understanding of the current application security posture. Organizations should implement a secure software development lifecycle, perform application security testing including both static and dynamic analysis, establish security requirements as part of the design phase, integrate security testing into the CI/CD pipeline, train developers on secure coding practices, and track security findings with a vulnerability management program."),

    # ─── NIST CSF 2.0 ──────────────────────────────────────────────────
    ("nist", "nist-csf-2.0", "framework_mapping",
     "Describe each of the six core functions defined in the NIST Cybersecurity Framework 2.0.",
     "The six core functions of NIST CSF 2.0 are Govern (GV), Identify (ID), Protect (PR), Detect (DE), Respond (RS), and Recover (RC). The Govern function is new in CSF 2.0 and establishes the organizational context, cybersecurity strategy, and oversight. Identify focuses on understanding current cybersecurity risks. Protect implements safeguards. Detect enables timely discovery of cybersecurity events. Respond takes action regarding detected incidents. Recover restores capabilities and services impaired by incidents."),

    ("nist", "nist-csf-2.0", "factual_definition",
     "Explain the purpose and scope of the Govern function introduced in NIST CSF 2.0.",
     "The Govern function is new in NIST CSF 2.0 and establishes and monitors the organization's cybersecurity risk management strategy, expectations, and policy. It informs how an organization implements and prioritizes the other five functions. Govern addresses the organizational context, risk management strategy, roles and responsibilities, policies, oversight, and cybersecurity supply chain risk management. It recognizes that cybersecurity is a significant source of enterprise risk alongside finance, legal, and other areas."),

    ("nist", "nist-csf-2.0", "factual_definition",
     "What does the Identify function cover in NIST CSF 2.0?",
     "The Identify function in NIST CSF 2.0 focuses on helping an organization understand its current cybersecurity risks. It encompasses asset management to identify and manage assets that enable the organization to achieve its business purposes, risk assessment to analyze potential cybersecurity risks, and improvement through identifying improvements to the organization's cybersecurity risk management processes and activities. Understanding these aspects is essential for prioritizing efforts consistent with the organization's risk management strategy."),

    ("nist", "nist-csf-2.0", "factual_definition",
     "What outcomes does the Protect function address in NIST CSF 2.0?",
     "The Protect function in NIST CSF 2.0 addresses safeguards to manage cybersecurity risks. Its categories include identity management, authentication, and access control to ensure authorized access; awareness and training to ensure personnel possess the knowledge and skills to perform their cybersecurity duties; data security to protect the confidentiality, integrity, and availability of data; platform security to protect hardware, software, and services; and technology infrastructure resilience to ensure security architectures are managed to protect assets."),

    ("nist", "nist-csf-2.0", "factual_definition",
     "What is the purpose of the Detect function in NIST CSF 2.0?",
     "The Detect function in NIST CSF 2.0 enables the timely discovery and analysis of anomalies, indicators of compromise, and other potentially adverse cybersecurity events that may indicate ongoing attacks or security incidents. It focuses on continuous monitoring of network and physical activities, maintaining detection processes to provide situational awareness of anomalous events, and ensuring detected events are analyzed to understand attack targets and methods."),

    ("nist", "nist-csf-2.0", "factual_definition",
     "What does the Respond function cover in NIST CSF 2.0?",
     "The Respond function in NIST CSF 2.0 supports the ability to contain the impact of cybersecurity incidents. Activities include incident management to detect, analyze, contain, eradicate, and recover from incidents; incident analysis to ensure adequate investigation and forensic capability; incident response reporting and communication to coordinate response activities with relevant stakeholders; and incident mitigation to prevent expansion of events, mitigate effects, and resolve incidents."),

    ("nist", "nist-csf-2.0", "factual_definition",
     "What is the Recover function in NIST CSF 2.0?",
     "The Recover function in NIST CSF 2.0 supports timely recovery to normal operations to reduce the impact from cybersecurity incidents. It includes incident recovery plan execution to ensure the organization implements recovery processes and procedures to restore systems and assets affected by incidents, as well as improvements where recovery planning and processes are refined by incorporating lessons learned into future activities."),

    ("nist", "nist-csf-2.0", "procedural_guidance",
     "How does NIST CSF 2.0 recommend organizations use CSF Profiles?",
     "NIST CSF 2.0 recommends organizations use CSF Profiles to describe their current or target cybersecurity posture in terms of the CSF Core outcomes. A Current Profile describes the cybersecurity outcomes currently being achieved. A Target Profile describes the desired outcomes. Comparing them reveals gaps to be addressed. Community Profiles are baseline profiles created for particular sectors or use cases. Organizations should tailor profiles to their specific risk environment, mission, and regulatory requirements."),

    ("nist", "nist-csf-2.0", "procedural_guidance",
     "How should organizations approach cybersecurity risk assessment according to NIST CSF 2.0?",
     "NIST CSF 2.0 recommends organizations identify and document their cybersecurity risks by considering threats to and vulnerabilities of enterprise assets, as well as the likelihood and potential impact of exploitation. Risk assessments should be updated regularly and whenever significant changes occur. Organizations should consider internal and external threat information, evaluate the potential consequences of risk scenarios, and use risk assessment results to inform risk response decisions including accepting, mitigating, transferring, or avoiding risks."),

    ("nist", "nist-csf-2.0", "control_guidance",
     "What does NIST CSF 2.0 say about cybersecurity supply chain risk management?",
     "NIST CSF 2.0 includes Cybersecurity Supply Chain Risk Management (C-SCRM) as a category within the Govern function. Organizations should establish a supply chain risk management strategy that includes plans, processes, and controls for managing cybersecurity risks throughout the supply chain. This involves identifying, assessing, and mitigating risks associated with suppliers and third-party service providers, including evaluating the security practices of suppliers, establishing requirements in contracts, and continuously monitoring supply chain risks."),

    ("nist", "nist-csf-2.0", "recommendation",
     "How does NIST CSF 2.0 recommend integrating cybersecurity with enterprise risk management?",
     "NIST CSF 2.0 recommends treating cybersecurity risk as a significant enterprise risk alongside financial, legal, operational, and other categories. The Govern function explicitly connects cybersecurity risk management to broader organizational governance. Senior leadership should establish cybersecurity risk management strategy and oversee its implementation. Risk tolerance levels should be determined and communicated across the organization. Cybersecurity risk information should be included in enterprise risk management reporting and decision-making processes."),

    ("nist", "nist-csf-2.0", "factual_definition",
     "What is the CSF Core in NIST Cybersecurity Framework 2.0?",
     "The CSF Core in NIST CSF 2.0 is a taxonomy of high-level cybersecurity outcomes organized into Functions, Categories, and Subcategories. The six Functions (Govern, Identify, Protect, Detect, Respond, Recover) provide a strategic view of cybersecurity risk management. Categories are groups of related cybersecurity outcomes within each Function. Subcategories further divide Categories into specific technical and management activities. The Core is not a checklist but provides a common language and systematic methodology for managing cybersecurity risk."),

    ("nist", "nist-csf-2.0", "factual_definition",
     "What are CSF Tiers in NIST CSF 2.0?",
     "CSF Tiers in NIST CSF 2.0 characterize the rigor of an organization's cybersecurity risk governance and management practices. There are four Tiers: Tier 1 (Partial) where risk management practices are ad hoc, Tier 2 (Risk Informed) where risk management practices are approved by management but may not be established organization-wide, Tier 3 (Repeatable) where practices are formally established and regularly updated, and Tier 4 (Adaptive) where the organization adapts practices based on lessons learned and predictive indicators. Tiers do not represent maturity levels."),

    # ─── CISA CPG 2.0 ──────────────────────────────────────────────────
    ("cisa", "cisa-cpg-2.0", "factual_definition",
     "What are CISA Cross-Sector Cybersecurity Performance Goals?",
     "CISA Cross-Sector Cybersecurity Performance Goals (CPGs) 2.0 are a prioritized subset of IT and operational technology cybersecurity practices that critical infrastructure owners and operators can implement to meaningfully reduce the likelihood and impact of known risks and adversary techniques. They are intended to be a minimum set of practices that entities of all sizes, across all critical infrastructure sectors, should implement. CPGs are outcome-oriented and designed to be applicable regardless of organizational size, resources, or sophistication."),

    ("cisa", "cisa-cpg-2.0", "control_guidance",
     "What does CISA CPG 2.0 recommend for multi-factor authentication?",
     "CISA CPG 2.0 recommends implementing phishing-resistant multi-factor authentication (MFA) for all accounts that have access to systems, data, and resources. MFA should be required for remote access, privileged accounts, and access to any critical systems. CISA specifically emphasizes phishing-resistant MFA methods such as FIDO2 security keys or PKI-based solutions over SMS or email-based MFA which can be intercepted. The goal is to reduce the risk of unauthorized access from compromised credentials."),

    ("cisa", "cisa-cpg-2.0", "control_guidance",
     "What guidance does CISA CPG 2.0 provide for managing default and shared credentials?",
     "CISA CPG 2.0 recommends that organizations change all default passwords on hardware and software upon installation, use unique and complex passwords for all accounts, eliminate the use of shared credentials, and implement a process for revoking credentials when personnel no longer require access. Default and shared credentials are among the most commonly exploited weaknesses that lead to unauthorized access. Organizations should maintain an inventory of all accounts with default credentials and prioritize their remediation."),

    ("cisa", "cisa-cpg-2.0", "factual_definition",
     "How are CISA CPG 2.0 goals structured and organized?",
     "CISA CPG 2.0 goals are structured to align with the NIST Cybersecurity Framework 2.0. The goals are organized according to CSF Functions including Govern, Identify, Protect, Detect, Respond, and Recover. Each goal specifies an outcome-oriented cybersecurity practice that organizations should implement. The goals are designed to be actionable and measurable, allowing organizations to assess their implementation status and track progress toward improved cybersecurity posture."),

    ("cisa", "cisa-cpg-2.0", "control_guidance",
     "What does CISA CPG 2.0 recommend for network segmentation?",
     "CISA CPG 2.0 recommends implementing network segmentation to separate critical systems and data from less sensitive areas. Organizations should segment operational technology networks from IT networks, implement proper access controls between segments, monitor traffic crossing network boundaries, and ensure that compromise of one segment does not automatically lead to compromise of other segments. Network segmentation is a key defensive measure to contain the spread of attacks and limit their impact."),

    ("cisa", "cisa-cpg-2.0", "control_guidance",
     "What does CISA CPG 2.0 recommend for vulnerability management?",
     "CISA CPG 2.0 recommends establishing a vulnerability management process that includes regular scanning of assets for known vulnerabilities, prioritizing remediation based on risk including CISA's Known Exploited Vulnerabilities catalog, applying security patches and updates in a timely manner, and maintaining an accurate asset inventory to ensure comprehensive vulnerability coverage. Organizations should remediate known exploited vulnerabilities on a priority basis as they represent the most immediate risk to operations."),

    ("cisa", "cisa-cpg-2.0", "control_guidance",
     "What guidance does CISA CPG 2.0 provide for incident reporting and response?",
     "CISA CPG 2.0 recommends that organizations establish an incident response plan that includes procedures for detecting, analyzing, containing, eradicating, and recovering from cybersecurity incidents. Organizations should report significant cybersecurity incidents to CISA and relevant sector risk management agencies. The plan should include communication procedures, roles and responsibilities, and regular testing through tabletop exercises. Organizations should also maintain relationships with law enforcement and CISA for coordinated incident response."),

    ("cisa", "cisa-cpg-2.0", "recommendation",
     "Why does CISA emphasize the importance of asset inventory in CPG 2.0?",
     "CISA CPG 2.0 emphasizes asset inventory because organizations cannot protect assets they do not know about. A comprehensive and accurate inventory of all hardware and software assets is foundational to effective cybersecurity. It enables vulnerability management, access control, incident response, and recovery planning. Without knowing what assets exist on the network, organizations cannot assess their attack surface, apply appropriate security controls, or detect unauthorized devices and software that may introduce risk."),

    ("cisa", "cisa-cpg-2.0", "control_guidance",
     "What does CISA CPG 2.0 recommend for secure remote access?",
     "CISA CPG 2.0 recommends implementing secure remote access solutions including VPN or zero trust network access with phishing-resistant MFA for all remote connections. Organizations should disable remote access protocols that are not in use, monitor remote access sessions for suspicious activity, implement session timeouts, and ensure remote access does not bypass network security controls. Remote access should be limited to only those users who require it for their job functions."),

    ("cisa", "cisa-cpg-2.0", "control_guidance",
     "What does CISA CPG 2.0 recommend for data protection and backup?",
     "CISA CPG 2.0 recommends implementing regular automated backups of critical data and systems, storing backups in a separate location from the source systems, testing backup restoration procedures regularly, encrypting backup data, and implementing access controls for backup systems. Organizations should ensure backups are sufficient to restore operations after a destructive attack such as ransomware. Backup integrity should be verified regularly and backup systems should not be directly accessible from the primary network."),

    ("cisa", "cisa-cpg-2.0", "recommendation",
     "How does CISA CPG 2.0 recommend organizations approach cybersecurity training?",
     "CISA CPG 2.0 recommends that organizations provide regular cybersecurity awareness training to all personnel. Training should cover recognition of phishing and social engineering attempts, proper handling of sensitive information, reporting procedures for suspicious activity, and role-specific security responsibilities. CISA emphasizes that human behavior is often the weakest link in cybersecurity defenses, making ongoing training and awareness critical to maintaining organizational security posture."),

    # ─── Cross-source / integration questions ──────────────────────────
    ("nist", "nist-csf-2.0", "framework_mapping",
     "How does NIST CSF 2.0 relate to other cybersecurity standards and frameworks?",
     "NIST CSF 2.0 is designed to complement and not replace other cybersecurity standards, guidelines, and practices. It provides Informative References that map CSF outcomes to specific sections of other documents such as NIST SP 800-53 security controls. The framework is applicable to organizations of all sizes and sectors. It can be used alongside sector-specific requirements, CISA's Cross-Sector Cybersecurity Performance Goals, and international standards. The CSF Core provides a common taxonomy for cybersecurity outcomes that can bridge different compliance frameworks."),

    ("cisa", "cisa-cpg-2.0", "framework_mapping",
     "How do CISA CPGs align with the NIST Cybersecurity Framework?",
     "CISA CPGs 2.0 are explicitly structured to align with the NIST Cybersecurity Framework 2.0. The goals are organized according to CSF Functions: Govern, Identify, Protect, Detect, Respond, and Recover. This alignment allows organizations to use CISA CPGs as a prioritized, actionable subset of CSF outcomes. Organizations implementing CISA CPGs can map their progress directly to CSF categories and subcategories, facilitating a comprehensive cybersecurity risk management approach using both frameworks together."),

    ("owasp", "owasp-top10-2025-intro", "factual_definition",
     "What methodology does OWASP use to determine the Top 10 categories?",
     "OWASP uses a data-driven methodology to determine the Top 10 categories. This includes collecting data from a variety of sources about application vulnerabilities, analyzing the frequency, severity, and exploitability of different vulnerability types, and considering both empirical data and input from the security community. Categories are ranked based on factors including how commonly the vulnerability is found in applications, the typical exploitability, the technical impact, and the related Common Weakness Enumerations (CWEs) that map to each category."),

    ("nist", "nist-csf-2.0", "procedural_guidance",
     "What implementation steps does NIST CSF 2.0 recommend for organizations new to cybersecurity?",
     "For organizations new to cybersecurity, NIST CSF 2.0 recommends starting by understanding the organizational context and establishing a governance structure for cybersecurity. Organizations should create a Current Profile to assess their existing cybersecurity posture, develop a Target Profile describing desired outcomes, compare the two to identify gaps, and create an action plan to address priority gaps. The framework emphasizes starting with the Govern function to establish leadership commitment and then working through Identify, Protect, Detect, Respond, and Recover."),

    ("cisa", "cisa-cpg-2.0", "recommendation",
     "What are the highest priority CISA CPG 2.0 practices for small organizations?",
     "For small organizations with limited resources, CISA CPG 2.0 emphasizes foundational practices that provide the greatest risk reduction. These include implementing phishing-resistant multi-factor authentication, changing default passwords, maintaining a basic asset inventory, applying security patches promptly (especially for known exploited vulnerabilities), implementing automated backups and testing restoration, establishing a basic incident response plan, and providing cybersecurity awareness training to all staff. These practices address the most common attack vectors and are designed to be achievable regardless of organizational size."),
]


def _normalize_question(q: str) -> str:
    """Normalize question for dedup/leakage comparison."""
    return " ".join(q.lower().split())


def _load_evaluation_questions() -> set[str]:
    """Load evaluation questions to prevent train/eval leakage."""
    questions = set()
    if EVAL_PATH.exists():
        with open(EVAL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    questions.add(_normalize_question(rec["question"]))
    return questions


def _build_records() -> list[dict]:
    """Build canonical records from DATASET."""
    records = []
    for idx, (family, doc_id, category, question, answer) in enumerate(DATASET, 1):
        record = {
            "id": f"ft_{idx:04d}",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "source_family": family,
            "source_document_id": doc_id,
            "category": category,
        }
        records.append(record)
    return records


def _deterministic_split(
    records: list[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Deterministic split by sorting on content hash. No randomness."""
    # Sort by SHA-256 of question text for determinism
    def sort_key(r: dict) -> str:
        q = r["messages"][1]["content"]
        return hashlib.sha256(q.encode("utf-8")).hexdigest()

    sorted_records = sorted(records, key=sort_key)
    n = len(sorted_records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = sorted_records[:n_train]
    val = sorted_records[n_train:n_train + n_val]
    test = sorted_records[n_train + n_val:]

    return train, val, test


def _check_leakage(
    records: list[dict],
    eval_questions: set[str],
) -> list[str]:
    """Check for question overlap with evaluation dataset."""
    leaks = []
    for rec in records:
        q_norm = _normalize_question(rec["messages"][1]["content"])
        if q_norm in eval_questions:
            leaks.append(rec["id"])
    return leaks


def _check_duplicates(records: list[dict]) -> list[str]:
    """Check for near-duplicate questions (normalized)."""
    seen: dict[str, str] = {}
    dupes = []
    for rec in records:
        q_norm = _normalize_question(rec["messages"][1]["content"])
        if q_norm in seen:
            dupes.append(f"{rec['id']} duplicates {seen[q_norm]}")
        else:
            seen[q_norm] = rec["id"]
    return dupes


def _check_fake_citations(records: list[dict]) -> list[str]:
    """Check for fake [SOURCE_N] citation patterns."""
    import re
    pattern = re.compile(r"\[SOURCE_\d+\]")
    violations = []
    for rec in records:
        answer = rec["messages"][2]["content"]
        if pattern.search(answer):
            violations.append(rec["id"])
    return violations


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _compute_stats(
    records: list[dict],
    train: list[dict],
    val: list[dict],
    test: list[dict],
) -> dict:
    """Compute dataset statistics."""
    def _split_stats(split: list[dict], name: str) -> dict:
        if not split:
            return {"name": name, "count": 0}
        q_lens = [len(r["messages"][1]["content"].split()) for r in split]
        a_lens = [len(r["messages"][2]["content"].split()) for r in split]
        families = {}
        categories = {}
        for r in split:
            families[r["source_family"]] = families.get(r["source_family"], 0) + 1
            categories[r["category"]] = categories.get(r["category"], 0) + 1
        return {
            "name": name,
            "count": len(split),
            "question_length_words": {
                "min": min(q_lens),
                "max": max(q_lens),
                "mean": round(sum(q_lens) / len(q_lens), 1),
            },
            "answer_length_words": {
                "min": min(a_lens),
                "max": max(a_lens),
                "mean": round(sum(a_lens) / len(a_lens), 1),
            },
            "source_families": families,
            "categories": categories,
        }

    return {
        "total": len(records),
        "splits": {
            "train": _split_stats(train, "train"),
            "validation": _split_stats(val, "validation"),
            "test": _split_stats(test, "test"),
        },
        "overall": _split_stats(records, "full"),
    }


def _select_review_sample(
    records: list[dict],
    n: int = 18,
) -> list[dict]:
    """Select a balanced review sample covering all three source families."""
    by_family: dict[str, list[dict]] = {"owasp": [], "nist": [], "cisa": []}
    for r in records:
        by_family.setdefault(r["source_family"], []).append(r)

    sample = []
    per_family = n // 3
    for family in ["owasp", "nist", "cisa"]:
        sample.extend(by_family[family][:per_family])

    # Fill remainder from largest family
    remaining = n - len(sample)
    if remaining > 0:
        for family in ["owasp", "nist", "cisa"]:
            available = by_family[family][per_family:]
            take = min(remaining, len(available))
            sample.extend(available[:take])
            remaining -= take
            if remaining <= 0:
                break

    return sample


def main() -> int:
    """Build and validate the fine-tuning dataset."""
    print("Phase 7 — Fine-Tuning Dataset Builder")
    print("=" * 50)

    # Build records
    records = _build_records()
    print(f"Total records built: {len(records)}")

    # Load evaluation questions for leakage check
    eval_questions = _load_evaluation_questions()
    print(f"Evaluation questions loaded: {len(eval_questions)}")

    # Check for leakage
    leaks = _check_leakage(records, eval_questions)
    if leaks:
        print(f"\nERROR: Evaluation leakage detected in: {leaks}")
        return 1

    # Check for duplicates
    dupes = _check_duplicates(records)
    if dupes:
        print(f"\nERROR: Duplicate questions found: {dupes}")
        return 1

    # Check for fake citations
    fake_cites = _check_fake_citations(records)
    if fake_cites:
        print(f"\nERROR: Fake [SOURCE_N] citations found in: {fake_cites}")
        return 1

    # Validate source IDs
    all_valid = set()
    for family_sources in VALID_SOURCES.values():
        all_valid.update(family_sources)
    bad_sources = [
        r["id"] for r in records if r["source_document_id"] not in all_valid
    ]
    if bad_sources:
        print(f"\nERROR: Unknown source document IDs in: {bad_sources}")
        return 1

    print("All validation checks passed.")

    # Split
    train, val, test = _deterministic_split(records)
    print(f"\nSplits: train={len(train)}, val={len(val)}, test={len(test)}")

    # Verify no split overlap
    train_ids = {r["id"] for r in train}
    val_ids = {r["id"] for r in val}
    test_ids = {r["id"] for r in test}
    assert not (train_ids & val_ids), "Train/val overlap!"
    assert not (train_ids & test_ids), "Train/test overlap!"
    assert not (val_ids & test_ids), "Val/test overlap!"

    # Write outputs
    _write_jsonl(OUT_FULL, records)
    _write_jsonl(OUT_TRAIN, train)
    _write_jsonl(OUT_VAL, val)
    _write_jsonl(OUT_TEST, test)
    print(f"\nDataset files written to {QA_DIR}")

    # Stats
    stats = _compute_stats(records, train, val, test)
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Statistics written to {OUT_STATS}")

    # Metadata
    metadata = {
        "version": "1.0",
        "phase": 7,
        "description": "Fine-tuning dataset for ContextForge domain-specific cybersecurity QA",
        "sources": {
            "owasp": "OWASP Top 10:2025 (CC BY-SA 4.0)",
            "nist": "NIST Cybersecurity Framework 2.0 (public domain)",
            "cisa": "CISA Cross-Sector Cybersecurity Performance Goals 2.0 (public domain)",
        },
        "system_prompt": SYSTEM_PROMPT,
        "message_format": "chat-style: system, user, assistant",
        "split_method": "deterministic SHA-256 hash sort, 70/15/15",
        "total_records": len(records),
        "valid_source_document_ids": {k: v for k, v in VALID_SOURCES.items()},
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadata written to {OUT_META}")

    # Review sample
    sample = _select_review_sample(records)
    _write_jsonl(OUT_REVIEW, sample)
    print(f"Human review sample: {len(sample)} examples -> {OUT_REVIEW}")

    # Summary
    print("\n" + "=" * 50)
    print("Dataset Summary:")
    for split_name in ["train", "validation", "test"]:
        s = stats["splits"][split_name]
        print(f"  {split_name}: {s['count']} records")
        if s["count"] > 0:
            print(f"    Sources: {s['source_families']}")
    print(f"  Total: {stats['total']}")
    print("\nPhase 7 dataset build COMPLETE.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
