# Security

## Prismalama deployments

Prismalama inherits Ollama’s **unauthenticated local API** model. Treat any reachable **`OLLAMA_HOST`** like an **open inference endpoint**: bind to loopback on single-user machines (the Arch package defaults to **`127.0.0.1:11434`** for **`OLLAMA_HOST`**), use firewalls and VPNs on LANs, or terminate TLS and authenticate at a reverse proxy for remote access.

Report security issues for **this fork** through the maintainers’ preferred private channel for **[github.com/piotroxp/prismalama](https://github.com/piotroxp/prismalama)** (e.g. GitHub Security Advisories if enabled).

---

## Upstream Ollama disclosure

The upstream Ollama maintainer team takes security seriously and will actively work to resolve security issues in the shared codebase.

### Reporting a vulnerability (upstream)

If you discover a vulnerability that belongs to **upstream Ollama**, please do not open a public issue there until coordinated disclosure. Instead, report by emailing **hello@ollama.com**. Include:

- A description of the vulnerability
- Steps to reproduce the issue
- Your assessment of the potential impact
- Any possible mitigations

### Security best practices (upstream)

- Regularly updating to the latest version of Ollama / Prismalama merge
- Securing access to hosted instances (same surface as above)
- Monitoring systems for unusual activity

### Contact (upstream)

Other upstream security questions: **hello@ollama.com**
