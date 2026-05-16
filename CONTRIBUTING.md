# Contributing to Prismalama

![Prismalama Logo](logo.jpg)

Thank you for your interest in contributing. **Prismalama** is an Ollama-compatible fork ([github.com/piotroxp/prismalama](https://github.com/piotroxp/prismalama)); changes that touch runners, GGML/Vulkan, AirLLM, or packaging should align with **[docs/DEVELOPER.md](./docs/DEVELOPER.md)** and **[docs/PRISMALAMA_PRINCIPLE.md](./docs/PRISMALAMA_PRINCIPLE.md)**.

## Set up

See **[docs/DEVELOPER.md](./docs/DEVELOPER.md)** first. Generic upstream build notes live in [development documentation](./docs/development.md).

### Ideal issues (Prismalama)

* [Bugs](https://github.com/piotroxp/prismalama/issues): crashes, incorrect routing (GGML vs AirLLM), load failures, or API regressions.
* Performance: throughput, VRAM use, or scheduling — include hardware and **`journalctl -u ollama`** snippets where useful.
* [Security](./SECURITY.md): do **not** open public issues for undisclosed vulnerabilities; use the process in **SECURITY.md**.

Upstream Ollama uses its own tracker for stock Ollama-only issues: [bugs](https://github.com/ollama/ollama/issues?q=is%3Aissue+is%3Aopen+label%3Abug), [performance](https://github.com/ollama/ollama/issues?q=is%3Aissue+is%3Aopen+label%3Aperformance).

### Issues that are harder to review

* New features: new features (e.g. API fields, environment variables) add surface area to Ollama and make it harder to maintain in the long run as they cannot be removed without potentially breaking users in the future.
* Refactoring: large code improvements are important, but can be harder or take longer to review and merge.
* Documentation: small updates to fill in or correct missing documentation are helpful, however large documentation additions can be hard to maintain over time.

### Issues that may not be accepted

* Changes that break backwards compatibility in Ollama's API (including the OpenAI-compatible API)
* Changes that add significant friction to the user experience
* Changes that create a large future maintenance burden for maintainers and contributors

## Proposing a (non-trivial) change

> By "non-trivial", we mean a change that is not a bug fix or small documentation update.

Open an issue on **[github.com/piotroxp/prismalama](https://github.com/piotroxp/prismalama/issues)** before a large PR so maintainers can agree on scope (dispatch, GGML, packaging). For **upstream Ollama–only** topics, the [Ollama Discord](https://discord.gg/ollama) remains the upstream forum.

Before opening a non-trivial Pull Request, discussion helps prevent duplicated work or changes that conflict with Prismalama’s architecture (**`docs/PRISMALAMA_PRINCIPLE.md`**).

Tips for proposals:

* Explain the problem you are trying to solve, not what you are trying to do.
* Explain why the change is important.
* Explain how the change will be used.
* Explain how the change will be tested.

Additionally, for bonus points: Provide draft documentation you would expect to
see if the changes were accepted.

## Pull requests

**Commit messages**

The title should look like:

    <package>: <short description>

The package is the most affected Go package. If the change does not affect Go
code, then use the directory name instead. Changes to a single well-known
file in the root directory may use the file name.

The short description should start with a lowercase letter and be a
continuation of the sentence:

      "This changes Prismalama to..."

Examples:

      llm/backend/mlx: support the llama architecture
      CONTRIBUTING: provide clarity on good commit messages, and bad

Bad Examples:

      feat: add more emoji
      fix: was not using famous web framework
      chore: generify code

**Tests**

Please include tests. Strive to test behavior, not implementation.

**New dependencies**

Dependencies should be added sparingly. If you are adding a new dependency,
please explain why it is necessary and what other ways you attempted that
did not work without it.

## Need help?

Use **[Prismalama issues](https://github.com/piotroxp/prismalama/issues)** for this fork. For upstream Ollama behavior, see **[discord.gg/ollama](https://discord.gg/ollama)** and **[github.com/ollama/ollama](https://github.com/ollama/ollama)**.
