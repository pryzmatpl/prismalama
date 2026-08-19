package version

// Version is the Prismalama server version. Bumped per release.
var Version string = "v0.4.1.r5053.4b15df6b"

// CompiledAt is the RFC3339 build timestamp, set via -ldflags at build time
// (e.g. -X github.com/ollama/ollama/version.CompiledAt=2026-08-13T10:00:00Z).
// Empty when the build did not inject the value — surfaced in
// /api/prismalama/capabilities under build.compiled_at only when present.
var CompiledAt string = ""
