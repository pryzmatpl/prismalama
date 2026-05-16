# `runner`

> **Prismalama:** production routing and **`DecideEngine`** (GGML vs AirLLM) live in **`dispatch.go`** and **`runner.go`** — see **`docs/RUNTIME_DISPATCH.md`**. This README describes the minimal subprocess HTTP runner shape.

> Note: this is a work in progress

A minimal runner for loading a model and running inference via an HTTP server.

```shell
./runner -model <model binary>
```

### Completion

```shell
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "hi"}' http://localhost:8080/completion
```

### Embeddings

```shell
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "turn me into an embedding"}' http://localhost:8080/embedding
```
