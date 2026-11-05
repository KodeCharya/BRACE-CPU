# Decisions Log — BRACE-CPU

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-11-05 | Isolated venv inside project; do not modify system Python | Avoid touching user's existing Python installation |
| 2026-11-05 | CPU-only PyTorch wheel (`torch --index-url .../cpu`) | No GPU/CUDA available in environment |
| 2026-11-05 | Start Phase 1 literature review | Mandatory before implementation per project directive |
