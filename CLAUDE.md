# rho-tts

Multi-provider TTS library with voice cloning, subprocess isolation, and Gradio UI.

## Project Layout

- **Source**: `src/rho_tts/` (hatchling build, `pyproject.toml`)
- **Tests**: `tests/` (pytest)
- **Venv**: `venv/bin/python` — always use this, not system python

## Development Commands

```bash
# Run all tests
venv/bin/python -m pytest tests/ -x -q

# Run specific test file
venv/bin/python -m pytest tests/test_<name>.py -v

# Lint / format
venv/bin/ruff check src/ --fix
venv/bin/ruff format src/

# Verify imports compile
venv/bin/python -c "import ast; ast.parse(open('src/rho_tts/<file>.py').read())"
```

## Architecture

- `BaseTTS` — abstract base with 2 abstract methods: `_generate_audio`, `generate` (+ `sample_rate` property)
- `TTSFactory` — provider registry with class-level state. Tests must save/restore `_providers`, `_isolated_providers`, `_default_providers_registered`
- `CancellationToken` — thread-safe cooperative cancellation
- Providers: `providers/qwen.py`, `providers/chatterbox.py`
- Isolation layer (`isolation/`): subprocess-based venv isolation for providers with conflicting deps. JSON-line IPC protocol over stdin/stdout
- UI: Gradio-based, state in `ui/state.py` (AppState singleton)

## Testing Conventions

- Mock torchaudio: `patch.dict("sys.modules", {"torchaudio": mock_ta})`
- Use class-based tests with `setup_method`/`teardown_method`
- Factory tests must save/restore class-level state in setup/teardown
- 107+ tests, target is all passing with 1 allowed skip (num2words)

## Key Rules

- Always use `venv/bin/python`, never system python
- Don't edit `.safetensors`, `.bin`, `.pkl`, or `requirements-frozen.txt`
- Provider code may have heavy deps (torch, transformers) — mock them in tests
