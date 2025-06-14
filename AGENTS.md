
# Project Agents.md Guide for OpenAI Codex

This `AGENTS.md` file provides comprehensive guidance for OpenAI Codex and other AI agents working with this codebase.

---

## 🧠 Project Structure for OpenAI Codex Navigation

- `/gui_modern.py`: Streamlit GUI (entry point, interface only)
- `/gui_core.py`: Core image processing logic and detector execution
- `/detectors_modern.py`: All edge detector classes (BDCN, RCF, HED, DexiNed, CASENet, Structured Forest)
- `/detector_base.py`: Abstract base class shared by all detectors
- `/utils_image.py`: Image I/O, format checking, resizing and grayscale handling
- `/log_modern.py`: Logger setup with file and console output
- `/config_modern.yaml`: Central configuration used across all modules
- `/images`: Input directory for all supported image types
- `/results`: Output directory with per-method subfolders
- `/tests`: Pytest-compatible test module for basic validation
- `/setup_script.bat`: Windows-only script to fetch models and configure structure

---

## 🧱 Coding Conventions for OpenAI Codex

- Use Python 3.10+ syntax
- Follow PEP-8 formatting and naming conventions
- Use `torch.device` and `.to(self.device)` for all model and tensor operations
- Always normalize edge output to 0–255 uint8
- Use `kornia.color.rgb_to_grayscale()` before feeding into CNNs
- Models should be loaded only once and reused (no reloading per image)
- Use `cv2.resize()` with `INTER_CUBIC` or `INTER_LINEAR` when resizing
- Use `tqdm` or Streamlit progress bar for all batch iterations
- Do not include preview, interactive display, or CLI interaction
- GPU fallback must never crash – always support CPU fallback

---

## 🧪 Testing Requirements for OpenAI Codex

OpenAI Codex should ensure any detector implementation supports dummy image test calls via:

```bash
python tests/test_detectors.py
```

Models must output proper grayscale maps (2D, uint8) of same size as input.

---

## 🔄 Pull Request Guidelines for OpenAI Codex

When OpenAI Codex contributes code, PRs must:

1. Clearly describe the detector or module added
2. Include any test cases or dummy image verifications
3. Pass linting and structural checks
4. Use Streamlit components without JavaScript or frontend hacks
5. Include docstrings for all class-level and method-level definitions

---

## ✅ Programmatic Checks for OpenAI Codex

Before submitting new detector code or core logic, ensure:

```bash
# Check type hints
mypy .

# Check formatting
black .

# Run functional test
python tests/test_detectors.py
```

All code must be functional in a local offline environment with no network dependency after models are downloaded.

Agents.md helps OpenAI Codex follow these rules when contributing to this advanced image-processing GUI system.
