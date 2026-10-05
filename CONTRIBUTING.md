# Contributing to logbook

Thank you for your interest in contributing! This project is research code
accompanying a published paper. We welcome contributions that improve
reproducibility, fix bugs, or extend the evaluation framework.

## Code of Conduct

By participating in this project, you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md). Please be respectful, constructive,
and collaborative.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear description of the problem
- Steps to reproduce (ideally a minimal example)
- Expected vs. actual behavior
- Environment details (Python version, CUDA version, GPU type)

### Proposing Changes

For significant changes (new features, architectural changes, etc.):
1. Open an issue first to discuss the proposed change
2. Wait for maintainer feedback before investing significant effort
3. Once approved, submit a pull request

For small fixes (typos, documentation, obvious bugs):
- Feel free to submit a PR directly

### Pull Request Guidelines

**Before submitting:**
- Run the test suite: `python -m unittest discover tests`
- Verify your changes don't break existing functionality
- Add tests for new functionality where appropriate
- Follow the existing code style (we use `ruff format` or `black`)

**PR content:**
- Provide a clear description of what changed and why
- Reference any related issues
- Keep PRs focused — one logical change per PR

**What we require for merges:**
- All tests must pass
- Changes to core metrics (`boundary_f1`, `frame_level_accuracy`,
  `event_based_f1`, `event_error_rate`) require paired tests
- Changes to `clean_annotations`, `single_stream_from_events`, or other
  data-prep invariants require paired tests
- No regressions on published baseline numbers (we'll verify)

### Testing

Tests live in `tests/` and use Python's `unittest` framework:
```bash
python -m unittest discover tests
```

Current baseline: 238 tests passing (4 skipped when vllm unavailable).

### Development Setup

```bash
git clone https://github.com/facebookresearch/logbook.git
cd logbook
pip install -e ".[gpu]"   # or [cpu] for data-prep only
```

For EnCLAP/MS-CLAP cascade work, see `scripts/install_las_enclap.sh`.

### Commit Messages

- Use the imperative mood ("Add feature" not "Added feature")
- First line: concise summary (<72 chars)
- Body (if needed): explain *why* the change was made, not *what* changed
  (the diff shows what)

### Areas We're Particularly Interested In

- **Reproducibility improvements**: clearer setup instructions, environment
  fixes, platform compatibility
- **Bug fixes**: correctness issues in metrics, data loading, or inference
- **Evaluation extensions**: new datasets, new metrics (with references),
  alternate label taxonomies
- **Performance**: faster data loading, better GPU utilization (with
  benchmarks showing the improvement)
- **Documentation**: clearer READMEs, API docs, usage examples

### What We Won't Accept

- Changes that break reproducibility of published numbers
- Removing or weakening tests
- Undocumented "magic number" changes to hyperparameters or thresholds
- PRs that mix unrelated changes (refactors + features + fixes all in one)

## License

By contributing, you agree that your contributions will be licensed under
the same CC BY-NC 4.0 license as the rest of the project.

## Questions?

Open an issue with the `question` label. Maintainers typically respond
within a few days.
