# UV Package Manager Migration Plan

This document outlines the plan to migrate the `cli-code` project from using `pip` and `venv` to using `uv` for environment and package management.

## 1. Rationale

Switching to `uv` aims to:

*   **Improve Development Speed:** Leverage `uv`'s significantly faster dependency resolution and installation capabilities compared to `pip`.
*   **Streamline Workflows:** Utilize `uv`'s unified command-line interface for environment creation (`uv venv`), installation (`uv pip install`, `uv sync`), and potentially locking (`uv pip compile`).
*   **Enhance CI/CD Efficiency:** Reduce the time taken for dependency installation steps in the GitHub Actions workflow.
*   **Align with Modern Tooling:** Adopt a actively developed tool from the creators of Ruff.

## 2. Migration Milestones

This migration will be performed on a dedicated feature branch (e.g., `feature/uv-migration`) following the established development practices.

*   **M1: Local Validation & Setup** (`feature/uv-migration`)
    *   Developers install `uv` locally.
    *   Verify core local workflows:
        *   Create virtual environment: `uv venv`
        *   Install dependencies from `pyproject.toml` for development: `uv pip install -e .[dev]` (or similar depending on group definitions)
        *   Run linters (Ruff, MyPy).
        *   Run tests (`pytest`).
    *   Confirm basic project functionality remains unchanged.
    *   Document any immediate issues or necessary adjustments.
    *   Commit initial findings and successful local workflow commands.

*   **M2: Update Documentation** (`feature/uv-migration`)
    *   Modify `README.md`: Update installation and setup sections to use `uv` commands.
    *   Modify `docs/install.md`: Update detailed installation and setup instructions.
    *   Modify `docs/contributing.md`: Update development environment setup guide.
    *   Ensure consistency across all documentation referencing environment setup or package installation.
    *   Commit documentation changes.

*   **M3: Update CI Workflow** (`feature/uv-migration`)
    *   Modify `.github/workflows/python-ci.yml`:
        *   Add a step to install `uv` itself early in the job.
        *   Replace `pip install` commands with equivalent `uv pip install` or `uv sync` commands for installing project dependencies and test requirements.
    *   Trigger CI runs and verify all jobs pass successfully using `uv`.
    *   Commit CI workflow changes.

*   **M4: Final Review & Merge** (`feature/uv-migration`)
    *   Review all project files (scripts, docs, configs) for any lingering `pip`-specific commands or references that should be updated or removed.
    *   Consider removing `requirements.txt` files if they were only used for `pip` installation and locking isn't managed by `uv pip compile` yet (if applicable).
    *   Perform a final check of local workflows and CI pipeline status.
    *   Submit Pull Request for review and merge into `main`.

## 3. Development Practices

For *this migration*:

1.  **Feature Branch:** All work will be done on the `feature/uv-migration` branch.
2.  **Small Commits:** Use small, logical commits for each milestone step.
3.  **Testing/Linting:** Ensure all existing tests and linter checks pass after switching local commands to `uv`.
4.  **CI Pipeline:** The primary goal is to ensure the CI pipeline passes using `uv` for dependency management in M3.
5.  **Pull Request:** Submit a Pull Request to `main` upon completion of M4.

## 4. Files Potentially Modified

*   `README.md`
*   `docs/install.md`
*   `docs/contributing.md`
*   `.github/workflows/python-ci.yml`
*   (Potentially remove/ignore `requirements*.txt` files if switching locking mechanism, TBD)

## 5. Considerations

*   **Lock Files:** This plan focuses on replacing `pip install` and `venv`. We can decide later whether to also replace `pip-tools` with `uv pip compile` for managing lock files (`requirements.txt`/`.lock`). For now, `uv pip install -r requirements.txt` can still be used if needed.
*   **Developer Adoption:** Ensure all active developers are aware of the switch and update their local environments/workflows.
