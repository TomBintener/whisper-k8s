## Description
<!-- Provide a brief summary of the changes introduced by this pull request. -->

## Motivation & Context
<!-- Why is this change required? What issue does it fix? -->
<!-- If fixing an open issue, link it here: Fixes #123 -->

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Performance optimization (P0–P4 enhancement)
- [ ] Documentation update
- [ ] CI/CD or build pipeline adjustment

## Verification & Testing
<!-- Describe the tests you ran to verify your changes. Include test commands and output summary. -->
- [ ] Ran zero-dependency test suite: `python -m unittest discover tests -v`
- [ ] Validated Docker Compose syntax: `docker compose config`
- [ ] Linted codebase: `ruff check app tests scripts`

## Security & Architectural Review
- [ ] Any external URL handling includes SSRF validation (`_validate_remote_url`)
- [ ] Any file/directory operations are path-traversal safe (`_safe_under`)
- [ ] GPU memory allocations respect configured VRAM limits

## Checklist:
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation in `docs/` and `README.md`
- [ ] My changes generate no new warnings or lint errors
