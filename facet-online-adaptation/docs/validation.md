# Validation

```bash
python -m pytest -q
python tools/check_release.py
```

The tests exercise action bounds, successful-intervention selection, target networks,
auxiliary wrench updates, checkpoint resume, frozen evaluation, policy contracts,
robot backend protocols and command guards. Hardware motion, force regulation and
success rates require separate robot evaluation.
