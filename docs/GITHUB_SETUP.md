# Group repository and collaboration

The project repository already exists:
https://github.com/naramritvij/-neu-self-improvising-assignment-8

Owner: `naramritvij`. Visibility: private as verified on 2026-11-16.
Keep the leading hyphen in clone URLs. Use `assignment-8` as the local folder:

```bash
git clone https://github.com/naramritvij/-neu-self-improvising-assignment-8.git assignment-8
cd assignment-8
```

Collaborators are added through Settings → Collaborators. Do not invent a
GitHub handle, change repository visibility or grant access to another person.

Use feature branches for independent work. Main holds the shared
implementation. Review each other's changes and record actual work in
the project log (docs/CONTRIBUTIONS.md).

The CI workflow runs CPU tests only. It does not authenticate to Modal, start
GPU jobs or download model weights. Its presence is not evidence that a specific
Actions run passed; inspect the Actions result.
