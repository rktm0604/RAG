# Security Policy

## Reporting a Vulnerability

If you discover a security issue, please open a GitHub issue with the label `security`.

Please **do not** post API keys, tokens, or credentials in issues. The `.env` file is gitignored and excluded from Docker builds via `.dockerignore`.

## Best Practices

- Never commit `.env` or any file containing secrets
- Use environment variables for all sensitive configuration
- Keep dependencies updated via `pip install --upgrade -r requirements.txt`
