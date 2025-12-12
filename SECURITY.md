# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in dLLM Serving, please report it responsibly:

1. **Do NOT** open a public GitHub issue
2. Email security concerns to: security@example.com
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will respond within 48 hours and work with you to understand and resolve the issue.

## Security Best Practices

When deploying dLLM Serving:

1. **Network Security**: Run behind a reverse proxy with TLS
2. **Authentication**: Enable API key authentication in production
3. **Container Security**: Use non-root user (default in our Dockerfile)
4. **Resource Limits**: Set appropriate CPU/memory limits
5. **Updates**: Keep dependencies updated regularly

