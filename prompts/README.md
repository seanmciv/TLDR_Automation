# Prompt templates for TLDR report

This folder holds the prompts used to generate the weekly TLDR AI summary. Keeping them here makes it easy to track changes over time.

## Files

- **tldr_report_system.txt** – System message (e.g. output format, constraints)
- **tldr_report_user.txt** – User prompt template; `Stories:` is followed by the scraped content

## Versioning

To record a prompt change, commit with a message like:

```
prompts: adjust tone / add bullets / narrow scope
```

You can also copy files before editing (e.g. `tldr_report_user_v2.txt`) if you want to keep old versions.
