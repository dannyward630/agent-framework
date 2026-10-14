# Get Started with Microsoft Agent Framework Foundry Local

Install this package as the extra for `agent-framework` and see the [Python README](https://github.com/microsoft/agent-framework/tree/main/python/README.md) for more information:

```bash
pip install agent-framework-foundry-local --pre
```

Python 3.10 uses Foundry Local SDK 0.5.1. Python 3.11 and newer use Foundry Local SDK `>=1.2.4,<3`, including the 2.x SDK. The current SDK starts its embedded OpenAI-compatible HTTP endpoint automatically; the client continues to use Agent Framework's OpenAI transport and middleware.

The `client.manager` property retains the previous `list_catalog_models`, `get_model_info`, `download_model`, and `load_model` helpers. It also exposes the SDK manager API, including `client.manager.catalog`.

## Foundry Local Sample

See the [Foundry Local provider sample](../../samples/02-agents/providers/foundry/foundry_local_agent.py) for a runnable example.
