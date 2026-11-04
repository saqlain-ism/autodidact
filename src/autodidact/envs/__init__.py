"""Sandbox applications the agent learns to use."""

# Bumped whenever an app's behaviour or its hidden tasks change, so results
# from different versions are never mixed in one run folder.
VERSIONS = {"inventory": "stockroom-v2"}


def version_of(app: str) -> str:
    return VERSIONS.get(app, app)
