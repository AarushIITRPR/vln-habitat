from habitat.core.registry import registry


def get_env_class(env_name: str):
    env_class = registry.get_env(env_name)
    if env_class is None:
        raise KeyError(f"Environment '{env_name}' is not registered")
    return env_class
