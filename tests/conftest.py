import os
import jax
import pytest
from jaxpropka.synthetic import synthetic_cache

# Finite-difference checks use float64; production examples use float32.
jax.config.update("jax_enable_x64", True)

@pytest.fixture
def cache():
    return synthetic_cache(n=3,neighbors=2,chains=2)


def pytest_collection_modifyitems(config,items):
    if os.environ.get("JAXPROPKA_REQUIRE_INTEGRATION") == "1":
        # Dependency failures should happen in CI, rather than produce a green
        # job containing skipped scientific integration tests.
        import biotite  # noqa: F401
    if os.environ.get("JAXPROPKA_REQUIRE_REFERENCE") == "1":
        import propka  # noqa: F401
