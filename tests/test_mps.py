import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jaxpropka import TitrationModel
from jaxpropka.synthetic import synthetic_cache


@pytest.mark.mps
def test_mps_float32_forward_reverse_and_midpoint():
    devices=[d for d in jax.devices() if d.platform=='mps']
    if not devices: pytest.skip('JAX MPS unavailable; MPS execution NOT validated')
    jax.config.update("jax_enable_x64",False)
    with jax.default_device(devices[0]):
        model=TitrationModel(synthetic_cache(n=8,chains=2))
        curve=model.curves([3.,7.,11.]);pka=model.pka_sites([(0,'HIS')])
        def loss(z):
            p=jax.nn.softmax(z,-1)
            return curve(p).total_charge.sum()+pka(p).value.sum()
        value,grad=jax.jit(jax.value_and_grad(loss))(jnp.zeros((8,20),jnp.float32))
        value.block_until_ready()
        assert np.isfinite(value) and np.isfinite(grad).all()
        assert next(iter(grad.devices())).platform=='mps'
