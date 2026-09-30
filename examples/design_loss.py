"""A generic sequence-gradient example, not a trained inverse-folding model."""
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from jaxpropka import prepare,TitrationModel,ResidueKey


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure")
    args=parser.parse_args()
    cache=prepare(args.structure)
    model=TitrationModel(cache)
    selected=[cache.keys[0]]  # substitute explicit keys, e.g. ResidueKey("A",42,"B")
    charge_fn=model.charge(7.0,selected)
    curves_fn=model.curves(np.linspace(4,10,25),selected)
    pka_fn=model.pka_sites([(selected[0],"HIS")])
    logits=jnp.log(.98*model.native_probabilities+.02/20)

    @jax.jit
    def loss(logits):
        p=model.probabilities_from_logits(logits)
        charge=charge_fn(p)
        pk=pka_fn(p)
        # A conditional target pKa is not multiplied by the focal identity
        # probability inside HH. This loss separately rewards that identity.
        histidine_probability=p[0,6]  # ALPHABET='ACDEFGHIKLMNPQRSTVWY'
        pka_error=jnp.where(pk.valid,jnp.square(pk.value-6.5),0).sum()
        return jnp.square(charge).sum()+.1*pka_error-.05*jnp.log(histidine_probability+1e-8)

    # Check diagnostics before optimizing; invalid root masks must not become a
    # way for an optimizer to evade a pKa objective. Reject/penalize invalid states
    # in the enclosing training pipeline rather than relying only on the mask.
    diagnostic=pka_fn(model.probabilities_from_logits(logits))
    if not bool(jnp.all(diagnostic.valid)):
        raise RuntimeError("initial pKa target is invalid/nonconverged")
    value,gradient=jax.value_and_grad(loss)(logits)
    print("loss",float(value),"gradient shape",gradient.shape)
    print("selected charge",charge_fn(model.probabilities_from_logits(logits)))
    print("curve shape",curves_fn(model.probabilities_from_logits(logits)).residue_charge.shape)

if __name__=="__main__":main()
