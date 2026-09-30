"""Command-line cache construction, prediction, and explicit reference comparison."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import jax
import numpy as np
from . import prepare, StructureCache, TitrationModel, ModelConfig, one_hot, ALPHABET, GROUPS


def main(argv=None):
    parser=argparse.ArgumentParser(prog="jaxpropka")
    sub=parser.add_subparsers(dest="command",required=True)
    prep=sub.add_parser("prepare",help="precompute structural constants with Biotite")
    prep.add_argument("structure");prep.add_argument("output")
    prep.add_argument("--backbone-only",action="store_true",help="permit rebuilding missing native sidechain atoms from CCD")
    prep.add_argument("--freeze-disulfides",action="store_true")
    prep.add_argument("--gap-policy",choices=["error","cap","free"],default="error")
    pred=sub.add_parser("predict",help="evaluate a saved cache")
    pred.add_argument("cache");pred.add_argument("--sequence",help="concatenated one-letter sequence in cache residue order")
    pred.add_argument("--ph-min",type=float,default=0);pred.add_argument("--ph-max",type=float,default=14)
    pred.add_argument("--points",type=int,default=29);pred.add_argument("--pka",action="store_true")
    pred.add_argument("--residues",help="comma-separated ZERO-based output indices; environment remains complete")
    pred.add_argument("--output",default="prediction.npz")
    reg=sub.add_parser("regress",help="compare the SAME exported native candidate structure against PROPKA")
    reg.add_argument("structure");reg.add_argument("--backend",choices=["legacy30","modern"],default="legacy30")
    reg.add_argument("--sequence",help="hard mutant in cache residue order; exported from the same frozen candidates")
    reg.add_argument("--legacy-root");reg.add_argument("--expected-commit")
    reg.add_argument("--output",default="reports/reference.json")
    reg.add_argument("--baseline",help="previously approved report to check; never auto-overwritten")
    reg.add_argument("--record-baseline",help="explicitly record a NEW approved numerical baseline")
    reg.add_argument("--max-mae",type=float,help="optional scientific acceptance threshold, in pKa units")
    for subparser in (pred,reg):
        subparser.add_argument("--steps",type=int,default=64)
        subparser.add_argument("--damping",type=float,default=.35)
        subparser.add_argument("--gate-width",type=float,default=20.)
    args=parser.parse_args(argv)
    config=None if args.command=="prepare" else ModelConfig(steps=args.steps,damping=args.damping,gate_width=args.gate_width)
    if args.command=="prepare":
        cache=prepare(args.structure,topology_options={"freeze_disulfides":args.freeze_disulfides,"gap_policy":args.gap_policy},
                      geometry_options={"missing_sidechain":"template" if args.backbone_only else "error"})
        cache.save(args.output)
        print(json.dumps({"residues":cache.n_residues,"chains":cache.chain_ids,"bytes":cache.nbytes,"cache":args.output},indent=2))
    elif args.command=="predict":
        if args.points<1 or args.ph_max<args.ph_min:
            parser.error("invalid pH grid")
        cache=StructureCache.load(args.cache);model=TitrationModel(cache,config)
        p=model.native_probabilities if args.sequence is None else one_hot(args.sequence)
        model.validate_probabilities(p)
        residues=None if args.residues is None else [int(x) for x in args.residues.split(",")]
        out=jax.device_get(model.curves(np.linspace(args.ph_min,args.ph_max,args.points),residues)(p))
        arrays={"curves_"+name:np.asarray(value) for name,value in out._asdict().items()}
        if args.pka:
            pka=jax.device_get(model.pka(residues)(p))
            arrays.update({"pka_"+name:np.asarray(value) for name,value in pka._asdict().items()})
        sel=cache.select(residues)
        arrays["metadata"]=np.asarray(json.dumps({"selected_residues":[vars(cache.keys[i]) for i in sel],"alphabet":ALPHABET,"groups":GROUPS,
                                                   "chain_ids":cache.chain_ids,"model_config":asdict(model.config),"cache_fingerprint":cache.fingerprint()}))
        with open(args.output,"wb") as f: np.savez_compressed(f,**arrays)
        print(json.dumps({"output":args.output,"max_fixed_point_residual":float(out.residual.max()),
                          "all_curve_states_converged":bool(out.converged.all())},indent=2))
    elif args.command=="regress":
        from .topology import load_topology
        from .geometry import build_candidates
        from .precompute import build_cache
        from .reference import write_reference_structure,run_reference,compare_reference,assert_baseline
        top=load_topology(args.structure,freeze_disulfides=True)
        geometry=build_candidates(top);cache=build_cache(top,geometry);model=TitrationModel(cache,config)
        p=model.native_probabilities if args.sequence is None else one_hot(args.sequence)
        model.validate_probabilities(p)
        sequence=np.argmax(np.asarray(p),axis=-1)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"normalized.pdb"
            mapping=write_reference_structure(top,geometry,path,sequence_indices=sequence)
            ref=run_reference(path,backend=args.backend,legacy_root=args.legacy_root,
                              mapping=mapping,expected_commit=args.expected_commit)
        report=compare_reference(model,p,ref,sequence_indices=sequence)
        dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(report,indent=2)+"\n")
        dest.with_suffix(".pka").write_text(ref.output_text)
        dest.with_suffix(".stdout.txt").write_text(ref.stdout)
        if args.baseline:
            assert_baseline(report,json.loads(Path(args.baseline).read_text()))
        if args.max_mae is not None and report["metrics"]["pka_mae"]>args.max_mae:
            raise SystemExit(f"pKa MAE {report['metrics']['pka_mae']:.4f} exceeds {args.max_mae}; report saved to {dest}")
        if args.record_baseline:
            baseline=Path(args.record_baseline)
            if baseline.exists():
                raise FileExistsError("refusing to overwrite an existing approved baseline")
            baseline.parent.mkdir(parents=True,exist_ok=True)
            baseline.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report["metrics"],indent=2))

if __name__=="__main__": main()
