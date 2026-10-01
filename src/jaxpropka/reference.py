"""External PROPKA adapters and discrepancy/regression reporting.

The default backend uses the pinned PyPI package installed by the ``reference``
extra. An explicit legacy30 backend remains available for historical comparisons.
No subprocess or Python reference code is ever called from a JAX trace.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import importlib.metadata
import json
import re
import shutil
import string
import subprocess
import sys
import tempfile
import time
import numpy as np
from .cache import ResidueKey
from .parameters import GROUPS, GROUP_AA, Q_DEPROT, THREE


@dataclass(frozen=True)
class ReferenceSite:
    key: ResidueKey
    group: str
    pka: float


@dataclass(frozen=True)
class ReferenceRun:
    sites: tuple[ReferenceSite,...]
    provenance: dict
    output_text: str
    stdout: str
    elapsed_seconds: float
    excluded_sites: tuple[dict, ...] = ()


def write_reference_structure(topology, candidates, path, sequence_indices=None):
    """Export the SAME frozen native/mutant candidate coordinates used by the model.

    Every peptide segment receives a unique short PDB chain code and every
    residue a globally unique consecutive number. PROPKA 3.5.1's terminal
    detection compares residue numbers without chain identity after OXT;
    restarting numbering at 1 misassigns termini after one-residue segments.
    A bijective map restores original multichar
    chain IDs, arbitrary numbering and insertion codes after PROPKA parsing.
    This prevents legacy PDB label limitations from conflating residues.
    """
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile
    sequence=topology.native_index if sequence_indices is None else np.asarray(sequence_indices)
    if sequence.shape!=(topology.n_residues,) or not np.issubdtype(sequence.dtype,np.integer):
        raise ValueError("sequence_indices must be a length-N integer array")
    if np.any((sequence<0)|(sequence>=20)):
        raise ValueError("identity outside alphabet")
    if np.any(sequence[topology.disulfide]!=topology.native_index[topology.disulfide]):
        raise ValueError("cannot change a frozen disulfide identity")
    # Bare PDB segments are interpreted by the reference as free termini. Do
    # not compare a capped/missing-segment model to uncapped reference chemistry.
    starts=topology.previous<0;ends=topology.following<0
    if not np.array_equal(topology.nterm,starts) or not np.array_equal(topology.cterm,ends):
        raise ValueError("reference export requires free peptide-segment termini; capped/gapped terminal chemistry is not supported by this adapter")
    codes=string.ascii_uppercase+string.ascii_lowercase+string.digits
    records=[];mapping={};segment=-1;number=0
    for i,key in enumerate(topology.keys):
        if topology.previous[i]<0:
            segment+=1
        if segment>=len(codes):
            raise ValueError("legacy PDB reference adapter supports at most 62 segments")
        number+=1
        if number>9999:
            raise ValueError("legacy PDB residue-number capacity exceeded")
        chain=codes[segment]
        mapping[chain,number]=key
        candidate=candidates.residues[i][int(sequence[i])]
        for name,xyz in candidate.coordinates.items():
            element=candidate.elements.get(name,"O" if name=="OXT" else name[0])
            records.append((chain,number,THREE[int(sequence[i])],name,element,xyz))
    if len(records)>99999:
        raise ValueError("legacy PDB atom-number capacity exceeded")
    atoms=struc.AtomArray(len(records))
    atoms.chain_id=np.asarray([x[0] for x in records])
    atoms.res_id=np.asarray([x[1] for x in records])
    atoms.ins_code=np.full(len(records),"")
    atoms.res_name=np.asarray([x[2] for x in records])
    atoms.atom_name=np.asarray([x[3] for x in records])
    atoms.element=np.asarray([x[4] for x in records])
    atoms.coord=np.asarray([x[5] for x in records],np.float32)
    atoms.hetero=np.zeros(len(records),bool)
    file=PDBFile();file.set_structure(atoms);file.write(path)
    return mapping


def parse_pka(text, mapping=None):
    """Parse summary values, not determinant contributions or repeated profiles."""
    # PROPKA's fixed-width labels have no separating space at residue 1000+.
    pattern=re.compile(r"^\s*(ASP|GLU|HIS|CYS|TYR|LYS|ARG|N\+|C-)\s*(-?\d+)\s+([A-Za-z0-9])\s+([-+]?\d+(?:\.\d+)?)")
    active=False;sites=[];seen=set()
    for line in text.splitlines():
        if "SUMMARY OF THIS PREDICTION" in line.upper():
            active=True;continue
        if not active:
            continue
        match=pattern.match(line)
        if match:
            name,number,chain,value=match.groups()
            group={"N+":"NTERM","C-":"CTERM"}.get(name,name)
            lookup=(chain,int(number))
            if mapping is not None:
                if lookup not in mapping:
                    raise ValueError(f"reference emitted an unmapped residue: {lookup}")
                key=mapping[lookup]
            else:
                key=ResidueKey(chain,int(number))
            if (key,group) in seen:
                raise ValueError(f"duplicate reference site {key} {group}")
            seen.add((key,group));sites.append(ReferenceSite(key,group,float(value)))
        elif sites and (line.strip().startswith("-----") or "Free energy" in line):
            break
    if not sites:
        raise ValueError("no recognized sites in PROPKA summary; inspect captured output")
    return tuple(sites)


def run_reference(path, *, backend="modern", legacy_root=None, mapping=None,
                  timeout=300, expected_commit=None):
    path=Path(path).resolve()
    provenance={"backend":backend,"input_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    if backend=="legacy30":
        if legacy_root is None:
            raise ValueError("legacy30 requires a PROPKA 3.0 checkout via legacy_root")
        root=Path(legacy_root).resolve()
        if not (root/"propka.py").is_file() or not (root/"Source"/"version.py").is_file():
            raise ValueError("not a legacy PROPKA 3.0 checkout")
        git=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],capture_output=True,text=True)
        if git.returncode:
            raise ValueError("reference checkout must retain Git provenance")
        commit=git.stdout.strip()
        if expected_commit is not None and commit!=expected_commit:
            raise ValueError(f"legacy reference commit mismatch: {commit} != {expected_commit}")
        dirty=subprocess.run(["git","-C",str(root),"status","--porcelain","--untracked-files=no"],capture_output=True,text=True)
        if dirty.stdout.strip():
            raise ValueError("reference checkout has tracked modifications; use a clean pinned checkout")
        provenance.update(version="3.0",subversion="Nov30 (upstream default)",commit=commit)
        command=[sys.executable,str(root/"propka.py")]
    elif backend=="modern":
        try:
            version=importlib.metadata.version("propka")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError("install the reference extra for modern PROPKA") from exc
        provenance.update(version=version,subversion="modern package default; NOT PROPKA 3.0")
        command=[sys.executable,"-c", "\n".join([
            "import sys, json",
            "from pathlib import Path",
            "from propka.run import single",
            "m = single(sys.argv[1], write_pka=True)",
            "excluded = [{'chain': g.atom.chain_id, 'number': g.atom.res_num, 'group': g.residue_type, 'coupled_label': g.coupled_titrating_group.label} for g in m.conformations['AVR'].groups if g.coupled_titrating_group and m.version.parameters.remove_penalised_group]",
            "Path('excluded.json').write_text(json.dumps(excluded))",
        ])]
    else:
        raise ValueError("backend must be legacy30 or modern")
    with tempfile.TemporaryDirectory(prefix="jaxpropka-reference-") as directory:
        work=Path(directory);input_path=work/"structure.pdb";shutil.copyfile(path,input_path)
        start=time.perf_counter()
        run=subprocess.run(command+[str(input_path)],cwd=work,capture_output=True,text=True,timeout=timeout)
        elapsed=time.perf_counter()-start
        if run.returncode:
            raise RuntimeError(f"PROPKA {backend} failed ({run.returncode})\n{run.stdout[-3000:]}\n{run.stderr[-3000:]}")
        files=list(work.glob("*.pka"))
        if len(files)!=1:
            raise RuntimeError(f"expected one .pka file, got {files}; stdout:\n{run.stdout[-2000:]}")
        text=files[0].read_text()
        excluded=[]
        if backend == "modern":
            for entry in json.loads((work/"excluded.json").read_text()):
                pair=(entry["chain"],entry["number"])
                key=mapping[pair] if mapping is not None else ResidueKey(*pair)
                excluded.append({"residue":asdict(key),
                    "group":{"N+":"NTERM","C-":"CTERM"}.get(entry["group"],entry["group"]),
                    "reason":"propka_covalent_coupling_suppression", "coupled_reference_label":entry["coupled_label"]})
    return ReferenceRun(parse_pka(text,mapping),provenance,text,run.stdout+run.stderr,elapsed,tuple(excluded))


def reference_charge(sites, ph):
    """HH charge reconstructed from the REPORTED reference pKas (not microstates)."""
    from scipy.special import expit
    ph=np.atleast_1d(ph)
    pka=np.array([s.pka for s in sites])
    q0=np.array([Q_DEPROT[GROUPS.index(s.group)] for s in sites])
    return q0[None,:]+expit(np.log(10)*(pka[None,:]-ph[:,None]))


def compare_reference(model, probabilities, reference, *, ph=None, sequence_indices=None):
    """Compare all expected physical native/hard-sequence sites without silent dropping.

    Returned pKa/curve errors are model-discrepancy metrics, not assertions of
    equivalence. Invalid midpoint solves fail loudly. To compare a mutant,
    export that same candidate sequence and pass sequence_indices explicitly.
    """
    import jax
    ph=np.arange(0.,15.) if ph is None else np.asarray(ph,float)
    cache=model.cache
    sequence=cache.native_index if sequence_indices is None else np.asarray(sequence_indices)
    model.validate_probabilities(probabilities)
    if sequence.shape!=(cache.n_residues,) or not np.issubdtype(sequence.dtype,np.integer):
        raise ValueError("sequence_indices must be a length-N integer vector")
    if np.any((sequence<0)|(sequence>=20)):
        raise ValueError("identity outside alphabet")
    if np.any(sequence[cache.frozen]!=cache.native_index[cache.frozen]):
        raise ValueError("cannot compare a mutation of a frozen disulfide site")
    expected_hot=np.eye(20)[sequence]
    if not np.allclose(probabilities,expected_hot,atol=1e-6):
        raise ValueError("a categorical PROPKA regression requires the same HARD sequence on both sides")
    expected=[]
    for i,key in enumerate(cache.keys):
        where=np.flatnonzero(GROUP_AA==sequence[i])
        if len(where) and cache.group_mask[i,int(where[0])]:
            expected.append((i,GROUPS[int(where[0])]))
        for g in (7,8):
            if cache.group_mask[i,g]:
                expected.append((i,GROUPS[g]))
    if not expected:
        raise ValueError("no active physical titratable sites in this hard sequence")
    lookup={(s.key,s.group):s for s in reference.sites}
    missing=[(str(cache.keys[i]),g) for i,g in expected if (cache.keys[i],g) not in lookup]
    if missing:
        raise ValueError(f"reference is missing required sites: {missing}")
    pk=jax.device_get(model.pka_sites(expected)(probabilities))
    if not np.all(pk.valid):
        failed=[(str(cache.keys[i]),g) for (i,g),valid in zip(expected,pk.valid) if not valid]
        raise ValueError(f"invalid/nonconverged midpoint solves: {failed}")
    ref_sites=[lookup[cache.keys[i],g] for i,g in expected]
    ref_pka=np.array([s.pka for s in ref_sites])
    difference=pk.value-ref_pka
    curves=jax.device_get(model.curves(ph)(probabilities))
    if not np.all(curves.converged):
        raise ValueError("occupancy solver did not converge over reference pH grid")
    ref_q=reference_charge(ref_sites,ph)
    actual_q=np.stack([curves.site_charge[:,i,GROUPS.index(g)] for i,g in expected],axis=-1)
    rows=[{"residue":asdict(cache.keys[i]),"group":g,"surrogate_pka":float(value),
           "reference_pka":float(ref),"delta":float(value-ref)}
          for (i,g),value,ref in zip(expected,pk.value,ref_pka)]
    extras=[{"residue":asdict(s.key),"group":s.group} for s in reference.sites
            if (s.key,s.group) not in {(cache.keys[i],g) for i,g in expected}]
    return {"schema":1,"reference":reference.provenance,"cache_fingerprint":cache.fingerprint(),
            "model_config":asdict(model.config),"n_sites":len(rows),"sites":rows,
            "excluded_reference_sites":extras,
            "metrics":{"pka_mae":float(np.abs(difference).mean()),
                       "pka_rmse":float(np.sqrt(np.mean(difference**2))),
                       "pka_max_abs":float(np.abs(difference).max()),
                       "site_charge_rmse":float(np.sqrt(np.mean((actual_q-ref_q)**2))),
                       "total_charge_rmse":float(np.sqrt(np.mean((actual_q.sum(-1)-ref_q.sum(-1))**2)))},
            "ph":ph.tolist(),"surrogate_total_charge":actual_q.sum(-1).tolist(),
            "reference_total_charge":ref_q.sum(-1).tolist(),
            "reference_seconds_including_process_startup":reference.elapsed_seconds,
            "interpretation":"mean-field surrogate vs PROPKA-reported-pKa HH curves; not microscopic parity"}


def assert_baseline(report, baseline, *, pka_atol=2e-4, charge_atol=2e-5):
    """Require identity/version-matched, previously approved full-structure regression."""
    if report["reference"]!=baseline["reference"]:
        raise AssertionError("reference version/commit/input changed; review a new baseline")
    if report["cache_fingerprint"]!=baseline["cache_fingerprint"]:
        raise AssertionError("structural constants changed; review a new baseline")
    if report["model_config"]!=baseline["model_config"]:
        raise AssertionError("model configuration changed; review a new baseline")
    ids=lambda x:[(r["residue"],r["group"]) for r in x["sites"]]
    if ids(report)!=ids(baseline):
        raise AssertionError("site identities/order changed")
    np.testing.assert_allclose(report["ph"],baseline["ph"],atol=0,rtol=0)
    if report.get("excluded_reference_sites")!=baseline.get("excluded_reference_sites"):
        raise AssertionError("excluded reference site set changed")
    np.testing.assert_allclose([x["reference_pka"] for x in report["sites"]],
                               [x["reference_pka"] for x in baseline["sites"]],atol=1e-8,rtol=0)
    np.testing.assert_allclose(report["reference_total_charge"],baseline["reference_total_charge"],atol=charge_atol,rtol=0)
    np.testing.assert_allclose([x["surrogate_pka"] for x in report["sites"]],
                               [x["surrogate_pka"] for x in baseline["sites"]],atol=pka_atol,rtol=0)
    np.testing.assert_allclose(report["surrogate_total_charge"],baseline["surrogate_total_charge"],atol=charge_atol,rtol=0)
