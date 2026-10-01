"""Real metal-containing targets must export the same protein-only candidates."""
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from foldbench_site_report import audit_reference_input


# All three deposited structures contain zinc (8ey3 is metal-free).
@pytest.mark.parametrize("pdb_id", ["7uba-assembly1", "8j9r-assembly1", "8jws-assembly1"])
def test_real_metal_targets_have_matched_exclusions(tmp_path, pdb_id):
    from biotite.structure.io import pdbx
    from jaxpropka.topology import load_topology
    from jaxpropka.geometry import build_candidates
    from jaxpropka.reference import write_reference_structure

    archive_path = Path(__file__).resolve().parents[1] / "ground_truth_1522.tar"
    if not archive_path.exists():
        pytest.skip("local FoldBench archive is not installed")
    with tarfile.open(archive_path) as archive:
        member = next(m for m in archive.getmembers() if Path(m.name).name == pdb_id + ".cif")
        cif = pdbx.CIFFile.read(io.StringIO(archive.extractfile(member).read().decode()))
    atoms = pdbx.get_structure(cif, model=1, altloc="occupancy", use_author_fields=False, include_bonds=True)
    assert np.any(np.char.upper(atoms.element) == "ZN")
    topology = load_topology(atoms[atoms.chain_id == "A"], gap_policy="free",
                             freeze_disulfides=True, ignore_nonprotein=True)
    assert set(np.char.upper(topology.atoms.element)) <= {"C", "N", "O", "S"}
    candidates = build_candidates(topology, missing_sidechain="template")
    path = tmp_path / "reference.pdb"
    mapping = write_reference_structure(topology, candidates, path)
    audit = audit_reference_input(path, topology, candidates, mapping)
    assert audit["passed"] and audit["metal_atoms"] == 0
    assert audit["reference_atoms"] == audit["native_candidate_atoms"]
    # The guard must catch contamination, not merely document an assumption.
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    index = next(i for i, line in enumerate(lines) if line.startswith("ATOM"))
    lines[index] = lines[index][:76] + "ZN" + lines[index][78:]
    path.write_text("".join(lines))
    with pytest.raises(ValueError, match="element mismatch"):
        audit_reference_input(path, topology, candidates, mapping)
