#!/usr/bin/env python3
"""Explicitly download a PDB fixture from RCSB and record/verify its SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("pdb_id",nargs="?",default="1UBQ")
    p.add_argument("--directory",default=".reference/structures")
    p.add_argument("--sha256",help="expected digest for immutable regression setup")
    a=p.parse_args();pid=a.pdb_id.upper()
    if len(pid)!=4 or not pid.isalnum():p.error("expected a four-character PDB accession")
    url=f"https://files.rcsb.org/download/{pid}.pdb"
    directory=Path(a.directory);directory.mkdir(parents=True,exist_ok=True)
    destination=directory/(pid+".pdb")
    with urllib.request.urlopen(url,timeout=60) as response:content=response.read()
    digest=hashlib.sha256(content).hexdigest()
    if a.sha256 and digest!=a.sha256:raise ValueError(f"input digest mismatch: {digest}")
    destination.write_bytes(content)
    destination.with_suffix(".provenance.json").write_text(json.dumps({"url":url,"sha256":digest},indent=2)+"\n")
    print(destination,digest)

if __name__=="__main__":main()
