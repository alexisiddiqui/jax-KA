#!/usr/bin/env python3
"""Explicitly fetch a legacy reference and record its EXACT Git revision.

A first --ref master invocation resolves and records the revision; subsequent
runs reuse the lock. For fully reproducible setup, pass a reviewed full SHA.
This script requires network access. It was not run in the delivery environment.
"""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref",default="master")
    p.add_argument("--directory",default=".reference/propka-3.0")
    p.add_argument("--lock",default=".reference/legacy-lock.json")
    a=p.parse_args();directory=Path(a.directory);lock=Path(a.lock)
    url="https://github.com/jensengroup/propka-3.0.git"
    ref=json.loads(lock.read_text())["commit"] if lock.exists() else a.ref
    directory.parent.mkdir(parents=True,exist_ok=True)
    if not (directory/".git").exists():
        subprocess.run(["git","clone","--no-checkout",url,str(directory)],check=True)
    subprocess.run(["git","-C",str(directory),"fetch","origin",ref],check=True)
    subprocess.run(["git","-C",str(directory),"checkout","--detach","FETCH_HEAD"],check=True)
    commit=subprocess.check_output(["git","-C",str(directory),"rev-parse","HEAD"],text=True).strip()
    provenance={"repository":url,"commit":commit,"parameter_subversion":"Nov30"}
    lock.parent.mkdir(parents=True,exist_ok=True)
    lock.write_text(json.dumps(provenance,indent=2)+"\n")
    print(json.dumps(provenance,indent=2))

if __name__=="__main__":main()
