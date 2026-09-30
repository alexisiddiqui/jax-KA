"""Deterministic numerical fixtures, NOT molecular accuracy benchmarks."""
import numpy as np
from .cache import StructureCache, ResidueKey
from .parameters import AA_TO_INDEX


def synthetic_cache(n=8, neighbors=4, chains=2, seed=7, separated=False):
    if not 1 <= chains <= n:
        raise ValueError("require 1 <= chains <= n")
    rng = np.random.default_rng(seed)
    ci = np.minimum(np.arange(n)*chains//n,chains-1).astype(np.int32)
    chain_ids = tuple(f"chain_{i}" for i in range(chains))
    keys = tuple(ResidueKey(chain_ids[ci[i]],int(np.sum(ci[:i]==ci[i]))+1) for i in range(n))
    gm = np.ones((n,9),dtype=bool); gm[:,7:]=False
    for c in range(chains):
        pos = np.flatnonzero(ci==c); gm[pos[0],7]=True;gm[pos[-1],8]=True
    edge_set = set()
    for i in range(n):
        for delta in range(1,min(n,neighbors//2+1)):
            j=(i+delta)%n
            if not separated or ci[i]==ci[j]:
                edge_set.add((i,j));edge_set.add((j,i))
    rows=[sorted(j for ii,j in edge_set if ii==i) for i in range(n)]
    ke=max(1,max(map(len,rows)))
    env_idx=np.zeros((n,ke),np.int32);env_mask=np.zeros((n,ke),bool)
    for i,row in enumerate(rows):
        env_idx[i,:len(row)]=row;env_mask[i,:len(row)]=True
    kc=ke+1
    idx=np.zeros((n,kc),np.int32);pm=np.zeros((n,kc,9,9),bool)
    cg=np.zeros((n,kc,9,9),np.float32)
    for i,row in enumerate(rows):
        for k,j in enumerate([i]+row):
            idx[i,k]=j
            mask=gm[i,:,None]&gm[j,None,:]
            if i==j:
                mask &= ~np.eye(9,dtype=bool);mask[:7,:7]=False
            pm[i,k]=mask
            # Exact reciprocity; modest coefficients after division by eps.
            r=5.5+0.1*abs(i-j)+0.03*(np.arange(9)[:,None]+np.arange(9)[None,:])
            cg[i,k]=np.where(mask,244.12/r*np.clip((10-r)/6,0,1),0)
    shape=(n,ke,9,20)
    volume=rng.uniform(.0001,.001,size=shape).astype(np.float32)*env_mask[:,:,None,None]
    mass=rng.uniform(1,8,size=shape).astype(np.float32)*env_mask[:,:,None,None]
    zng=np.zeros((n,9),np.float32)
    native=np.array([AA_TO_INDEX["DEHK"[i%4]] for i in range(n)],dtype=np.int32)
    return StructureCache(keys,chain_ids,native,ci,gm,np.zeros(n,bool),env_idx,env_mask,
                          volume,mass,np.zeros(shape,np.float32),np.zeros((n,9,20),np.float32),zng.copy(),
                          np.full((n,9),150,np.float32),zng.copy(),zng.copy(),
                          idx,pm,cg,np.zeros_like(cg),np.zeros_like(cg),
                          {"fixture":"synthetic numerical graph", "seed":seed,
                           "geometry":"not molecular", "chains_separated":separated}).validate()
