import numpy as np, json, sys
from src.qht import Chain, MLP
from src.hybrid import DressedCircuit
from src.run_study import CFG, targets_from_h, fit_scale_and_score, learning_error

T=CFG["T"]; n=CFG["n"]
models = {
 "WEIGHTED": Chain(n=n, caps=np.array(CFG["weighted_caps"]), C_min=CFG["weighted_C_min"],
                   a0=CFG["a0"], b0=CFG["b0"], gamma=CFG["gamma"], eta=CFG["eta"]),
 "KOFN": Chain(n=n, k_min=CFG["kofn_k"], a0=CFG["a0"], b0=CFG["b0"],
               gamma=CFG["gamma"], eta=CFG["eta"]),
}
out={}
for nm,ch in models.items():
    h=ch.exact_h(T); p=float(h[T,ch.x0]); vn=p*(1-p); tg=targets_from_h(ch,h,T)
    X={s:np.concatenate([ch.bits.astype(float),np.full((ch.S,1),s/T)],1) for s in range(1,T+1)}
    print(f"\n=== {nm}   p_T={p:.4e} ===", flush=True)
    print(f"{'model':<24}{'params':>7}{'VRF':>13}{'KL':>10}", flush=True)
    rows=[]
    for hid in (4,6,8,12,16,24):
        for sd in (0,1):
            m=MLP(n,hid,seed=sd); m.fit(ch.bits.astype(float),tg,T,steps=CFG["mlp_steps"],lr=CFG["mlp_lr"])
            r=fit_scale_and_score(ch,lambda s,m=m: m.probs(ch.bits.astype(float),max(s,1),T),T,"mlp")
            ke=learning_error(ch,lambda s,m=m: m.probs(ch.bits.astype(float),max(s,1),T),h,T)
            rows.append(["MLP",hid,sd,m.n_params(),vn/r["var"],ke])
        b=max([r for r in rows if r[1]==hid],key=lambda r:r[4])
        print(f"{'MLP hidden='+str(hid):<24}{b[3]:>7}{b[4]:>13,.2f}{b[5]:>10.4f}", flush=True)
    hrows=[]
    for nq,L in ((3,2),(4,2),(4,3),(6,2)):
        for sd in (0,1):
            hy=DressedCircuit(n+1,n_q=nq,L=L,scale=CFG["hyb_scale"],seed=sd)
            hy.fit(X,tg,steps=CFG["hyb_steps"],lr=CFG["hyb_lr"])
            r=fit_scale_and_score(ch,lambda s,hy=hy: hy.probs(X[max(s,1)]),T,"hyb")
            ke=learning_error(ch,lambda s,hy=hy: hy.probs(X[max(s,1)]),h,T)
            hrows.append(["HYB",f"nq{nq}L{L}",sd,hy.n_params(),vn/r["var"],ke])
        b=max([r for r in hrows if r[1]==f"nq{nq}L{L}"],key=lambda r:r[4])
        print(f"{'hybrid nq'+str(nq)+'L'+str(L):<24}{b[3]:>7}{b[4]:>13,.2f}{b[5]:>10.4f}", flush=True)
    bm=max(rows,key=lambda r:r[4]); bh=max(hrows,key=lambda r:r[4])
    print(f"  BEST MLP    hidden={bm[1]} seed={bm[2]} params={bm[3]} VRF={bm[4]:,.2f} KL={bm[5]:.4f}", flush=True)
    print(f"  BEST HYBRID {bh[1]} seed={bh[2]} params={bh[3]} VRF={bh[4]:,.2f} KL={bh[5]:.4f}", flush=True)
    out[nm]=dict(mlp=rows,hyb=hrows,p_T=p,var_naive=vn)
json.dump(out,open("results_sweep.json","w"),indent=2,default=float)
print("\nDONE", flush=True)