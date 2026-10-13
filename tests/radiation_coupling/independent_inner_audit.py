#!/usr/bin/env python3
"""Independent 170-digit inner-chart audit, mpmath used only by this test.
Reference solves in gas-temperature space, with arbitrary-precision bisection.
Run: python tests/independent_inner_audit.py /path/to/independent_inner_probe
"""
import collections,json,math,pathlib,random,subprocess,sys
import mpmath as mp
mp.mp.dps=170
rng=random.Random(53183)
cases=[]
for newton in (0,1):
    for Aexp in (-200,-40,0,40,200):
        for Dexp in (-200,-40,0,40,200):
            for ratio in (2.0**-160,0.125,0.5,math.nextafter(1.0,0.0),1.0,math.nextafter(1.0,2.0),2.0,2.0**160):
                cases.append((f'grid_{len(cases)}',2.0**Aexp,2.0**Dexp,1.0,ratio,newton))
for i in range(320):
    T=2.0**rng.randint(-200,200)
    x=T*2.0**rng.randint(-200,200)
    cases.append((f'random_{i}',2.0**rng.randint(-200,200),2.0**rng.randint(-200,200),T,x,i%2))
wire=''.join(' '.join(map(str,c[1:]))+'\n' for c in cases)
r=subprocess.run([sys.argv[1]],input=wire,text=True,capture_output=True,check=True)
lines=r.stdout.splitlines()
assert len(lines)==len(cases)
lam=-mp.log1p(-mp.mpf(2)**-53)
counts=collections.Counter();bad=[];worst_t=mp.mpf(0);worst_q=mp.mpf(0)
for case,line in zip(cases,lines):
    name,A,D,T,x,newton=case
    status,th,qh,balance,iters=line.split();counts[status]+=1
    if status!='accepted_conditional':continue
    A,D,T,x=map(mp.mpf,(A,D,T,x));th,qh=map(lambda s:mp.mpf(float(s)),(th,qh))
    if x==T:
        if th!=T or qh!=0:bad.append((name,'equilibrium'))
        continue
    lo,hi=min(T,x),max(T,x)
    for _ in range(1800):
        t=(lo+hi)/2
        residual=D*mp.sqrt(t)*(t-x)+A*(t-T)
        if residual<0:lo=t
        else:hi=t
        if hi-lo<mp.mpf('1e-125')*min(t,abs(t-T)):break
    t=(lo+hi)/2;q=A*abs(t-T)
    et=abs(mp.log(th/t))/lam;eq=abs(mp.log(qh/q))/lam
    worst_t=max(worst_t,et);worst_q=max(worst_q,eq)
    if et>52 or eq>52:bad.append((name,float(et),float(eq),case[1:],line))
summary={'cases':len(cases),'status_counts':dict(counts),'maximum_temperature_log_error_in_lambda':float(worst_t),'maximum_transfer_log_error_in_lambda':float(worst_q),'violations':bad}
print(json.dumps(summary,indent=2))
sys.exit(bool(bad))
