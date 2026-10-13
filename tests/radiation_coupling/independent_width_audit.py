#!/usr/bin/env python3
"""Check actual width acceptance, root containment, and component certificates.
Run: python tests/independent_width_audit.py /path/to/independent_width_probe
"""
import json,subprocess,sys
import mpmath as mp
mp.mp.dps=170
fields=subprocess.run([sys.argv[1]],capture_output=True,text=True,check=True).stdout.split()
assert fields[0]=='accepted_conditional' and fields[1]=='2',fields
xhat,uhat,ehat,blo,bhi,dx,du,de=[mp.mpf(float(s)) for s in fields[2:]]
h=mp.mpf(2)**-40
lo,hi=mp.mpf(1),1+2*h
for _ in range(700):
    t=(lo+hi)/2
    x=t+(t-1)/mp.sqrt(t)
    e=(2+h*x**4)/(1+h)
    if t-1+e-2<0:lo=t
    else:hi=t
    if hi==lo or hi-lo<=mp.eps*hi:break
t=(lo+hi)/2;x=t+(t-1)/mp.sqrt(t);e=(2+h*x**4)/(1+h)
errors=[abs(xhat/x-1),abs(uhat/t-1),abs(ehat/e-1)]
bounds=[dx,du,de]
passed=blo<=x<=bhi and blo<=xhat<=bhi and all(a<=b for a,b in zip(errors,bounds))
result={'passed':bool(passed),'stop':'width','root_in_bracket':bool(blo<=x<=bhi),'test_precision_decimal_digits':170,'actual_relative_errors':dict(zip(('dust','gas','group'),map(float,errors))),'reported_relative_bounds':dict(zip(('dust','gas','group'),map(float,bounds))),'reference_dust':mp.nstr(x,40),'bracket':[float(blo),float(bhi)]}
print(json.dumps(result,indent=2));sys.exit(not passed)
