#!/usr/bin/env python3
"""Exact-rational audit of outward certificate arithmetic, no oracle assumed.
Run: python tests/independent_certificate_audit.py /path/to/certificate_probe
"""
import fractions,json,math,random,struct,subprocess,sys
rng=random.Random(207223)
def bits(x):return struct.unpack('Q',struct.pack('d',x))[0]
def number(x):return struct.unpack('d',struct.pack('Q',x))[0]
inputs=[0,1,2,3,bits(2.0**-1022),bits(2.0**-53),bits(.5)-1]
inputs += [bits(n*2.0**-53) for n in (14,30,32,53,78,94,207,223,467)]
inputs += [rng.randrange(bits(.5)) for _ in range(10000)]
p=subprocess.run([sys.argv[1]],input=''.join(str(x)+'\n' for x in inputs),text=True,capture_output=True,check=True)
outputs=list(map(int,p.stdout.splitlines()));assert len(outputs)==len(inputs)
bad=[]
for b,r in zip(inputs,outputs):
    x=fractions.Fraction(number(b));bound=fractions.Fraction(number(r))
    # b/(1-b) is mathematically >= exp(b)-1 on this interval.
    if bound<x/(1-x):bad.append({'b_bits':b,'bound_bits':r})
result={'cases':len(inputs),'violations':bad,'target':'returned bound >= exact rational b/(1-b) >= exp(b)-1'}
print(json.dumps(result,indent=2));sys.exit(bool(bad))
