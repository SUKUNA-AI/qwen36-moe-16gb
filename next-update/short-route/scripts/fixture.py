import json
d=json.load(open('/home/USER/qwen36-api/optimization/config/teacher-fixture.json'))
print(list(d.keys()))
for k,v in d.items():
    if isinstance(v,dict):
        for name,c in list(v.items())[:30]:
            if isinstance(c,dict): print(name,{kk:(len(vv) if isinstance(vv,(list,str)) else vv) for kk,vv in c.items()})
            else: print(name, type(c).__name__, len(c) if hasattr(c,'__len__') else c)
    else: print(k, type(v).__name__, len(v) if hasattr(v,'__len__') else v)
