import json,socket
results=[]
for port in [1919,1920]:
    try:
        with socket.create_connection(('192.0.2.10',port),timeout=3):reachable=True
    except OSError:reachable=False
    results.append({'from':'edge 192.0.2.10','to':'Windows 192.0.2.10','port':port,'reachable':reachable})
assert not any(r['reachable'] for r in results),results
print(json.dumps({'backend_and_distributed_port_not_reachable_over_LAN':True,'results':results,'public_external_scan':'NOT_RUN'}))
