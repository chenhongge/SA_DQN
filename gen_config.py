import json
import sys

with open('defaults.json', 'r') as f:
  d = json.load(f)
with open(sys.argv[1]) as f:
  old = json.load(f)

new = {}

def search(d, u, new):
  for k in u.keys(): 
    if d.get(k, {}) != u[k]:
      if isinstance(u[k], dict):
        new[k] = {}
        new[k] = search(d.get(k, {}), u[k], new[k]) 
      else:
        new[k] = u[k]
    if new[k] == {}:
      del new[k]
  return new
new = search(d, old, new)
with open(sys.argv[2], 'w') as f:
  json.dump(new, f, indent = 4)

