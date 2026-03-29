import chromadb
from collections import Counter
client = chromadb.PersistentClient(path='chroma_db')
col = client.get_collection('chunks_overlap')
results = col.get(
    where={'course_key': {'$eq': 'deep_learning'}},
    include=['metadatas']
)
units = Counter(m['unit_key'] for m in results['metadatas'])
methods = Counter(m.get('resolution_method','unknown') for m in results['metadatas'])
print('Deep Learning unit distribution:')
for unit, count in sorted(units.items()):
    print(f'  {unit}: {count}')
print()
print('Resolution methods:')
for method, count in sorted(methods.items()):
    print(f'  {method}: {count}')
