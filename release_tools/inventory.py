"""Freeze an explicit reviewed-file inventory; never stage or publish."""
from pathlib import Path
import csv,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
GENERATED={'provenance/PUBLIC_FILE_INVENTORY.csv','provenance/file_hashes.json','provenance/reviewed_paths.nul','results/github_publication_manifest.csv'}
def approved_shape(path):
 rel=path.relative_to(ROOT);parts=rel.parts
 if any(p in ['.git','__pycache__','.ipynb_checkpoints','verification_outputs'] for p in parts):return False
 if path.name=='.DS_Store' or path.suffix in ['.pyc','.log','.tmp']:return False
 if len(parts)==1:return path.name in ['README.md','LICENSE','CITATION.cff','.gitignore','pyproject.toml','requirements.txt','requirements-reviewer.txt']
 if parts[0] in ['src','scripts','tests','release_tools']:return path.suffix=='.py'
 if parts[0]=='configs':return path.suffix in ['.yaml','.toml','.json']
 if parts[0]=='docs':return path.suffix=='.md'
 if parts[0]=='reports':return path.suffix in ['.md','.py'] or path.name.endswith('.md.in')
 if parts[0]=='notebooks':return path.suffix=='.ipynb'
 if parts[0]=='evidence':return path.suffix in ['.csv','.json']
 if parts[0]=='provenance':return path.suffix in ['.json','.csv','.txt','.nul']
 if parts[0]=='results':
  return (len(parts)>2 and parts[1]=='publication_tables' and path.suffix in ['.csv','.md','.json']) or (len(parts)>2 and parts[1]=='publication_figures' and path.suffix in ['.png','.pdf']) or (len(parts)==2 and path.name in ['github_publication_audit.md','github_publication_manifest.csv','github_reproducibility_validation.md','github_publication_approval_report.md'])
 return False

def build():
 records=[]
 for p in sorted(ROOT.rglob('*')):
  if not p.is_file() or not approved_shape(p) or str(p.relative_to(ROOT)) in GENERATED:continue
  if p.is_symlink():raise ValueError('Symlinks require separate review: '+str(p))
  rel=str(p.relative_to(ROOT));raw=p.read_bytes()
  records.append(dict(path=rel,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
      approval_status='pending_final_inventory_approval',category='subject/recording-derived aggregate or identifier review' if rel.startswith('reports/historical/') or rel in ['evidence/subject_confusions.json','evidence/splits/subject_splits.csv','evidence/splits/recording_splits.csv','evidence/splits/subject_split_freeze.json'] else 'source/documentation/aggregate evidence'))
 path=ROOT/'provenance/PUBLIC_FILE_INVENTORY.csv';path.parent.mkdir(exist_ok=True)
 controls=[dict(path=name,bytes=(ROOT/name).stat().st_size if (ROOT/name).exists() else 0,
     sha256='non-recursive control file; see reviewed inventory digest',approval_status='pending_final_inventory_approval',
     category='proposed staging control; excluded from recursive payload hashes') for name in sorted(GENERATED)]
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records+controls)
 digest=hashlib.sha256(path.read_bytes()).hexdigest()
 (ROOT/'results/github_publication_manifest.csv').write_bytes(path.read_bytes())
 (ROOT/'provenance/reviewed_paths.nul').write_bytes(b''.join((r['path']+'\0').encode() for r in records)+b''.join((r+'\0').encode() for r in sorted(GENERATED)))
 value=dict(inventory_sha256=digest,payload_files=len(records),payload_bytes=sum(r['bytes'] for r in records),
     files={r['path']:{'sha256':r['sha256'],'bytes':r['bytes']} for r in records},
     self_reference_note='The four inventory/hash/path-list control files are listed for staging but excluded from their own recursive payload hashes. Review their exact bytes and inventory digest separately.')
 (ROOT/'provenance/file_hashes.json').write_text(json.dumps(value,indent=2)+'\n')
 return value

if __name__=='__main__':
 r=build();print(json.dumps({k:v for k,v in r.items() if k!='files'},indent=2))
