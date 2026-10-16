"""After explicit approval only: validate hashes and stage a fixed file allowlist.

Never commits, adds a remote, pushes, changes visibility or reads credentials.
"""
from pathlib import Path
import argparse,csv,hashlib,json,subprocess
import yaml
from inventory import ROOT,GENERATED,approved_shape

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reviewed-manifest-sha256',required=True)
 p.add_argument('--approval-record',type=Path,required=True)
 action=p.add_mutually_exclusive_group(required=True)
 action.add_argument('--stage',action='store_true');action.add_argument('--check-only',action='store_true')
 a=p.parse_args();record=json.loads(a.approval_record.read_text())
 required=['authors_and_citation_complete','code_license_approved','subject_confusion_counts_release_approved','frozen_split_identifiers_release_approved']
 if a.stage:required+=['publication_inventory_approved']
 if not all(record.get(k) is True for k in required):raise PermissionError('Explicit inventory, genuine author/citation, code-license and derived-evidence approval are required')
 citation=yaml.safe_load((ROOT/'CITATION.cff').read_text())
 if not citation.get('authors') or 'DRAFT PROPOSAL' in (ROOT/'LICENSE').read_text():raise PermissionError('Complete genuine author/CFF metadata and activate the approved code license before staging')
 manifest=ROOT/'provenance/PUBLIC_FILE_INVENTORY.csv';sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
 if a.reviewed_manifest_sha256!=sha or record.get('inventory_sha256')!=sha:raise PermissionError('Approval belongs to a different exact inventory')
 sealed=json.loads((ROOT/'provenance/file_hashes.json').read_text())
 if sealed['inventory_sha256']!=sha:raise ValueError('Hash manifest differs')
 with manifest.open(newline='') as f:entries=list(csv.DictReader(f))
 if len({r['path'] for r in entries})!=len(entries):raise ValueError('Duplicate inventory paths')
 expected={r['path']:{'bytes':int(r['bytes']),'sha256':r['sha256']} for r in entries if r['path'] not in GENERATED}
 if sealed['files']!=expected:raise ValueError('Payload hashes differ from the reviewed CSV')
 if (ROOT/'results/github_publication_manifest.csv').read_bytes()!=manifest.read_bytes():raise ValueError('Public inventory mirrors differ')
 for row in entries:
  f=ROOT/row['path']
  if not f.resolve().is_relative_to(ROOT.resolve()) or not approved_shape(f) or f.is_symlink():raise ValueError('Unsafe inventory path')
  if not f.is_file() or f.stat().st_size!=int(row['bytes']):raise ValueError('Inventory size differs: '+row['path'])
 for name,v in sealed['files'].items():
  f=ROOT/name
  if f.is_symlink() or not f.is_file() or f.stat().st_size!=v['bytes'] or hashlib.sha256(f.read_bytes()).hexdigest()!=v['sha256']:raise ValueError('Reviewed file changed: '+name)
 listed=(ROOT/'provenance/reviewed_paths.nul').read_bytes().decode().split('\0');listed=[s for s in listed if s]
 if len(listed)!=len(set(listed)):raise ValueError('Duplicate staging paths')
 paths=set(listed)
 if paths!=set(sealed['files'])|GENERATED:raise ValueError('Explicit staging path list differs')
 if a.check_only:
  print(json.dumps(dict(successful=True,inventory_sha256=sha,files=len(paths),bytes=sum(int(r['bytes']) for r in entries),
      final_inventory_approved=record.get('publication_inventory_approved') is True,git_called=False,staged=False,committed=False,pushed=False)))
  return
 existing=subprocess.run(['git','rev-parse','--show-toplevel'],cwd=ROOT,capture_output=True,text=True)
 if existing.returncode==0:
  if Path(existing.stdout.strip()).resolve()!=ROOT:raise ValueError('An ancestor Git repository requires inspection; refusing nested initialization')
  if subprocess.run(['git','rev-parse','--verify','HEAD'],cwd=ROOT,capture_output=True).returncode==0:raise ValueError('Existing Git history needs its own sensitive-file/history audit before staging this proposal')
 elif not (ROOT/'.git').exists():subprocess.run(['git','init','-b','main',str(ROOT)],check=True)
 if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode!=0:raise ValueError('Pre-existing staged changes require review; refusing to mix inventories')
 subprocess.run(['git','add','--pathspec-from-file=provenance/reviewed_paths.nul','--pathspec-file-nul'],cwd=ROOT,check=True)
 print('Approved exact allowlist staged. No commit, remote creation or push performed. Inspect the cached diff before committing.')

if __name__=='__main__':main()
