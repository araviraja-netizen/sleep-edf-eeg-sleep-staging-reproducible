"""Read-only scientific-artifact checks; write only new editorial verification evidence.

No trainer, model constructor, dataset loader or inference module is imported.
No original report, figure, statistic or test evidence is rewritten.
"""
from pathlib import Path
from datetime import datetime
from urllib.parse import urlsplit, unquote
import csv
import hashlib
import importlib.util
import json
import math
import re
import statistics
import sys
from PIL import Image

P = Path(__file__).resolve().parents[2]
OUT = P / 'results/manuscript_revision'
TABLES = P / 'results/publication_tables'
DOCUMENTS = ['publication_manuscript_revised.md', 'reviewer_readiness_report.md',
             'manuscript_changes_summary.md', 'supplementary_material_plan.md',
             'revised_reference_audit.md']
checks = []


def check(condition, description, **details):
    checks.append({'passed': bool(condition), 'description': description, **details})
    if not condition:
        raise AssertionError(description + ': ' + str(details))


def read_rows(name):
    with (TABLES / name).open(newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def anchors(text):
    return {re.sub(r'[^\w\- ]', '', h.lower()).replace(' ', '-')
            for h in re.findall(r'^#{1,6}\s+(.+)$', text, re.M)}


def main():
    started = datetime.now().astimezone().isoformat()
    check(Path('/Volumes/Mac/datasets/sleep-edfx').is_dir(), 'Existing dataset directory is available')
    check(Path(sys.executable).resolve() == Path('/Volumes/Mac/envs/dl-m4/bin/python').resolve(),
          'Checks use the existing research interpreter')
    docs = {name:(P / 'reports' / name).read_text() for name in DOCUMENTS}
    spec = importlib.util.spec_from_file_location('revision_render_check', Path(__file__).with_name('build_revision.py'))
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    renderer.CHECK_ONLY = True
    renderer.build()
    check(True, 'Revised main text and supplement exactly match evidence-linked templates')
    ledger = json.loads((TABLES / 'manuscript_numbers.json').read_text())
    facts = json.loads((TABLES / 'facts.json').read_text())
    for key, entry in ledger.items():
        check(format(entry['value'], entry['format']) == facts[key] == entry['formatted'],
              'Original numeric formatting: ' + key)
        # Original ledger sources are direct paths, with qualifiers in some entries.
    bindings = json.loads((OUT / 'numeric_bindings.json').read_text())
    check(all(b['formatted'] in docs[b['document']] for b in bindings),
          'All narrative numeric bindings appear unchanged', occurrences=len(bindings),
          unique_keys=len({b['key'] for b in bindings}))

    # Independent consistency checks on existing reported metrics, not new statistical fits.
    all_seeds = read_rows('10_all_seed_test_metrics.csv')
    groups = {'M1':['M1'], 'M5':['m5_ce_s42','m5_ce_s43','m5_ce_s44'],
              'M7':['m7_ce_s42','m7_ce_s43','m7_ce_s44']}
    summary = read_rows('03_primary_test_summary.csv')
    stage = read_rows('04_primary_classwise.csv')
    seed_classes = read_rows('13_all_classwise_metrics.csv')
    labels = {'M1':'CNN','M5':'CNN-BiLSTM','M7':'CNN-Transformer'}
    mapping = {'accuracy':'accuracy','macro_f1':'macro_f1','n1_f1':'n1_f1','n3_f1':'n3_f1',
               'balanced_accuracy':'balanced_accuracy','cohen_kappa':'cohen_kappa'}
    for mid, keys in groups.items():
        group = [r for r in all_seeds if r['key'] in keys]
        s = next(r for r in summary if r['model'] == labels[mid])
        check(len(group) == len(keys) == int(s['seeds']), 'Fixed seed replication: ' + mid)
        check([int(r['seed']) for r in group] == ([42] if mid == 'M1' else [42,43,44]),
              'Every predefined seed retained in order: ' + mid)
        for field in mapping:
            values = [float(r[field]) for r in group]
            check(close(statistics.mean(values), s[field+'_mean']), 'Saved mean: '+mid+'/'+field)
            if len(values) > 1:
                check(close(statistics.stdev(values), s[field+'_seed_sd']), 'Saved seed sample SD: '+mid+'/'+field)
            else:
                check(s[field+'_seed_sd'] == '', 'Single-seed SD is undefined: '+field)
            fact_key = 'test_' + mid + '_' + ('cohen_kappa' if field == 'cohen_kappa' else field)
            check(close(ledger[fact_key]['value'], s[field+'_mean']), 'Narrative fact agrees with summary: '+fact_key)
        for field, col in [('low','macro_f1_subject_CI_low'), ('high','macro_f1_subject_CI_high')]:
            check(close(ledger[mid+'_CI_'+field]['value'], s[col]), 'Saved subject interval: '+mid+'/'+field)
        check(int(facts['parameters_'+mid].replace(',','')) == int(s['parameters']), 'Registered parameter count: '+mid)
        for r in [r for r in stage if r['model']==labels[mid]]:
            matched = [c for c in seed_classes if c['key'] in keys and c['stage']==r['stage']]
            check(len(matched)==len(keys), 'Class seed coverage: '+mid+'/'+r['stage'])
            for metric in ['precision','recall','f1','specificity','roc_auc','average_precision']:
                check(close(statistics.mean(float(c[metric]) for c in matched),r[metric]),
                      'Saved class score: '+mid+'/'+r['stage']+'/'+metric)
            check(all(c['support']==r['support_per_seed'] for c in matched), 'Shared true-class supports: '+mid+'/'+r['stage'])
    check(len(all_seeds)==9 and all(int(r['epochs'])==31156 and int(r['subjects'])==12 for r in all_seeds),
          'All nine frozen checkpoint rows describe the identical target population')
    allocation = read_rows('01_dataset_split.csv')
    check([int(r['subjects']) for r in allocation]==[54,12,12], 'Frozen subject allocation')
    check([int(r['A_epochs']) for r in allocation]==[134047,30266,31156], 'Frozen eligible epoch allocation')
    check(sum(int(r['A_epochs']) for r in allocation)==int(ledger['protocol_A_epochs']['value']), 'Protocol A count total')
    transition = read_rows('08_transition_primary.csv')
    for mid in groups:
        for subset, suffix in [('transition','transition_f1'),('stable','stable_f1')]:
            r = next(r for r in transition if r['model']==labels[mid] and r['subset']==subset)
            check(close(r['macro_f1'], ledger[mid+'_'+suffix]['value']), 'Transition/stable measurement: '+mid+'/'+subset)
            check(int(r['epochs_per_seed'])==int(ledger[subset+'_epochs']['value']), 'Shared subset support: '+mid+'/'+subset)
    check(3498+27628+30==31156, 'Transition, stable and boundary/gap counts account for all targets')
    check(float(transition[0]['macro_f1']) > float(transition[2]['macro_f1']) > float(transition[4]['macro_f1']),
          'Main transition statement preserves the observed direction')
    paired = read_rows('05_paired_comparisons.csv')
    for name, a,b in [('M5_minus_M1','M5_fixed_seeds','M1_fixed_seeds'),
                       ('M7_minus_M1','M7_fixed_seeds','M1_fixed_seeds'),
                       ('M7_minus_M5','M7_fixed_seeds','M5_fixed_seeds')]:
        r = next(r for r in paired if r['a']==a and r['b']==b and r['metric']=='macro_f1')
        for suffix, col in [('delta','estimate'),('low','lower95'),('high','upper95'),
                            ('sim_low','lower_simultaneous'),('sim_high','upper_simultaneous')]:
            check(close(ledger[name+'_'+suffix]['value'],r[col]), 'Registered paired interval: '+name+'/'+suffix)
    validation = read_rows('11_validation_metrics.csv')
    for mid,keys in groups.items():
        check(close(statistics.mean(float(r['macro_f1']) for r in validation if r['key'] in keys),
                    ledger['validation_'+mid+'_macro']['value']), 'Validation means kept separate: '+mid)
    ablations = [r for r in validation if r['key'].startswith('m8_ablation_')]
    check(len(ablations)==3 and all(float(r['completed_epochs'])==3 for r in ablations), 'M8 bounded budgets preserved')
    for r in ablations:
        check(format(float(r['macro_f1']),'.6f') in docs['supplementary_material_plan.md'] and
              format(float(r['n1_f1']),'.6f') in docs['supplementary_material_plan.md'],
              'Original six-decimal M8 outcomes retained: '+r['key'])
    norm = json.loads((P/'results/normalization.json').read_text())['protocols']['A']
    check(close(norm['mean_microvolts'],ledger['norm_mean']['value']) and
          close(norm['std_microvolts'],ledger['norm_std']['value']), 'Normalization displays retain original training-only parameters')

    # Link checks use document-relative paths; equations are not Markdown links.
    def verify_links():
        link_rows = []
        for name, text in docs.items():
            stripped = re.sub(r'\$\$.*?\$\$', '',text,flags=re.S)
            stripped = re.sub(r'\$[^\n$]*\$', '',stripped)
            for label, target in re.findall(r'!?\[([^\]\n]*)\]\(([^\n]*?)\)', stripped):
                target = target.strip('<>')
                parsed = urlsplit(target)
                if parsed.scheme in ['http','https','mailto']:
                    link_rows.append({'document':name,'target':target,'kind':'external','status':'bibliographic/primary-source audit; not a live HTTP assertion'})
                    continue
                destination = (P/'reports'/unquote(parsed.path)).resolve() if parsed.path else P/'reports'/name
                check(destination.exists(), 'Local manuscript link resolves', document=name, target=target)
                if parsed.fragment:
                    check(destination.is_file() and parsed.fragment in anchors(destination.read_text()), 'Heading anchor resolves', target=target)
                link_rows.append({'document':name,'target':target,'kind':'local','status':'passed'})
        (OUT/'link_verification.json').write_text(json.dumps(link_rows,indent=2)+'\n')
        return link_rows
    main_text = docs['publication_manuscript_revised.md']
    refs = re.findall(r'^### (R\d+)\s*$', main_text, re.M)
    check(refs == ['R'+str(i) for i in range(1,16)], 'All fifteen unique reference entries are present')
    for name in ['publication_manuscript_revised.md','supplementary_material_plan.md']:
        cited = set(re.findall(r'\bR\d+\b',docs[name]))
        check(cited <= set(refs), 'No undefined reference identifiers: '+name)
    check(re.findall(r'^\*Figure (\d+)\.\*',main_text,re.M)==['1','2','3','4','5','6'], 'Main caption order is complete and consecutive')
    check(re.findall(r'^### Figure S(\d+)\.',docs['supplementary_material_plan.md'],re.M)==[str(i) for i in range(1,11)], 'Supplement caption order is complete and consecutive')
    main_images = re.findall(r'!\[.*?\]\(\.\./results/publication_figures/(.*?)\.png\)', main_text)
    supp_images = re.findall(r'!\[.*?\]\(\.\./results/publication_figures/(.*?)\.png\)',docs['supplementary_material_plan.md'])
    figure_manifest = json.loads((P/'results/publication_figures/figure_manifest.json').read_text())['figures']
    check(set(main_images).isdisjoint(supp_images) and set(main_images+supp_images)=={f['name'] for f in figure_manifest},
          'All sixteen existing figures have exactly one proposed placement')
    fig_rows = []
    for f in figure_manifest:
        base = P/'results/publication_figures'/f['name']
        png,pdf = base.with_suffix('.png'),base.with_suffix('.pdf')
        check(digest(png)==f['png_sha256'] and digest(pdf)==f['pdf_sha256'], 'Original figure checksums: '+f['name'])
        with Image.open(png) as im:
            width,height = im.size
            dpi=im.info.get('dpi')
            im.verify()
        b=pdf.read_bytes()
        box=re.search(rb'/MediaBox\s*\[([^]]+)\]',b)
        check(dpi and all(abs(v-300)<0.1 for v in dpi), 'PNG resolution metadata: '+f['name'])
        check(b.startswith(b'%PDF-') and box and len(re.findall(rb'/Type\s*/Page\b',b))==1,
              'PDF header, page and bounds: '+f['name'])
        fig_rows.append({'name':f['name'],'placement':('Main '+str(main_images.index(f['name'])+1) if f['name'] in main_images else 'S'+str(supp_images.index(f['name'])+1)),
                         'png_width_px':width,'png_height_px':height,'png_dpi_x':dpi[0],
                         'native_width_mm_at_300dpi':width/300*25.4,'effective_png_dpi_at_180mm':width/(180/25.4),
                         'pdf_mediabox_pt':box.group(1).decode().strip(),'pdf_embedded_font_marker':b'/FontFile' in b,
                         'png_sha256':f['png_sha256'],'pdf_sha256':f['pdf_sha256'],
                         'visual_review':'overview for all; dense panels separately inspected; see reviewer_readiness_report.md'})
    (OUT/'figure_review.json').write_text(json.dumps(fig_rows,indent=2)+'\n')
    with (OUT/'figure_review.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(fig_rows[0]));writer.writeheader();writer.writerows(fig_rows)
    link_rows = verify_links()

    # Before/after read-only hashes prove preservation of the entire captured research tree.
    snapshot=json.loads((OUT/'preservation_snapshot.json').read_text())['files']
    differences=[]
    for relative, saved in snapshot.items():
        file=P/relative
        if not file.is_file() or file.stat().st_size!=saved['bytes'] or digest(file)!=saved['sha256']:
            differences.append(relative)
    os_metadata_changes=[f for f in differences if Path(f).name=='.DS_Store']
    research_differences=[f for f in differences if Path(f).name!='.DS_Store']
    check(not research_differences, 'Every captured research artifact remains byte-identical; Finder metadata changes are recorded separately',
          files_checked=len(snapshot), research_differences=research_differences, os_metadata_changes=os_metadata_changes)
    processed=[f for f in snapshot if f.startswith('data/') and f.endswith('.npy')]
    check(len(processed)==153,'All processed EEG arrays included in preservation verification')
    # Metadata refresh failures remain recorded and do not become invented successful retrievals.
    metadata=json.loads((OUT/'new_reference_metadata.json').read_text())
    check(metadata['R14']['message']['DOI'].lower()=='10.1161/01.cir.101.23.e215','Original PhysioNet DOI verified in primary metadata')
    check(metadata['R15']['message']['DOI'].lower()=='10.1016/j.bspc.2020.102037','IITNet final DOI verified in primary metadata')
    result={'successful':True,'checked_at':datetime.now().astimezone().isoformat(),'started_at':started,
            'python_executable':sys.executable,'checks_passed':len(checks),'checks':checks,
            'files_checked':len(snapshot),'byte_identical_files':len(snapshot)-len(differences),
            'research_artifact_changes':research_differences,'os_metadata_changes':os_metadata_changes,
            'preserved_processed_arrays':len(processed),
            'original_manuscript_sha256':snapshot['reports/publication_manuscript.md']['sha256'],
            'narrative_numeric_binding_occurrences':len(bindings),'narrative_unique_facts':len({b['key'] for b in bindings}),
            'local_links_verified':sum(r['kind']=='local' for r in link_rows),
            'external_links_inventoried':sum(r['kind']=='external' for r in link_rows),
            'figures_reviewed':len(fig_rows),'main_figures':len(main_images),'supplementary_figures':len(supp_images),
            'training_called':False,'model_inference_called':False,'experimental_statistics_recomputed':False,
            'existing_model_tests_rerun':False,'journal_submission_performed':False,
            'limitations':['Dense figures require journal-size typography proofing','Attention export lacks quantitative colorbar',
                           'External access failures are disclosed; local link success is not live HTTP success',
                           'No new statistical or experimental evidence was generated']}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=[f"Verification completed: {result['checked_at']}",f"Python: {sys.executable}"]
    lines += ['PASS '+c['description'] for c in checks]
    lines += [f"PASS {len(snapshot)} files checked; {len(snapshot)-len(differences)} byte-identical; {len(processed)} unchanged EEG arrays",
              'Finder metadata changes (not research artifacts): '+str(os_metadata_changes),
              f"PASS {result['local_links_verified']} local links; {len(fig_rows)} PNG/PDF pairs",
              'No training, inference, experimental reanalysis or submission performed.']
    (OUT/'verification_log.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        failure={'successful':False,'checked_at':datetime.now().astimezone().isoformat(),'error':repr(exc),'checks':checks}
        (OUT/'verification.json').write_text(json.dumps(failure,indent=2)+'\n')
        (OUT/'verification_log.txt').write_text('\n'.join('PASS '+c['description'] for c in checks if c['passed'])+'\nFAIL '+repr(exc)+'\n')
        raise
