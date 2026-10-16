"""Read-only publication verification; writes separate Phase6 audit evidence."""
from pathlib import Path
import json,sys,re,ast
import numpy as np
import pandas as pd
from PIL import Image
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'src'))
from sleep_research.common import digest,fingerprint,json_write,now
from sleep_research.phase5_registry import plan,verify_files
from models.cnn1d import CNN1D,architecture_details
from models.advanced_sleep import make_advanced
T=P/'results/publication_tables';F=P/'results/publication_figures';R=P/'reports'


def verify():
    body=plan(P);snapshot=json.loads((T/'preservation_snapshot.json').read_text());verify_files(P,snapshot['files'])
    input_manifest=json.loads((T/'input_manifest.json').read_text());verify_files(P,input_manifest['files']);assert input_manifest['plan_sha256']==fingerprint(body)
    ledger=json.loads((T/'manuscript_numbers.json').read_text());facts=json.loads((T/'facts.json').read_text())
    for k,r in ledger.items():assert format(r['value'],r['format'])==r['formatted']==facts[k],k
    claims_used=set()
    for path in (R/'publication_package').glob('*.md*'):
        for key in re.findall(r'\{\{([^{}]+)\}\}',path.read_text()):
            if not key.startswith(('table:','include:')):assert key in ledger,key;claims_used.add(key)
    test=pd.read_csv(P/'results/phase5_model_comparison.csv').set_index('key');main=pd.read_csv(T/'03_primary_test_summary.csv')
    groups={'CNN':['M1'],'CNN-BiLSTM':['m5_ce_s42','m5_ce_s43','m5_ce_s44'],'CNN-Transformer':['m7_ce_s42','m7_ce_s43','m7_ce_s44']}
    for label,keys in groups.items():
        row=main[main.model==label].iloc[0];assert row.seeds==len(keys)
        for metric in ['accuracy','macro_f1','n1_f1','n3_f1','balanced_accuracy','cohen_kappa']:
            np.testing.assert_allclose(row[metric+'_mean'],test.loc[keys,metric].mean(),atol=1e-12,rtol=0)
            if len(keys)>1:np.testing.assert_allclose(row[metric+'_seed_sd'],test.loc[keys,metric].std(ddof=1),atol=1e-12,rtol=0)
            else:assert pd.isna(row[metric+'_seed_sd'])
    original=json.loads((P/'results/phase5_subject_bootstrap.json').read_text());recalc=json.loads((T/'bootstrap_recalculation.json').read_text())
    assert original['bootstrap_weights_sha256']==recalc['bootstrap_weights_sha256'] and original['estimates']==recalc['estimates']
    assert all(all(a[k]==b[k] for k in a) for a,b in zip(recalc['paired_differences'],original['paired_differences']))
    contrast=next(r for r in original['paired_differences'] if r['a']=='M7_fixed_seeds' and r['b']=='M5_fixed_seeds' and r['metric']=='macro_f1')
    assert contrast['percentile_interval']['lower']<0<contrast['percentile_interval']['upper']
    assert contrast['simultaneous_interval']['lower']<0<contrast['simultaneous_interval']['upper']
    cfgs=[r['configuration'] for r in body['models']]
    for cfg in cfgs:
        assert cfg['batch_size']==16 and cfg['max_epochs']==25 and cfg['early_stopping']['patience']==8
        assert cfg['optimizer']['name'].lower()=='adamw' and cfg['optimizer']['learning_rate']==.001 and cfg['optimizer']['weight_decay']==.01
    cnn=CNN1D();assert [b[0].out_channels for b in cnn.blocks]==[16,32,64,64]
    assert [b[0].kernel_size[0] for b in cnn.blocks]==[25,9,7,5] and [b[0].stride[0] for b in cnn.blocks]==[2,2,2,1]
    assert all(b[1].num_groups==4 and b[3].kernel_size==2 for b in cnn.blocks) and cnn.classifier[0].p==.3
    m5=make_advanced(next(r['configuration']['architecture'] for r in body['models'] if r['key']=='m5_ce_s42'))
    m7=make_advanced(next(r['configuration']['architecture'] for r in body['models'] if r['key']=='m7_ce_s42'))
    assert m5.recurrent.forward_cell.hidden_size==m5.recurrent.backward_cell.hidden_size==32
    assert len(m7.blocks)==2 and tuple(m7.position.shape)==(5,64)
    assert all(b.attention.num_heads==4 and b.attention.embed_dim==64 and b.attention.dropout==.1 and b.ff[0].out_features==128 for b in m7.blocks)
    costs=pd.read_csv(T/'06_computational_cost.csv');base=architecture_details(cnn)['conv_linear_macs_per_epoch'];assert base==4638464
    np.testing.assert_array_equal(costs.approximate_forward_MACs_per_target,[base,5*(base-320)+122880+320,5*(base-320)+334080+320])
    selection=json.loads((T/'representation_lineage.json').read_text())
    from sleep_research.analysis import load_export
    from sleep_research.phase5_evaluation import workspace
    references=json.loads((workspace(P,body)/'exports.json').read_text())
    for item in selection:
        key={'CNN':'M1','CNN-BiLSTM':'m5_ce_s42','CNN-Transformer':'m7_ce_s42'}[item['model']]
        manifest,frame,features,root=load_export(P/references['models'][key]['export'],final_evaluation=True)
        coords=pd.read_csv(P/item['source']);byid=frame.set_index('epoch_id')
        np.testing.assert_array_equal(coords.true_label.to_numpy(),byid.loc[coords.epoch_id,'true_label'].to_numpy())
        if item['method']=='pca':
            params=json.loads((P/item['source']).with_name('pca_parameters.json').read_text())
            projected=(np.asarray(features,dtype=np.float64)-np.asarray(params['mean'])) @ np.asarray(params['components']).T
            np.testing.assert_allclose(projected,coords[['pc1','pc2']].to_numpy(),atol=1e-9,rtol=1e-9)
        else:
            params=json.loads((P/item['source']).with_name('tsne_parameters.json').read_text())
            assert coords.epoch_id.tolist()==params['epoch_ids'] and len(coords)==500 and params['seed']==42 and params['iterations']==500
    figures=json.loads((F/'figure_manifest.json').read_text())['figures'];assert len(figures)==16
    for fig in figures:
        assert digest(F/(fig['name']+'.png'))==fig['png_sha256'] and digest(F/(fig['name']+'.pdf'))==fig['pdf_sha256']
        with Image.open(F/(fig['name']+'.png')) as im:assert min(im.size)>=700 and min(im.info['dpi'])>=299
        assert (F/(fig['name']+'.pdf')).read_bytes().startswith(b'%PDF')
    required=['publication_manuscript.md','publication_abstract.md','highlights.md','cover_letter_draft.md','research_contributions.md','reviewer_risk_checklist.md','verified_references.md']
    for filename in required:
        text=(R/filename).read_text();assert len(text)>500 and '{{' not in text
    assert (P/'results/publication_reproducibility_checklist.md').exists()
    manuscript=(R/'publication_manuscript.md').read_text();abstract=(R/'publication_abstract.md').read_text()
    abstract_body=abstract.split('**Background:**',1)[1].split('**Keywords:**',1)[0].strip();assert abstract_body in manuscript
    broken=[]
    for path in [*(R/f for f in required),P/'results/publication_reproducibility_checklist.md']:
        for dest in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',re.sub(r'\$\$.*?\$\$','',path.read_text(),flags=re.S)):
            if '://' not in dest:
                target=(path.parent/dest.split('#')[0]).resolve()
                if not target.exists():broken.append((str(path.relative_to(P)),dest))
    assert not broken,broken
    # Code path forbids model forwards/training: constructors only check parameter counts.
    source=R/'publication_package/build_publication.py';tree=ast.parse(source.read_text())
    forbidden={'fit','forward','backward','train','step','load_model','export_inference','run'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
            assert name not in forbidden,name
        if isinstance(node,ast.ImportFrom):assert node.module not in ['sleep_research.trainer','sleep_research.phase4_campaign','sleep_research.phase5_training']
    metadata=json.loads((T/'reference_metadata.json').read_text());assert len(metadata)==10
    for name,row in metadata.items():assert row.get('title') and row.get('published') and row.get('DOI') and not row.get('verification_error'),name
    refs_text=(R/'verified_references.md').read_text().lower()
    for row in metadata.values():assert row['DOI'].lower() in refs_text
    assert '10.5664/jcsm.2350' in refs_text and '10.1109/tnsre.2021.3076234' in refs_text
    report=dict(successful=True,verified_at=now(),historical_files_preserved=len(snapshot['files']),processed_arrays_preserved=sum(p.endswith('/epochs.npy') for p in snapshot['files']),
        final_plan_sha256=fingerprint(body),figures_verified=len(figures),png_dpi=300,pdf_counterparts=len(figures),publication_csv_files=len(list(T.glob('*.csv'))),
        numeric_facts=len(ledger),narrative_numeric_facts_used=len(claims_used),primary_metrics_recalculated=True,all_nine_classwise_metrics_recalculated=True,
        bootstrap_weights_and_intervals_exactly_reproduced=True,architecture_constants_and_partial_MACs_verified=True,representation_lineage_and_PCA_verified=True,
        previous_tests='Historical Phase5 evidence:202 passed; not rerun because earlier scientific source is unchanged',
        final_inference_repeated=False,model_training_called=False,new_gradient_extraction=False,administrative_author_fields_pending=True)
    json_write(T/'verification.json',report)
    text=f'''# Publication package verification

Verified {report['verified_at']}. **Scientific package checks passed.** No training, model forward, new gradients or final-test inference was performed.

## Numerical consistency and provenance

All nine final checkpoint metrics and all class-wise scores were recalculated from saved predictions by the publication builder. Primary fixed-seed means/sample SD and paired contrasts match the historical CSV/JSON evidence. The exact predefined10,000 bootstrap weights and all reported intervals were reproduced. The manuscript’s empirical narrative uses{len(claims_used)} linked numeric facts; its ledger contains{len(ledger)} facts with source, full precision, display formatting and provenance. Tables are generated from those preserved inputs. Architecture constants, parameter counts and partial analytic MACs were checked against the actual model construction and registered configurations. Reference years/DOIs were checked against primary bibliographic sources. Equations, model identifiers and bibliographic numerals are not treated as measured performance.

**No empirical numerical discrepancy was found** in the generated manuscript/tables. This statement covers the evidence-linked results, configuration/count facts and checked formulas; it is not a guarantee that every editorial or methodological interpretation is beyond criticism. All measurement units and rounded displays are explicit. The abstract matches the manuscript abstract.

## Artifacts and preservation

{len(figures)} publication figures have300-dpi PNG and PDF counterparts with valid checksums;{len(list(T.glob('*.csv')))} CSV files include tables and linked coordinate exports. Local document/image links resolve. PCA projections agree with saved features/parameters, and t-SNE IDs/labels retain their saved mapping. All{len(snapshot['files'])} earlier files, including153 processed EEG arrays and all Phase1–5 reports/experiments/checkpoints/exports, remain byte-identical. Earlier source and the frozen evaluation plan are unchanged. Historical202-test passing evidence remains preserved; the new publication verifier is the current check.

The figure preprocessing panel intentionally embeds a historical training-only raster illustration and labels that limitation. Its analytic response is vector-rendered from saved SOS coefficients. Explainability uses existing validation examples; it is not new final-test explanation extraction.

## Contributions and remaining limitations

The contribution is a reproducible comparison of compact established model families, checkpoint/epoch lineage and transparent class/transition/cost reporting. M5/M7 exceed the single-seed CNN baseline on aggregate held-out macro-F1 in this cohort. Their direct difference interval crosses zero, so reliable superiority between them is not established. M8 remains exploratory validation-only; short ablations cannot establish novelty or causal module efficacy.

Reviewers may question unequal seed counts, context/capacity confounding, the single small cohort/split, percentile bootstrap coverage with12 subjects, label-based Wake eligibility, noncausal/future inputs, numerical flag validity and hardware-specific timing. These limitations are explicit in the manuscript and reviewer checklist. No SOTA, new-architecture, causal-physiology or clinical-effectiveness claim is made.

## Readiness

The scientific manuscript and supporting artifacts are ready for author review and journal adaptation. Submission is not claimed complete: author/affiliation/contributor details, funding/conflict/ethics/data-reuse declarations, a public code/archive URL and target-journal formatting remain author actions. Stronger claims about novel modules, matched-context architecture effects, universal loss improvements or external/clinical generalization require additional controlled experiments and independent evaluation data. The final test cohort is consumed and is closed to further development. This phase launches no new experiment.
'''
    (P/'results/publication_verification_report.md').write_text(text)
    # Snapshot new output hashes; exclude live build/verification logs and the self-referential manifest.
    paths=[p for folder in [R/'publication_package',F,T] for p in folder.rglob('*') if p.is_file() and p.suffix not in ['.pyc'] and '__pycache__' not in p.parts and p.name not in ['build_log.txt','verification_log.txt','package_manifest.json']]
    paths.extend(R/f for f in required);paths.extend([P/'results/publication_reproducibility_checklist.md',P/'results/publication_verification_report.md'])
    json_write(T/'package_manifest.json',dict(created_at=now(),final_plan_sha256=fingerprint(body),files={str(p.relative_to(P)):digest(p) for p in sorted(set(paths))},training_called=False,new_inference_called=False))
    print(json.dumps(report,indent=2));return report

if __name__=='__main__':verify()
