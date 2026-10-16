"""Render editorial documents from preserved facts; never import research/training code."""
from pathlib import Path
import csv
import json
import re

PROJECT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = PROJECT / 'results/manuscript_revision'
FACTS = json.loads((PROJECT / 'results/publication_tables/facts.json').read_text())
LEDGER = json.loads((PROJECT / 'results/publication_tables/manuscript_numbers.json').read_text())
bindings = []
CHECK_ONLY = False


def rows(name):
    with (PROJECT / 'results/publication_tables' / name).open(newline='') as f:
        return list(csv.DictReader(f))


def table(headers, values):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in values])


def render(name, text):
    def replace(m):
        key = m.group(1)
        entry = LEDGER[key]
        assert FACTS[key] == format(entry['value'], entry['format']) == entry['formatted']
        bindings.append({'document': name, 'key': key, **entry})
        return FACTS[key]
    value = re.sub(r'\{\{(\w+)\}\}', replace, text)
    assert '[[' not in value and '{{' not in value
    destination = PROJECT / 'reports' / name
    if CHECK_ONLY:
        assert destination.read_text() == value, f'Rendered content drift: {name}'
    else:
        destination.write_text(value)


def build():
    original = (PROJECT / 'reports/publication_manuscript.md').read_text()
    architecture = original.split('### 6.1. Shared epoch encoder and CNN baseline\n', 1)[1].split('![Figure 3.', 1)[0]
    architecture = '#### Shared epoch encoder and CNN baseline\n\n' + architecture
    architecture = re.sub(r'### 6\.\d\. ', '#### ', architecture)
    # The full, already verified architecture text is retained rather than rederived.
    reference_text = (PROJECT / 'reports/verified_references.md').read_text()
    reference_text = '### R1\n' + reference_text.split('## R1\n', 1)[1].split('## Verification notes', 1)[0]
    reference_text = re.sub(r'^## R', '### R', reference_text, flags=re.M)
    reference_text = reference_text.replace('&amp;', '&')
    reference_text += '''### R14

Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals. *Circulation*. 2000;101(23):e215–e220. [DOI: 10.1161/01.CIR.101.23.e215](https://doi.org/10.1161/01.CIR.101.23.e215). [Original PhysioNet publication record](https://archive.physionet.org/publications/).

### R15

Seo H, Back S, Lee S, Park D, Kim T, Lee K. Intra- and inter-epoch temporal context network (IITNet) using sub-epoch features for automatic sleep scoring on raw single-channel EEG. *Biomedical Signal Processing and Control*. 2020;61:102037. [DOI: 10.1016/j.bspc.2020.102037](https://doi.org/10.1016/j.bspc.2020.102037). [Author manuscript](https://arxiv.org/abs/1902.06562).
'''
    dataset = table(['Split', 'Subjects', 'Recordings', 'Protocol A epochs', 'Protocol B epochs'],
                    [[r['split'].capitalize()] + [f"{int(r[k]):,}" for k in ['subjects', 'recordings', 'A_epochs', 'B_epochs']]
                     for r in rows('01_dataset_split.csv')])
    primary = []
    for mid, label in [('M1', 'M1: CNN'), ('M5', 'M5: CNN–BiLSTM'), ('M7', 'M7: CNN–Transformer')]:
        f = FACTS
        macro = f[f'test_{mid}_macro_f1'] + (f" ± {f[f'test_{mid}_macro_f1_sd']}" if mid != 'M1' else ' (one seed)')
        primary.append([label, '1' if mid == 'M1' else '3', f[f'test_{mid}_accuracy_percent'], macro,
                        f[f'test_{mid}_n1_f1'], f"[{f[mid+'_CI_low']}, {f[mid+'_CI_high']}]"])
    primary = table(['Configuration', 'Seeds (n)', 'Accuracy (%)', 'Macro-F1 (seed SD)', 'N1 F1', 'Macro-F1 subject 95% CI'], primary)
    template = (HERE / 'revised_template.md').read_text()
    for key, value in [('ARCHITECTURE_METHODS', architecture), ('REFERENCES', reference_text),
                       ('DATASET_TABLE', dataset), ('PRIMARY_TABLE', primary)]:
        template = template.replace('[['+key+']]', value)
    render('publication_manuscript_revised.md', template)

    # Supplement retains all original less-central narrative sections, without rewriting numbers.
    segments = []
    for num, title in [(12, 'Secondary weighted-loss comparisons'),
                       (14, 'Computational complexity and parameter efficiency'),
                       (15, 'Convergence and exploratory ablations'),
                       (16, 'Representations and explainability')]:
        start = f'## {num}. {title}\n'
        text = original.split(start, 1)[1].split(f'## {num+1}. ', 1)[0]
        # Figures/captions are assigned afresh below; underlying saved assets remain unchanged.
        text = re.sub(r'!\[Figure[^\n]+\n\n\*Figure[^\n]+\n', '', text)
        text = text.replace('Temporal smoothing is a plausible interpretation, not an experimentally isolated cause or a newly optimized mechanism.',
                            'No causal mechanism was experimentally isolated.')
        segments.append((title, text.strip()))
    bounded = [r for r in rows('11_validation_metrics.csv') if r['key'].startswith('m8_ablation_')]
    values = []
    control = float(bounded[0]['macro_f1'])
    for r in bounded:
        values.append([r['ablation'], f"{int(r['parameters']):,}", '3', f"{float(r['macro_f1']):.6f}",
                       f"{float(r['n1_f1']):.6f}", f"{float(r['macro_f1'])-control:+.6f}"])
    ablation = table(['M8 variant', 'Parameters', 'Completed epochs', 'Validation macro-F1', 'Validation N1 F1', 'Macro-F1 difference from bounded control'], values)
    supplement = (HERE / 'supplement_template.md').read_text()
    supplement = supplement.replace('[[ABLATION_TABLE]]', ablation)
    for (title, text), key in zip(segments, ['SECONDARY', 'COST', 'M8', 'REPRESENTATIONS']):
        supplement = supplement.replace('[['+key+']]', text)
    manifest = json.loads((PROJECT / 'results/publication_figures/figure_manifest.json').read_text())['figures']
    by_name = {f['name']:f for f in manifest}
    supplementary = ['02_stage_distribution', '03_preprocessing', '08_training_convergence',
                     '09_parameter_cost', '10_N1_errors', '15_subjectwise', '11_hypnograms',
                     '12_embeddings', '13_explanations', '14_attention']
    figures = []
    for index, name in enumerate(supplementary, 1):
        item = by_name[name]
        caption = item['caption']
        if name == '14_attention':
            caption += ' The existing image has no quantitative colorbar; interpret color patterns qualitatively, not as calibrated cross-head magnitudes.'
        if name == '13_explanations':
            caption += ' Context heatmap slots 0–4 correspond to relative original-epoch offsets −2, −1, 0, +1, +2; slot 2 is central.'
        if name == '11_hypnograms':
            caption += ' The displayed recording is SC4081E.'
        figures.append(f"### Figure S{index}. {name}\n\n![Figure S{index}. {name}](../results/publication_figures/{name}.png)\n\n{caption}\n\n[PDF export](../results/publication_figures/{name}.pdf). Source: {item['sources']}. Original manuscript figure: {item['manuscript_figure_number']}.\n")
    supplement = supplement.replace('[[SUPPLEMENT_FIGURES]]', '\n'.join(figures))
    render('supplementary_material_plan.md', supplement)
    if CHECK_ONLY:
        assert json.loads((OUT / 'numeric_bindings.json').read_text()) == bindings
        return
    (OUT / 'numeric_bindings.json').write_text(json.dumps(bindings, indent=2)+'\n')
    (OUT / 'editorial_lineage.json').write_text(json.dumps({
        'architecture_verbatim_source': 'reports/publication_manuscript.md, Sections 6.1–6.4; src/models/cnn1d.py; src/models/advanced_sleep.py',
        'dataset_table': 'results/publication_tables/01_dataset_split.csv',
        'primary_table': 'results/publication_tables/03_primary_test_summary.csv; facts.json',
        'ablation_table': 'results/publication_tables/11_validation_metrics.csv; results/phase4_ablation_results.md',
        'supplement_verbatim_sections': [12,14,15,16],
        'additional_method_constants': ['configs/preprocessing.yaml', 'configs/wake_trimming.yaml',
                                        'results/phase2_preprocessing_report.md', 'results/phase4_architecture_protocol.md',
                                        'results/phase5_evaluation_plan.json'],
        'no_empirical_recomputation': True,
    }, indent=2)+'\n')
    print('Rendered separate revised manuscript and proposed supplement from preserved evidence.')


if __name__ == '__main__':
    build()
