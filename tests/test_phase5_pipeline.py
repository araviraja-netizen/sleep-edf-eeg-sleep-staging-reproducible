"""Temporary synthetic end-to-end final analysis wiring, never real test EEG."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from models.cnn1d import CNN1D
from sleep_research.synthetic import SyntheticEpochs,synthetic_contract
from sleep_research.experiments import Experiment
from sleep_research.trainer import Trainer
from sleep_research.phase5_registry import KEYS
from sleep_research.phase5_evaluation import chunk_export,merge_chunks,checked_export,workspace
from sleep_research.phase5_analysis import regenerate
from sleep_research.common import json_write,fingerprint,digest
from test_phase5 import metadata
P=Path(__file__).resolve().parents[1]


class FinalAnalysisPipelineTests(unittest.TestCase):
    def test_all_final_reports_from_synthetic_saved_artifacts_without_training_imports_or_new_inference(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='phase5-synthetic-analysis-',dir=P/'results') as tmp:
            root=Path(tmp);(root/'results').mkdir();(root/'reports').mkdir()
            data=metadata(n=15);data.index.to_csv(root/'results/protocol_a_index.csv',index=False)
            cfg=dict(architecture={'name':'CNN1D'},device='cpu',seed=42,batch_size=5,max_epochs=1,early_stopping={'patience':8},progress_batches=0,num_workers=0,synthetic=True)
            a,b=SyntheticEpochs('train',10,3000),SyntheticEpochs('validation',10,3000);model=CNN1D()
            exp=Experiment(root,'cnn',cfg,synthetic_contract(a,b),'synthetic-final-analysis')
            trainer=Trainer(model,torch.optim.AdamW(model.parameters()),None,a,b,exp,cfg);trainer.fit(resume='none')
            checkpoint=Path(trainer.manager.discover()['best']);row=dict(key='synthetic',model_id='M1',checkpoint_sha256=digest(checkpoint))
            refs=chunk_export(model,data,exp,row,5,5,'cpu');export=merge_chunks(exp,row,refs,data.index,'synthetic-only')
            evidence=root/'results/synthetic_validation_explanations';evidence.mkdir();json_write(evidence/'manifest.json',dict(checkpoint_sha256=digest(checkpoint),files={},synthetic=True))
            models=[]
            for key in KEYS:models.append(dict(key=key,model_id=key if key in ['M1','M4'] else key.split('_')[0].upper(),role='synthetic',seed=42 if key in ['M1','M4'] else int(key.rsplit('s',1)[1]),parameters=40501,best_epoch_1_based=1,training_seconds=1.,checkpoint_sha256=digest(checkpoint),validation_explanations=str(evidence.relative_to(root))))
            body=dict(plan_id='synthetic-only',models=models,analysis=dict(tsne_max_samples=8,tsne_perplexity=2,tsne_iterations=300,seed=42),
                primary_groups={'M1':['M1'],'M5':['m5_ce_s42','m5_ce_s43','m5_ce_s44'],'M7':['m7_ce_s42','m7_ce_s43','m7_ce_s44']},
                primary_comparisons=[['M5','M1'],['M7','M1'],['M7','M5']],secondary_comparisons=[['M4','M1'],['m5_wce_s42','m5_ce_s42']],
                statistics=dict(resamples=40,seed=3,confidence=.95,primary_simultaneous_confidence=1-.05/3,procedure='Synthetic subject-paired integration fixture'))
            home=workspace(root,body);home.mkdir(parents=True)
            json_write(home/'exports.json',dict(plan_sha256=fingerprint(body),models={key:dict(export=str(export.relative_to(root)),manifest_sha256=digest(export/'manifest.json'),checkpoint_sha256=digest(checkpoint)) for key in KEYS}))
            pd.DataFrame([dict(key=key,parameters=40501,accuracy=.1,macro_f1=.1,n1_f1=.1,n3_f1=.1) for key in KEYS]).to_csv(root/'results/phase4_models_comparison.csv',index=False)
            (root/'results/phase5_reproducibility_report.md').write_text('# Synthetic fixture only\n')
            def smallfig(fig,path):fig.savefig(path.with_suffix('.png'),dpi=30);plt.close(fig)
            def synthetic_checked(path,canonical,sha):return checked_export(path,canonical,sha,allow_synthetic=True)
            with patch('sleep_research.phase5_analysis.plan',return_value=body),patch('sleep_research.phase5_analysis.authorization',return_value={'synthetic':True}),patch('sleep_research.phase5_analysis.checked_export',side_effect=synthetic_checked),patch('sleep_research.phase5_analysis.savefig',side_effect=smallfig),patch('sleep_research.phase5_analysis.validate_saved_context'),patch('sleep_research.inference.export_inference',side_effect=AssertionError('Analysis cannot infer')):
                result=regenerate(root,True,fingerprint(body))
            self.assertTrue(result['successful']);self.assertFalse(result['training_called'])
            self.assertEqual(len(pd.read_csv(root/'results/phase5_model_comparison.csv')),9)
            self.assertEqual(len(pd.read_csv(root/'results/phase5_classwise_metrics.csv')),45)
            self.assertEqual(len(pd.read_csv(root/'results/phase5_seed_variability.csv')),18)
            for name in ['phase5_final_test_results.md','phase5_statistical_analysis.md','phase5_subject_bootstrap.json']:
                self.assertTrue((root/'results'/name).is_file())
            marker=json.loads((home/'analysis_complete.json').read_text());self.assertEqual(marker['plan_sha256'],fingerprint(body))
            for relative,expected in marker['files'].items():self.assertEqual(digest(root/relative),expected)
            report=(root/'reports/final_sleep_stage_classification_research_report.md').read_text();self.assertIn('Frozen final-test results',report)

if __name__=='__main__':unittest.main()
