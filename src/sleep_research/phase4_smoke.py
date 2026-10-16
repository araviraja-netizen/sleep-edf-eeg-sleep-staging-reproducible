"""Short real-data MPS benchmarks and interrupted tiny-subset checks, never full runs."""
import argparse
import copy
import json
from pathlib import Path
import resource
import time
import uuid
import statistics

import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from models.advanced_sleep import make_advanced
from .phase4_registry import PROJECT,environment,settings,configuration,source_identity,read_registry,COUNTS
from .phase4_campaign import real_datasets,build_trainer
from .phase4_data import ContextDataset
from .objectives import ResearchLoss,build_optimizer
from .experiments import frozen_dataset_contract
from .common import json_write,now
from .trainer import SimulatedInterruption
from .checkpoints import capture_rng


def sync():torch.mps.synchronize()


def benchmark(project=PROJECT):
    project=Path(project);environment()
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable: run this short benchmark in local Mac Terminal before registering/full campaign')
    cfg=configuration(project,'M5','cross_entropy');train,val=real_datasets(project,cfg);rng=np.random.default_rng(42)
    # Fixed real central targets, uniform within frozen train/validation; not a performance experiment.
    ti=np.sort(rng.choice(len(train),128,False)).tolist();vi=np.sort(rng.choice(len(val),32,False)).tolist()
    records={}
    cold_start=time.perf_counter()
    for indices,dataset in [(ti,train),(vi,val)]:
        for index in indices:dataset[index]
    cold_load_seconds=time.perf_counter()-cold_start
    try:
        for model,ablation in [('M5',None),('M6',None),('M7',None),('M8',None),('M8','no_multiscale'),('M8','no_context')]:
            key=model if ablation is None else model+'_'+ablation;adaptations=[]
            for batchsize in [16,8]:
                cfg=configuration(project,model,'cross_entropy',ablation=ablation,batch_size=batchsize)
                net=make_advanced(cfg['architecture']).to('mps');opt=build_optimizer(net.parameters(),cfg['optimizer']);loss=ResearchLoss(dict(name='weighted_cross_entropy'),COUNTS).to('mps')
                loader=DataLoader(Subset(train,ti),batch_size=batchsize,shuffle=False,num_workers=0);iterator=iter(loader);elapsed=[];loads=[]
                try:
                    for step in range(6):
                        start=time.perf_counter();batch=next(iterator);loads.append(time.perf_counter()-start)
                        x=batch['eeg'].to('mps');y=batch['label'].to('mps');sync();start=time.perf_counter()
                        opt.zero_grad(set_to_none=True);logits=net(x);objective=loss(logits,y)
                        reference=torch.nn.functional.cross_entropy(logits,y,weight=loss.alpha)
                        torch.testing.assert_close(objective,reference,atol=1e-6,rtol=1e-6)
                        objective.backward()
                        if logits.shape!=(len(y),5) or not bool(torch.isfinite(objective)) or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in net.parameters()):raise ValueError('Nonfinite or disconnected model gradients')
                        opt.step();sync();elapsed.append(time.perf_counter()-start)
                    net.eval();vb=next(iter(DataLoader(Subset(val,vi),batch_size=batchsize)));vx=vb['eeg'].to('mps');times=[]
                    with torch.no_grad():
                        for _ in range(4):sync();start=time.perf_counter();out=net(vx);sync();times.append(time.perf_counter()-start)
                    assert out.shape==(batchsize,5) and bool(torch.isfinite(out).all())
                    # Exact frozen normalization for one real central EEG, and valid context packet.
                    central=np.asarray(train.base[ti[0]]['eeg']).reshape(3000)
                    np.testing.assert_array_equal(train[ti[0]]['eeg'][2,:3000].numpy(),central)
                    step_s=statistics.median(elapsed[1:]);load_s=statistics.median(loads[1:]);val_s=statistics.median(times[1:])
                    seconds=np.ceil(134047/batchsize)*(step_s+load_s)+np.ceil(30266/batchsize)*(val_s+load_s)
                    records[key]=dict(parameters=sum(p.numel() for p in net.parameters()),batch_size=batchsize,
                        train_step_seconds=step_s,warm_load_seconds=load_s,validation_forward_seconds=val_s,
                        estimated_epoch_seconds=float(seconds),estimated_25epoch_minutes=float(seconds*25/60),
                        mps_allocated_bytes=torch.mps.current_allocated_memory(),mps_driver_bytes=torch.mps.driver_allocated_memory(),
                        cumulative_process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        finite_logits_loss_gradients=True,weighted_ce_matches_torch=True,adaptations=adaptations)
                    break
                except RuntimeError as exc:
                    if 'out of memory' not in str(exc).lower() or batchsize==8:raise
                    adaptations.append(dict(failed_batch=batchsize,reason=str(exc),predefined_fallback=8))
                finally:del net,opt;torch.mps.empty_cache()
        result=dict(successful=True,generated_at=now(),device='mps',source_identity=source_identity(project),models=records,
                    training_subset_targets=128,validation_subset_targets=32,normalization_exact=True,cold_subset_load_seconds=cold_load_seconds,full_training_launched=False,
                    estimate_limits='Six updates/variant, median warm steps and data fetches; Both fixed subsets are primed before model timing; cold subset loading is recorded separately. Epoch estimates include estimated validation loading and are not measured full epochs. Cumulative macOS RSS high-water and MPS end values are not isolated peak GPU measurements.')
        json_write(project/'results/phase4_benchmarks.json',result)
        return result
    finally:train.close();val.close()


def smoke(project=PROJECT):
    project=Path(project);environment();body=read_registry(project)
    if not torch.backends.mps.is_available():raise RuntimeError('MPS required for real smoke')
    train,val=real_datasets(project,next(iter(body['runs'].values()))['configuration']);rng=np.random.default_rng(42)
    ti=np.sort(rng.choice(len(train),32,False)).tolist();vi=np.sort(rng.choice(len(val),16,False)).tolist();records=[]
    try:
        for model in ['M5','M6','M7','M8']:
            row=next(r for r in body['runs'].values() if r['kind']=='initial' and r['configuration']['model_id']==model and r['configuration']['loss']['name']=='weighted_cross_entropy')
            cfg=copy.deepcopy(row['configuration']);cfg.update(stage='smoke',development_smoke_subset=True,max_epochs=2,progress_batches=0)
            contract=copy.deepcopy(body['dataset_contract']);contract['phase4_smoke_targets']={'train':train.index.iloc[ti].epoch_id.tolist(),'validation':val.index.iloc[vi].epoch_id.tolist()}
            run_id='phase4-smoke-'+model.lower()+'-'+uuid.uuid4().hex[:8]
            def build():return build_trainer(project,cfg,contract,Subset(train,ti),Subset(val,vi),run_id)
            trainer=build()
            def interrupt(epoch,batch,step):
                if epoch==1 and batch==0:raise SimulatedInterruption('Real Phase4 short interruption')
            try:trainer.fit(resume='none',interrupt_callback=interrupt)
            except SimulatedInterruption:pass
            state=trainer.manager.recover();assert state['next_epoch']==1
            restored=build();restored.restore('auto');assert restored.history==state['history'] and restored.early_stopping==state['early_stopping']
            current=capture_rng(True)
            assert current['python']==state['rng']['python'] and current['numpy']==state['rng']['numpy']
            for name in ['torch_cpu','torch_mps']:torch.testing.assert_close(current[name],state['rng'][name],rtol=0,atol=0)
            for param,values in state['optimizer']['state'].items():
                for name,value in values.items():torch.testing.assert_close(restored.optimizer.state_dict()['state'][param][name].cpu(),value.cpu(),rtol=0,atol=0)
            restored.fit(resume='auto');assert restored.next_epoch==2 and restored.global_step==2*np.ceil(32/cfg['batch_size'])
            # End-to-end bounded artifact path on actual MPS, not just training arithmetic.
            from .inference import export_inference,load_model
            from .analysis import analyze_export,load_export
            from .phase4_analysis import EXTRAS
            from .phase4_explanations import save_explanations
            export=export_inference(restored.model,restored.validation_loader,restored.experiment,device='mps',streaming=True,
                intermediate_layers=[dict(name='intermediate',pooling='flatten')],metadata_fields=EXTRAS)
            manifest,frame,features,_=load_export(export)
            assert len(frame)==16 and features.shape==(16,64)
            assert frame.epoch_id.tolist()==val.index.iloc[vi].epoch_id.tolist()
            analyze_export(export,tsne_max_samples=16,tsne_perplexity=3,tsne_iterations=300,seed=42,hypnograms=False)
            loaded,_,checkpoint=load_model(restored.model,restored.experiment,'best','mps')
            batch=torch.utils.data.default_collate([val[vi[0]]])
            save_explanations(loaded,batch,checkpoint,restored.experiment.path/'explanations'/export.name)
            records.append(dict(model=model,run_id=run_id,streamed_export_rows=16,embedding_shape=[16,64],
                saved_saliency_gradcam_attention_supported=True,bounded_analysis_executed=True,train_targets=32,validation_targets=16,completed_epochs=2,
                optimizer_rng_history_early_stop_restored=True,interrupted_epoch_repeated=True))
            del trainer,restored;torch.mps.empty_cache()
        result=dict(successful=True,generated_at=now(),source_identity=source_identity(project),campaign_id=body['campaign_id'],
                    device='mps',models=records,full_training_launched=False,metrics_are_smoke_only=True,mps_bitwise_guarantee=False)
        json_write(project/'results/phase4_smoke_validation.json',result);return result
    finally:train.close();val.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['benchmark','smoke']);args=parser.parse_args()
    print(json.dumps(benchmark() if args.command=='benchmark' else smoke(),indent=2))
