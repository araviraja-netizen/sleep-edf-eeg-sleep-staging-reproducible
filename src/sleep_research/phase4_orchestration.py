"""Sequential fail-closed state machine; callbacks allow bounded synthetic tests."""
from .common import json_write,atomic_file,now
from .phase4_registry import sealed,envelope,active_keys


class CampaignEngine:
    def __init__(self,body,path,run,verify,has_result,selection):
        self.body,self.path,self.run_callback,self.verify,self.has_result,self.selection_callback=body,path,run,verify,has_result,selection
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.exists():
            self.state=envelope(self.path)
            if self.state['campaign_id']!=body['campaign_id']:raise ValueError('State belongs to another immutable registry')
        else:
            self.state=dict(campaign_id=body['campaign_id'],status='pending',updated_at=now(),current=None,
                runs={key:dict(status='conditional' if row['kind']=='conditional_repeat' else 'pending') for key,row in body['runs'].items()})
        self.save()
    def save(self):
        self.state['updated_at']=now();sealed(self.path,self.state)
        lines=['# Phase4 campaign status',f"Campaign {self.body['campaign_id']}; status {self.state['status']}; current {self.state['current']}; updated {self.state['updated_at']}.",'',
               '| Key | Status | Detail |','| --- | --- | --- |']
        for key,row in self.state['runs'].items():lines.append('| '+key+' | '+row['status']+' | '+str(row.get('error','')).replace('|','/').replace('\n',' ')+' |')
        with atomic_file(self.path.parent/'phase4_campaign_status.md') as f:f.write('\n'.join(lines)+'\n')
    def execute(self,key):
        node=self.state['runs'][key]
        try:
            if node['status']=='completed' or self.has_result(key):
                result=self.verify(key);node.update(status='completed',verified_at=now());self.save();return result
            self.state.update(status='running',current=key);node.update(status='running',started_at=now());node.pop('error',None);self.save()
            self.run_callback(key)
            result=self.verify(key);node.update(status='completed',verified_at=now());self.save();return result
        except BaseException as exc:
            status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
            node.update(status=status,error=type(exc).__name__+': '+str(exc));self.state['status']=status;self.save();raise
    def run(self):
        primary=[self.execute(key) for key in self.body['initial_keys']]
        selection=self.selection_callback(primary)
        for key in active_keys(self.body,selection)[len(self.body['initial_keys']):]:self.execute(key)
        self.state.update(status='completed',current=None);self.save();return selection
