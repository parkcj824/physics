"""Post-confirmation diagnostics on frozen predictions/models; no optimizer updates."""

import json
import numpy as np
import torch
from experiments.session0004 import ROOT,restore,tensor,write
from experiments.session0004_transport import TransportCoGeneration


def main():
    rows=json.loads((ROOT/'transport_confirmation.json').read_text())['records'];positivity=[]
    for row in rows:
        if row['case']!='direction':continue
        with np.load(ROOT/row['arrays']) as data:
            p=data['prediction']
            positivity.append({'mode':row['mode'],'seed':row['seed'],'minimum_prediction':float(p.min()),
                'minimum_reference':float(data['target'].min()),'negative_nodes_below_minus_1e12':int((p< -1e-12).sum()),
                'negative_samples':int((p.min(1)< -1e-12).sum()),'samples':len(p)})
    write('transport_positivity_audit.json',{'scope':'post-confirmation diagnostics on previously saved predictions; positive source and zero boundary; no retraining','records':positivity})
    infos=json.loads((ROOT/'frozen_models.json').read_text())['models']+json.loads((ROOT/'transport_frozen.json').read_text())['models']
    if (ROOT/'capacity_frozen.json').exists():infos+=json.loads((ROOT/'capacity_frozen.json').read_text())['models']
    account=[]
    for info in infos:
        if info['seed']!=1004:continue
        family=info['family']
        if info['mode']=='direct_width64':
            from src.operators.session0004 import CoGenerationModel
            model=(CoGenerationModel if family=='G' else TransportCoGeneration)(mode='direct',width=64).double()
            model.load_state_dict(torch.load(ROOT/info['checkpoint'],weights_only=True)['state'])
        elif family=='H':
            model=TransportCoGeneration(mode=info['mode']).double()
            model.load_state_dict(torch.load(ROOT/info['checkpoint'],weights_only=True)['state'])
        else:model=restore(info)
        with np.load(ROOT/'data'/(family+'_train.npz')) as data:
            if family=='E':loss=model(tensor(data['lam'][:2]),tensor(data['x'])).square().sum()
            elif family=='F':loss=model(tensor(data['inputs'][:2]),tensor(data['x'])).square().sum()
            else:
                t,j=model(tensor(data['inputs'][:2]));loss=t.square().sum()+j.square().sum()
        loss.backward()
        account.append({'family':family,'mode':info['mode'],'total_parameters':sum(p.numel() for p in model.parameters()),
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'parameters_with_gradient_in_probe':sum(p.numel() for p in model.parameters() if p.grad is not None),
            'scope':'one representative forward/backward on training inputs, no optimizer update'})
    write('parameter_accounting.json',account)
    print(json.dumps({'positivity_records':len(positivity),'parameter_probes':len(account)}))


if __name__=='__main__':
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);main()
