"""Post-confirmation diagnostics of saved models/predictions. No fitting or model selection."""
import json
import numpy as np
import torch
from experiments.session0005_common import ROOT,load_data,tensor,write,sha,preserve
from experiments.session0005_controls import build
from experiments.session0005_interaction import labels
from src.operators.session0005_controls import physical_flux
from src.operators.session0005_normalized import NormalizedControl,normalize


def trace(model,raw,kind):
    h=torch.tanh(model.encode((raw-model.center)/model.scale));r0=None;stats=[]
    if kind in ['static_physics','norm_static']:
        u,j=model.generate(h,raw);r0=j-physical_flux(raw,u,model.family)
    for layer in range(model.depth):
        u,j=model.generate(h,raw);r=j-physical_flux(raw,u,model.family)
        if kind.startswith('norm_'):
            s=normalize(model.feature_projection(h))
            if kind=='norm_physical':s=normalize(s+normalize(r))
            if kind=='norm_static':s=normalize(s+normalize(r0))
        elif kind=='learned_feature':s=model.feature_projection(h)
        elif kind=='static_physics':s=r0
        elif kind=='physics_plus_feature':s=r+model.feature_projection(h)
        else:s=r
        h=torch.tanh(model.evolve(h)+model.mismatch(s)+torch.tanh(model.gate(h))*torch.tanh(model.reaction(s)))
        stats.append({'layer':layer,'physical_constitutive_defect_rms':float(r.square().mean().sqrt()),
            'used_feature_rms':float(s.square().mean().sqrt()),'hidden_fraction_abs_above_099':float((h.abs()>.99).double().mean())})
    return model.generate(h,raw),stats


def main():
    if (ROOT/'posthoc_audit.json').exists():raise FileExistsError('posthoc audit exists')
    infos=json.loads((ROOT/'controls_frozen.json').read_text())['models']+json.loads((ROOT/'normalized_frozen.json').read_text())['models']
    trace_rows=[]
    with torch.no_grad():
        for info in infos:
            kind=info['kind'];family=info['family']
            if kind in ['direct_matched','learned_jacobi']:continue
            m=NormalizedControl(kind,family).double() if kind.startswith('norm_') else build(kind,family)
            assert sha(ROOT/info['checkpoint'])==info['sha256'];m.load_state_dict(torch.load(ROOT/info['checkpoint'],weights_only=True)['state'])
            d=load_data(f'C0b_{family}_confirm');raw=tensor(d['inputs'])
            (u,j),stats=trace(m,raw,kind);ue,je=m(raw)
            torch.testing.assert_close(u,ue,atol=1e-12,rtol=1e-12);torch.testing.assert_close(j,je,atol=1e-12,rtol=1e-12)
            trace_rows.append({'family':family,'kind':kind,'seed':info['seed'],'layers':stats})
    frows=json.loads((ROOT/'interaction_confirmation.json').read_text())['records'];models=json.loads((ROOT/'interaction_frozen.json').read_text())['models']
    bykey={(i['kind'],i['seed']):i for i in models};decomp=[];set_errors=[]
    for row in frows:
        with np.load(ROOT/row['arrays']) as d:
            base,single,truth,cross=labels({k:d[k] for k in d.files})
            encoder_error=d['encoded_base']+d['encoded_responses'].sum(1)-base-single.sum(1)
            cross_error=d['predicted_mixed']-cross;whole=d['prediction']-truth
            np.testing.assert_allclose(whole,encoder_error+cross_error,atol=2e-15,rtol=1e-12)
            decomp.append({'kind':row['kind'],'seed':row['seed'],'case':row['case'],
                'mean_encoder_component_error_l2':float(np.linalg.norm(encoder_error,axis=1).mean()),
                'mean_mixed_component_error_l2':float(np.linalg.norm(cross_error,axis=1).mean()),
                'mean_whole_field_error_l2':float(np.linalg.norm(whole,axis=1).mean()),
                'mean_squared_cross_term':float(np.mean(2*np.sum(encoder_error*cross_error,axis=1)))})
            if row['kind']=='reaction_gate':
                state=torch.load(ROOT/bykey[(row['kind'],row['seed'])]['checkpoint'],weights_only=True)['state']
                r=d['encoded_responses'];s=r.sum(1);k=d['conductivity'];lam=d['lam'];context=np.c_[np.ones(len(k)),lam/k,np.log(k)]
                h=np.maximum(lam[:,None]*np.exp(d['encoded_base'])*(np.expm1(s)-np.expm1(r).sum(1)),0)
                threshold=state['threshold'].numpy();tau=np.logaddexp(0,threshold[0]+threshold[1]*lam/k)[:,None]
                kernel=np.logaddexp(0,np.einsum('bc,cnk->bnk',context,state['kernel'][0].numpy()))
                rec=d['x']*(1-d['x'])*np.einsum('bnk,bk->bn',kernel,np.maximum(h-tau,0))
                error=float(abs(rec-d['predicted_mixed']).max());assert error<1e-12
                set_errors.append({'seed':row['seed'],'case':row['case'],'pooled_set_reconstruction_max_error':error,
                    'tau_min':float(tau.min()),'tau_max':float(tau.max()),
                    'nonlinear_only_activation_fraction':float(d['activation'][lam>0].mean()) if (lam>0).any() else None})
    write('posthoc_audit.json',{'feature_traces':trace_rows,'field_error_decomposition':decomp,'pooled_set_equivalence':set_errors,
        'scope':'saved predictions/frozen checkpoints only; post-confirmation explanatory diagnostics, not new independent confirmation; no optimizer updates',
        'activation_note':'non-reaction baselines store evaluated slots=1, not a learned physical activation fraction; dense kernels run even when gate signal is zero',
        'prior_files_preserved':preserve()})
    print(json.dumps({'feature_trace_batches':len(trace_rows),'error_decomposition_batches':len(decomp),'set_equivalence_batches':len(set_errors)}))


if __name__=='__main__':
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1);main()
