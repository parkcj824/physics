"""Read-only replay of scientific assets; writes only Session 0005 verification output."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import torch
from scipy.optimize import root
from scipy.linalg import expm

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.session0005_common import ROOT,sha,tensor,load_data,preserve,write,now
from experiments.session0005_controls import build,classical
from experiments.session0005_interaction import encoder_restore,labels
from src.operators.session0005_normalized import NormalizedControl
from src.operators.session0005_interaction import Interaction


def close(a,b,atol=1e-10,rtol=1e-9):
    np.testing.assert_allclose(a,b,atol=atol,rtol=rtol)


def physical(k,u,g,v=0.):
    full=np.r_[g[0],u,g[1]];h=1/(len(u)+1)
    return -k*np.diff(full)/h+max(v,0)*full[:-1]+min(v,0)*full[1:]


def dense_reference(k,q,g=(0.,0.),lam=0.,v=0.):
    """Independent dense assembly and scipy hybr root; no banded training solver."""
    n=len(q);zero=np.zeros(n)
    offset=np.diff(physical(k,zero,g,v))*(n+1)
    a=np.column_stack([np.diff(physical(k,np.eye(n)[i],g,v))*(n+1)-offset for i in range(n)])
    rhs=q-offset;initial=np.linalg.solve(a,rhs)
    if lam:
        sol=root(lambda u:a@u-rhs-lam*np.exp(u),initial,
                 jac=lambda u:a-np.diag(lam*np.exp(u)),tol=1e-11)
        u=sol.x
    else:u=initial
    residual=a@u-rhs-lam*np.exp(u)
    assert max(abs(residual))<2e-8,(lam,max(abs(residual)))
    return u,physical(k,u,g,v),float(max(abs(residual)))


def cm(raw,u,j,target,family):
    fields={k:[] for k in ['pde','balance','constitutive','energy']}
    for z,t,f in zip(raw,u,j):
        k=np.exp(z[:16]);q=z[16:31]+(z[-1]*np.exp(t) if family=='G' else 0)
        p=physical(k,t,z[31:33],z[-1] if family=='H' else 0.)
        fields['pde'].append(max(abs(np.diff(p)*16-q)))
        fields['balance'].append(max(abs(np.diff(f)*16-q)))
        fields['constitutive'].append(max(abs(f-p)))
        fields['energy'].append(abs(f[-1]-f[0]-sum(q)/16))
    required=(raw[:,16:31].min(1)>=0)&(abs(raw[:,31:33]).max(1)==0)
    return {'mean_relative_l2':float(np.mean(np.linalg.norm(u-target,axis=1)/np.maximum(np.linalg.norm(target,axis=1),1e-14))),
        'max_absolute_error':float(abs(u-target).max()),'mean_pde_residual_inf':float(np.mean(fields['pde'])),
        'max_pde_residual_inf':float(max(fields['pde'])),'mean_constitutive_mismatch_inf':float(np.mean(fields['constitutive'])),
        'max_balance_residual_inf':float(max(fields['balance'])),'max_generated_energy_balance_abs':float(max(fields['energy'])),
        'minimum_state':float(u.min()),'positive_required_samples':int(required.sum()),
        'positivity_violating_samples':int(((u.min(1)<-1e-12)&required).sum()),'boundary_error_abs':0.}


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    out={'verified_utc':now(),'prior_tracked_files_unchanged':preserve(),'dataset_files':0,'independent_reference_checks':0,
         'max_reference_field_difference':0.,'max_reference_pde_residual':0.,'checkpoint_files':0,
         'control_prediction_batches':0,'interaction_prediction_batches':0,'classical_prediction_batches':0,
         'snapshot_hash_checks':0,'snapshot_current_source_checks':0}
    data_by_hash={}
    def refcheck(k,q,g,lam,v,target,flux=None):
        u,j,r=dense_reference(k,q,g,lam,v);close(u,target,atol=1e-8,rtol=1e-8)
        if flux is not None:close(j,flux,atol=1e-8,rtol=1e-8)
        out['independent_reference_checks']+=1
        out['max_reference_field_difference']=max(out['max_reference_field_difference'],float(max(abs(u-target))))
        out['max_reference_pde_residual']=max(out['max_reference_pde_residual'],r)
    for manifest in ['controls_dataset_manifest','interaction_dataset_manifest','normalized_dataset_manifest']:
        j=json.loads((ROOT/(manifest+'.json')).read_text())
        assert not any(j.get('reference_failures',j.get('failures',{})).values())
        for name,h in j['sha256'].items():
            if manifest=='normalized_dataset_manifest':name=f'C0b_{name}_confirm'
            assert sha(ROOT/'data'/(name+'.npz'))==h
            d=load_data(name);out['dataset_files']+=1;data_by_hash[h]=d
            if 'inputs' in d:
                family=name.split('_')[1]
                for z,t,f in zip(d['inputs'],d['target'],d['flux']):
                    refcheck(np.exp(z[:16]),z[16:31],z[31:33],z[-1] if family=='G' else 0.,z[-1] if family=='H' else 0.,t,f)
            else:
                for i,m in enumerate(d['source_count']):
                    for mask in range(1<<int(m)):
                        assert d['valid_subsets'][i,mask]
                        q=sum((a*np.exp(-.5*((d['x']-c)/w)**2) for j,(a,c,w) in enumerate(d['sources'][i]) if mask&(1<<j)),np.zeros(15))
                        refcheck(np.full(16,d['conductivity'][i]),q,(0,0),d['lam'][i],0.,d['target_subsets'][i,mask])
                    assert not d['valid_subsets'][i,1<<int(m):].any()
    cinfo=json.loads((ROOT/'controls_frozen.json').read_text())
    ninfo=json.loads((ROOT/'normalized_frozen.json').read_text())
    finfo=json.loads((ROOT/'interaction_frozen.json').read_text())
    einfo=json.loads((ROOT/'interaction_encoder_frozen.json').read_text())
    allinfos=cinfo['models']+ninfo['models']+finfo['models']+[einfo['model']]
    assert len({i['checkpoint'] for i in allinfos})==70
    for i in allinfos:
        assert sha(ROOT/i['checkpoint'])==i['sha256'];out['checkpoint_files']+=1
    controls=json.loads((ROOT/'controls_confirmation.json').read_text())
    norm=json.loads((ROOT/'normalized_confirmation.json').read_text())
    fconf=json.loads((ROOT/'interaction_confirmation.json').read_text())
    assert cinfo['frozen_utc']<controls['evaluated_utc']<ninfo['frozen_utc']<norm['evaluated_utc']
    assert einfo['frozen_utc']<finfo['frozen_utc']<fconf['evaluated_utc']
    modelcache={}
    with torch.no_grad():
        for row in controls['records']+norm['records']:
            key=row['checkpoint']
            if key not in modelcache:
                kind=row['kind'];family=row['family']
                model=NormalizedControl(kind,family).double() if kind.startswith('norm_') else build(kind,family)
                model.load_state_dict(torch.load(ROOT/key,weights_only=True)['state']);modelcache[key]=model
            assert sha(ROOT/row['checkpoint'])==row['checkpoint_sha256']
            with np.load(ROOT/row['arrays']) as a:
                for dk,dv in data_by_hash[row['input_sha256']].items():close(a[dk],dv,atol=0,rtol=0)
                u,j=[v.numpy() for v in modelcache[key](tensor(a['inputs']))]
                close(u,a['prediction']);close(j,a['predicted_flux'])
                stats=cm(a['inputs'],u,j,a['target'],row['family'])
                for k,v in stats.items():close(v,row[k],atol=1e-9)
            out['control_prediction_batches']+=1
        for row in controls['classical_records']:
            with np.load(ROOT/row['arrays']) as a:
                u,j=classical(a['inputs'],row['family'],row['kind'])
                close(u,a['prediction']);close(j,a['predicted_flux'])
                for k,v in cm(a['inputs'],u,j,a['target'],row['family']).items():close(v,row[k],atol=1e-9)
            out['classical_prediction_batches']+=1
        encoder=encoder_restore();modelcache={}
        for row in fconf['records']:
            if row['checkpoint'] not in modelcache:
                m=Interaction(row['kind']).double();m.load_state_dict(torch.load(ROOT/row['checkpoint'],weights_only=True)['state']);modelcache[row['checkpoint']]=m
            m=modelcache[row['checkpoint']]
            with np.load(ROOT/row['arrays']) as z:
                d={k:z[k] for k in z.files}
                assert sha(ROOT/row['checkpoint'])==row['checkpoint_sha256']
                for dk,dv in data_by_hash[row['input_sha256']].items():close(d[dk],dv,atol=0,rtol=0)
                base,single,true,cross=labels(d)
                k=tensor(d['conductivity']);lam=tensor(d['lam']);src=tensor(d['sources']);x=tensor(d['x'])
                b,r=encoder(k,lam,src,x);cp,activation=m(b,r,src,k,lam,x);pred=b+r.sum(1)+cp
                close(b.numpy(),d['encoded_base']);close(r.numpy(),d['encoded_responses']);close(cp.numpy(),d['predicted_mixed'])
                close(pred.numpy(),d['prediction']);close(activation.numpy(),d['activation']);close(true,d['reference']);close(cross,d['reference_mixed'])
                hr=cross.copy();hp=cp.numpy().copy()
                for i in range(4):
                    for j in range(i+1,4):
                        active=d['source_count']>j
                        if not active.any():continue
                        pair=d['target_subsets'][:,(1<<i)+(1<<j)]-d['target_subsets'][:,1<<i]-d['target_subsets'][:,1<<j]+base
                        hr-=pair*active[:,None]
                        rr=torch.zeros_like(r);rr[:,[i,j]]=r[:,[i,j]]
                        ss=src.clone();ss[...,0]=0.;ss[:,[i,j],0]=src[:,[i,j],0]
                        pp=m(b,rr,ss,k,lam,x)[0].numpy();hp-=pp*active[:,None]
                close(hr,d['reference_higher']);close(hp,d['predicted_higher'])
                p=pred.numpy();cr=np.linalg.norm(cross,axis=1);ce=np.linalg.norm(cp.numpy()-cross,axis=1)
                hn=np.linalg.norm(hr,axis=1);he=np.linalg.norm(hp-hr,axis=1)
                vals={'mean_relative_l2':np.mean(np.linalg.norm(p-true,axis=1)/np.maximum(np.linalg.norm(true,axis=1),1e-14)),
                    'max_absolute_error':abs(p-true).max(),'mean_cross_reference_l2':cr.mean(),'max_cross_reference_l2':cr.max(),
                    'mean_cross_absolute_l2_error':ce.mean(),'mean_cross_relative_l2':np.mean(ce/np.maximum(cr,1e-12)),
                    'nontrivial_cross_count':sum(cr>1e-8),'mean_higher_reference_l2':hn.mean(),'mean_higher_absolute_l2_error':he.mean(),
                    'nontrivial_higher_count':sum(hn>1e-10),'mean_cross_to_full_norm_ratio':np.mean(cr/np.maximum(np.linalg.norm(true,axis=1),1e-14)),
                    'minimum_state':p.min(),'negative_samples':sum(p.min(1)<-1e-12),'activation_fraction':activation.numpy().mean()}
                for k0,v0 in vals.items():close(v0,row[k0])
                for key,numer,denom,cut in [('mean_cross_relative_l2_nontrivial',ce,cr,1e-8),('mean_higher_relative_l2_nontrivial',he,hn,1e-10)]:
                    if (denom>cut).any():close(np.mean(numer[denom>cut]/denom[denom>cut]),row[key])
                    else:assert row[key] is None
                pdes=[];energies=[]
                for i,u in enumerate(p):
                    q=sum((a*np.exp(-.5*((d['x']-c)/w)**2) for a,c,w in d['sources'][i] if a>0),np.zeros(15))
                    j=physical(np.full(16,d['conductivity'][i]),u,(0,0));q=q+d['lam'][i]*np.exp(u)
                    pdes.append(max(abs(np.diff(j)*16-q)));energies.append(abs(j[-1]-j[0]-sum(q)/16))
                close(np.mean(pdes),row['mean_pde_residual_inf']);close(np.mean(energies),row['mean_energy_balance_abs'])
            out['interaction_prediction_batches']+=1
    # Analytical counterexamples independently reconstructed from saved physical fields.
    jresult=json.loads((ROOT/'compatibility.json').read_text());arr=np.load(ROOT/'arrays/compatibility_counterexamples.npz')
    for row in jresult['records']:
        name=row['case'];u=arr[name+'_prediction'];j=arr[name+'_flux'];k=np.full(16,row['conductivity']);v=row['velocity']
        b=np.diff(j)*16-1.;c=j-physical(k,u,(0,0),v)
        close(np.diff(physical(k,u,(0,0),v))*16-1,b-np.diff(c)*16,atol=1e-11)
        close(max(abs(b)),row['balance_inf']);close(max(abs(c)),row['constitutive_inf'])
        assert row['gradient_equals_normal_equations_error']<1e-10
        assert row['gradient_history'][-1]['defect_squared_norm']<=row['gradient_history'][0]['defect_squared_norm']
    ca=json.loads((ROOT/'causal.json').read_text());a=np.load(ROOT/'arrays/causal_counterexamples.npz');p=expm(ca['dt']*a['generator'])
    close(p,a['transition']);close(ca['learned_ar2_coefficients'],[np.trace(p),-np.linalg.det(p)])
    coef=np.linalg.lstsq(np.c_[a['train_first'][:,0],a['train_initial'][:,0]],a['train_second'][:,0],rcond=None)[0]
    close(np.c_[a['confirm_first'][:,0],a['confirm_initial'][:,0]]@coef,a['prediction'])
    close((a['confirm_first']@p.T)[:,0],a['reference']);assert ca['alias_observation_gap']<1e-12 and ca['after_alias_future_gap']>.04
    cl=json.loads((ROOT/'closure_counterexample.json').read_text());a=np.load(ROOT/'arrays/closure_counterexample.npz')
    equation_audit=json.loads((ROOT/'closure_equation_audit.json').read_text())
    out['closure_equation_checks']=0
    for row in cl['records']:
        n=row['n'];x=np.arange(1,n+1)/(n+1);base,_j,_res=dense_reference(np.ones(n+1),np.zeros(n),lam=1.5)
        for label in ['A','B']:
            q=a[f'n{n}_{label}_sources'];r=a[f'n{n}_{label}_responses'];t=a[f'n{n}_{label}_reference']
            for qi,ri in zip(q,r):refcheck(np.ones(n+1),qi,(0,0),1.5,0.,base+ri)
            refcheck(np.ones(n+1),q.sum(0),(0,0),1.5,0.,t)
            h=1.5*np.exp(base)*(np.expm1(r.sum(0))-np.expm1(r).sum(0));close(h,a[f'n{n}_{label}_mixed_defect'])
            delta=t-base-r.sum(0)
            close(delta,a[f'n{n}_{label}_required_correction'])
            # Independent dense diffusion applied through flux differences.
            residual=np.diff(physical(np.ones(n+1),delta,(0,0)))*(n+1)-1.5*np.exp(base+r.sum(0))*np.expm1(delta)-h
            assert max(abs(residual))<1e-9
            saved=[v for v in equation_audit['records'] if v['n']==n and v['case']==label][0]
            assert saved['exact_correction_equation_residual_inf']<1e-9
            out['closure_equation_checks']+=1
        gap=np.linalg.norm(a[f'n{n}_A_required_correction']-a[f'n{n}_B_required_correction'])
        close(gap,row['required_correction_l2_difference']);close(gap/2,row['H_only_worst_case_l2_lower_bound'])
        assert row['H_input_max_difference']<1e-12 and row['minimum_source']>0
    for file in sorted((ROOT/'source_snapshot').glob('*/sha256.json')):
        for path,h in json.loads(file.read_text()).items():
            assert sha(file.parent/path)==h;out['snapshot_hash_checks']+=1
    for phase in ['controls_confirm','interaction_confirm','compatibility','causal','closure_counterexample','normalized_control','cost_replay','closure_equation_audit']:
        for path,h in json.loads((ROOT/'source_snapshot'/phase/'sha256.json').read_text()).items():
            assert sha(path)==h,(phase,path);out['snapshot_current_source_checks']+=1
    audit=json.loads((ROOT/'posthoc_audit.json').read_text())
    assert len(audit['feature_traces'])==42 and len(audit['field_error_decomposition'])==135
    assert max(r['pooled_set_reconstruction_max_error'] for r in audit['pooled_set_equivalence'])<1e-12
    report=ROOT/'validation/pytest.xml'
    if report.exists():
        suites=ET.parse(report).getroot();nodes=list(suites) if suites.tag=='testsuites' else [suites]
        out['pytest_tests']=sum(int(n.attrib['tests']) for n in nodes)
        assert sum(int(n.attrib.get('failures',0))+int(n.attrib.get('errors',0)) for n in nodes)==0
    else:out['pytest_tests']=None
    assets=ROOT/'validation/assets_sha256.json'
    if assets.exists():
        manifest=json.loads(assets.read_text())
        for path,h in manifest.items():assert sha(path)==h,path
        out['session0005_asset_files_unchanged']=len(manifest)
    else:out['session0005_asset_files_unchanged']=None
    out['status']='PASS';out['scope']='checkpoint/array replay, independent dense references, metrics and counterexamples; no refitting or historical output writes'
    write('validation/integrity.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
