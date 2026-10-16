"""Check result provenance and numerical record integrity without rerunning research."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


def verify(root):
    root=Path(root)
    run=json.loads((root/'run.json').read_text())
    assert run['status']=='completed'
    assert run['cycle_count']==12
    assert run['invention_selected'] is False
    rows=json.loads((root/'metrics/all_cases.json').read_text())
    assert len(rows)==run['recorded_case_count']==198
    assert set(row['cycle'] for row in rows)==set(range(1,13))
    names=set()
    for row in rows:
        path=row['predictions_file']
        assert path not in names
        names.add(path)
        assert row['latency_median_seconds']>=0 and row['traced_peak_bytes']>=0
        with np.load(root/path) as data:
            prediction,reference=data['prediction'],data['reference']
            assert prediction.shape==reference.shape
            assert np.isfinite(prediction).all() and np.isfinite(reference).all()
            difference=prediction-reference
            expected=np.linalg.norm(difference)/max(np.linalg.norm(reference),1e-14)
            np.testing.assert_allclose(row['relative_l2'],expected,rtol=1e-12,atol=1e-15)
            np.testing.assert_allclose(row['max_absolute_error'],np.max(abs(difference)),rtol=1e-12,atol=1e-15)
            if 'source' in data and 'conductivity' in data and data['source'].ndim==2:
                sample=np.linalg.norm(difference,axis=1)/np.maximum(np.linalg.norm(reference,axis=1),1e-14)
                if 'mean_sample_relative_l2' in row:
                    np.testing.assert_allclose(row['mean_sample_relative_l2'],sample.mean(),rtol=1e-12)
    hashes=json.loads((root/'source_snapshot/sha256.json').read_text())
    for name,digest in hashes.items():
        assert hashlib.sha256((root/'source_snapshot'/name).read_bytes()).hexdigest()==digest
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest, f'current source differs: {name}'
    plot_count=0
    for path in (root/'plots').glob('*.png'):
        with Image.open(path) as image: image.verify()
        plot_count+=1
    assert plot_count>=3
    convergence=json.loads((root/'metrics/fem_manufactured_convergence.json').read_text())
    assert convergence[-1]['relative_l2'] < convergence[0]['relative_l2']/8
    report=Path('artifacts/validation/research_pytest.xml')
    suites=list(ET.parse(report).getroot().iter('testsuite'))
    test_count=sum(int(s.attrib['tests']) for s in suites)
    assert test_count>=21
    assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
    result={'status':'verified','records':len(rows),'cycles':12,'pytest_passed':test_count,
            'pytest_failed':0,'pytest_skipped':0,'validated_png_count':plot_count,
            'checks':['all saved error metrics recomputed from predictions and references',
                      'all predictions finite and timed','source snapshot and current code hashes match',
                      'PNG files decoded','manufactured continuum convergence confirmed',
                      'nonzero passing pytest result inspected'],
            'limitations':['record integrity is not proof of architectural novelty',
                          'traced memory omits some native allocations; RSS is a process high-water mark']}
    (root/'integrity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output')
    verify(parser.parse_args().output)
