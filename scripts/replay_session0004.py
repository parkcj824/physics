"""Run the saved scientific functions in a fresh output directory without overwrites."""

import argparse
from pathlib import Path
import shutil
import torch

from experiments import session0004 as study
from experiments import session0004_diagnostics as diagnostic
from experiments import session0004_transport as transport
from scripts import verify_session0004 as audit
from scripts import audit_session0004 as posthoc
from experiments import session0004_capacity as capacity


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('phase',choices=['initialize','prepare','train','train_oracle','train_matched','confirm','transport','counterexamples','controls','capacity','audit','verify','all'])
    args=parser.parse_args();canonical=study.ROOT.resolve();target=args.output.resolve()
    if target==canonical:raise ValueError('replay output must differ from the preserved canonical session')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    if args.phase in ['initialize','all']:
        target.mkdir(parents=True,exist_ok=False)
        (target/'arrays').mkdir()
        for name in ['prior_file_sha256.json','environment.json','prior_validation.json']:
            shutil.copy2(canonical/name,target/name)
    if not target.exists():raise FileNotFoundError('initialize a fresh replay first')
    study.ROOT=diagnostic.ROOT=transport.ROOT=audit.ROOT=posthoc.ROOT=capacity.ROOT=target
    operations={'prepare':study.prepare,'train':study.train,'train_oracle':study.train_oracle,
                'train_matched':study.train_matched,'confirm':study.confirm,'transport':transport.main,
                'counterexamples':diagnostic.counterexamples,'controls':diagnostic.controls,
                'capacity':capacity.main,'audit':posthoc.main,'verify':audit.verify}
    if args.phase=='all':
        for name in ['prepare','counterexamples','train','train_oracle','train_matched','confirm','transport','controls','capacity','audit']:
            operations[name]()
        print('Replay finished. Run pytest with a JUnit XML under this output/validation, then run the verify phase.')
    elif args.phase!='initialize':operations[args.phase]()
    else:print(f'Initialized isolated replay: {target}')


if __name__=='__main__':main()
