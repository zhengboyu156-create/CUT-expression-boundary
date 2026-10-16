"""Regressions exposed by six new pairs; no patient-level independence assumed."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from expression.manifest import prepare,read_manifest
from expression.runner import make_options,make_dataset,load_generator,load_risk,generated_batch,evaluate_generator,seed_everything
from expression.calibration import gate,fit_and_certify
from expression.protocol import policy_hash
from models.expression_cut_model import ExpressionCUTModel
from expression.artifact_io import save_png

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/multipair_debug'


class MultiPairRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything();cls.rows=read_manifest(BASE/'full_manifest.jsonl')
        cls.opt=make_options(BASE/'train_manifest.jsonl',BASE/'test_tmp',split='demo',ngf=8,batch_size=2)

    def test_faint_IHC_is_not_all_rejected(self):
        for pid in ['2M_6_14','2M_9_19','2M_12_23','2M_13_15','2M_13_17']:
            rows=[r for r in self.rows if r['pair_id']==pid]
            self.assertEqual(len(rows),64);self.assertGreater(sum(r['valid'] for r in rows),50,pid)
        missing=[r for r in self.rows if r['pair_id']=='2M_11_11' and r['reason']=='reference_unobserved']
        self.assertGreater(len(missing),20)

    def test_all_new_windows_keep_identity_coordinates(self):
        self.assertEqual(len([r for r in self.rows if r['pair_id'].startswith('2M_')]),384)
        for row in self.rows:self.assertEqual(row['he_bbox'],row['ihc_bbox'])

    def test_factory_does_not_parse_host_program_argv(self):
        with patch.object(sys,'argv',['unrelated_program','--ngf','NOT_AN_INTEGER','--help']):
            opt=make_options(BASE/'train_manifest.jsonl',BASE/'test_tmp',split='demo',ngf=8)
        self.assertEqual(opt.ngf,8)

    def test_window_size_inferred_from_prepared_manifest(self):
        pair=json.loads((BASE/'pairs.json').read_text())[0]
        with tempfile.TemporaryDirectory() as directory:
            manifest=Path(directory)/'window256.jsonl';prepare([pair],manifest,window=256)
            opt=make_options(manifest,directory,split='demo',ngf=8)
            self.assertEqual(opt.stats_window,256)
            self.assertEqual(tuple(make_dataset(opt)[0]['A'].shape),(3,256,256))

    def test_within_bin_mean_can_represent_near_zero(self):
        opt=copy.deepcopy(self.opt);opt.learn_bin_means=True
        model=ExpressionCUTModel(opt);g=model.netG.eval();a=make_dataset(self.opt)[0]['A'][None]
        with torch.no_grad():
            g.distribution_head.weight.zero_();g.distribution_head.bias.fill_(-20);g.distribution_head.bias[0]=20
            g.within_bin_head.weight.zero_();g.within_bin_head.bias.fill_(-12)
            _,aux=g(a,return_aux=True)
        self.assertLess(float(aux['expected_condition']),.002)

    def test_training_can_use_H_and_E_only_predicted_condition(self):
        opt=copy.deepcopy(self.opt);opt.teacher_probability=0
        model=ExpressionCUTModel(opt);batch=next(iter(DataLoader(make_dataset(opt),batch_size=2)))
        model.set_input(batch);model.set_training_progress(1);model.forward()
        self.assertTrue(torch.equal(model.expression_aux['condition'],model.expression_aux['expected_condition'].detach()))

    def test_unvalidated_training_strategies_are_opt_in(self):
        self.assertFalse(self.opt.learn_bin_means)
        self.assertEqual(self.opt.lambda_moment,0.)
        self.assertEqual(self.opt.teacher_probability,1.)

    def test_MC_variance_is_invariant_to_renaming(self):
        model,_,_=load_generator(ROOT/'artifacts/attachment/expression_cut/generator.pt')
        a=make_dataset(self.opt)[0]['A'][None]
        first=generated_batch(model.netG,a,['original_name'],samples=3)
        renamed=generated_batch(model.netG,a,['renamed_copy'],samples=3)
        self.assertTrue(torch.equal(first[1],renamed[1]));self.assertTrue(torch.equal(first[3],renamed[3]))
        with self.assertRaisesRegex(ValueError,'protocol'):
            load_risk(ROOT/'artifacts/attachment/expression_cut/risk.pt',ROOT/'artifacts/attachment/expression_cut/generator.pt')

    def test_record_only_evaluation_retains_no_image_tensors(self):
        ds=make_dataset(self.opt);ds.rows=ds.rows[:2]
        model=ExpressionCUTModel(self.opt);seen=[]
        records,outputs=evaluate_generator(model.netG,ds,samples=1,keep_outputs=False,output_callback=lambda o,r:seen.append(r['window_id']))
        self.assertEqual(len(records),2);self.assertEqual(outputs,[]);self.assertEqual(len(seen),2)

    def test_image_cache_is_bounded_and_owned_by_dataset(self):
        ds=make_dataset(self.opt);paths=sorted({r[k] for r in ds.rows for k in ['he','ihc']})
        for path in paths:ds.image(path)
        misses=ds.cache_misses
        for path in paths:ds.image(path)
        self.assertEqual(ds.cache_misses,misses);self.assertEqual(ds.cache_hits,len(paths))
        second=make_dataset(self.opt);self.assertEqual(len(second._images),0)
        second.cache_limit=2
        for path in paths:second.image(path)
        self.assertLessEqual(len(second._images),2)

    def test_stale_protocol_and_invalid_scores_are_rejected(self):
        def fixture(split):
            return [dict(group_id=f'{split}_{i}',group_kind='patient',split=split,
                         window_id=str(i),score=i/100,event=0) for i in range(100)]
        cert=fit_and_certify(fixture('calibrate'),fixture('certify'),dict(training_groups=[],generator_hash='g',risk_hash='h',protocol_hash=policy_hash()))
        keep,_=gate([-1.,2.,float('nan')],cert,'g','h');self.assertFalse(keep.any())
        cert['provenance']['protocol_hash']='old_policy'
        keep,reason=gate([.1],cert,'g','h');self.assertFalse(keep.any());self.assertEqual(reason,'protocol_mismatch')

    def test_large_PNG_roundtrip_is_complete_and_lossless(self):
        source=Image.open(ROOT/'examples/multipair/2M2103108_13_17(2).jpg').convert('RGB')
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'last_export.png';save_png(output,source)
            restored=Image.open(output);restored.load()
            self.assertTrue(np.array_equal(np.array(source),np.array(restored)))
            self.assertFalse(output.with_name(output.name+'.tmp').exists())


if __name__=='__main__':unittest.main()
