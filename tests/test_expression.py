"""Real attachment integration tests plus clearly synthetic calibration unit fixtures."""
import copy
import json
from pathlib import Path
import unittest
import tempfile
import numpy as np
import torch
from torch.utils.data import DataLoader
from expression.manifest import read_manifest,check_leakage
from expression.statistics import StainStatistics
from expression.runner import seed_everything,make_options,make_dataset
from expression.networks import ErrorPredictor
from expression.calibration import fit_and_certify,gate,representative_per_group
from models.expression_cut_model import ExpressionCUTModel
from models.networks import ResnetBlock
from expression.protocol import policy_hash

MANIFEST=Path('artifacts/attachment/manifest.jsonl')


class AttachmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything();cls.rows=read_manifest(MANIFEST)
        cls.opt=make_options(MANIFEST,'artifacts/test','expression_cut','demo',ngf=8,batch_size=2,learn_bin_means=True,lambda_moment=2.)
        cls.dataset=make_dataset(cls.opt)

    def test_full_grid_identity_and_coverage(self):
        cover=np.zeros((1024,1024),dtype=int)
        for r in self.rows:
            self.assertEqual(r['he_bbox'],r['ihc_bbox']);x,y,w,h=r['he_bbox'];cover[y:y+h,x:x+w]+=1
        self.assertEqual(len(self.rows),64);self.assertTrue(np.all(cover==1))

    def test_paired_reference_survives_random_B(self):
        first=self.dataset[7]
        for _ in range(4):
            other=self.dataset[7]
            self.assertTrue(torch.equal(first['paired_B'],other['paired_B']))
            self.assertTrue(torch.equal(other['bbox'],other['paired_bbox']))
        real_ref,_=self.dataset.crop(self.dataset.rows[7],'ihc')
        self.assertTrue(torch.equal(first['paired_B'],real_ref))

    def test_numpy_torch_measurement_agree(self):
        s=StainStatistics()
        for i in range(0,len(self.dataset),5):
            sample=self.dataset[i];q,valid=s(sample['paired_B'][None],sample['tissue_mask'][None])
            self.assertTrue(bool(valid.all()));self.assertAlmostEqual(float(q),self.dataset.rows[i]['q'],places=4)

    def test_empty_tissue_is_invalid(self):
        q,valid=StainStatistics()(torch.zeros(1,3,128,128),torch.zeros(1,1,128,128))
        self.assertFalse(bool(valid.any()));self.assertTrue(bool(torch.isfinite(q).all()))

    def test_teacher_cannot_change_distribution(self):
        model=ExpressionCUTModel(self.opt);g=model.netG.eval();a=self.dataset[0]['A'][None]
        with torch.no_grad():
            lo,auxlo=g(a,condition=torch.full((1,1,1,1),.1),return_aux=True)
            hi,auxhi=g(a,condition=torch.full((1,1,1,1),.9),return_aux=True)
            plain=g.backbone(a)
        self.assertTrue(torch.equal(auxlo['logits'],auxhi['logits']))
        self.assertTrue(torch.equal(auxlo['pixel_logits'],auxhi['pixel_logits']))
        self.assertTrue(torch.equal(auxlo['expected_condition'],auxhi['expected_condition']))
        self.assertTrue(torch.allclose(lo,plain,atol=1e-6))
        self.assertTrue(torch.allclose(auxlo['probabilities'].sum(1),torch.ones(1,1,1)))

    def test_all_new_modules_receive_real_image_gradients(self):
        model=ExpressionCUTModel(self.opt);batch=next(iter(DataLoader(self.dataset,batch_size=2)))
        before=model.netG.film[-1].weight.detach().clone()
        model.data_dependent_initialize(batch);model.set_input(batch);model.optimize_parameters()
        for name,module in [('distribution',model.netG.distribution_head),('within_bin',model.netG.within_bin_head),('film',model.netG.film),
                            ('decoder_film',model.netG.decoder_film),
                            ('pixel_proxy',model.netG.pixel_head),('backbone',model.netG.backbone),('NCE_projector',model.netF)]:
            gradients=[p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients,name);self.assertTrue(all(torch.isfinite(g).all() for g in gradients),name)
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients),0,name)
        self.assertFalse(torch.equal(before,model.netG.film[-1].weight))

    def test_default_width64_six_blocks_forward_and_backward(self):
        opt=make_options(MANIFEST,'artifacts/test64','expression_cut','demo',ngf=64,batch_size=1)
        model=ExpressionCUTModel(opt)
        self.assertEqual(sum(isinstance(m,ResnetBlock) for m in model.netG.backbone.model),6)
        sample=self.dataset[0];out,aux=model.netG(sample['A'][None],return_aux=True)
        loss=out.square().mean()+aux['logits'].square().mean();loss.backward()
        self.assertEqual(tuple(out.shape),(1,3,128,128));self.assertEqual(tuple(aux['probabilities'].shape),(1,5,1,1))
        self.assertTrue(torch.isfinite(model.netG.backbone.model[1].weight.grad).all())

    def test_error_head_interface_needs_no_IHC(self):
        h=ErrorPredictor();a=self.dataset[0]['A'][None];fake=torch.zeros_like(a);variance=torch.zeros_like(a[:,:1])
        logits=h(a,fake,variance);logits.sum().backward()
        self.assertEqual(tuple(logits.shape),(1,1,1,1));self.assertTrue(torch.isfinite(logits).all())

    def test_same_image_content_cannot_cross_splits(self):
        a=copy.deepcopy(self.rows[0]);b=copy.deepcopy(self.rows[1])
        b.update(group_id='different_name',pair_id='different_pair',split='test')
        with self.assertRaises(ValueError):check_leakage([a,b])
        b['split']=a['split']
        with self.assertRaises(ValueError):check_leakage([a,b])

    def test_manifest_rejects_changed_image_content(self):
        row=copy.deepcopy(self.rows[0]);row['he_sha256']='0'*64
        with tempfile.TemporaryDirectory() as directory:
            manifest=Path(directory)/'invalid.jsonl';manifest.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'content changed'):read_manifest(manifest)

    def test_one_source_pair_cannot_become_many_calibration_groups(self):
        rows=[dict(r,score=.2,event=0) for r in self.rows]
        self.assertEqual(len(representative_per_group(rows)),1)
        with self.assertRaises(ValueError):
            fit_and_certify(rows,rows,dict(training_groups=[],generator_hash='g',risk_hash='h'))
        allowed,reason=gate([.01,.9],None,'g','h')
        self.assertFalse(allowed.any());self.assertEqual(reason,'missing_certificate')


class StatisticalUnitFixtures(unittest.TestCase):
    """These generated counts validate software, not the biological experiment."""
    def test_certificate_bound_and_checkpoint_binding(self):
        def fixture(split):
            return [dict(group_id=f'{split}_{i}',group_kind='patient',split=split,
                         window_id=f'{split}_{i}:R1C1',score=i/100,event=0) for i in range(100)]
        cert=fit_and_certify(fixture('calibrate'),fixture('certify'),
            dict(training_groups=['train'],generator_hash='g',risk_hash='h',protocol_hash=policy_hash()))
        self.assertTrue(cert['release_enabled']);self.assertLessEqual(cert['chosen']['upper'],.1)
        allowed,reason=gate([.1,.9],cert,'stale_generator','h')
        self.assertFalse(allowed.any());self.assertEqual(reason,'checkpoint_mismatch')
        bad=fixture('certify');bad[0]['group_id']='calibrate_0'
        with self.assertRaises(ValueError):fit_and_certify(fixture('calibrate'),bad,dict(training_groups=[]))


if __name__=='__main__':unittest.main()
