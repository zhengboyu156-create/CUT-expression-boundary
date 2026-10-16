"""Independent regression checks; synthetic fixtures are never biological evidence."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.stats import beta
from expression.challenge import retrieve,fit_morphology
from expression.calibration import fit_and_certify,gate
from expression.manifest import read_manifest
from expression.protocol import policy_hash

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/multipair_debug'


class CompleteEvidenceTests(unittest.TestCase):
    def test_radius_search_does_not_drop_dense_cluster_conflicts(self):
        rows=[]
        for i in range(40):
            rows.append(dict(window_id=str(i),valid=True,split='demo',group_id='synthetic',
                features=[0.],q=0.05 if i<20 else 0.45,
                he_bbox=[i*4,0,4,4],ihc_bbox=[i*4,0,4,4]))
        state=dict(mean=[0.],scale=[1.],centers=[[0.]],demo_only=True,
                   fit_groups=['synthetic'],distance_threshold=1.5,delta_threshold=.2)
        pairs=retrieve(rows,state,demo=True,neighbors=1,query_batch_size=3)
        self.assertEqual(len(pairs),400)
        self.assertEqual(len({tuple(sorted((p['a'],p['b']))) for p in pairs}),400)
        for p in pairs:
            self.assertEqual(p['he_a'],p['ihc_a']);self.assertEqual(p['he_b'],p['ihc_b'])

    def test_radius_search_matches_exhaustive_actual_data(self):
        rows=[r for r in read_manifest(BASE/'full_manifest.jsonl') if r['valid']]
        state=fit_morphology([r for r in rows if r['debug_partition']=='train'],demo=True)
        z=(np.asarray([r['features'] for r in rows])-state['mean'])/state['scale']
        labels=((z[:,None]-np.asarray(state['centers']))**2).sum(2).argmin(1)
        expected=set()
        for i in range(len(rows)):
            for j in range(i+1,len(rows)):
                if labels[i]==labels[j] and np.linalg.norm(z[i]-z[j])<=state['distance_threshold'] and abs(rows[i]['q']-rows[j]['q'])>=state['delta_threshold']:
                    expected.add(tuple(sorted((rows[i]['window_id'],rows[j]['window_id']))))
        actual={tuple(sorted((p['a'],p['b']))) for p in retrieve(rows,state,demo=True)}
        self.assertEqual(actual,expected);self.assertEqual(len(actual),106)

    def test_full_images_are_covered_once_before_quality_filter(self):
        rows=read_manifest(BASE/'full_manifest.jsonl')
        for pid in {r['pair_id'] for r in rows}:
            coverage=np.zeros((1024,1024),dtype=np.uint8)
            for row in [r for r in rows if r['pair_id']==pid]:
                self.assertEqual(row['he_bbox'],row['ihc_bbox'])
                x,y,w,h=row['he_bbox'];coverage[y:y+h,x:x+w]+=1
            self.assertTrue(np.all(coverage==1),pid)

    def test_formal_challenge_refuses_demo_representation(self):
        rows=[r for r in read_manifest(BASE/'full_manifest.jsonl') if r['valid']]
        state=fit_morphology([r for r in rows if r['debug_partition']=='train'],demo=True)
        with self.assertRaises(ValueError):retrieve(rows,state)

    def test_actual_roles_cannot_be_renamed_into_independent_calibration(self):
        rows=[r for r in read_manifest(BASE/'full_manifest.jsonl') if r['valid']]
        a=[dict(r,score=.1,event=0,split='calibrate') for r in rows if r['debug_partition']=='dev']
        b=[dict(r,score=.1,event=0,split='certify') for r in rows if r['debug_partition']=='holdout']
        with self.assertRaisesRegex(ValueError,'disjoint'):
            fit_and_certify(a,b,dict(training_groups=[],generator_hash='g',risk_hash='h',protocol_hash=policy_hash()))

    def test_zero_error_binomial_upper_matches_closed_form(self):
        alpha=.05;count=7;budget=.1
        for n in [1,2,10,46,47,100]:
            exact=1-(alpha/count)**(1/n)
            self.assertAlmostEqual(float(beta.ppf(1-alpha/count,1,n)),exact,places=12)
        self.assertGreater(1-(alpha/count)**(1/46),budget)
        self.assertLessEqual(1-(alpha/count)**(1/47),budget)

    def test_no_or_failing_certificate_never_releases(self):
        scores=[.0,.01,.2,1.]
        accepted,reason=gate(scores,None,'g','h')
        self.assertFalse(accepted.any());self.assertEqual(reason,'missing_certificate')
        def fixture(role):
            return [dict(group_id=f'synthetic_{role}_{i}',group_kind='patient',split=role,
                         window_id=str(i),score=.3,event=1) for i in range(100)]
        cert=fit_and_certify(fixture('calibrate'),fixture('certify'),
             dict(training_groups=[],generator_hash='g',risk_hash='h',protocol_hash=policy_hash()))
        self.assertIsNone(cert['chosen']);self.assertFalse(gate(scores,cert,'g','h')[0].any())

    def test_invalid_certificate_release_flag_does_not_override_risk_bound(self):
        cert=dict(provenance=dict(generator_hash='g',risk_hash='h',protocol_hash=policy_hash()),
                  chosen=dict(threshold=1.,upper=.5),budget=.1,release_enabled=True,
                  calibrator=dict(x=[0.,1.],y=[0.,1.]))
        allowed,reason=gate([0.],cert,'g','h')
        self.assertFalse(allowed.any());self.assertEqual(reason,'invalid_certificate_policy')


if __name__=='__main__':unittest.main()
