"""Small pure-Python math/control tests; no model or worker launch."""
import json
import math
import unittest
from critical_batch import Controller, Q, noise_moments, window


class NoiseTests(unittest.TestCase):
    def test_equal_pair_matches_reference(self):
        a,b=[1.,2.,-1.],[3.,0.,1.]
        dot=sum(x*y for x,y in zip(a,b))
        local=(sum(x*x for x in a)+sum(x*x for x in b))/2
        mean=sum(((x+y)/2)**2 for x,y in zip(a,b))
        signal,noise=noise_moments(local,mean,20,2)
        self.assertAlmostEqual(signal,dot)
        self.assertAlmostEqual(noise,10*(local-dot))

    def test_unequal_size_noise_is_unbiased_in_expectation(self):
        sizes=[2,3,5,7,11,13,17,19];n=sum(sizes);g2=3.;sigma=7.
        # Exact expected norms under independent unit noise, NOT sampled data.
        local=sum(k/n*(g2+sigma/k) for k in sizes)
        mean=g2+sigma/n
        signal,noise=noise_moments(local,mean,n,len(sizes))
        self.assertAlmostEqual(signal,g2)
        self.assertAlmostEqual(noise,sigma)

    def test_ema_moments_not_average_of_ratios(self):
        c=Controller(beta=.5,minimum_observations=2)
        c.observe(-1.,9.,100.,allow_change=False)
        r=c.observe(3.,3.,100.,allow_change=False)
        self.assertAlmostEqual(r['noise_scale_target_tokens'],3.)
        self.assertAlmostEqual(r['noise_scale_sequence_equivalent'],.03)

    def test_invalid_measurement_does_not_change_state(self):
        c=Controller();before=c.state_dict()
        r=c.observe(float('nan'),1.,1.)
        self.assertFalse(r['valid']);self.assertEqual(before,c.state_dict())
        c.observe(-1.,1.,100.)
        self.assertEqual(c.units,1)

    def test_hysteresis_and_bounded_growth(self):
        c=Controller(fraction=1.,minimum_observations=1)
        for want in [2,4,8,8]:
            c.observe(1.,10000.,1.)
            self.assertEqual(c.units,want)
        c=Controller(fraction=1.,minimum_observations=1)
        c.observe(1.,23.,1.) # below guarded increase threshold27.15
        self.assertEqual(c.units,1)

    def test_shadow_does_not_act_and_state_roundtrip(self):
        c=Controller(mode='shadow',minimum_observations=1)
        for _ in range(12):c.observe(1.,10000.,1.)
        self.assertEqual(c.units,1)
        restored=Controller(**json.loads(json.dumps(c.state_dict())))
        self.assertEqual(c.observe(2.,30.,2.),restored.observe(2.,30.,2.))

    def test_warmup_freezes_actions(self):
        c=Controller(minimum_observations=1)
        c.observe(1.,10000.,1.,allow_change=False)
        self.assertEqual(c.units,1)

    def test_complete_token_coverage_and_event_alignment(self):
        events=[5*Q,9*Q,20*Q,31*Q];seen=step=0;visited=[]
        while seen<31*Q:
            row=window(seen,step,8,events,31*Q)
            self.assertEqual(row['start_tokens'],seen)
            self.assertLessEqual(row['end_tokens'],min(x for x in events if x>seen))
            self.assertEqual(row['width'],16*row['accumulation'])
            visited.extend(range(seen//8192,row['end_tokens']//8192))
            seen=row['end_tokens'];step=row['update']
        self.assertEqual(visited,list(range(31*16)))
        with self.assertRaises(ValueError):window(1,0,1,events,31*Q)


if __name__=='__main__':unittest.main()
