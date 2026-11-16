import unittest
import torch
from search_r1_team.algorithms import group_advantages,clipped_policy_loss,reference_kl,gae,clipped_value_loss,prepare_batch,update_batch
from search_r1_team.config import Config
from search_r1_team.rollout import Trajectory
from toy_backend import TinyAdapterPolicy

torch.set_num_threads(1)

class TestAlgorithms(unittest.TestCase):
    def test_group_center_and_scale(self):
        a=group_advantages(torch.tensor([0.,1.,1.,1.]),2)
        self.assertTrue(torch.allclose(a,torch.tensor([-1.,1.,0.,0.]),atol=1e-5))
    def test_bad_groups(self):
        with self.assertRaises(ValueError): group_advantages(torch.ones(3),2)
    def test_gae_known_return(self):
        a,r=gae(torch.tensor([0.,0.,1.]),torch.zeros(3),gamma=1,lam=1)
        self.assertTrue(torch.equal(r,torch.ones(3))); self.assertTrue(torch.equal(a,r))
    def test_gae_uses_value(self):
        a,r=gae(torch.tensor([0.,1.]),torch.tensor([.2,.4]),gamma=1,lam=1)
        self.assertTrue(torch.allclose(a,torch.tensor([.8,.6])))
    def test_policy_clip_positive_and_negative(self):
        loss=clipped_policy_loss(torch.log(torch.tensor([2.,.5])),torch.zeros(2),torch.tensor([1.,-1.]),.2)
        self.assertAlmostEqual(float(loss),-.2,places=5)
    def test_reference_kl_zero_at_reference(self):
        x=torch.tensor([-2.,-3.]); self.assertEqual(float(reference_kl(x,x)),0.)
    def test_reference_kl_nonnegative(self):
        self.assertGreaterEqual(float(reference_kl(torch.tensor([-2.,-4.]),torch.tensor([-3.,-2.]))),0)
    def test_value_clipping(self):
        self.assertAlmostEqual(float(clipped_value_loss(torch.tensor([2.]),torch.tensor([0.]),torch.tensor([1.]),.2)),.5)
    def test_shape_guard(self):
        with self.assertRaises(ValueError): clipped_policy_loss(torch.ones(2),torch.ones(1),torch.ones(2),.2)
    def test_grpo_updates_only_actor_adapter(self):
        self._check_update('grpo')
    def test_ppo_updates_actor_and_critic(self):
        self._check_update('ppo')
    def _check_update(self,algorithm):
        p=TinyAdapterPolicy(ppo=algorithm=='ppo')
        tr=[Trajectory([10,11,12,13],[0,1,0,1],'synthetic'),Trajectory([10,14,15,16],[0,1,0,1],'synthetic')]
        cfg=Config(algorithm=algorithm,group_size=2,learning_rate=.01,critic_learning_rate=.01)
        base=[x.clone() for x in p.frozen_parameters()]; old_b=p.b.clone(); old_v=p.v.weight.clone()
        batch=prepare_batch(p,tr,[0.,1.],cfg)
        ao=torch.optim.AdamW(p.actor_parameters(),lr=.01)
        co=torch.optim.AdamW(p.critic_parameters(),lr=.01) if algorithm=='ppo' else None
        logs=update_batch(p,batch,cfg,ao,co)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(base,p.frozen_parameters())))
        self.assertFalse(torch.equal(old_b,p.b))
        if algorithm=='ppo': self.assertFalse(torch.equal(old_v,p.v.weight))
        self.assertTrue(all(torch.isfinite(torch.tensor(x)) for x in logs.values()))
        self.assertEqual(batch[0].old_logprobs.numel(),2) # retrieval token never has an action loss
    def test_ppo_warmup_freezes_actor(self):
        p=TinyAdapterPolicy(ppo=True); cfg=Config(algorithm='ppo')
        trs=[Trajectory([1,2,3],[0,1,1],'x'),Trajectory([1,3,4],[0,1,1],'y')]
        batch=prepare_batch(p,trs,[0,1],cfg); before=p.b.clone()
        update_batch(p,batch,cfg,torch.optim.Adam(p.actor_parameters()),torch.optim.Adam(p.critic_parameters()),actor_update=False)
        self.assertTrue(torch.equal(before,p.b))
    def test_reference_unchanged_after_actor_update(self):
        p=TinyAdapterPolicy(); tr=Trajectory([1,2,3],[0,1,1],'x')
        ref=p.logprobs(tr,reference=True).clone()
        with torch.no_grad(): p.b.add_(1)
        self.assertTrue(torch.equal(ref,p.logprobs(tr,reference=True)))
