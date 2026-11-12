"""Boundary checks for paired full/incremental forecasting data."""
import unittest
import numpy as np
import pandas as pd
from prepare import MarketPanel
from prepare_trajectory_comparison import fit_scaling, align_pairs, align_labels, CHANNELS


class ComparisonDataTests(unittest.TestCase):
    def test_scaling_never_uses_validation_or_test(self):
        values = np.arange(6*10*3,dtype=float).reshape(6,10,3)
        split=np.array([0,0,1,1,2,2])
        mean,std=fit_scaling(values,split)
        changed=values.copy();changed[split!=0]=1e15
        new_mean,new_std=fit_scaling(changed,split)
        np.testing.assert_array_equal(mean,new_mean)
        np.testing.assert_array_equal(std,new_std)
        np.testing.assert_array_equal(mean,values[:2].mean((0,1)))
        np.testing.assert_array_equal(std,values[:2].std((0,1)))

    def test_reference_prestate_is_independent_of_future(self):
        panel=object.__new__(MarketPanel)
        panel.calendar=pd.bdate_range('2020-01-01',periods=150)
        panel.codes=np.array(['000001','600001']); panel.length=10; panel.expanded=True
        panel.industry_policy='update_proxy'; panel._last_state=None
        market=np.sin(np.arange(150)*.41)*.01
        panel.market=np.repeat(market[:,None],2,axis=1)
        panel.returns=.001+panel.market*np.array([.8,1.2])[None,:]+np.cos(np.arange(150)[:,None])*.002
        panel.turnover=np.arange(300,dtype=float).reshape(150,2)/100+1
        panel.log_volume=np.log(np.arange(300,dtype=float).reshape(150,2)+1000)
        class Attributes:
            def at(self,*args):return np.ones(2),np.array(['a','a'])
        panel.attributes=Attributes()
        day=110; before=panel.state(day)
        assert before.valid.all()
        x,alpha,beta=before.x.copy(),before.alpha.copy(),before.beta.copy()
        for name in ['returns','market','turnover','log_volume']:
            getattr(panel,name)[day:]=np.nan
        panel._last_state=None
        after=panel.state(day)
        np.testing.assert_array_equal(after.x,x)
        np.testing.assert_array_equal(after.alpha,alpha)
        np.testing.assert_array_equal(after.beta,beta)

    def test_single_control_identity_and_weight_required(self):
        events=pd.DataFrame({'event_id':['e1','e2'],'stock':['000001','000002'],'event_date':['2025-01-02']*2})
        pairs=pd.DataFrame({'event_id':['e2','e1'],'control':['600002','600001'],'weight':[1.,1.],
                            'selected_at':['2025-01-02']*2,'retained_for_model':[True]*2,'included_after_outcome_check':[True]*2})
        self.assertEqual(align_pairs(pairs,events).control.tolist(),['600001','600002'])
        with self.assertRaises(ValueError):align_pairs(pd.concat([pairs,pairs.iloc[:1]]),events)
        invalid=pairs.copy(); invalid.loc[0,'weight']=.5
        with self.assertRaises(ValueError):align_pairs(invalid,events)

    def test_raw_label_grid_and_availability(self):
        events=pd.DataFrame({'event_id':['e1','e2'],'event_date':['2025-01-02']*2,'end_date':['2025-01-16']*2})
        rows=[]
        for event in events.event_id:
            for tau in range(11):
                for channel in CHANNELS:
                    rows.append(dict(event_id=event,event_date='2025-01-02',tau=tau,channel=channel,
                                     y=float(tau+2),baseline=1.,delta=float(tau+1),label_version='v',target_available_at='2025-01-17'))
        frame=pd.DataFrame(rows).sample(frac=1,random_state=1)
        y,b,d=align_labels(frame,events,'v')
        np.testing.assert_array_equal(y-b,d)
        invalid=frame.copy();invalid.iloc[0,invalid.columns.get_loc('target_available_at')]='2025-01-16'
        with self.assertRaises(ValueError):align_labels(invalid,events,'v')
        with self.assertRaises(ValueError):align_labels(frame.iloc[:-1],events,'v')

if __name__=='__main__':unittest.main()
