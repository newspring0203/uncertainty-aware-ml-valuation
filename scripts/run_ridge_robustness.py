# -*- coding: utf-8 -*-
"""Fixed development ablation: reuse six folds; do not select a final model."""
from pathlib import Path
import runpy,csv,json,math,statistics,hashlib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.linear_model import Ridge
HERE=Path(__file__).resolve().parent
OUT=HERE/'ridge_robustness_20261106'
M=runpy.run_path(str(HERE/'run_matched_cross_section.py'))
CANDIDATES={'Ridge_alpha1':(1.,False),'Ridge_alpha10':(10.,False),'Ridge_alpha100':(100.,False),'Ridge_clip5_alpha0.1':(.1,True)}
def clip_numeric(z):
 a=z.copy();a[:,:7]=np.clip(a[:,:7],-5,5);return a

def main():
 if OUT.exists():raise SystemExit('기존 실험 보존: 출력 폴더가 이미 있습니다.')
 OUT.mkdir()
 original=HERE/'matched_cross_section_20261106'
 batches=[original]+sorted((HERE/'split_stability_20261106').glob('partition_*'))
 plan={'candidates':CANDIDATES,'partitions':list(map(str,batches)),'selection':'none; four fixed candidates compared as exploratory ablations','clipping':'numeric features only, ±5 after train-only imputation and scaling; labels unchanged','HGB_and_baselines':'reuse saved matched predictions; not refit','final_data_used':False}
 (OUT/'PROTOCOL.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
 vintage={M['key'](r):r for r in M['read'](M['SOURCES']['vintage'])}
 shares={M['key'](r):r for r in json.loads(M['SOURCES']['shares'].read_text())}
 rows=[r for r in M['read'](M['SOURCES']['panel']) if vintage[M['key'](r)]['status']=='PASS']
 groups=sorted({r['market']+'/'+r['industry_2019_unverified'] for r in rows})
 for r in rows:
  a=M['numeric'](r['assets_krw']);p=M['numeric'](r['parent_equity_krw']);s=M['numeric'](r['sales_krw']);profit=M['numeric'](r['parent_profit_krw']);g=r['market']+'/'+r['industry_2019_unverified']
  def ratio(x,y):return x/y if math.isfinite(y) and y>0 else np.nan
  r['X']=[math.log(a) if a>0 else np.nan,ratio(M['numeric'](r['liabilities_krw']),a),ratio(p,a),ratio(M['numeric'](r['cash_krw']),a),ratio(s,a),ratio(profit,s),ratio(profit,p)]+[float(g==x) for x in groups]
  r['target']=float(r['market_cap_krw'])/p
  cs=shares[M['key'](r)].get('classes',[])[:-1];c=[x for x in cs if '보통주' in M['norm'](x['label'])]
  r['screen']=len(c)==1 and all(x['issued']==0 for x in cs if x not in c)
 # Check clipping preserves nonnumeric cohort dummies and input array.
 z=np.array([[20.]*7+[1.,0.,0.,0.]]);zz=clip_numeric(z)
 assert np.all(z[:,:7]==20) and np.all(zz[:,:7]==5) and np.array_equal(z[:,7:],zz[:,7:])
 results=[];summaries=[]
 for partition,folder in enumerate(batches):
  folds={r['firm_id']:int(r['fold']) for r in M['read'](folder/'FIRM_FOLDS.csv')}
  predictions=M['read'](folder/'PREDICTIONS.csv');lookup={(r['subset'],r['firm_id'],r['valuation_date']):r for r in predictions}
  for subset in ['all','single_class_screen']:
   sample=rows if subset=='all' else [r for r in rows if r['screen']]
   predictions_local=[]
   for date in sorted({r['valuation_date'] for r in sample}):
    rr=[r for r in sample if r['valuation_date']==date]
    for fold in range(5):
     train=[r for r in rr if folds[r['firm_id']]!=fold];test=[r for r in rr if folds[r['firm_id']]==fold]
     if not test:continue
     assert not {r['firm_id'] for r in train}&{r['firm_id'] for r in test}
     X=np.array([r['X'] for r in train]);y=np.log([r['target'] for r in train]);xt=np.array([r['X'] for r in test])
     values={}
     for name,(alpha,clip) in CANDIDATES.items():
      steps=[SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()]
      if clip:steps.append(FunctionTransformer(clip_numeric,validate=False))
      steps.append(Ridge(alpha=alpha));model=make_pipeline(*steps).fit(X,y)
      values[name]=np.exp(model.predict(xt));assert np.isfinite(values[name]).all() and (values[name]>0).all()
     for i,r in enumerate(test):
      prev=lookup[subset,r['firm_id'],date]
      assert math.isclose(r['target'],float(prev['actual']),rel_tol=1e-12)
      record=dict(partition=partition,subset=subset,firm_id=r['firm_id'],valuation_date=date,actual=r['target'],Ridge_original=float(prev['Ridge']),HGB=float(prev['HGB']),Industry=float(prev['Industry']),Global=float(prev['Global']))
      record.update({name:float(v[i]) for name,v in values.items()});predictions_local.append(record)
   assert len(predictions_local)==len(sample)
   scores={}
   for name in ['Ridge_original','HGB','Industry','Global']+list(CANDIDATES):
    es=[abs(r[name]-r['actual']) for r in predictions_local];logs=[abs(math.log(r[name]/r['actual'])) for r in predictions_local]
    scores[name]={'mae':statistics.mean(es),'medae':statistics.median(es),'mean_absolute_log_ratio':statistics.mean(logs)}
   summaries.append(dict(partition=partition,subset=subset,n=len(sample),scores=scores));results.extend(predictions_local)
  print('분할',partition+1,'/6 후보 비교 완료',flush=True)
 with (OUT/'PREDICTIONS.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
 (OUT/'PARTITION_SCORES.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2))
 aggregate={}
 for subset in ['all','single_class_screen']:
  ss=[r for r in summaries if r['subset']==subset];out={}
  for model in ss[0]['scores']:
   es=[r['scores'][model]['mae'] for r in ss]
   out[model]={'min_mae':min(es),'median_mae':statistics.median(es),'max_mae':max(es),'lower_mae_than_original_ridge':sum(r['scores'][model]['mae']<r['scores']['Ridge_original']['mae'] for r in ss),'median_log_error':statistics.median(r['scores'][model]['mean_absolute_log_ratio'] for r in ss)}
  aggregate[subset]=out
 (OUT/'SUMMARY.json').write_text(json.dumps(aggregate,ensure_ascii=False,indent=2))
 print('저장:',OUT)
if __name__=='__main__':main()
