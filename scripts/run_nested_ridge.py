# -*- coding: utf-8 -*-
"""Nested issuer CV: choose Ridge only inside each same-date outer training sample."""
from pathlib import Path
import csv,json,math,statistics,hashlib,runpy
from collections import Counter
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.linear_model import Ridge
HERE=Path(__file__).resolve().parent
OUT=HERE/'nested_ridge_20261106'
M=runpy.run_path(str(HERE/'run_matched_cross_section.py'))
R=runpy.run_path(str(HERE/'run_ridge_robustness.py'))
CANDIDATES={'alpha0.1':(.1,False),'alpha1':(1.,False),'alpha10':(10.,False),'alpha100':(100.,False),'clip5_alpha0.1':(.1,True)}
def model_for(name):
 alpha,clip=CANDIDATES[name];steps=[SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()]
 if clip:steps.append(FunctionTransformer(R['clip_numeric'],validate=False))
 return make_pipeline(*steps,Ridge(alpha=alpha))
def prepare():
 vintage={M['key'](r):r for r in M['read'](M['SOURCES']['vintage'])};shares={M['key'](r):r for r in json.loads(M['SOURCES']['shares'].read_text())}
 rows=[r for r in M['read'](M['SOURCES']['panel']) if vintage[M['key'](r)]['status']=='PASS'];groups=sorted({r['market']+'/'+r['industry_2019_unverified'] for r in rows})
 for r in rows:
  a=M['numeric'](r['assets_krw']);p=M['numeric'](r['parent_equity_krw']);s=M['numeric'](r['sales_krw']);profit=M['numeric'](r['parent_profit_krw']);g=r['market']+'/'+r['industry_2019_unverified'];r['cohort']=g
  def ratio(x,y):return x/y if math.isfinite(y) and y>0 else np.nan
  r['X']=[math.log(a) if a>0 else np.nan,ratio(M['numeric'](r['liabilities_krw']),a),ratio(p,a),ratio(M['numeric'](r['cash_krw']),a),ratio(s,a),ratio(profit,s),ratio(profit,p)]+[float(g==x) for x in groups]
  r['target']=float(r['market_cap_krw'])/p
  cs=shares[M['key'](r)].get('classes',[])[:-1];c=[x for x in cs if '보통주' in M['norm'](x['label'])];r['screen']=len(c)==1 and all(x['issued']==0 for x in cs if x not in c)
 return rows

def select_candidate(train,partition,outerfold):
 # This function has no test rows or test targets in its interface.
 inner={}
 for g in sorted({r['cohort'] for r in train}):
  firms=sorted({r['firm_id'] for r in train if r['cohort']==g},key=lambda f:hashlib.sha256((f'NESTED-INNER|{partition}|{outerfold}|'+f).encode()).hexdigest())
  for i,firm in enumerate(firms):inner[firm]=i%3
 scores=[]
 for name in CANDIDATES:
  logs=[];maes=[]
  for fold in range(3):
   a=[r for r in train if inner[r['firm_id']]!=fold];b=[r for r in train if inner[r['firm_id']]==fold]
   if not b:continue
   assert len(a)>=3 and not {r['firm_id'] for r in a}&{r['firm_id'] for r in b}
   fit=model_for(name).fit(np.array([r['X'] for r in a]),np.log([r['target'] for r in a]))
   # Score in log space to avoid exp overflow affecting selection.
   predictions=fit.predict(np.array([r['X'] for r in b]))
   logs.extend(abs(float(z)-math.log(r['target'])) for z,r in zip(predictions,b))
   for z,r in zip(predictions,b):maes.append(abs(math.exp(float(z))-r['target']) if z<700 else float('inf'))
  assert len(logs)==len(train)
  scores.append(dict(candidate=name,inner_n=len(logs),inner_mean_absolute_log_ratio=statistics.mean(logs),inner_mae=statistics.mean(maes)))
 # Predeclared order is tie-breaker. Inner MAE is diagnostic, not selection objective.
 chosen=min(scores,key=lambda r:r['inner_mean_absolute_log_ratio'])['candidate']
 return chosen,scores

def main():
 if OUT.exists():raise SystemExit('기존 결과 보존: 출력 폴더가 이미 있습니다.')
 OUT.mkdir();rows=prepare()
 batches=[HERE/'matched_cross_section_20261106']+sorted((HERE/'split_stability_20261106').glob('partition_*'))
 (OUT/'PROTOCOL.json').write_text(json.dumps({'candidates':CANDIDATES,'selection_metric':'pooled inner mean absolute log ratio','tie_break':'candidate insertion order','inner_folds':3,'outer_folds':5,'outer_partitions':6,'training_data':'same-date outer reference firms only; all imputation/scaling refit inside inner folds','comparators':'saved fixed HGB and median predictions on identical outer observations','limits':['2022 and development labels previously inspected; exploratory','Ridge receives inner tuning; HGB remains fixed and is not nested-tuned','single_class_screen is not certified common-equity scope','no independent final test or coverage claim'],'source_hashes':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in M['SOURCES'].items()},'final_2023_2025_used':False},ensure_ascii=False,indent=2))
 result=[];selection=[]
 # Selector must not mutate input or depend on external held-out targets.
 for partition,folder in enumerate(batches):
  folds={r['firm_id']:int(r['fold']) for r in M['read'](folder/'FIRM_FOLDS.csv')}
  old={(r['subset'],r['firm_id'],r['valuation_date']):r for r in M['read'](folder/'PREDICTIONS.csv')}
  for subset in ['all','single_class_screen']:
   sample=rows if subset=='all' else [r for r in rows if r['screen']]
   for date in sorted({r['valuation_date'] for r in sample}):
    yr=[r for r in sample if r['valuation_date']==date]
    for fold in range(5):
     train=[r for r in yr if folds[r['firm_id']]!=fold];test=[r for r in yr if folds[r['firm_id']]==fold]
     if not test:continue
     assert not {r['firm_id'] for r in train}&{r['firm_id'] for r in test}
     chosen,scores=select_candidate(train,partition,fold)
     for sc in scores:selection.append(dict(partition=partition,subset=subset,valuation_date=date,outer_fold=fold,chosen=chosen,selected=sc['candidate']==chosen,**sc))
     fit=model_for(chosen).fit(np.array([r['X'] for r in train]),np.log([r['target'] for r in train]))
     predictions=np.exp(fit.predict(np.array([r['X'] for r in test])))
     assert np.isfinite(predictions).all() and (predictions>0).all()
     for i,r in enumerate(test):
      prev=old[subset,r['firm_id'],date];assert math.isclose(r['target'],float(prev['actual']),rel_tol=1e-12)
      result.append(dict(partition=partition,subset=subset,firm_id=r['firm_id'],valuation_date=date,outer_fold=fold,selected=chosen,actual=r['target'],Nested_Ridge=float(predictions[i]),Ridge_original=float(prev['Ridge']),HGB=float(prev['HGB']),Industry=float(prev['Industry']),Global=float(prev['Global']),outer_train_n=len(train)))
  print('중첩 분할',partition+1,'/6 완료',flush=True)
 def score(rr):
  return {model:{'n':len(rr),'mae':statistics.mean(abs(r[model]-r['actual']) for r in rr),'medae':statistics.median(abs(r[model]-r['actual']) for r in rr),'mean_absolute_log_ratio':statistics.mean(abs(math.log(r[model]/r['actual'])) for r in rr)} for model in ['Nested_Ridge','Ridge_original','HGB','Industry','Global']}
 per=[];aggregate={}
 for subset in ['all','single_class_screen']:
  ss=[]
  for partition in range(6):
   rr=[r for r in result if r['subset']==subset and r['partition']==partition]
   assert len(rr)==(161 if subset=='all' else 134) and len({(r['firm_id'],r['valuation_date']) for r in rr})==len(rr)
   item={'subset':subset,'partition':partition,'scores':score(rr)};ss.append(item);per.append(item)
  counts=Counter(r['chosen'] for r in selection if r['subset']==subset and r['selected'])
  aggregate[subset]={'selection_counts':dict(counts),'models':{m:{'median_mae':statistics.median(x['scores'][m]['mae'] for x in ss),'min_mae':min(x['scores'][m]['mae'] for x in ss),'max_mae':max(x['scores'][m]['mae'] for x in ss),'median_log_error':statistics.median(x['scores'][m]['mean_absolute_log_ratio'] for x in ss)} for m in ss[0]['scores']},'nested_lower_mae_than':{m:sum(x['scores']['Nested_Ridge']['mae']<x['scores'][m]['mae'] for x in ss) for m in ['Ridge_original','HGB','Industry','Global']}}
 for filename,rr in [('PREDICTIONS.csv',result),('INNER_SELECTION.csv',selection)]:
  with (OUT/filename).open('w',encoding='utf-8-sig',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 (OUT/'PARTITION_SCORES.json').write_text(json.dumps(per,ensure_ascii=False,indent=2));(OUT/'SUMMARY.json').write_text(json.dumps(aggregate,ensure_ascii=False,indent=2))
 print('저장:',OUT)
if __name__=='__main__':main()
