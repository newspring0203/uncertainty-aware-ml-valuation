# -*- coding: utf-8 -*-
"""Exploratory same-date, issuer-separated fixed-parameter comparison."""
from pathlib import Path
import json,csv,hashlib,math,statistics
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
import sklearn
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
OUT=HERE/'matched_cross_section_20261106'
SOURCES={
 'panel':ROOT/'data/input/panel.csv',
 'vintage':ROOT/'data/input/filing_vintage.csv',
 'shares':ROOT/'data/input/share_tables.json'}
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def key(r):return r['firm_id'],r['valuation_date']
def norm(s):return ''.join(s.split())
def numeric(v):
 try:x=float(v);return x if math.isfinite(x) else np.nan
 except (TypeError,ValueError):return np.nan

def main():
 if OUT.exists():raise SystemExit('기존 결과 보존: 출력 폴더가 이미 있습니다.')
 OUT.mkdir()
 vintage={key(r):r for r in read(SOURCES['vintage'])};shares={key(r):r for r in json.loads(SOURCES['shares'].read_text())}
 rows=[r for r in read(SOURCES['panel']) if vintage[key(r)]['status']=='PASS']
 assert len(rows)==161
 groups=sorted({r['market']+'/'+r['industry_2019_unverified'] for r in rows})
 for r in rows:
  r['cohort']=r['market']+'/'+r['industry_2019_unverified']
  a=numeric(r['assets_krw']);p=numeric(r['parent_equity_krw']);sales=numeric(r['sales_krw']);profit=numeric(r['parent_profit_krw'])
  def ratio(x,y):return x/y if math.isfinite(y) and y>0 else np.nan
  r['features']=[math.log(a) if a>0 else np.nan,ratio(numeric(r['liabilities_krw']),a),ratio(p,a),ratio(numeric(r['cash_krw']),a),ratio(sales,a),ratio(profit,sales),ratio(profit,p)]+[float(r['cohort']==g) for g in groups]
  cap=numeric(r['market_cap_krw']);r['target']=cap/p
  assert r['target']>0 and math.isfinite(r['target'])
  cs=shares[key(r)].get('classes',[])[:-1];common=[x for x in cs if '보통주' in norm(x['label'])]
  r['single_class_screen']=len(common)==1 and all(c['issued']==0 for c in cs if c not in common)
 # Deterministic stratification by fixed historical cohort, grouped by issuer across all years.
 firmgroups={}
 for r in rows:
  if r['firm_id'] in firmgroups:assert firmgroups[r['firm_id']]==r['cohort']
  firmgroups[r['firm_id']]=r['cohort']
 folds={}
 for g in groups:
  ids=sorted((f for f,c in firmgroups.items() if c==g),key=lambda f:hashlib.sha256(('MATCHED-CS-20261106|'+f).encode()).hexdigest())
  for i,f in enumerate(ids):folds[f]=i%5
 protocol={'status':'exploratory fixed protocol written before this run; 2022 already inspected',
 'target':'KRX common listed market cap / prior-year parent book proxy',
 'split':'5 fixed issuer folds, stratified within historical cohort; same issuer role across years',
 'information_budget':'each fit and both median baselines use exactly the same other-fold issuers at the same valuation date; no other years pooled',
 'parameters':{'Ridge':{'alpha':.1},'HGB':{'max_leaf_nodes':3,'min_samples_leaf':5,'l2_regularization':1.,'learning_rate':.05,'max_iter':100,'early_stopping':False,'random_state':20261106}},
 'target_transform':'Ridge/HGB fit log target, predict exp; baselines median raw target',
 'subsets':['all B02 PASS','same-date reference and evaluation rows both restricted to single-class screen'],
 'screen':'one explicit common class, all other reported issued quantities zero (inferred zero included with provenance); no claim of full rights certification',
 'industry_fallback':'if fewer than 3 same-cohort training rows use global training median; count fallbacks',
 'source_hashes':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in SOURCES.items()},'sklearn_version':sklearn.__version__}
 (OUT/'PROTOCOL.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
 result=[]
 for subset in ['all','single_class_screen']:
  sample=rows if subset=='all' else [r for r in rows if r['single_class_screen']]
  for date in sorted({r['valuation_date'] for r in sample}):
   yearrows=[r for r in sample if r['valuation_date']==date]
   for fold in range(5):
    train=[r for r in yearrows if folds[r['firm_id']]!=fold];test=[r for r in yearrows if folds[r['firm_id']]==fold]
    if not test:continue
    assert not {r['firm_id'] for r in train}&{r['firm_id'] for r in test}
    assert len(train)>=5
    X=np.array([r['features'] for r in train]);y=np.log([r['target'] for r in train]);Xt=np.array([r['features'] for r in test])
    ridge=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=.1))
    hgb=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),HistGradientBoostingRegressor(max_leaf_nodes=3,min_samples_leaf=5,l2_regularization=1.,learning_rate=.05,max_iter=100,early_stopping=False,random_state=20261106))
    ridge.fit(X,y);hgb.fit(X,y);rp=np.exp(ridge.predict(Xt));hp=np.exp(hgb.predict(Xt))
    globalmedian=statistics.median(r['target'] for r in train)
    for i,r in enumerate(test):
     peers=[a['target'] for a in train if a['cohort']==r['cohort']];fallback=len(peers)<3
     result.append(dict(subset=subset,firm_id=r['firm_id'],valuation_date=date,fold=fold,cohort=r['cohort'],actual=r['target'],Ridge=float(rp[i]),HGB=float(hp[i]),Industry=globalmedian if fallback else statistics.median(peers),Global=globalmedian,train_n=len(train),industry_peers=len(peers),industry_fallback=fallback))
  print('완료:',subset,len(sample),'개 관측치',flush=True)
 def score(rr):
  out={}
  for model in ['Industry','Global','Ridge','HGB']:
   es=[abs(r[model]-r['actual']) for r in rr];logs=[abs(math.log(r[model]/r['actual'])) for r in rr]
   out[model]={'n':len(rr),'mae':statistics.mean(es),'medae':statistics.median(es),'mean_absolute_log_ratio':statistics.mean(logs)}
  return out
 summary={}
 for subset in ['all','single_class_screen']:
  rr=[r for r in result if r['subset']==subset]
  assert len({(r['firm_id'],r['valuation_date']) for r in rr})==len(rr)
  summary[subset]={'observations':len(rr),'firms':len({r['firm_id'] for r in rr}),'scores':score(rr),'industry_fallbacks':sum(r['industry_fallback'] for r in rr),'by_year':{year:score([r for r in rr if r['valuation_date'][:4]==year]) for year in sorted({r['valuation_date'][:4] for r in rr})}}
 # Same evaluated screened rows under the all-sample fit: isolate composition from refit effects.
 selected={key(r) for r in rows if r['single_class_screen']}
 summary['all_fits_evaluated_on_screen_only']=score([r for r in result if r['subset']=='all' and key(r) in selected])
 with (OUT/'PREDICTIONS.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
 (OUT/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 with (OUT/'FIRM_FOLDS.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['firm_id','cohort','fold']);w.writerows((firm,firmgroups[firm],folds[firm]) for firm in sorted(folds))
 for s in ['all','single_class_screen']:
  print(s,json.dumps(summary[s]['scores'],ensure_ascii=False))
 print('저장:',OUT)
if __name__=='__main__':main()
