# -*- coding: utf-8 -*-
"""Five predetermined alternative issuer partitions; no parameter tuning."""
from pathlib import Path
import json,hashlib,contextlib,io,statistics
HERE=Path(__file__).resolve().parent
OUT=HERE/'split_stability_20261106'
SALTS=[f'MATCHED-CS-STABILITY-{x}' for x in [20261107,20261108,20261109,20261110,20261111]]
def main():
 if OUT.exists():raise SystemExit('기존 결과 보존: 출력 폴더가 이미 있습니다.')
 OUT.mkdir()
 source=(HERE/'run_matched_cross_section.py').read_text()
 (OUT/'SOURCE_SNAPSHOT.py').write_text(source)
 (OUT/'PROTOCOL.json').write_text(json.dumps({'salts':SALTS,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'parameters':'unchanged from matched cross-section','purpose':'partition sensitivity, not independent samples or confidence interval','prespecified_before_runs':True,'number_of_additional_partitions':5,'final_2023_2025_used':False},ensure_ascii=False,indent=2))
 summaries=[]
 # Execute the same algorithm with only fold salt/output directory substituted.
 for i,salt in enumerate(SALTS):
  dest=OUT/('partition_'+str(i+1))
  text=source.replace("OUT=HERE/'matched_cross_section_20261106'",'OUT=Path('+repr(str(dest))+')').replace('MATCHED-CS-20261106|',salt+'|')
  namespace={'__name__':'stability_partition','__file__':str(HERE/'run_matched_cross_section.py')}
  exec(compile(text,'matched_cross_section_partition','exec'),namespace)
  with contextlib.redirect_stdout(io.StringIO()):namespace['main']()
  s=json.loads((dest/'SUMMARY.json').read_text());summaries.append(s)
  print(f'분할 {i+1}/5 완료: 전체 HGB MAE={s["all"]["scores"]["HGB"]["mae"]:.4f}; 선별 HGB MAE={s["single_class_screen"]["scores"]["HGB"]["mae"]:.4f}',flush=True)
 baseline=json.loads((HERE/'matched_cross_section_20261106/SUMMARY.json').read_text())
 allruns=[baseline]+summaries;result={}
 for subset in ['all','single_class_screen']:
  info={}
  for model in ['Industry','Global','Ridge','HGB']:
   values=[r[subset]['scores'][model]['mae'] for r in allruns]
   info[model]={'min_mae':min(values),'median_mae':statistics.median(values),'max_mae':max(values),'values':values}
  for comparator in ['Industry','Global']:
   info['HGB_lower_MAE_than_'+comparator]=sum(r[subset]['scores']['HGB']['mae']<r[subset]['scores'][comparator]['mae'] for r in allruns)
  result[subset]=info
 (OUT/'SUMMARY.json').write_text(json.dumps({'partitions_including_original':6,'results':result,'note':'Six dependent reuses of the same observations, not six independent experiments. Win counts are descriptive, not p-values.'},ensure_ascii=False,indent=2))
 print('저장:',OUT)
if __name__=='__main__':main()
