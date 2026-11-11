from pathlib import Path
import requests,concurrent.futures,json,datetime,xml.etree.ElementTree as ET,re
ROOT=Path(__file__).resolve().parent
NS={'a':'http://www.w3.org/2005/Atom','o':'http://a9.com/-/spec/opensearch/1.1/'}
opd='(all:"on-policy distillation" OR all:"on-policy self-distillation" OR all:"on policy distillation")'
queries=[
('cross_family',opd+' AND (all:"cross-family" OR all:"cross family" OR all:"cross-model")',150),
('cross_tokenizer',opd+' AND (all:"cross-tokenizer" OR all:"tokenizer alignment" OR all:"vocabulary alignment")',150),
('steering',opd+' AND (all:steering OR all:steered OR all:"activation intervention")',150),
('representation',opd+' AND (all:"hidden state" OR all:"representation distillation" OR all:"latent distillation" OR all:"subspace")',100),
('compatibility',opd+' AND (all:compatibility OR all:compatible OR all:"distribution gap" OR all:"policy gap")',100),
('cold_start',opd+' AND (all:"warm-up" OR all:"cold start" OR all:"warmup")',100),
('new_week',opd+' AND submittedDate:[202611010000 TO 202611072359]',120),
('steer_distill_cross','(all:"distillation" AND (all:"steering vector" OR all:"activation steering") AND (all:"cross-family" OR all:"cross-model" OR all:"student"))',100),
('latent_intervention','(all:"distillation" AND (all:"activation intervention" OR all:"latent steering" OR all:"residual injection" OR all:"training-time steering"))',100),
('title_cross','(ti:"cross-tokenizer" OR ti:"cross-family" OR ti:"cross-model") AND (all:distillation)',100),
('title_steer','(ti:steering OR ti:steer OR ti:activation) AND (all:"on-policy" OR all:"self-distillation")',100),
]
def run(item):
 key,q,n=item;d={'key':key,'query':q,'requested_max':n,'retrieved_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 try:
  r=requests.get('https://export.arxiv.org/api/query',params={'search_query':q,'max_results':n,'sortBy':'submittedDate','sortOrder':'descending'},timeout=55)
  d.update(status=r.status_code,url=r.url);r.raise_for_status();(ROOT/(key+'.xml')).write_text(r.text)
  tree=ET.fromstring(r.text);d['total_results']=int(tree.findtext('o:totalResults','0',NS));records=[]
  for e in tree.findall('a:entry',NS):
   row={k:' '.join(e.findtext('a:'+k,'',NS).split()) for k in ['id','title','summary','published','updated']}
   row['authors']=[a.findtext('a:name','',NS) for a in e.findall('a:author',NS)];row['version']=row['id'].split('/')[-1];row['arxiv_id']=re.sub(r'v\d+$','',row['version']);records.append(row)
  d['records']=records
 except Exception as exc:d.update(error=str(exc),records=[])
 (ROOT/(key+'.json')).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 return d
if __name__=='__main__':
 pool={}
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
  for d in ex.map(run,queries):
   print(d['key'],d.get('status'),d.get('total_results'),len(d['records']),d.get('error',''),flush=True)
   for r in d['records']:
    item=pool.setdefault(r['arxiv_id'],dict(r,matched_queries=[]));item['matched_queries'].append(d['key'])
 (ROOT/'search_pool.json').write_text(json.dumps(pool,ensure_ascii=False,indent=2)+'\n')
 print('unique',len(pool),flush=True)
