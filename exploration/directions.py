"""Direction designs independent of the layer/alpha/operator parameter grid."""
import torch
import torch.nn.functional as F


def unit(v):
 if not torch.isfinite(v).all() or v.norm()<1e-8:raise ValueError('Degenerate exploration direction')
 return v/v.norm()


def build_directions(ids,student_pools,teacher_pools,global19):
 if student_pools.shape!=teacher_pools.shape or len(ids)!=len(student_pools):raise ValueError('Unpaired pools')
 differences=student_pools-teacher_pools;g=differences.mean(0);common=(student_pools.mean(0)+teacher_pools.mean(0))/2
 nuisance_removed=g-torch.dot(g,unit(common))*unit(common)
 # Within-response-domain covariance across paired questions; ridge shrinks weak PCs.
 x=student_pools-student_pools.mean(0);y=teacher_pools-teacher_pools.mean(0)
 covariance=(x.T@x+y.T@y)/(2*(len(ids)-1))
 ridge=.1*covariance.trace()/covariance.shape[0]
 eigenvalues,eigenvectors=torch.linalg.eigh(covariance.double())
 whitened=eigenvectors@((eigenvectors.T@g.double())/(eigenvalues.clamp_min(0)+ridge).sqrt())
 # Nearby question profiles, excluding self; shared/global prior reduces sample-specific noise.
 profiles=F.normalize(y,dim=-1);similarities=profiles@profiles.T;similarities.fill_diagonal_(-float('inf'))
 k=min(64,len(ids)-1);neighbors=similarities.topk(k,dim=-1).indices
 local=.5*g+.5*differences[neighbors].mean(1)
 specs={
  'reference_replay':{'eligible':False,'kind':'control','layers':[29],'alpha':.1,'position_mode':'prediction'},
  'negative_direction_control':{'eligible':False,'kind':'control','layers':[29],'alpha':.1,'position_mode':'prediction'},
  'official_answer_input_mask':{'eligible':True,'kind':'mask_control','layers':[29],'alpha':.1,'position_mode':'answer_input'},
  'nuisance_orthogonal':{'eligible':True,'kind':'direction','layers':[29],'alpha':.1,'position_mode':'prediction'},
  'covariance_whitened':{'eligible':True,'kind':'direction','layers':[29],'alpha':.1,'position_mode':'prediction'},
  'knn_shrink_global':{'eligible':True,'kind':'direction','layers':[29],'alpha':.1,'position_mode':'prediction'},
  'source_adaptive_protect':{'eligible':True,'kind':'source_adaptive_strength','layers':[29],'alpha':.2,'position_mode':'prediction'},
  'multilayer_19_29':{'eligible':True,'kind':'multi_layer','layers':[19,29],'alpha':.1/(2**.5),'position_mode':'prediction'}
 }
 vectors={'reference_replay':{'global':{29:unit(g)}},'negative_direction_control':{'global':{29:-unit(g)}},
  'official_answer_input_mask':{'global':{29:unit(g)}},'nuisance_orthogonal':{'global':{29:unit(nuisance_removed)}},
  'covariance_whitened':{'global':{29:unit(whitened.float())}},
  'knn_shrink_global':{qid:{29:unit(local[i])} for i,qid in enumerate(ids)},
  'source_adaptive_protect':{'global':{29:unit(g)}},
  'multilayer_19_29':{'global':{19:unit(global19),29:unit(g)}}}
 geometry={'n':len(ids),'hidden_size':len(g),'global_raw_norm':float(g.norm()),'common_mean_norm':float(common.norm()),
  'global_cos_common':float(torch.dot(unit(g),unit(common))),'covariance_ridge':float(ridge),
  'covariance_whitened_cos_global':float(torch.dot(unit(whitened.float()),unit(g))),
  'nuisance_orthogonal_cos_global':float(torch.dot(unit(nuisance_removed),unit(g))),
  'per_question_cos_global_mean':float((F.normalize(differences,dim=-1)@unit(g)).mean()),
  'knn_cos_global_mean':float((F.normalize(local,dim=-1)@unit(g)).mean()),
  'knn_neighbors':k,'knn_self_excluded':True,'knn_global_weight':.5,'fit_pool':'same_all_1000_as_original_global_SV'}
 return specs,vectors,geometry


def native_strengths(native,plan,baseline_overlap,case,spec):
 n=len(native['token_ids'])
 if case=='no_sv_replay':return torch.zeros(n,dtype=torch.float32)
 if case=='source_adaptive_protect':
  result=torch.zeros(n,dtype=torch.float32)
  for u in plan['units']:
   if not u['valid']:continue
   a,b=u['ranges']['student'];c,d=u['ranges']['teacher']
   value=float(baseline_overlap[a])
   # Frozen source only: <=9/16, 10..12/16, >=13/16.
   result[c:d]=.3 if value<=9/16 else .2 if value<13/16 else 0.
  return result
 result=torch.full((n,),float(spec['alpha']))
 if spec['position_mode']=='answer_input':result[0]=0.
 return result
