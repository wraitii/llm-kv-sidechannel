import json
from pathlib import Path
import numpy as np
import mlx.core as mx
from llmz.runtime import load_config, model_and_tokenizer
from llmz.train import load_model_checkpoint
from llmz.inspect_attention import build_sequence
from llmz.model import causal_mask

out=Path('runs/controlled-scored/inspection')
rows=[e['row'] for e in json.loads((out/'results.json').read_text())[0]['examples_detail']]
cfg=load_config('configs/controlled/scored.json')
results=[]
for step in (900,1000):
    model,tok=model_and_tokenizer(cfg)
    load_model_checkpoint(Path(f'runs/controlled-scored/checkpoint-{step:07d}.npz'),model)
    aggregates=[[] for _ in model.blocks]
    details=[]
    for row in rows:
        seq,valid,bound,spans,labels,groups=build_sequence(tok,row,cfg)
        x=mx.array(seq); valid=mx.array(valid)
        h=model.embed(x)
        mask=causal_mask(valid,h.dtype)
        pos=mx.broadcast_to(mx.arange(h.shape[1]),valid.shape)
        sep=int(bound[0])-1
        layers=[]
        for li,block in enumerate(model.blocks):
            r=block.retention
            q,k,v=block._qkv(h)
            raw=r.sequence_scores(block.n1(h),v)
            scores=np.asarray(raw)[0,:-r.window]
            aggregates[li].append(scores)
            keep,_=r.select_sequence(raw,valid,pos,pos)
            support=np.asarray(keep)[0]
            atsep=np.flatnonzero(support[sep,:max(0,sep-r.window+1)])
            final=np.flatnonzero(support[-2,:max(0,len(labels)-2-r.window+1)])
            layers.append(dict(layer=li,min=float(scores.min()),max=float(scores.max()),median=float(np.median(scores)),old_at_sep=[dict(pos=int(i),label=labels[i],move=groups[i],score=float(scores[i])) for i in atsep],old_at_last_prediction=[dict(pos=int(i),label=labels[i],move=groups[i],score=float(scores[i])) for i in final]))
            h=block(h,mask,valid=valid,positions=pos)
        details.append(dict(file_index=row['file_index'],source_tokens=int(bound[0])-2,layers=layers))
    stats=[]
    for li,arrays in enumerate(aggregates):
        s=np.concatenate(arrays).astype(float)
        sig=1/(1+np.exp(-np.clip(s,-700,700)))
        actual_sig=np.asarray(mx.sigmoid(mx.array(s.astype(np.float32))))
        stats.append(dict(layer=li,count=len(s),finite=bool(np.isfinite(s).all()),min=float(s.min()),p01=float(np.quantile(s,.01)),median=float(np.median(s)),p99=float(np.quantile(s,.99)),max=float(s.max()),mean=float(s.mean()),std=float(s.std()),abs_gt_5=float(np.mean(abs(s)>5)),abs_gt_10=float(np.mean(abs(s)>10)),mean_sigmoid_derivative=float(np.mean(sig*(1-sig))),float32_sigmoid_exact_one=float(np.mean(actual_sig==1)),float32_sigmoid_exact_zero=float(np.mean(actual_sig==0)),weight_norms={name:float(mx.sqrt(mx.sum(getattr(model.blocks[li].retention,name).weight.astype(mx.float32)**2)).item()) for name in ('query','key','priority')}))
    result=dict(step=step,examples=len(rows),layers=stats,details=details)
    results.append(result)
    print(json.dumps(dict(step=step,layers=stats)),flush=True)
(out/'scores.json').write_text(json.dumps(results,indent=2))
