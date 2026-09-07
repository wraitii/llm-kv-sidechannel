import json
from pathlib import Path
import numpy as np
import mlx.core as mx
from llmz.runtime import load_config, model_and_tokenizer
from llmz.train import load_model_checkpoint
from llmz.eval_chess import teacher_forced_nll, generate_batch
from llmz.board_eval import score_board_outputs
from llmz.inspect_attention import build_sequence
from llmz.attention_html import render

out = Path('runs/controlled-scored/inspection')
cfg = load_config(Path('configs/controlled/scored.json'))
model, tok = model_and_tokenizer(cfg)
checkpoint = Path('runs/controlled-scored/checkpoint-0001000.npz')
state = load_model_checkpoint(checkpoint, model)
retentions = [b.retention for b in model.blocks]
rng = np.random.default_rng(1337)
rows, seen = [], 0
with (Path(cfg['data_dir']) / 'val.jsonl').open() as f:
    for index, line in enumerate(f):
        row = json.loads(line)
        if len(tok.encode_source(row['asm'])) > cfg['max_source_tokens'] or len(tok.encode_target_local(row['code'])) > cfg['max_target_tokens']:
            continue
        row = dict(row, file_index=index)
        seen += 1
        if len(rows) < 32:
            rows.append(row)
        else:
            j = int(rng.integers(seen))
            if j < 32:
                rows[j] = row
lengths = np.array([len(tok.encode_source(r['asm'])) for r in rows])
order = np.argsort(lengths)
selected = [int(order[i]) for i in (0, 16, 31)]
results = []
for name, recent, memory, window, mode in [
    ('scored32', 24, 8, None, 'preserve'),
    ('scored48', 36, 12, None, 'preserve'),
    ('full', None, None, None, 'preserve'),
    ('swa32', None, None, 32, 'preserve'),
    ('scored32-restart', 24, 8, None, 'restart'),
]:
    for block, retention in zip(model.blocks, retentions):
        block.retention = retention if recent else None
        if recent:
            retention.window, retention.memory = recent, memory
    losses, counts, predictions = [], [], []
    for start in range(0, len(rows), 8):
        batch = rows[start:start+8]
        loss, count = teacher_forced_nll(model, tok, batch, cfg['max_source_tokens'], window, cache_mode=mode)
        losses.extend(loss.tolist()); counts.extend(count.tolist())
        predictions.extend(generate_batch(model, tok, batch, cfg['max_source_tokens'], cfg['max_target_tokens'], 0, 1, window, cache_mode=mode))
    parsed, valid, exact, squares, metadata = score_board_outputs(predictions, [r['code'] for r in rows])
    result = dict(condition=name, checkpoint=str(checkpoint), seed=1337, split='val', examples=len(rows), recent=recent, memory=memory, window=window, cache_mode=mode, nll=sum(losses)/sum(counts), parseable=parsed, valid=valid, exact=exact, mean_board_square_error=squares/parsed if parsed else None,
        examples_detail=[dict(row=r, source_tokens=int(lengths[i]), prediction=predictions[i], nll=losses[i]/counts[i]) for i,r in enumerate(rows)])
    results.append(result)
    (out / 'results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'examples_detail'}), flush=True)
    if mode != 'preserve':
        continue
    for i in selected:
        row = rows[i]
        sequence, valid_mask, boundaries, spans, labels, groups = build_sequence(tok, row, cfg)
        x, v, b = mx.array(sequence), mx.array(valid_mask), mx.array(boundaries)
        hidden, maps = model.attention_maps(x, v, b, sliding_window=window)
        reference = model.hidden(x, v, b, sliding_window=window)
        error = float(mx.max(mx.abs(hidden-reference)).item())
        attention = np.stack([np.asarray(p, dtype=np.float32)[0] for p in maps])
        n_source = int(boundaries[0])-2
        sep = n_source+1
        layer_stats=[]
        for layer, p in enumerate(attention):
            support=(p > 0).any(axis=0)
            old = np.flatnonzero(support[sep, :max(0,sep-(recent or 32)+1)])
            mean = p.mean(axis=0)
            # SEP predicts first FEN token; final FEN token predicts EOS.
            target_mass=mean[sep:-1]
            layer_stats.append(dict(layer=layer, max_support=int(support.sum(axis=-1).max()), old_at_sep=[dict(position=int(k), label=labels[k], move=groups[k], attention=float(mean[sep,k])) for k in old], source_mass=float(target_mass[:,:sep].sum(axis=-1).mean()), bos_mass=float(target_mass[:,0].mean()), sep_mass=float(target_mass[:,sep].mean()), top_at_sep=[dict(position=int(k),label=labels[k],mass=float(mean[sep,k])) for k in np.argsort(-mean[sep])[:5]]))
        stem=f'{name}-val-{row["file_index"]}'
        (out / f'{stem}.html').write_text(render(attention,labels,groups,n_source,row,state,checkpoint,cfg))
        (out / f'{stem}.json').write_text(json.dumps(dict(condition=name,row=row,source_tokens=n_source,capture_max_error=error,layers=layer_stats),indent=2))
        print(f'inspected {stem}: {n_source} source tokens, capture error {error}',flush=True)
