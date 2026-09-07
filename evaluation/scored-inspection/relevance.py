import json
from pathlib import Path
import chess
from llmz.runtime import load_config, tokenizer_from_config
from llmz.inspect_attention import source_pieces, char_move_map
p=Path('runs/controlled-scored/inspection')
cfg=load_config('configs/controlled/scored.json');tok=tokenizer_from_config(cfg)
results=json.loads((p/'results.json').read_text())[0]
scores=json.loads((p/'scores.json').read_text())[-1]
for idx in (23289,21935):
 row=next(e['row'] for e in results['examples_detail'] if e['row']['file_index']==idx)
 diag=next(e for e in scores['details'] if e['file_index']==idx)
 board=chess.Board(); identities={s:chess.square_name(s) for s in board.piece_map()}; last={}; events=[]
 for i,uci in enumerate(row['asm'].split()):
  m=chess.Move.from_uci(uci); assert m in board.legal_moves
  ident=identities.pop(m.from_square); captured=None
  cap_square=m.to_square
  if board.is_en_passant(m): cap_square=m.to_square+(-8 if board.turn else 8)
  captured=identities.pop(cap_square,None)
  if board.is_castling(m):
   rank=chess.square_rank(m.from_square); king_side=m.to_square>m.from_square
   rs=chess.square(7 if king_side else 0,rank); rt=chess.square(5 if king_side else 3,rank)
   rook=identities.pop(rs);identities[rt]=rook;last[rook]=i
  identities[m.to_square]=ident;last[ident]=i
  events.append(dict(move=i,uci=uci,identity=ident,captured=captured))
  board.push(m)
 assert board.fen()==row['code']
 live=set(identities.values()); cmap=char_move_map(row['asm'])
 pieces=source_pieces(tok,row['asm'])
 pos_moves={pos:{cmap[c] for c in range(start,end) if c in cmap} for pos,(_,start,end) in enumerate(pieces,1)}
 print('\nEXAMPLE',idx,'final',board.fen())
 for layer in diag['layers']:
  moves=set().union(*(pos_moves.get(k['pos'],set()) for k in layer['old_at_sep']))
  print('layer',layer['layer'],[(i,events[i]['uci'],'LAST' if last.get(events[i]['identity'])==i and events[i]['identity'] in live else 'live-history' if events[i]['identity'] in live else 'gone', 'capture' if events[i]['captured'] else '') for i in sorted(moves)])
 print('LAST moves of surviving pieces outside recent window:')
 for square,ident in identities.items():
  if ident not in last: continue
  move=last[ident];positions=[pos for pos,ms in pos_moves.items() if move in ms]
  recent_start=diag['source_tokens']+2-24
  if any(pos>=recent_start for pos in positions): continue
  kept=[l['layer'] for l in diag['layers'] if any(k['pos'] in positions for k in l['old_at_sep'])]
  print(board.piece_at(square).symbol(),chess.square_name(square),events[move]['uci'],'move',move,'tokens',positions,'kept layers',kept)
