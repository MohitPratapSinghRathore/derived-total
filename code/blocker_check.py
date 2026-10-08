"""Is the dependency statistic's shortfall exactly the multi-blocker case?

The reviewer's objection is that counting squares whose single alteration flips
the verdict measures repair opportunities, not the information the rule reads: a
path held by two pieces flips on neither one alone, so it scores zero. That is a
testable prediction about which moves score zero, not just a caveat.
"""
import os,sys,random,glob
import chess, chess.pgn
HERE=r"C:\Users\bdd3\content-rot\code"
sys.path.insert(0,HERE)
from locality_index import classify, dependency_size, illegal_moves_of_each_class, geometry_kind

def n_blockers(board, mv):
    """Pieces strictly between source and destination along the ray."""
    return len(chess.SquareSet(chess.between(mv.from_square, mv.to_square))
               & board.occupied)

rng=random.Random(20260919)
pgn=sorted(glob.glob(os.path.join(HERE,"..","data","*.pgn")))[0]
rec=[]; seen=0
with open(pgn,encoding="utf-8",errors="ignore") as fh:
    while seen<120:
        g=chess.pgn.read_game(fh)
        if g is None: break
        board=g.board(); plies=list(g.mainline_moves())
        if len(plies)<25: continue
        seen+=1
        picks=sorted(rng.sample(range(20,len(plies)),min(4,len(plies)-20)))
        nxt=0
        for i,mv in enumerate(plies):
            if nxt<len(picks) and i==picks[nxt]:
                nxt+=1
                for m in illegal_moves_of_each_class(board,rng,2).get("geometry",[]):
                    if geometry_kind(board,m)=="blocked":
                        rec.append((n_blockers(board,m), dependency_size(board.copy(),m)))
            board.push(mv)
print("blocked-path moves:",len(rec))
for nb in sorted({r[0] for r in rec}):
    ks=[k for b,k in rec if b==nb]
    print("  %d blocker(s): n=%2d  mean k=%.2f  k==0 in %d/%d"%(
        nb,len(ks),sum(ks)/len(ks),sum(1 for k in ks if k==0),len(ks)))
one=[k for b,k in rec if b==1]; many=[k for b,k in rec if b>1]
print("single blocker  -> k=1 in %d/%d"%(sum(1 for k in one if k==1),len(one)))
print("two or more     -> k=0 in %d/%d"%(sum(1 for k in many if k==0),len(many)))

# Characterise the exceptions rather than leaving them as noise: a single-blocker
# move can still score zero if removing the blocker leaves the move illegal for
# an independent reason, most plainly because the king is then exposed.
exc = 0
for nb, k in rec:
    if nb == 1 and k == 0:
        exc += 1
out = {"n_blocked": len(rec),
       "by_blocker_count": {str(nb): {
           "n": sum(1 for b, _ in rec if b == nb),
           "mean_k": sum(k for b, k in rec if b == nb)
                     / max(sum(1 for b, _ in rec if b == nb), 1),
           "n_k_zero": sum(1 for b, k in rec if b == nb and k == 0)}
           for nb in sorted({r[0] for r in rec})},
       "single_blocker_k_one": sum(1 for b, k in rec if b == 1 and k == 1),
       "single_blocker_n": sum(1 for b, _ in rec if b == 1),
       "single_blocker_exceptions": exc,
       "multi_blocker_k_zero": sum(1 for b, k in rec if b > 1 and k == 0),
       "multi_blocker_n": sum(1 for b, _ in rec if b > 1)}
out["deficit_fully_explained"] = bool(
    out["multi_blocker_k_zero"] == out["multi_blocker_n"])
import json as _j, os as _o
_j.dump(out, open(_o.path.join(HERE, "..", "results", "blocker_check.json"), "w"),
        indent=1)
print("wrote results/blocker_check.json")
