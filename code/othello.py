"""OthelloGPT adapter: the boundary test for the differential scaling claim.

Chess gives a clean local/global split. A move can be illegal because the source
square is empty, which is determined by one square, or because it leaves the king
in check, which is determined by the whole board. Our claim is that scale removes
the local kind much faster than the global kind.

Othello is the interesting case precisely because its legality is inherently
non-local. A move is legal only if it flanks a line of opponent discs terminating
in one of your own, so legality is never decidable from the target square alone.
There are still two classes, and they still differ in locality, but far less
sharply than in chess:

  occupied   the chosen square already holds a disc. Decidable from that square.
  no_flank   the square is empty but flanks nothing. Requires scanning eight
             directions over the whole board.

Two outcomes and both are worth reporting. If no_flank scales away more slowly
than occupied, the claim survives a game whose rules are not chess-shaped. If the
two scale alike, that is the boundary of the claim, and a boundary is a result.

The released weights have LayerNorm folded into the following linear layers, so
there are no LayerNorm parameters and normalisation is parameter-free. The
forward below reproduces TransformerLens exactly under that convention; as
everywhere else in this project, it is not trusted until the model plays legally.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(__file__), "..")
EXT = os.path.join(ROOT, "data", "ext")

CENTER = (27, 28, 35, 36)
VALID = [s for s in range(64) if s not in CENTER]      # 60 squares
SQ2TOK = {s: i + 1 for i, s in enumerate(VALID)}       # token 0 is pass
TOK2SQ = {i + 1: s for i, s in enumerate(VALID)}
DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

BLACK, WHITE, EMPTY = 1, -1, 0


def start_board():
    b = np.zeros(64, dtype=np.int8)
    b[27] = WHITE
    b[36] = WHITE
    b[28] = BLACK
    b[35] = BLACK
    return b


def flips(board, sq, colour):
    """Discs that would flip if `colour` plays `sq`; empty if illegal."""
    if board[sq] != EMPTY:
        return []
    r0, c0 = divmod(sq, 8)
    out = []
    for dr, dc in DIRS:
        r, c = r0 + dr, c0 + dc
        run = []
        while 0 <= r < 8 and 0 <= c < 8 and board[r * 8 + c] == -colour:
            run.append(r * 8 + c)
            r += dr
            c += dc
        if run and 0 <= r < 8 and 0 <= c < 8 and board[r * 8 + c] == colour:
            out.extend(run)
    return out


def legal_moves(board, colour):
    return [s for s in VALID if board[s] == EMPTY and flips(board, s, colour)]


def apply_move(board, sq, colour):
    b = board.copy()
    f = flips(board, sq, colour)
    b[sq] = colour
    for s in f:
        b[s] = colour
    return b


def random_game(rng, max_moves=59):
    """A legal self-play game, which is the distribution the synthetic model saw."""
    b = start_board()
    colour = BLACK
    moves, boards, colours = [], [], []
    while len(moves) < max_moves:
        lm = legal_moves(b, colour)
        if not lm:
            colour = -colour
            lm = legal_moves(b, colour)
            if not lm:
                break
        boards.append(b.copy())
        colours.append(colour)
        sq = int(rng.choice(lm))
        moves.append(sq)
        b = apply_move(b, sq, colour)
        colour = -colour
    return moves, boards, colours


# ---------------------------------------------------------------- the model
class OthelloGPT(torch.nn.Module):
    """Minimal TransformerLens forward for fold_ln weights (no LN parameters)."""

    def __init__(self, sd, n_layer=8, n_head=8, d_head=64, device="cuda"):
        super().__init__()
        self.sd = {k: v.to(device) for k, v in sd.items()}
        self.n_layer, self.n_head, self.d_head = n_layer, n_head, d_head
        self.device = device

    @staticmethod
    def ln(x):
        x = x - x.mean(-1, keepdim=True)
        return x / (x.pow(2).mean(-1, keepdim=True) + 1e-5).sqrt()

    def forward(self, toks):
        sd = self.sd
        T = toks.shape[1]
        x = sd["embed.W_E"][toks] + sd["pos_embed.W_pos"][:T]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        for i in range(self.n_layer):
            p = f"blocks.{i}."
            h = self.ln(x)
            q = torch.einsum("btd,hdk->bhtk", h, sd[p + "attn.W_Q"]) \
                + sd[p + "attn.b_Q"][None, :, None]
            k = torch.einsum("btd,hdk->bhtk", h, sd[p + "attn.W_K"]) \
                + sd[p + "attn.b_K"][None, :, None]
            v = torch.einsum("btd,hdk->bhtk", h, sd[p + "attn.W_V"]) \
                + sd[p + "attn.b_V"][None, :, None]
            att = (q @ k.transpose(-1, -2)) / np.sqrt(self.d_head)
            att = att.masked_fill(~mask, -1e9).softmax(-1)
            z = att @ v
            x = x + torch.einsum("bhtk,hkd->btd", z, sd[p + "attn.W_O"]) \
                + sd[p + "attn.b_O"]
            h = self.ln(x)
            m = h @ sd[p + "mlp.W_in"] + sd[p + "mlp.b_in"]
            m = 0.5 * m * (1 + torch.tanh(np.sqrt(2 / np.pi)
                                          * (m + 0.044715 * m ** 3)))
            x = x + m @ sd[p + "mlp.W_out"] + sd[p + "mlp.b_out"]
        return self.ln(x) @ sd["unembed.W_U"] + sd["unembed.b_U"]


def load(name="synthetic_model.pth", device="cuda"):
    sd = torch.load(os.path.join(EXT, name), map_location="cpu",
                    weights_only=False)
    if not isinstance(sd, dict) or "embed.W_E" not in sd:
        sd = sd.get("model", sd)
    return OthelloGPT(sd, device=device).eval()


@torch.no_grad()
def verify(model, n_games=20, seed=0, device="cuda"):
    """The model must choose legal moves, or the adapter is wrong."""
    rng = np.random.default_rng(seed)
    ok = tot = 0
    for _ in range(n_games):
        moves, boards, colours = random_game(rng)
        if len(moves) < 12:
            continue
        toks = [SQ2TOK[m] for m in moves]
        for t in range(5, min(len(moves) - 1, 40)):
            x = torch.tensor([toks[:t + 1]], dtype=torch.long, device=device)
            pred = int(model(x)[0, -1].argmax())
            sq = TOK2SQ.get(pred)
            lm = legal_moves(boards[t + 1], colours[t + 1])
            tot += 1
            if sq is not None and sq in lm:
                ok += 1
    return ok, tot


def main():
    print("Othello engine self-test")
    rng = np.random.default_rng(0)
    lens = [len(random_game(rng)[0]) for _ in range(200)]
    print(f"  200 random legal games, mean length {np.mean(lens):.1f}, "
          f"min {min(lens)}, max {max(lens)}")
    b = start_board()
    print(f"  opening legal moves for black: "
          f"{sorted(legal_moves(b, BLACK))} (expect [19, 26, 37, 44])")

    print("\nloading OthelloGPT")
    model = load()
    ok, tot = verify(model)
    print(f"  legal-move accuracy: {ok}/{tot} = {ok/max(tot,1):.4f}")
    print("  " + ("ADAPTER VERIFIED" if ok / max(tot, 1) > 0.9 else
                  "REJECTED -- forward or vocabulary is wrong"))


if __name__ == "__main__":
    main()
