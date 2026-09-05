"""State-Consistency Regularisation: an oracle-free objective for context rot.

Paper 1 established three things about search-free chess transformers. The
internal state decays with reasoning depth. Errors are coherent with the decayed
state. And editing the state representation changes the action, so the policy
reads it. Together those say the failure is a state-maintenance failure, and that
the state is a real, writable object rather than an epiphenomenon.

The obvious repair is to supervise the state against an oracle. Paper 1 tried
that and found it improves the MEASUREMENT of state far more than behaviour, and
it cannot generalise: outside chess there is no rules engine to supervise with.

This is the alternative. A correct state must satisfy a consistency condition
that needs no oracle at all:

    the state after reading token x_{t+1} must be a function of the state before
    it and of x_{t+1} itself.

If the model's own representation violates that, it is drifting, and the
violation is detectable from the representation alone. We therefore learn a small
transition operator on a projected subspace of the residual stream and train the
model so that its own state evolves consistently under it.

    z_t   = P h_t                          projection to a state subspace
    z^_t+1 = normalise(A z_t + B e(x_t+1))  predicted next state
    loss  = InfoNCE(z^_t+1 , z_t+1)         against same-timestep negatives

InfoNCE rather than a squared error because a regression target invites the
trivial solution of collapsing z to a constant, which satisfies consistency
perfectly and carries nothing. A contrastive target requires z to stay
discriminative between positions while remaining predictable across one step.

Nothing here consults the oracle. The move token is already in the input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateConsistency(nn.Module):
    """Projection, transition operator, and the contrastive consistency loss."""

    def __init__(self, d_model, d_state=64, vocab=None, d_move=64,
                 temperature=0.1):
        super().__init__()
        self.proj = nn.Linear(d_model, d_state, bias=False)
        self.move = nn.Embedding(vocab, d_move)
        self.trans = nn.Linear(d_state + d_move, d_state)
        # start close to identity in the state part, so the operator begins as
        # "carry the state forward unchanged" and learns the update from there
        with torch.no_grad():
            self.trans.weight.zero_()
            self.trans.weight[:, :d_state].copy_(torch.eye(d_state))
            self.trans.bias.zero_()
        self.temperature = temperature
        self.d_state = d_state

    def project(self, h):
        """Project to the state subspace.

        An earlier version subtracted the batch mean at each position, intending
        to remove positional information. That was both unnecessary and wrong.
        Unnecessary because the negatives already share the timestep, so any
        component depending only on position is identical across candidates and
        cannot help discriminate. Wrong because it leaves the representation
        undefined for a single sequence: at inference the mean is taken over a
        batch of one, the projection returns zeros, and the state vanishes. The
        representation has to be computable from one sequence to be usable at
        inference at all.
        """
        return F.normalize(self.proj(h), dim=-1)

    def forward(self, h, x_next, valid):
        """h: (B,T,d) hidden states. x_next: (B,T) token read after position t.
        valid: (B,T) bool, positions where both t and t+1 are real.

        Negatives are the SAME timestep in other sequences of the batch. That
        choice is what makes the task non-trivial, and it is what neutralises
        position. Two earlier designs failed: pooling negatives across the batch
        lets the model win by encoding which game it is in, and drawing them
        from the same sequence lets it win by encoding position, since the
        residual stream carries positional embeddings. Here every candidate
        shares the timestep exactly, so a component that depends only on
        position is identical across candidates and carries no information. The
        only way to identify the right target is to represent what distinguishes
        this game's state at this moment.
        """
        z = self.project(h)                                  # (B,T,k)
        m = self.move(x_next)                                # (B,T,d_move)
        pred = F.normalize(self.trans(torch.cat([z, m], dim=-1)), dim=-1)

        B, T, k = z.shape
        if B < 2:
            return h.new_zeros(()), h.new_zeros(())
        p = pred[:, :-1]                                     # (B,T-1,k)
        g = z[:, 1:]                                         # (B,T-1,k)
        vm = valid[:, :-1]                                   # (B,T-1)

        # per timestep: (B,B) similarity between each prediction and every
        # sequence's realised next state at that same timestep
        logits = torch.einsum("btk,ctk->tbc", p, g) / self.temperature
        labels = torch.arange(B, device=z.device).expand(T - 1, B)   # (T-1,B)

        # keep only anchors whose own next position is real; a candidate that is
        # padding must not be selectable either
        cand_ok = vm.t().unsqueeze(1)                        # (T-1,1,B)
        logits = logits.masked_fill(~cand_ok, float("-inf"))
        keep = vm.t()                                        # (T-1,B)
        lg = logits[keep]
        lb = labels[keep]
        if lg.numel() == 0:
            return h.new_zeros(()), h.new_zeros(())
        loss = F.cross_entropy(lg, lb)
        with torch.no_grad():
            acc = (lg.argmax(-1) == lb).float().mean()
        return loss, acc


@torch.no_grad()
def consistency_error(scr, h, x_next, valid):
    """Diagnostic used at evaluation: 1 minus cosine between the predicted next
    state and the realised one, per position. Needs no oracle, so it can be
    computed at inference."""
    z = scr.project(h)
    m = scr.move(x_next)
    pred = F.normalize(scr.trans(torch.cat([z, m], dim=-1)), dim=-1)
    cos = (pred[:, :-1] * z[:, 1:]).sum(-1)
    return 1.0 - cos

class MultiStepConsistency(StateConsistency):
    """Consistency over k steps, not one.

    The one-step operator reaches 0.975 on its own task and still degrades the
    state when iterated: rolling twelve steps forward produces a board that
    decodes worse than the model's current state, and worse than the unrolled
    stale state. A one-step contrastive objective constrains a single
    application; it says nothing about what happens when the operator is
    composed with itself, and the errors compound.

    This trains the composition directly. Each anchor is rolled forward a random
    number of steps and matched against the realised state at that horizon, with
    negatives again drawn from the same timestep in other sequences. The
    operator is thereby asked for the property the repair actually needs.
    """

    def __init__(self, *a, max_k=12, **kw):
        super().__init__(*a, **kw)
        self.max_k = max_k

    def forward(self, h, x_next, valid, k=None):
        z = self.project(h)
        B, T, _ = z.shape
        if B < 2 or T < self.max_k + 2:
            return h.new_zeros(()), h.new_zeros(())
        if k is None:
            k = int(torch.randint(1, self.max_k + 1, (1,)).item())

        # anchors that can be rolled k steps and still land on a real position
        last = T - k
        zk = z[:, :last]                                     # (B,last,d_state)
        for j in range(k):
            mv = self.move(x_next[:, j:j + last])
            zk = F.normalize(self.trans(torch.cat([zk, mv], dim=-1)), dim=-1)

        tgt = z[:, k:k + last]
        vm = valid[:, :last] & valid[:, k:k + last]
        if vm.sum() == 0:
            return h.new_zeros(()), h.new_zeros(())

        logits = torch.einsum("btd,ctd->tbc", zk, tgt) / self.temperature
        labels = torch.arange(B, device=z.device).expand(last, B)
        cand_ok = vm.t().unsqueeze(1)
        logits = logits.masked_fill(~cand_ok, float("-inf"))
        keep = vm.t()
        lg, lb = logits[keep], labels[keep]
        if lg.numel() == 0:
            return h.new_zeros(()), h.new_zeros(())
        loss = F.cross_entropy(lg, lb)
        with torch.no_grad():
            acc = (lg.argmax(-1) == lb).float().mean()
        return loss, acc

