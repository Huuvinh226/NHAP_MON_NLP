"""
mcDPO loss cho InImageTrans (Zuo et al., ACL Findings 2025), Muc 4.2, Cong thuc (5), (6), (7).

Moi ham nhan LOG-XAC-SUAT CUA CA CHUOI DAP AN (tong log-prob cac token dap an), tinh boi:
  - policy : mo hinh dang huan luyen (pi_theta)
  - ref    : ban sao dong bang (pi_ref). Bai ghi "reference model"; viec ref la
             ban sao cua mo hinh sau SFT la CACH HIEU CUA GIAO VIEN, can kiem tra lai trong repo cua bai.

Ky hieu (dung theo bai):
  I_w: anh goc | I_l: anh bi che ~20% chu | y_w: nhan dung | y_l: nhan lap (rejected)
  beta = 0.1 (bai, Muc A.3)

TRANG THAI: cong thuc da duoc doi chieu voi bai. Ban torch nay CHUA chay thu o sandbox cua giao vien
(sandbox khong co torch). Hay chay phan self-test o cuoi file tren Colab truoc khi dung.
"""
import torch
import torch.nn.functional as F


def sequence_logprob(logits, input_ids, answer_mask):
    """Tong log-prob cac token DAP AN.
    logits: [B, T, V] | input_ids: [B, T] | answer_mask: [B, T], 1 tai vi tri token dap an.
    """
    logits = logits[:, :-1, :].float()          # du doan token t+1 tu vi tri t
    labels = input_ids[:, 1:]
    mask = answer_mask[:, 1:].float()
    logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(dim=-1)            # [B]


def _dpo_term(pol_good, ref_good, pol_bad, ref_bad, beta):
    # -log sigma( beta*log(pi(good)/ref(good)) - beta*log(pi(bad)/ref(bad)) )
    margin = (pol_good - ref_good) - (pol_bad - ref_bad)
    return -F.logsigmoid(beta * margin).mean()


def rpo_loss(pol_yw_Iw, ref_yw_Iw, pol_yl_Iw, ref_yl_Iw, beta=0.1):
    """Cong thuc (5): cung anh I_w, so sanh nhan dung y_w voi nhan lap y_l."""
    return _dpo_term(pol_yw_Iw, ref_yw_Iw, pol_yl_Iw, ref_yl_Iw, beta)


def vpo_loss(pol_yw_Iw, ref_yw_Iw, pol_yw_Il, ref_yw_Il, beta=0.1):
    """Cong thuc (6): cung nhan y_w, so sanh anh goc I_w voi anh bi che I_l."""
    return _dpo_term(pol_yw_Iw, ref_yw_Iw, pol_yw_Il, ref_yw_Il, beta)


def mcdpo_loss(pol_yw_Iw, ref_yw_Iw, pol_yl_Iw, ref_yl_Iw,
               pol_yw_Il, ref_yw_Il, beta=0.1):
    """Cong thuc (7): l_mcDPO = l_rPO + l_vPO.
    Moi mau can 3 luot forward cho policy va 3 cho ref:
      (I_w, y_w), (I_w, y_l), (I_l, y_w).
    """
    l_r = rpo_loss(pol_yw_Iw, ref_yw_Iw, pol_yl_Iw, ref_yl_Iw, beta)
    l_v = vpo_loss(pol_yw_Iw, ref_yw_Iw, pol_yw_Il, ref_yw_Il, beta)
    return l_r + l_v, l_r.detach(), l_v.detach()


if __name__ == "__main__":
    # Self-test (chay tren Colab)
    torch.manual_seed(0)
    a = torch.randn(8) - 20.0                    # log-prob gia lap
    b = torch.randn(8) - 30.0
    # 1) policy == ref  ->  moi so hang = log 2 ~ 0.6931
    l, lr, lv = mcdpo_loss(a, a, b, b, a, a)
    print("policy==ref:", float(lr), float(lv), "(ky vong ~0.6931 moi so hang)")
    assert abs(float(lr) - 0.6931) < 1e-3 and abs(float(lv) - 0.6931) < 1e-3
    # 2) tang xac suat nhan dung -> loss giam
    l2, _, _ = mcdpo_loss(a + 5, a, b, b, a + 5, a)
    print("sau khi tang p(y_w):", float(l2), "<", float(l))
    assert float(l2) < float(l)
    # 3) sequence_logprob chay duoc voi tensor gia lap
    logits = torch.randn(2, 6, 10)
    ids = torch.randint(0, 10, (2, 6))
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1]])
    print("sequence_logprob:", sequence_logprob(logits, ids, mask).shape, "(ky vong torch.Size([2]))")
    print("OK")
