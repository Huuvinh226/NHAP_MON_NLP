"""
Do luong ao giac LAP / BO SOT va chat luong dich (bam theo Muc 6.1, A.5 cua bai InImageTrans).

TRUNG THUC ve nhung gi la cua bai va nhung gi la cua giao vien:
- Repetition_4 (cong thuc 8): 1 - |unique(4-gram)| / |4-gram|, tinh tren KY TU. Bai viet
  "unique(4_grams) = bon ky tu lien tiep chua bi lap"; o day hieu la SO 4-gram KHAC NHAU (cach
  hieu cua giao vien, can doi chieu voi Xu et al. 2022).
- Co lap (repetition candidate): bai dung "do dai dau ra vuot xa tham chieu va co 10 lan lap lien tiep",
  sau do nguoi doc xac nhan. O day: regex bat chuoi 1-20 ky tu lap >= 10 lan lien tiep (cach cai dat cua
  giao vien). Khong co buoc nguoi xac nhan.
- Co bo sot (omission candidate): bai dung "do dai dau ra NGAN HON NHIEU so voi tham chieu". Nguong
  "nhieu" bai khong neu; o day dung ti le do dai < 0.5 (GIA DINH cua giao vien). Day chi la ung vien,
  khong phai ket luan bo sot.
- BLEU: bai dung sacreBLEU; cach tach tu tieng Trung bai khong neu. O day dung tokenize="zh" (can kiem tra).
"""
import re

_REP_RE = re.compile(r"(.{1,20}?)\1{9,}", re.S)


def has_repetition(text):
    return _REP_RE.search(text) is not None


def repetition_4(text):
    text = re.sub(r"\s+", "", text)
    grams = [text[i:i + 4] for i in range(len(text) - 3)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def length_ratio(hyp, ref):
    return len(re.sub(r"\s+", "", hyp)) / max(1, len(re.sub(r"\s+", "", ref)))


def evaluate(hyps, refs, over=2.0, under=0.5):
    assert len(hyps) == len(refs) and len(hyps) > 0
    n = len(hyps)
    ratios = [length_ratio(h, r) for h, r in zip(hyps, refs)]
    rep_flags = [has_repetition(h) and ra > over for h, ra in zip(hyps, ratios)]
    rep_any = [has_repetition(h) for h in hyps]
    omit = [ra < under for ra in ratios]
    out = {
        "n": n,
        "repetition_rate": sum(rep_flags) / n,          # lap + dai vuot xa tham chieu (theo bai)
        "repetition_rate_loose": sum(rep_any) / n,      # chi can co chuoi lap >= 10 lan
        "omission_candidate_rate": sum(omit) / n,       # UNG VIEN bo sot (ti le do dai < under)
        "repetition_4": sum(repetition_4(h) for h in hyps) / n,
        "len_ratio_mean": sum(ratios) / n,
        "bleu": None,
    }
    try:
        import sacrebleu
        out["bleu"] = sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh").score
    except ImportError:
        pass
    return out


if __name__ == "__main__":
    ref = "本校园内禁止吸烟，请勿在建筑物内使用任何烟草制品。"
    good = ref
    rep = "本校园内禁止" + "吸烟。" * 40
    short = "本校园"
    r = evaluate([good, rep, short], [ref, ref, ref])
    print(r)
    assert has_repetition(rep) and not has_repetition(good)
    assert repetition_4(rep) > 0.8 and repetition_4(good) < 0.2
    assert abs(r["repetition_rate"] - 1 / 3) < 1e-9 and abs(r["omission_candidate_rate"] - 1 / 3) < 1e-9
    print("OK")
