"""ADMET, safety proxy, and synthesizability scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdmetProfile:
    qed: float
    sa_score: float
    mw: float
    logp: float
    tpsa: float
    hbd: int
    hba: int
    rotatable_bonds: int
    lipinski_violations: int
    veber_pass: bool
    admet_score: float
    synthesis_score: float
    safety_proxy_score: float
    main_risk: str


def lipinski_violations(mw: float, logp: float, hbd: int, hba: int) -> int:
    return int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)


def veber_pass(tpsa: float, rotatable_bonds: int) -> bool:
    return tpsa <= 140 and rotatable_bonds <= 10


def bounded_score(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 6)


def score_admet_from_descriptors(
    qed: float,
    sa_score: float,
    mw: float,
    logp: float,
    tpsa: float,
    hbd: int,
    hba: int,
    rotatable_bonds: int,
    has_reactive_alert: bool = False,
    slc19a3_similarity_risk: bool = False,
) -> AdmetProfile:
    lipinski = lipinski_violations(mw, logp, hbd, hba)
    veber = veber_pass(tpsa, rotatable_bonds)
    property_score = 1.0
    property_score -= 0.18 * lipinski
    property_score -= 0.15 * int(not veber)
    property_score -= 0.10 * int(mw > 900)
    property_score -= 0.10 * int(tpsa > 200)
    property_score -= 0.10 * int(logp > 7)
    property_score = bounded_score(0.55 * property_score + 0.45 * qed)

    synthesis_score = bounded_score(1.0 - ((sa_score - 1.0) / 9.0))
    safety_proxy = 1.0
    safety_proxy -= 0.35 * int(has_reactive_alert)
    safety_proxy -= 0.25 * int(slc19a3_similarity_risk)
    safety_proxy -= 0.10 * int(logp > 6.5)
    safety_proxy = bounded_score(safety_proxy)

    risks = []
    if lipinski:
        risks.append(f"{lipinski}_lipinski_violations")
    if not veber:
        risks.append("veber_fail")
    if has_reactive_alert:
        risks.append("reactive_or_medchem_alert")
    if slc19a3_similarity_risk:
        risks.append("slc19a3_proxy_similarity_risk")
    if not risks:
        risks.append("no_major_descriptor_risk")

    return AdmetProfile(
        qed=float(qed),
        sa_score=float(sa_score),
        mw=float(mw),
        logp=float(logp),
        tpsa=float(tpsa),
        hbd=int(hbd),
        hba=int(hba),
        rotatable_bonds=int(rotatable_bonds),
        lipinski_violations=lipinski,
        veber_pass=veber,
        admet_score=property_score,
        synthesis_score=synthesis_score,
        safety_proxy_score=safety_proxy,
        main_risk=";".join(risks),
    )


def profile_to_dict(profile: AdmetProfile) -> dict:
    return {
        "qed": profile.qed,
        "sa_score": profile.sa_score,
        "mw": profile.mw,
        "logp": profile.logp,
        "tpsa": profile.tpsa,
        "hbd": profile.hbd,
        "hba": profile.hba,
        "rotatable_bonds": profile.rotatable_bonds,
        "lipinski_violations": profile.lipinski_violations,
        "veber_pass": profile.veber_pass,
        "admet_score": profile.admet_score,
        "synthesis_score": profile.synthesis_score,
        "safety_proxy_score": profile.safety_proxy_score,
        "main_risk": profile.main_risk,
    }
