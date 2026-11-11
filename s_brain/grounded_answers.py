"""S-Brain grounded-answer verifier: source ID integrity and claim screening.

Source markers in generated text are not evidence that the quoted source
supports the claim. This module verifies whether every citation points to an
included, current and unmodified local memory. A conservative lexical claim
support check can flag ungrounded statements; it is NOT semantic entailment.

A safe user-facing deployment should require independent entailment testing
for material claims and avoid claiming factual verification from keyword
overlap. Raw memory text is untrusted and never executed as tool instructions.
"""
from __future__ import annotations

import dataclasses
import hashlib
import re
from collections import Counter
from typing import Iterable

from .memory_store import MemoryStore, tokens
from .rag_inference import AnswerPacket, EvidenceExcerpt, PromptAssembly

MARKER=re.compile(r"\[mem:([0-9a-fA-F-]{36})\]")
CLAIM_BOUNDARY=re.compile(r"(?<=[.!?])\s+|\n+")
STOP_WORDS={
    "the","a","an","of","is","are","and","or","to","for","on",
    "it","as","with","in","that","this","be","was","were","by",
    "what","how","can","does","did","from","at","which","who",
    "we","i","you","they","he","she","but","not","yes","no",
}


@dataclasses.dataclass(frozen=True)
class VerificationPolicy:
    require_citations: bool = True
    prohibit_unknown_citations: bool = True
    require_current_memory: bool = True
    min_lexical_coverage: float = 0.35
    min_supported_claim_fraction: float = 0.60
    max_claims: int = 32
    allow_uncited_intro: bool = True

    def validate(self):
        if not 0<=self.min_lexical_coverage<=1:
            raise ValueError("Lexical threshold must be a fraction")
        if not 0<=self.min_supported_claim_fraction<=1:
            raise ValueError("Claim fraction must be a fraction")
        if self.max_claims<1:
            raise ValueError("max_claims must be positive")


@dataclasses.dataclass(frozen=True)
class ClaimCheck:
    sentence: str
    cited_ids: tuple[str,...]
    valid_ids: tuple[str,...]
    unknown_ids: tuple[str,...]
    best_lexical_coverage: float
    supported_by_heuristic: bool
    reasons: tuple[str,...]

    def public(self):
        return {
            "claim_sha256":hashlib.sha256(self.sentence.encode()).hexdigest(),
            "cited_ids":list(self.cited_ids),
            "valid_ids":list(self.valid_ids),
            "unknown_ids":list(self.unknown_ids),
            "lexical_coverage":self.best_lexical_coverage,
            "passes_keyword_heuristic":self.supported_by_heuristic,
            "reasons":list(self.reasons),
        }


@dataclasses.dataclass
class VerificationReport:
    accepted: bool
    source_integrity: bool
    unknown_citations: list[str]
    stale_sources: list[str]
    claims_checked: int
    claims_passing_heuristic: int
    claim_fraction: float
    checks: list[ClaimCheck]
    limitations: tuple[str,...]

    def to_dict(self):
        return {
            "accepted":self.accepted,
            "source_integrity":self.source_integrity,
            "unknown_citations":self.unknown_citations,
            "stale_sources":self.stale_sources,
            "claims_checked":self.claims_checked,
            "claims_passing_heuristic":self.claims_passing_heuristic,
            "claim_fraction":self.claim_fraction,
            "checks":[item.public() for item in self.checks],
            "limitations":list(self.limitations),
        }


def extract_markers(answer:str)->list[str]:
    if not isinstance(answer,str):
        raise TypeError("Answer must be text")
    return [match.group(1).lower() for match in MARKER.finditer(answer)]


def remove_markers(answer:str)->str:
    return MARKER.sub("",answer)


def split_claims(answer:str,max_claims:int=32)->list[str]:
    if max_claims<1:
        raise ValueError("Invalid claim bound")
    plain=remove_markers(answer)
    fragments=[text.strip() for text in CLAIM_BOUNDARY.split(plain) if text.strip()]
    return fragments[:max_claims]


def important_words(sentence:str)->set[str]:
    return {
        term for term in tokens(sentence)
        if term not in STOP_WORDS and len(term)>=2
    }


def lexical_coverage(claim:str,source:str)->float:
    terms=important_words(claim)
    if not terms:
        return 0.0
    return len(terms & important_words(source))/len(terms)


def lexical_best(claim:str,sources:Iterable[str])->float:
    return max((lexical_coverage(claim,source) for source in sources),default=0.0)


def retrieve_current_source(
    store:MemoryStore,scope:str,excerpt:EvidenceExcerpt,
)->bool:
    if excerpt.scope!=scope:
        return False
    current=store.get(scope,excerpt.memory_id)
    return bool(
        current
        and current.source_sha256==excerpt.source_hash
        and current.content_sha256==excerpt.content_hash
        and current.revision==excerpt.revision
        and current.content==excerpt.content
    )


def check_source_integrity(
    excerpts:Iterable[EvidenceExcerpt],
    *,store:MemoryStore|None=None,scope:str="S",
    require_current:bool=True,
)->dict:
    included=list(excerpts)
    stale=[]
    duplicate=[]
    known=set()
    for item in included:
        if item.memory_id in known:
            duplicate.append(item.memory_id)
        known.add(item.memory_id)
        if item.scope!=scope:
            stale.append(item.memory_id)
        elif require_current:
            if store is None or not retrieve_current_source(store,scope,item):
                stale.append(item.memory_id)
    return {
        "valid":not stale and not duplicate,
        "stale":sorted(set(stale)),
        "duplicates":sorted(set(duplicate)),
        "included":len(included),
    }


def check_claim(
    claim:str,
    citations:list[str],
    allowed:dict[str,EvidenceExcerpt],
    policy:VerificationPolicy,
)->ClaimCheck:
    policy.validate()
    ids=tuple(sorted(set(citations)))
    valid=tuple(x for x in ids if x in allowed)
    unknown=tuple(x for x in ids if x not in allowed)
    sources=[allowed[item].content for item in valid]
    score=lexical_best(claim,sources)
    reasons=[]
    if policy.require_citations and not ids:
        reasons.append("missing_citation")
    if policy.prohibit_unknown_citations and unknown:
        reasons.append("unknown_source")
    if valid and score<policy.min_lexical_coverage:
        reasons.append("low_lexical_coverage")
    if policy.require_citations and not valid:
        reasons.append("no_valid_source")
    accepted=not reasons
    return ClaimCheck(
        sentence=claim,cited_ids=ids,valid_ids=valid,
        unknown_ids=unknown,best_lexical_coverage=score,
        supported_by_heuristic=accepted,reasons=tuple(reasons),
    )


def group_citations_by_sentence(answer:str)->list[tuple[str,list[str]]]:
    """Retain adjacent citation markers while splitting sentences."""
    fragments=[s.strip() for s in CLAIM_BOUNDARY.split(answer) if s.strip()]
    groups=[]
    for fragment in fragments:
        citations=extract_markers(fragment)
        claim=remove_markers(fragment).strip()
        if claim:
            groups.append((claim,citations))
        elif citations and groups:
            old,prior=groups[-1]
            groups[-1]=(old,prior+citations)
    return groups


def verify_answer(
    packet:AnswerPacket,excerpts:Iterable[EvidenceExcerpt],
    *,
    store:MemoryStore|None=None,scope:str="S",
    policy:VerificationPolicy|None=None,
)->VerificationReport:
    policy=policy or VerificationPolicy()
    policy.validate()
    sources=list(excerpts)
    integrity=check_source_integrity(
        sources,store=store,scope=scope,
        require_current=policy.require_current_memory,
    )
    allowed={item.memory_id:item for item in sources}
    # The generator may not cite an item that wasn't included in the prompt.
    if set(allowed)-set(packet.allowed_source_ids):
        raise ValueError("Verifier received excerpts outside injected prompt")
    citations=extract_markers(packet.answer)
    unknown=sorted(set(citations)-set(allowed))
    groups=group_citations_by_sentence(packet.answer)
    if len(groups)>policy.max_claims:
        raise ValueError("Generated answer contains too many claims to verify")
    checks=[
        check_claim(claim,ids,allowed,policy)
        for claim,ids in groups
    ]
    successful=sum(item.supported_by_heuristic for item in checks)
    fraction=successful/max(1,len(checks))
    accepted=(
        integrity["valid"]
        and (not unknown or not policy.prohibit_unknown_citations)
        and fraction>=policy.min_supported_claim_fraction
        and (not policy.require_citations or bool(citations))
    )
    return VerificationReport(
        accepted=accepted,source_integrity=integrity["valid"],
        unknown_citations=unknown,stale_sources=integrity["stale"],
        claims_checked=len(checks),
        claims_passing_heuristic=successful,claim_fraction=fraction,
        checks=checks,
        limitations=(
            "Keyword overlap cannot establish semantic entailment or factual correctness",
            "Retrieved memory can be incomplete, inaccurate or attacker-controlled",
            "A model can copy the citation marker without relying on a source",
        ),
    )


def verify_and_gate(
    packet:AnswerPacket,excerpts:list[EvidenceExcerpt],
    *,
    store:MemoryStore|None=None,scope:str="S",
    policy:VerificationPolicy|None=None,
    fallback:str="Insufficient verified evidence to answer this request.",
)->AnswerPacket:
    report=verify_answer(
        packet,excerpts,store=store,scope=scope,policy=policy,
    )
    packet.verified=report.accepted
    packet.evidence["verification"]=report.to_dict()
    packet.cited_sources=[
        item.public() for item in excerpts
        if item.memory_id in extract_markers(packet.answer)
    ]
    if not report.accepted:
        packet.rejection_reason="source_or_claim_verification_failed"
        packet.answer=fallback
        packet.cited_sources=[]
    return packet


def citation_precision(
    answers:list[AnswerPacket],
    reference_ids:list[set[str]],
)->dict:
    if len(answers)!=len(reference_ids) or not answers:
        raise ValueError("Aligned answer/gold sets required")
    totals=Counter()
    for packet,gold in zip(answers,reference_ids):
        selected=set(extract_markers(packet.answer))
        totals["cited"]+=len(selected)
        totals["correct"]+=len(selected & gold)
        totals["gold"]+=len(gold)
    return {
        "precision":totals["correct"]/max(1,totals["cited"]),
        "recall":totals["correct"]/max(1,totals["gold"]),
        "cited":totals["cited"],
        "gold":totals["gold"],
    }


def detect_unsupported_numbers(answer:str,excerpts:list[EvidenceExcerpt])->list[str]:
    """Simple exact-number review flag, not universal numeric entailment."""
    plain=remove_markers(answer)
    claim_numbers=set(re.findall(r"\b\d+(?:\.\d+)?\b",plain))
    source_text="\n".join(item.content for item in excerpts)
    source_numbers=set(re.findall(r"\b\d+(?:\.\d+)?\b",source_text))
    return sorted(claim_numbers-source_numbers)


def source_coverage(packet:AnswerPacket,excerpts:list[EvidenceExcerpt])->dict:
    mentioned=set(extract_markers(packet.answer))
    available={item.memory_id for item in excerpts}
    return {
        "cited":len(mentioned),
        "available":len(available),
        "unknown":sorted(mentioned-available),
        "cited_fraction":len(mentioned&available)/max(1,len(available)),
    }
