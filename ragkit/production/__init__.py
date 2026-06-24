"""ragkit.production — Part IV: making a correct system survivable in production (Book Ch 13, 15, 16).

    generation/     ground · cite · abstain · arbitrate conflicting evidence            (Ch 13)
    security/       retrieval-time ACL trimming · multi-tenancy · PII · injection depth (Ch 15)
    serving/        vLLM continuous batching · prompt-cache discipline · model routing  (Ch 16)
    observability/  OTel GenAI + OpenInference spans · drift detection · feedback loop   (Ch 16)

The discipline of this Part: everything here is invisible to an offline nDCG@10 sweep and visible
only to a user, an auditor, or an adversary. A correct answer is the demo; a generator that fails
*honestly*, a retrieval layer that *trims before the model sees a chunk*, a tail bounded at P99,
and a system that *sees its own drift* are what make it shippable. Evaluation — the harness that
proves any of this moved a number — is its own top-level module, `ragkit.eval` (Ch 14).

Phase-1 scaffold. Phase 2 exports the top-level conveniences listed below.
"""

from ragkit.production.generation import GroundedAnswer, GroundedGenerator
from ragkit.production.observability import psi, psi_alert, trace
from ragkit.production.security import InjectionDetector, PIIRedactor, enforce_acl
from ragkit.production.serving import ModelRouter, SemanticCache

__all__ = [
    "GroundedGenerator",
    "GroundedAnswer",
    "PIIRedactor",
    "InjectionDetector",
    "enforce_acl",
    "SemanticCache",
    "ModelRouter",
    "psi",
    "psi_alert",
    "trace",
]
