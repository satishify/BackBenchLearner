---
title: "Human-in-the-Loop Workflows"
description: "Insert human approval and correction at high-risk steps so agents stay fast on the safe path and accountable on the dangerous one."
---

**What is this for?** To show how **human-in-the-loop (HITL)** checkpoints make agentic systems safe enough for production.

**Why does it exist?** Agents can loop, get stuck, or try risky actions. In enterprise settings, we need pause points where a person approves before the system acts—especially for actions that are hard to undo.

## Intuition

Let the agent draft freely. Pause when the next action is hard to undo.

- A support agent can auto-tag tickets.
- A human should still green-light a $2,000 refund.
- An infrastructure agent that wants to **restart a pod** should pause at an approval node, wait for Slack approval, then continue.

HITL turns "autonomous" from all-or-nothing into a **risk dial**.

| Plain-English idea | What it means |
| --- | --- |
| **HITL checkpoint** | Graph pauses before a risky action; human approves or rejects |
| **Recursion limit** | Hard cap that stops infinite agent back-and-forth |
| **Summarize node** | Compresses old messages so state does not bloat |

:::key
HITL is not a failure of automation—it is how you ship agents where mistakes cost money, rights, or trust.
:::

```mermaid
flowchart TD
    A[Agent drafts action] --> B{High risk?}
    B -- Yes --> C[Human review]
    C --> D[Approve / edit / reject]
    B -- No --> E[Auto execute]
    D --> E
```

## How it works

### Where HITL pays off

- Financial transactions and credit changes.
- Legal, medical, or compliance-facing text.
- Security and account-access changes.
- External communications in sensitive contexts.
- Irreversible deletes or production deploys.
- Novel cases outside the eval distribution.

### HITL patterns

| Pattern | Behavior | Good for |
| --- | --- | --- |
| Approve / reject | Binary gate | Refunds, sends |
| Edit then run | Human fixes draft | Customer emails |
| Tool allow on ask | Agent requests elevation | Rare admin ops |
| Review sampling | Spot-check auto path | Quality audits |
| Escalation on confidence | Low score → human | Ambiguous intents |

### Production guardrails beyond HITL

Three guardrails often work together:

1. **HITL** — a human approves a risky action before execution.
2. **Recursion limit** — stops infinite agent arguments and runaway token burn.
3. **Summarize node** — keeps message history small enough to stay usable (**state bloat** is when history gets too large and slows the system).

### Designing the queue

Humans are a scarce resource. Rank the queue by risk and uncertainty, not first-in-first-out alone. Show the reviewer: goal, proposed action, evidence, policy citations, and blast radius. Capture the decision as structured feedback so you can train graders and tighten auto paths later.

### Autonomy schedule

1. **Shadow:** agent proposes, human always acts.
2. **Supervised:** auto on allowlisted low-risk; HITL otherwise.
3. **Autonomous scoped:** auto inside sandbox + budgets; HITL for elevations.

Move a class of actions to auto only when offline probes and online error rates stay under threshold for a defined window.

## In code

Risk-aware gating with an approval record.

```python
from dataclasses import dataclass
from typing import Callable

RISKY = {"create_refund", "delete_user", "send_external_email", "restart_pod"}

@dataclass
class Proposal:
    tool: str
    args: dict
    evidence: str
    risk_score: float  # 0..1

@dataclass
class Decision:
    status: str  # approved | rejected | edited
    args: dict
    reviewer: str

def needs_hitl(p: Proposal) -> bool:
    return p.tool in RISKY or p.risk_score >= 0.6

def human_review(p: Proposal) -> Decision:
    if p.tool == "create_refund" and p.args.get("amount_cents", 0) <= 5000:
        return Decision("approved", p.args, reviewer="alex")
    return Decision("rejected", p.args, reviewer="alex")

def execute(tool: str, args: dict) -> str:
    return f"ran {tool}({args})"

def run(p: Proposal, reviewer_fn: Callable[[Proposal], Decision] = human_review) -> str:
    if needs_hitl(p):
        d = reviewer_fn(p)
        if d.status == "rejected":
            return f"blocked by {d.reviewer}"
        args = d.args
    else:
        args = p.args
    return execute(p.tool, args)

print(run(Proposal("tag_ticket", {"id": "T1"}, "keyword", 0.1)))
print(run(Proposal("create_refund", {"amount_cents": 2000}, "policy#12", 0.7)))
print(run(Proposal("restart_pod", {"name": "api-7"}, "OOM alert", 0.85)))
```

Wire `human_review` to a real queue UI; keep the policy (`needs_hitl`) in version control next to evals.

## What goes wrong

- **Rubber-stamping.** Reviewers approve everything; HITL becomes theater.
- **Alert fatigue.** Too many low-value gates; humans stop reading evidence.
- **Unclear UI.** Missing blast radius and citations → inconsistent decisions.
- **No feedback loop.** Rejections never become golden tests.
- **All-human bottleneck.** Refusing to auto anything prevents learning where risk is low.
- **Silent auto path.** Users not told when a human did vs did not intervene.

## Putting it into practice

Define a promotion rule in writing: an action class may go auto when (a) offline probes pass for N days, (b) HITL reject rate stays under R%, and (c) zero Sev-1 incidents in the window. Demote automatically if any condition breaks.

Store decisions with reason codes (`policy_mismatch`, `amount_too_high`, `unclear_evidence`) so you can mine them for new eval cases.

Train reviewers like graders, not like firefighters. A five-minute rubric beats heroic judgment under load. Publish HITL latency as a product metric; if approvals take hours, users will route around the agent entirely.

## UX for trust

Show users when a human is in the loop ("Pending specialist review") and when the agent acted alone ("Completed automatically under policy #14"). For edited approvals, show the diff between model draft and shipped text in internal tools so reviewers learn which instructions to tighten.

## One-line summary

Put humans on the high-risk joint of the workflow—with ranked queues, structured decisions, recursion limits, and a path to graduate safe actions to auto—so agents stay useful without becoming unaccountable.

## Key terms

- **HITL (human-in-the-loop):** human approval or correction before risky actions.
- **Checkpoint:** pause point in a workflow graph (e.g., before restart pod).
- **Recursion limit:** hard cap that stops infinite loops in agent workflows.
- **State bloat:** when message history gets too large and slows the system.
- **Summarize node:** step that compresses older dialogue into a shorter summary.
- **Risk score:** estimate that an action needs review.
- **Autonomy schedule:** staged expansion of what runs without humans.
