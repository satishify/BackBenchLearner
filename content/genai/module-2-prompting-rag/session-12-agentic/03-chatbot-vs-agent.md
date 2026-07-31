---
title: "Difference Between a Chatbot and an Agent"
description: "Chatbots respond; agents complete tasks—compare turns, tools, autonomy, and when each is the right product shape."
---

**What is this for?** To clarify when you need a **chatbot** (answers in a thread) vs an **agent** (finishes a job in the world).

**Why does it exist?** Both can look like a chat window, but they promise different things. Mixing them up leads to under-powered "agents" that only talk, or over-armed chatbots that should never have had write access.

## Intuition

**Chatbot:** "How do I reset my password?" → explains the steps.

**Agent:** verifies identity → triggers the reset link → updates the ticket → confirms completion.

Same surface (a chat window), different contract with the user. Chatbots optimize for helpful messages. Agents optimize for completed outcomes—and therefore need tools, loops, and stop rules.

| Aspect | Chatbot | Agent |
| --- | --- | --- |
| Main job | Answer questions | Complete tasks (and answer) |
| Tools | Optional / limited | Core capability |
| Style | Mostly single turn | Multi-step loop |
| Autonomy | Low | Higher (scoped) |
| Success metric | Customer satisfaction, deflection | Task success rate |
| Failure mode | Wrong advice | Wrong side effect |

:::key
A chatbot ends when it sends a reply. An agent ends when a checkable outcome happens—or when a human or budget stops it.
:::

### Why linear RAG is not an agent

A simple RAG (retrieval-augmented generation) pipeline:

```
query  ->  retrieve documents  ->  generate answer
```

That can read evidence once. It cannot naturally pause, reflect, or loop back when the first chunk is incomplete.

An agent loop:

```
observe  ->  think  ->  act  ->  observe  ->  ...  ->  answer
```

**DevOps example:** A payment-gateway failure may need a code diff, log traces, and past incident notes. If the first retrieved chunk is incomplete, the system must search again—not blindly answer. That is agent territory, not static RAG.

## How it works

### Chatbot shape

1. User message in.
2. Optional retrieval for FAQs.
3. Model generates a reply.
4. Done.

State is conversational. Side effects, if any, are rare and manually triggered by the user following instructions.

### Agent shape

1. User states a goal.
2. System plans steps.
3. Tools execute under policy.
4. Results feed the next decision.
5. Ends on success, escalation, or budget.

State includes plan progress and external system status ("ticket #442 closed").

### Hybrid products (most of reality)

Many apps are chatbots with a few agentic buttons: "Create the ticket for me." That is fine. Label autonomy honestly. When the UI promises "I'll handle it," you owe the user agent-grade controls: confirmations, receipts, and undo paths.

### When to choose which

Choose a **chatbot** when:

- The user must learn or decide.
- Actions are better done in existing UIs.
- Risk of automated side effects is high relative to benefit.

Choose an **agent** when:

- The workflow is repetitive and well-specified.
- APIs exist for each step.
- Success is objectively checkable.
- You can afford human-in-the-loop (HITL) on the risky subset.

### Example contrast

Password reset again:

- **Chatbot:** paste the help-center steps; user clicks around.
- **Agent:** calls `start_reset(user_id)` after auth checks; writes audit log; replies with "link sent to the email on file" without exposing the address fully.

If your "agent" cannot call `start_reset`, it is a chatbot wearing a cape.

## In code

Two handlers, same user utterance—different contracts.

```python
def chatbot_password_reset(user_msg: str) -> str:
    return (
        "To reset your password: open Settings -> Security -> Reset. "
        "We will email a link. I cannot trigger it from chat."
    )

def agent_password_reset(user_id: str, verified: bool) -> dict:
    if not verified:
        return {"status": "need_verification", "next": "otp_challenge"}
    # host-side tool — not free-form model text
    receipt = {"action": "start_reset", "user_id": user_id, "ok": True}
    return {
        "status": "done",
        "user_visible": "A reset link is on its way to your email on file.",
        "audit": receipt,
    }

print(chatbot_password_reset("reset my password"))
print(agent_password_reset("u_123", verified=True))
```

Product question: which return type did you promise in the UI copy?

## What goes wrong

- **Marketing rename.** Calling every bot an agent without tools or loops.
- **Chatbot with god tools.** Still single-turn mentally, but one injection away from disaster.
- **Agent without receipts.** Users cannot tell what changed in CRM, email, or bank.
- **No escalation path.** Agents that never hand off create support debt.
- **Wrong metric.** Optimizing chat satisfaction while task success collapses.
- **Hidden multi-step.** Long silent tool runs with no progress UI feel broken.

## Putting it into practice

Audit your UI strings. Phrases like "I'll take care of that" imply agent semantics; "Here's how you can..." implies chatbot semantics. Mismatch is a trust bug.

For each flow, pick a primary metric: deflection or satisfaction for chatbots, task success and undo rate for agents. Review weekly which metric moved after prompt or model changes.

A useful migration path is progressive disclosure of agency: answer first, then offer a button "Do this for me" that enters the tool loop with an explicit receipt. Users who only wanted information never pay the latency or risk tax; users who wanted completion get a clear contract.

## Boundary cases

Some products sit on the line: a FAQ bot that can create a ticket is mostly a chatbot with one agentic escape hatch. Treat each side effect as its own mini-agent contract—schema, auth, receipt—even if the rest of the UI stays conversational.

Conversely, an "agent" that only drafts pull-request text without opening the PR is still a chatbot for shipping purposes. Name capabilities after what changes in external systems, not after the model vendor's marketing tier.

## One-line summary

Use chatbots to inform and guide; use agents when the product must execute multi-step, tool-backed work to a checkable outcome under policy and budgets.

## Key terms

- **Chatbot:** conversational system optimized for replies.
- **Agent:** goal-seeking loop that acts through tools.
- **Linear RAG:** one-shot retrieve-then-generate pipeline.
- **Task success rate:** fraction of goals completed correctly.
- **Side effect:** durable change outside the chat transcript.
- **Escalation:** handoff to a human or safer workflow.
- **Receipt / audit:** record of what the agent changed.
