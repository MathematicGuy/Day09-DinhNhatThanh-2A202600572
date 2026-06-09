# Guardrail Graceful Fallback Design

## Goal

Add a low-cost guardrail in front of supervisor routing so hostile comments receive a short polite default response without invoking policy or data workers.

## Scenarios

1. Abusive comment toward the assistant or service
   - Example: "You are stupid and useless"
   - Expected category: `abusive`
   - Expected behavior: return a calm response inviting the user to ask a shopping-support question.
   - Token-saving requirement: skip policy retrieval and data lookup.

2. Identity-based hateful comment
   - Example: "I hate all muslim people"
   - Expected category: `hate`
   - Expected behavior: decline hateful content and redirect to shopping-support help.
   - Token-saving requirement: skip policy retrieval and data lookup.

3. Normal frustrated shopping question
   - Example: "This delivery is terrible, where is order 1971?"
   - Expected category: none
   - Expected behavior: continue normal supervisor-worker routing because the user still asks an actionable support question.

## User Stories

1. As a shopper who writes a rude message, I receive a short respectful fallback so the assistant does not escalate the tone.
2. As a business owner, I avoid spending policy/data worker calls on abusive or hateful comments.
3. As an evaluator, I can inspect the trace and see that the supervisor blocked the turn before downstream workers.
4. As a Streamlit demo user, I can see the configured supervisor model and guardrail behavior in the live chat tab.

## Design

The supervisor calls a deterministic `classify_guardrail()` check before normal route classification. If the guardrail returns `abusive` or `hate`, `route_question()` returns status `guardrail_blocked`, both worker flags false, and a default response. The graph routes directly to the response worker, which emits the default response. Existing Streamlit chat rendering then displays the response and trace under the assistant message.

The Streamlit app defaults the supervisor model environment to `gpt-5.4-nano` when no model is already configured. Existing environment variables still win, so local overrides remain possible.
