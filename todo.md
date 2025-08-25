# TODOs

For now I've copied the code from the examples/prompt_evals folder. Which is the code for docs/experimental/tutorials/prompt.md

Now we want to create a detailed how to guide on how to evaluate, do error analysis, and iterate on a prompt.

In the folder examples/iterate_prompt you can find the code for the prompt.py and evals.py files.
- [x] Fix prompt.py based on below context
- [x] Run it
- [ ] Fix evals.py based on below context
- [ ] Run it for prompt 1
- [ ] Analyze the failures
- [ ] Create prompt 2
- [ ] Run evals for that. 
- [ ] See results and compare
- [ ] Choose one
- [ ] Write docs/experimental/howtos/iterate_prompt.md based on the code and the results.



# Overall plan

### Summary

A lean “How to iterate and improve your prompt” guide using **Customer Support Ticket Triage**. We classify tickets into **multi-label categories** plus a **single priority**. We keep metrics minimal and do **two iterations** only: (1) add crisp definitions; Analyze failures, (2) add few-shots to handle ambiguity and edge cases. Dataset is small (20 rows) but intentionally hard.

### Why this use case

- Common across SaaS/support teams
- Multi-label mirrors reality (`Billing;HowTo`, `Account;ProductIssue`)
- Priority is practical for routing/SLA
- **ProductIssue** ties directly to engineering/incident workflows

### Labels and priority (authoritative definitions)

- **Billing** — charges, taxes (GST/VAT), invoices, plans, credits
- **Account** — login/SSO, password reset, identity/email/account merges
- **ProductIssue** — malfunction: crashes, errors, won’t load, data loss, loops, outages
- **HowTo** — usage questions (“where/how do I…”, “where to find…”)
- **Feature** — new capability or improvement request
- **RefundCancel** — cancel/terminate and/or refund requests
- **AbuseSpam** — insults/profanity/spam; not just mild frustration

**Priority (single value)**

- **P0 (High)** — blocked from core action or money/data at risk
- **P1 (Normal)** — degraded/needs timely help, not fully blocked
- **P2 (Low)** — minor/info/how-to/feature

**Escalation rules**

- Never escalate on tone alone
- Use **P0** only with clear block/impact/time pressure; **P2** for informational/feature; otherwise **P1**

### How we’ll make Iteration 1 fail (by design)

- **Mixed intents**: a ticket can legitimately map to 2 labels (e.g., `Billing;HowTo`, `Account;ProductIssue`).
- **Buried cues**: the decisive signal appears at the **end** (“incognito works”, “not a billing error”).
- **Time pressure vs tone**: explicit deadlines (“audit tomorrow”) trigger **P1** without being blocked; tone alone should not.
- **Workarounds**: presence of a workaround ⇒ **P1** not **P0** (mini models often over-escalate).
- **Jargon + logs**: error codes (`429`, `invalid_token`), browser versions, regions, snippets—forces reading comprehension.
- **Red herrings**: words like “urgent”, “shipping”, “refund” mentioned in non-action contexts to tempt wrong labels.
- **Ambiguous phrasing**: feature vs how-to (“Is there a setting for… or can you add…”).

### Output contract (strict JSON)

```json
{"labels":["Billing","HowTo"], "priority":"P1"}

```

### Harder, longer dataset (20 rows)

CSV columns: `id,text,labels,priority`

```
1,"Upgraded to Plus on July 2 and my bank statement (ending 5021) shows two charges for the same day. I attached a screenshot to the email thread. No plan change since then—just want the duplicate reversed.","Billing","P1"
2,"SSO via Okta succeeds then bounces me back to /login with no session. Colleagues can sign in. I tried clearing cookies; same result. Error in devtools: state mismatch. I’m blocked from our boards.","Account;ProductIssue","P0"
3,"I need to export a board to PDF with comments and page numbers for our audit pack. I found ‘Export’ but comments didn’t appear in the file—am I missing a setting? Deadline is next week, not today.","HowTo","P2"
4,"Android app crashes when I tap Share on the board menu (Pixel 7, Android 14). Repro: open Board → Share → App closes. Crash dump attached; reinstall didn’t help. I can still use desktop meanwhile.","ProductIssue","P1"
5,"Please cancel our Team plan for Acme LLC. Finance asked for a refund of last month since we stopped using it after the pilot. Keep the workspace accessible until the end of this week for archiving.","RefundCancel","P1"
6,"Dashboard hangs on a spinner in Chrome 126.0 but the same account opens fine in Safari and Edge. Network tab shows a 504 from /projects. Not completely blocked, but it’s slowing down the team.","ProductIssue","P1"
7,"Is there a built-in way to schedule dark mode to follow sunset? If not, consider this a feature request; our designers swap themes daily and would love automation.","Feature","P2"
8,"For our EU teammates the web app sits on ‘Initializing…’ since ~09:10 CET. US teammates are fine. Status page shows no incident. We can’t access any boards on the EU side.","ProductIssue","P0"
9,"GST is getting added at checkout. I’m paying with a US card from NYC. I originally created the account while in Bangalore last year—do I need to update something so GST doesn’t apply?","Billing","P1"
10,"I signed up with my personal Gmail and later invited my work email. Can you move ownership of all projects to my work account and merge the seats so I don’t pay twice?","Account","P1"
11,"After sync, notes disappeared from two devices. I saw them briefly then they vanished—no trash entry. This is client work and we don’t have a backup. Please advise; we’re effectively stuck.","ProductIssue","P0"
12,"Do you offer a student discount on annual plans? I saw a community post from 2023 but the link is 404 now. If there is a verification step, what documents do you need?","Billing","P2"
13,"Following up on my cancellation—emailed on the 3rd and again on the 6th. Please confirm termination and ensure no further auto-charges. We’re closing the cost center this month.","RefundCancel","P1"
14,"I don’t have a billing issue; I just need to download invoices with a GST breakdown for Q2 FY24-25. Where exactly is the button in the new UI? Our audit is tomorrow morning.","Billing;HowTo","P1"
15,"Password reset emails rarely arrive; when one finally did, clicking produced ‘invalid_token’. Cleared cache, different browser, same behavior. I can’t access our workspace today.","Account;ProductIssue","P0"
16,"Offline mode would help when we review boards on flights. Ideally comments remain editable and sync when we reconnect. If that’s already possible, point me to the doc; otherwise please consider.","Feature","P2"
17,"Your login is garbage—keeps looping. Funny thing: it works in **Incognito** but not my normal profile even after disabling extensions. I can get in, but it’s wasting time. Fix it.","Account;ProductIssue;AbuseSpam","P1"
18,"We want to switch from monthly to annual without losing ~350 credits that rolled over from Q2. Is there a self-serve path, or do you need to migrate the balance manually?","Billing;HowTo","P1"
19,"Trial expired yesterday and we were auto-charged despite pausing the workspace last week (Workspace ID: acme-eu-prod). Please refund this cycle and prevent future charges.","Billing;RefundCancel","P1"
20,"Order webhooks started failing around 10:20 UTC with 429 ‘rate_limit exceeded’. Payload sizes unchanged. Should we raise limits on our plan or backoff differently? Orders aren’t syncing to ERP.","ProductIssue;HowTo","P0"

```

### Why this dataset stresses Iteration 1 (definitions-only)

- **Items 14 & 18** look like pure Billing, but the real ask is **where/how** → need `Billing;HowTo` and nuanced **P1** (time pressure) vs **P2**.
- **Item 17** is **not blocked** (incognito works) → should be **P1**, but mini models often jump to **P0** on strong tone.
- **Items 2, 15** combine `Account` with a genuine malfunction (`ProductIssue`) and clear **P0** language.
- **Items 6 & 4** include **workarounds** (other browser/desktop) → **P1** not **P0**.
- **Items 8 & 11** are unmistakable **P0**: team-wide outage and data loss.
- **Item 20** mixes **incident + plan guidance** (rate limits) → `ProductIssue;HowTo` (mini models often choose only one).
- **Items 9 & 12** are Billing but easily confused with Refund/Account without examples.

### Metrics (minimal)

- **Exact-set accuracy (labels)** — strict match of label sets.
- **Priority accuracy** — correct `P0/P1/P2`.
- *(Optional)* **Average Jaccard (labels)** — partial credit for overlap: `|pred ∩ gold| / |pred ∪ gold|`.

### Iterations (only two)

- **Iteration 1 — Definitions-only**
    
    Paste the one-line label/priority definitions and rules; enforce strict JSON. Expect failures on 14, 17, 20, etc.
    
- **Iteration 2 — Few-shots for ambiguity & edge cases**
    
    Include compact examples specifically encoding:
    
    - `Account+ProductIssue` with clear **P0** (item 15-style)
    - `Billing+HowTo` with time pressure ⇒ **P1** (item 14-style)
    - `ProductIssue` with workaround ⇒ **P1** (item 6-style)
    - `Feature` clean **P2** (item 7-/16-style)
    - `ProductIssue+HowTo` rate-limit (item 20-style)
    - **Abuse** without escalation (item 17-style)

### Pasteable prompts

### Iteration 1 (definitions-only)

```
You categorize a short customer support ticket into (a) one or more labels and (b) a single priority.

Allowed labels (multi-label):
- Billing: charges, taxes (GST/VAT), invoices, plans, credits.
- Account: login/SSO, password reset, identity/email/account merges.
- ProductIssue: malfunction (crash, error code, won't load, data loss, loops, outages).
- HowTo: usage questions (“where/how do I…”, “where to find…”).
- Feature: new capability or improvement request.
- RefundCancel: cancel/terminate and/or refund requests.
- AbuseSpam: insults/profanity/spam (not mild frustration).

Priority (exactly one):
- P0 (High): blocked from core action or money/data at risk.
- P1 (Normal): degraded/needs timely help, not fully blocked.
- P2 (Low): minor/info/how-to/feature.

Rules:
- Use only labels from the list. If none apply, return [].
- Do not escalate on tone alone; escalate only if block/impact/time is explicit.
- Respond with STRICT JSON only, no commentary.

Return exactly:
{"labels":[<labels>], "priority":"P0"|"P1"|"P2"}
Ticket: "<PASTE_TICKET_TEXT_HERE>"

```

### Iteration 2 (few-shots added)

```
You will label a customer ticket with (a) one or more categories and (b) a single priority.

LABELS & PRIORITY: (same definitions as before)

FEW-SHOTS:
Q: "Password reset emails rarely arrive; when one finally did, clicking produced ‘invalid_token’. I can’t access our workspace."
A: {"labels":["Account","ProductIssue"], "priority":"P0"}

Q: "I don’t have a billing issue; I just need to download invoices with a GST breakdown for Q2. Our audit is tomorrow."
A: {"labels":["Billing","HowTo"], "priority":"P1"}

Q: "Dashboard hangs on a spinner in Chrome 126 but works in Safari."
A: {"labels":["ProductIssue"], "priority":"P1"}

Q: "Is there a built-in way to schedule dark mode to follow sunset? If not, consider this a feature request."
A: {"labels":["Feature"], "priority":"P2"}

Q: "Order webhooks failing with 429 ‘rate_limit exceeded’. Should we raise limits on our plan?"
A: {"labels":["ProductIssue","HowTo"], "priority":"P0"}

Q: "Your login is garbage—loops forever. It works in Incognito though."
A: {"labels":["Account","ProductIssue","AbuseSpam"], "priority":"P1"}

OUTPUT (strict JSON):
{"labels":[<labels from allowed set>], "priority":"P0"|"P1"|"P2"}

Now label this ticket:
Ticket: "<PASTE_TICKET_TEXT_HERE>"

Respond ONLY with JSON.

```

### Evaluation flow (Ragas, concept-only)

- Use the 20-row CSV above
- Define scorers:
    - **labels_exact**: predicted label set equals gold
    - **priority_acc**: predicted priority equals gold
    - *(Optional)* **labels_jaccard** for partial credit on multi-label
- Run Iteration 1 (definitions) and
- 1. Analyse the failed samples
2. Does a small error analysis to understand where it fails, then devices a “plan” ( which here is few shot examples) to  fix the issue. Ideally you should only add examples for cases where you know the system fails ( doesn’t learn with 0 shot)
- Iteration 2 (few-shots) on the same dataset
- Compare metrics and inspect the hardest failures (ambiguous Billing/HowTo; Account+ProductIssue; Abuse without escalation)

So the loop is always
measure → observe → hypothesise → experiment → measure…