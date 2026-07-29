# Harbourline Freight Systems — canonical facts

Internal scaffolding for the eval corpus. NOT ingested — lives outside `docs/corpus/`.
Every figure in `eval/corpus_src/` must trace to a line in this file.

## Company

Harbourline Freight Systems. B2B logistics software. Founded 2014.
HQ Melbourne; offices Sydney, Perth, Auckland. 340 staff.

## People

| Fact | Value | Document |
|---|---|---|
| Annual leave | 25 days per year | leave_policy |
| Personal/carer's leave | 12 days per year | leave_policy |
| Leave request notice | 10 business days for 5+ consecutive days | leave_policy |
| Leave balance cap | 40 days, excess forfeited each 30 June | leave_policy |
| Leave request response | 5 business days | leave_policy |
| Medical certificate required | 3+ consecutive days, or any absence immediately before/after a public holiday | leave_policy |
| Parental leave, primary carer | 18 weeks at full pay | parental_leave |
| Parental leave, secondary carer | 6 weeks at full pay | parental_leave |
| Parental leave eligibility | 12 months continuous service | parental_leave |
| Parental leave, superannuation | Paid on unpaid portion for up to 12 months | parental_leave |
| Parental leave split | May be split into up to 2 blocks with manager agreement | parental_leave |
| Parental leave, return to role | Same role, or a comparable role at the same level and pay | parental_leave |
| Remote days | Up to 3 days per week | remote_work |
| Core hours | 10:00–15:00 AEST | remote_work |
| Fully remote | Requires Executive Leadership Team approval | remote_work |
| Home office allowance | $650 once per 24 months | remote_work |
| Review cycle | Twice yearly, March and September | performance_review |
| Promotion nominations close | 14 February and 14 August | performance_review |
| Late promotion nomination | Held over to the following cycle | performance_review |
| Rating scale | 1–5, where 3 is "meets expectations" | performance_review |
| Rating consequence, 1-2 | Triggers a support plan agreed with the manager | performance_review |
| Rating consequence, 4-5 | Required to be considered for promotion that cycle | performance_review |
| Calibration | Panel of 3 department heads | performance_review |
| Gift declaration threshold | $200 | code_of_conduct |
| Conflict declaration window | 5 business days | code_of_conduct |
| Secondary employment | Written approval required | code_of_conduct |
| Non-retaliation protection | Reports handled confidentially; no retaliation against the reporter | code_of_conduct |

## Finance

| Fact | Value | Document |
|---|---|---|
| Expense submission window | 30 days from spend | expense_reimbursement |
| Receipt required above | $75 | expense_reimbursement |
| Reimbursement pay run | 15th of each month | expense_reimbursement |
| Domestic meal allowance | $85 per day | travel_policy |
| International meal allowance | $120 per day | travel_policy |
| Hotel cap, Sydney/Melbourne | $280 per night | travel_policy |
| Hotel cap, elsewhere domestic | $220 per night | travel_policy |
| Business class | Flights over 4 hours, ELT only | travel_policy |
| Corporate card limit, standard | $5,000 per month | corporate_card |
| Corporate card limit, manager | $15,000 per month | corporate_card |
| Card reconciliation window | 10 business days | corporate_card |
| Card suspension | After 2 missed reconciliations | corporate_card |
| Procurement, manager approval | Under $5,000 | procurement_approval |
| Procurement, department head | $5,000–$25,000 | procurement_approval |
| Procurement, CFO | $25,000–$100,000 | procurement_approval |
| Procurement, board | Above $100,000 | procurement_approval |
| New vendor security review | Required if handling customer data | procurement_approval |
| Standard payment terms | 30 days | invoicing_payment_terms |
| Enterprise payment terms | 45 days | invoicing_payment_terms |
| Late payment fee | 1.5% per month | invoicing_payment_terms |
| Purchase order required above | $2,500 | invoicing_payment_terms |

## Engineering

| Fact | Value | Document |
|---|---|---|
| P1 internal acknowledgement | 15 minutes | incident_response |
| P1 resolution target | 4 hours | incident_response |
| P2 acknowledgement | 1 hour | incident_response |
| P2 resolution target | 12 hours | incident_response |
| P3 acknowledgement | 4 hours | incident_response |
| P3 resolution target | 5 business days | incident_response |
| P1 status page update | Within 30 minutes | incident_response |
| P1 incident commander | Mandatory | incident_response |
| On-call rotation length | 7 days | on_call_rotation |
| On-call handover | Wednesday 10:00 AEST | on_call_rotation |
| On-call allowance | $900 per week | on_call_rotation |
| On-call frequency cap | 1 rotation per 4 weeks | on_call_rotation |
| Deploy window | Monday–Thursday, no Friday deploys | release_deployment |
| Canary stage | 10% of traffic for 30 minutes | release_deployment |
| Automatic rollback | Error rate above 2% | release_deployment |
| Production deploy approval | Release manager | release_deployment |
| Production access | Time-boxed 8-hour elevation | access_control |
| Access review cadence | Quarterly | access_control |
| Offboarding revocation | Within 2 hours | access_control |
| SSO | Mandatory, no local passwords | access_control |
| Customer shipment data retention | 7 years | data_retention |
| Application log retention | 90 days | data_retention |
| Audit log retention | 3 years | data_retention |
| Backup retention | 35 days | data_retention |
| Deletion request fulfilment | Within 30 days | data_retention |

## Commercial

| Fact | Value | Document |
|---|---|---|
| Discount, account executive | Up to 10% | discount_approval |
| Discount, sales manager | 10–20% | discount_approval |
| Discount, VP Sales | 20–30% | discount_approval |
| Discount, CFO and CEO | Above 30% | discount_approval |
| Uptime, standard tier | 99.9% | sla_service_credits |
| Uptime, enterprise tier | 99.95% | sla_service_credits |
| Service credit, below 99.9% | 10% of monthly fee | sla_service_credits |
| Service credit, below 99.0% | 25% of monthly fee | sla_service_credits |
| Service credit, below 95% | 50% of monthly fee | sla_service_credits |
| Contractual P1 response | 30 minutes | sla_service_credits |
| Credit claim window | 30 days from incident | sla_service_credits |
| Service credit bands | Not cumulative; customer receives the credit for the lowest band their uptime falls into | sla_service_credits |
| Money-back window | 30 days, new annual contracts | refund_policy |
| Pro-rata cancellation notice | 90 days | refund_policy |
| Usage overages | Non-refundable | refund_policy |
| Breach notification | Within 72 hours | data_processing |
| Default data residency | ap-southeast-2 | data_processing |
| EU data residency option | eu-west-1 | data_processing |
| Sub-processor change notice | 30 days | data_processing |
| Partner tiers | Registered, Silver, Gold | partner_program |
| Gold tier requirement | $500,000 referred ARR | partner_program |
| Partner margin | 15% / 22% / 30% by tier | partner_program |
| Deal registration validity | 90 days | partner_program |

## Planted near-misses

Deliberate traps. A weak retriever confuses these; they are what makes
`recall_at_k` move.

| # | Collision | Documents |
|---|---|---|
| 1 | P1 response: 15 min internal ack vs 30 min contractual | incident_response vs sla_service_credits |
| 2 | "30 days" means four different things | expense_reimbursement, invoicing_payment_terms, refund_policy, data_retention |
| 3 | "90 days" means two different things | refund_policy vs partner_program |
| 4 | Spending limits appear in three finance documents | expense_reimbursement, travel_policy, corporate_card |
| 5 | Retention periods differ by data class within one document | data_retention |

## Deliberately absent

Never appears in any corpus document. Used for unanswerable golden items.

- Employee stock option or equity policy
- Annual professional development / training budget
- Bereavement leave entitlement
- Preferred airline or travel booking vendor
