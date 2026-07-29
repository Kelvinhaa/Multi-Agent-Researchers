---
doc_id: HFS-ENG-003
title: Release and Deployment Policy
version: "5.2"
effective: 2026-05-20
---

# Release and Deployment Policy

## Purpose

This policy sets out when Harbourline Freight Systems engineering may
deploy to production, how a release is rolled out safely, and who must
approve a production deployment.

## Deploy windows

Production deployments are permitted Monday to Thursday. No deploys are
made on Fridays, so that any issue arising from a release has a full
business day of team availability behind it rather than running into the
weekend. Deployments outside the standard window require sign-off from the
release manager and the on-call primary.

| Day | Deploys permitted |
|---|---|
| Monday–Thursday | Yes |
| Friday | No |

## Canary rollout

Every production release is first rolled out to a canary slice of 10% of
traffic, held for 30 minutes before proceeding to full rollout. The canary
period allows error rates and latency to be observed on a limited blast
radius before the release reaches all customers.

## Automatic rollback

If the error rate observed during the canary period, or after full
rollout, rises above 2%, the release is automatically rolled back to the
previous version without waiting for manual intervention. An automatic
rollback pages the on-call primary, who follows the Incident Response
Policy to assess customer impact and classify the event.

## Approval

Every production deployment requires approval from the release manager
before it proceeds, in addition to the standard code review process. The
release manager checks that the canary and rollback configuration for the
release are correctly set before granting approval.
