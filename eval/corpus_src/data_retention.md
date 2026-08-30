---
doc_id: HFS-ENG-005
title: Data Retention Policy
version: "3.3"
effective: 2026-02-15
---

# Data Retention Policy

## Purpose

This policy sets how long Harbourline Freight Systems retains different
classes of data, and how quickly a deletion request is fulfilled once
received.

## Retention periods by data class

Retention periods differ significantly by data class, reflecting different
legal, operational and security needs. Customer shipment data, which
underpins billing history and customer disputes, is retained for 7 years.
Application logs, used mainly for short-term debugging, are retained for
90 days. Audit logs, which record security-relevant events such as access
changes, are retained for 3 years. Database backups are retained for 35
days on a rolling basis, with each new backup cycle expiring the oldest.

| Data class | Retention period |
|---|---|
| Customer shipment data | 7 years |
| Application logs | 90 days |
| Audit logs | 3 years |
| Backups | 35 days |

## Deletion requests

Where a customer or former employee submits a valid deletion request for
their personal data, Harbourline fulfils that request within 30 days,
removing the data from production systems and scheduling its removal from
backups as they naturally expire. Data that Harbourline is required to keep
for legal or regulatory reasons, such as certain financial records, is
retained beyond a deletion request until the applicable retention period
above has elapsed.

## Backups and disaster recovery

Backups exist to support disaster recovery and are not treated as a
long-term archive. Restoring from a backup older than the 35-day retention
window is not possible once that backup has expired from the rolling
schedule.
