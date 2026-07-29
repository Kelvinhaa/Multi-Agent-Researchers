---
doc_id: HFS-ENG-004
title: Access Control Policy
version: "4.0"
effective: 2026-03-01
---

# Access Control Policy

## Purpose

This policy sets out how Harbourline Freight Systems controls access to
production systems, from everyday authentication through to how access is
reviewed and revoked.

## Single sign-on

Access to all Harbourline systems is authenticated through single sign-on.
Local passwords are not permitted for any production or corporate system,
so that access can be centrally granted, reviewed and revoked through a
single identity provider rather than a patchwork of separate credentials.

| Requirement | Detail |
|---|---|
| Authentication | SSO only |
| Local passwords | Not permitted |

## Production access elevation

Standing access to production systems is not granted by default. Engineers
who need production access request a time-boxed elevation lasting up to 8
hours, scoped to the systems required for the task at hand. Elevated access
expires automatically at the end of the 8-hour window and must be
re-requested if more time is needed.

## Access reviews

Access to production systems and sensitive internal tools is reviewed
quarterly. Each review confirms that every engineer's standing permissions
still match their current role, and any access no longer required is
removed as part of the review rather than left in place.

## Offboarding

When an employee leaves Harbourline or changes role in a way that removes
their need for production access, that access is revoked within 2 hours of
the offboarding or role change being processed by People & Culture. This
target applies to both voluntary departures and involuntary terminations.
