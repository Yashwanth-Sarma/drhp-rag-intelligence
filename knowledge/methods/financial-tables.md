---
type: Playbook
title: Financial table evidence requirements
description: Preserve units, headers, signs, restatement and accounting basis.
status: stable
generated: {by: "finsight/0.3", at: "2026-09-16T00:00:00Z"}
tags: [financial, revenue, profit, cash, balance, tables, units]
---
# Before extracting a metric

Bind a cell to its row label, all applicable column headers, period, currency,
scale, accounting basis and footnotes. Retain page coordinates and source crop.
Repeated headers and landscape pages require explicit layout handling.
Parentheses may indicate negative numbers; a dash is not automatically zero.
Restated and originally reported figures are distinct observations.

Native text block retrieval does not reconstruct table semantics. Ask for
review or abstain when the cell-to-header relation is uncertain. Never build a
chart from an unbound numeric token. This playbook is application-authored and
has not received professional accounting review.
