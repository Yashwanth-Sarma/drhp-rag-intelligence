---
type: Playbook
title: Period and financial metric comparability
description: Application checks for comparing revenue, profit, cash flow and growth.
status: stable
generated: {by: "finsight/0.3", at: "2026-09-16T00:00:00Z"}
tags: [revenue, growth, comparison, financial, profit, cash]
---
# Comparison prerequisites

Match issuer for growth, metric definition, currency, scale, consolidated or
standalone basis, fiscal period boundaries and duration. Preserve restatement
status. An interim quarter must not be presented as a full year. Balance sheet
dates describe instants; income and cash-flow periods describe durations.

Use the application's Decimal computation after validating operands against
filing evidence. A nonpositive growth baseline requires a different explanation;
do not emit a conventional growth percentage. Do not infer currency conversion.

This is unverified application methodology. It supplies no company facts.
See [table extraction](/methods/financial-tables.md).
