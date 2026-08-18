Text-to-SQL is one of the most practical applications of LLMs, translating human language into database queries. But naively prompting an LLM often leads to incorrect queries: wrong filters, missing group-by, or misuse of metrics. In this article, we build a robust end-to-end Text-to-SQL pipeline that goes beyond raw generation. Our workflow combines schema retrieval (to scope queries to relevant tables and columns), business rules (to enforce canonical metric definitions), a reasoning planner (to convert natural language into structured JSON plans), a SQL writer (to reliably generate queries), and an LLM-Judge (to evaluate correctness via static and execution-based checks). This layered approach reduces hallucinations, enforces consistency, and makes the system extensible for enterprise-scale use cases where accuracy and governance matter as much as automation.
Why this works better than naive prompting:
Scoped schema snippets keep the LLM focused, preventing it from “inventing” tables/columns. Also, the information passed to prompt is contextually relevant.
Business rules ensure ambiguous terms (like “Average Forecast” or “At Risk”) map to canonical SQL expressions.
Two-agent split (Planner + SQL writer) improves reasoning and reduces syntax/logic errors.
LLM-Judge provides measurable correctness scores, enabling iterative improvement.
Step 1: Schema Extraction
This is the first (and only occasionally repeated) step of the Text-to-SQL pipeline: Capturing the database’s schema into a clean JSON artifact the rest of the system can reliably use.
Why a schema snapshot?
Decouple LLM planning from live DBs.
Deterministic generation (no “what tables exist today?” flakiness).
Faster iteration (LLM sees a small, scoped snippet instead of the entire catalog).
Safer (We can redact/exclude sensitive tables/columns).
What this module produces:
A single JSON per database (e.g., SALES_schema.json) with:
DB metadata (dialect/driver/database)
Schemas → Tables
For each table:
Columns (types, PK/FK flags, comments)
Indexes (unique + columns)
Foreign keys (lightweight)
(Optional) Approx row counts
(Optional) Value sketches (top categories, numeric stats)
This is the ground truth our retriever + planner will consume.
Configuration knobs (env + schema_config)
Key environment variables:
DB_SCHEMAS: hard-include only these schemas (comma-separated).
DB_INCLUDE_SCHEMAS / DB_EXCLUDE_SCHEMAS: allow-list / deny-list at schema level.
DB_EXCLUDE_TABLE_KEYWORDS: case-insensitive substrings to drop noisy tables (e.g., snapshot,tmp,backup).
MYSQL_SCAN_MODE: current (only current db) vs “all schemas”.
DB_STATEMENT_TIMEOUT_MS: enforce per-statement execution caps during introspection.
TOPK_CATS / NUMERIC_STATS / ROWCOUNT_SAMPLE: feature toggles for sketches/rowcounts.
Step 2: Schema Retrieval (Scoping the Search Space)
One of the biggest reasons LLMs fail at Text-to-SQL is schema overload. Enterprise databases often have hundreds of tables with cryptic names (tbl_cfg_inv_hist), duplicate columns (qty, quantity_value), and inconsistent metadata. If we just dump the entire schema into the model prompt, two things happen:
Context bloat — the prompt becomes too large to handle efficiently.
Hallucinations — the LLM may “invent” relationships or pick irrelevant tables.
The solution is a Schema Retriever: a lightweight IR (information retrieval) module that dynamically selects only the relevant subset of the schema for each query.
Why it matters?
Keeps prompts small and focused.
Reduces hallucinations by excluding irrelevant tables/columns.
Respects business rules (e.g., use “Average Forecast” instead of “Forecast”).
Provides the SQL Agent with only what it needs to reason correctly.
How it works?
Schema Metadata Loader:
Load the schema JSON into structured TableMeta objects.
Each table stores description, columns, indexes, FKs, and row counts.
2. Normalization & Tagging
Normalize column names (item_id → item id).
Tag columns as numeric, datetime, or categorical using helpers (is_numeric, is_datetime, is_dimension).
Create compact column signatures like "pk, fk, num".
3. Document Builder
Construct a textual representation of each table: table name, description, column names, and column types.
This text becomes the input to a TF-IDF retriever for semantic matching.
4. Intent Detection
Parse the user query for aggregation (sum, avg), ranking (top, lowest), or trend (daily, monthly).
Intent guides column scoring (e.g., boost numeric columns for aggregation).
5. Table Ranking
Use TF-IDF + cosine similarity between the query and table docs to select the top-K candidate tables.
6. Column Ranking
Within each candidate table, score columns based on:
Direct text matches.
Column comments/descriptions.
Intent alignment (numeric for sums, datetime for trends).
PK/FK importance.
Heuristics (e.g., boost amount, qty, date).
7. Expand by Foreign Keys
Pull in related tables one hop away (via FK), so joins are possible.
8. Build Schema Snippet
Output a scoped schema snippet in YAML-like form, e.g.:
dialect: postgresql
tables:
— name: text2sqlusr.suppliers
desc: Supplier master table
columns:
— supplier_id (pk,cat)
— supplier_name (cat)
— region (cat)
— name: text2sqlusr.configurations
columns:
— config_id (pk,cat)
— supplier_id (fk,cat)
— item (cat)
relationships:
— text2sqlusr.configurations.supplier_id -> text2sqlusr.suppliers.supplier_id
business_context:
note: Use ‘Average Forecast’ instead of ‘Forecast’; Projected Forecast indicates risk.
Why this step is critical?
Planner-friendly: The reasoning model has just enough schema to produce a correct JSON plan.
SQL-agent ready: The SQL writer won’t hallucinate unknown tables.
Business-safe: Canonical rules are baked into the snippet.
Step 3: Running the SQL Agent (Planner + SQL Writer)
Once we have a scoped schema snippet from the Schema Retriever, the next step is to run the SQL Agent.
This involves two distinct roles:
Planner LLM — interprets the user’s query and produces a structured JSON plan.
SQL Writer LLM — converts that JSON plan + schema snippet into a valid SQL statement.
This two-step design is critical because it enforces determinism, rule-checking, and business alignment before SQL is written.
Why not go directly from Query → SQL?
If we let the LLM jump straight from “What parts have the lowest forecast on July 4th?” → SQL, it might hallucinate column names, miss join keys, or misuse aggregations.
Instead, the Planner forces it to explicitly commit to:
Intent (aggregation, top-k, trend, listing)
Tables and Joins
Filters (date ranges, canonical mappings like “at risk” → %Data Measure% + color_code=RED)
Metrics (always wrapped in COALESCE for safety)
Group By / Order By / Limit
Only post this we let the SQL Writer generate the SQL.
How the Agent Works:
1. Planner Prompt
We build a system message with hard rules:
No SELECT * allowed.
Every aggregation must include proper GROUP BY.
Canonical mappings (e.g., “At Risk” → Data Measure + Color Code for At Risk).
Dates normalized to ISO (single day → week_day='D', weeks → week_day='W').
Self-joins enforced for mismatch queries (e.g., Supply On-Hand vs Supply On-Hand (Actual)).
The user message includes:
User query.
Scoped schema snippet.
Business rules.
Date context (today, timezone).
Few-shot examples (Q/A pairs).
The Planner LLM returns JSON like this:
Get Abhishek Sharma’s stories in your inbox
Join Medium for free to get updates from this writer.

Subscribe

Remember me for faster sign in
{
“intent”: “aggregation”,
“tables”: [“text2sqlusr.postgresql”],
“filters”: [
“level_type = ‘XXX’”,
“data_measure = ‘Forecast’”,
“week_day = ‘D’”,
“CAST(start_date AS DATE) = DATE ‘2025–07–04’”
],
“metrics”: [
{“expr”: “MIN(COALESCE(quantity_value,0))”, “alias”: “min_quantity”}
],
“group_by”: [“site”, “supplier”, “item”, “data_measure”],
“order_by”: [{“expr”: “min_quantity”, “dir”: “ASC”}],
“projections”: [“site”, “supplier”, “item”, “data_measure”, “min_quantity”],
“limit”: 1
}
2. SQL Writer Prompt
The SQL Writer takes the plan + snippet and outputs SQL like this:
SELECT site, supplier, item, data_measure,
MIN(COALESCE(quantity_value,0)) AS min_quantity
FROM text2sqlusr.postgresql
WHERE level_type = ‘XXX’
AND data_measure = ‘Forecast’
AND week_day = ‘D’
AND (CAST(start_date AS DATE) = DATE ‘2025–07–04’ OR CAST(end_date AS DATE) = DATE ‘2025–07–04’)
GROUP BY site, supplier, item, data_measure
ORDER BY min_quantity ASC
LIMIT 1;
End Result: Instead of brittle “direct-to-SQL”, the two-phase SQL Agent gives us:
Structured reasoning (Planner JSON).
Robust SQL generation (SQL Writer).
Safe defaults + retries.
Step 4: Evaluation with Custom LLM Judge (LLM-J)
After generating SQL, the final challenge is evaluating correctness.
Traditionally, this means running queries against the database and comparing row-sets. But that’s brittle:
Gold queries may include extra projections that aren’t necessary.
Aggregations may use slightly different limits or aliases.
The schema may evolve, breaking comparisons.
Instead, we built a Custom LLM Judge (LLMJ) that performs static + semantic evaluation of SQL pairs:
Static checks (cheap regex-level validation).
Semantic rubric-based grading (LLM reasoning with explicit JSON rubric).
Why do we need LLM-J?
Gold SQL is human-written → prone to noise (extra columns, arbitrary order by).
Row-set comparison is expensive (requires DB access, slow for big tables).
We care about semantic intent, not byte-for-byte SQL equality.
LLMJ solves this by combining:
Static rules (schema scope, group by correctness, no SELECT *).
Business-aware rubric (SUM vs COUNT mismatches matter, aliases don’t).
Score scaling (normalized [0–1] → scaled [1–5] for human readability).
1. Static Checks
Before the LLM judge runs, we run lightweight parsing:
Does it parse? (SELECT ... FROM exists)
Scoped schema only? (no hallucinated tables/columns)
Projection policy (no SELECT *)
LIMIT check (required if gold has it, but exact number ignored)
GROUP BY sanity (must exist if aggregation + non-aggregated projections)
DISTINCT parity (gold has DISTINCT → model must too)
ORDER BY check (required if gold has ORDER BY)
Result is a StaticChecks object:
{
“parses”: true,
“uses_scoped_schema_only”: true,
“no_select_star”: true,
“limit_policy_ok”: false,
“groupby_ok”: true,
“distinct_parity_ok”: true,
“orderby_needed_present”: false,
“notes”: [
“Missing LIMIT while gold has LIMIT”,
“Gold has ORDER BY; model lacks ORDER BY”
]
}
2. Judge Rubric
We then prompt the Judge LLM with a strict JSON rubric:
LIMIT Policy: If gold has LIMIT, model must have one. Exact value ignored.
Aggregation Policy: SUM ≠ COUNT ≠ AVG (strict mismatch).
Forecast vs Average Forecast: penalize only if query explicitly said “Average”.
Projection Policy: Only penalize if essential columns/metrics are missing, not if gold added extra.
Table/View flexibility: Allow swapping equivalent sources.
Alias differences: Ignore (rows vs item_count).
3. Judge Output (Strict JSON)
The Judge returns structured JSON like this:
{
“overall_score”: 0.75,
“verdict”: “partially_correct”,
“reasons”: [
“Gold used SUM, model used COUNT”,
“Missing LIMIT while gold had one”
],
“action_items”: [
“Use SUM instead of COUNT”,
“Add LIMIT when gold query requires it”
],
“penalties”: {
“limit_missing_when_required”: true,
“wrong_groupby”: false,
“wrong_orderby_for_ranking”: true,
“filter_mismatch”: false,
“aggregation_mismatch”: true,
“projection_missing”: false,
“uses_out_of_scope_schema”: false
}
}
4. Why This Approach is Better
Compared to naive rowset matching or string diff:
Flexible but strict: allows harmless differences (aliases, order, extra projections) while catching meaningful mismatches.
Scalable: works even without a live DB connection.
Explainable: Judge returns reasons + penalties, not just “wrong”.
Aligned with business rules: encodes domain-specific mappings
Overall Pipeline:
Schema Extraction:
Connects to Postgres/MySQL/MSSQL via SQLAlchemy.
Pulls table/column metadata, PK/FK relationships, row counts, top-k values.
Outputs JSON schema for downstream use.
2. Schema Retriever:
Loads schema JSON.
Uses TF-IDF + business rules to rank relevant tables/columns.
Expands schema one-hop along foreign keys.
Produces a scoped schema snippet.
3. SQL Agent:
Planner LLM → structured JSON plan (tables, filters, metrics, groupby).
Validation → enforce groupby, limits, defaults, level_type=’XXX’.
SQL Writer LLM → generates actual SQL string.
Optionally executes SQL to fetch results.
4. Evaluation (LLMJ):
Static checks for syntax, scope, groupby, limit, star-selection.
Judge LLM applies rubric: semantic equivalence vs mismatches.
Produces JSON verdict with score, reasons, penalties.
Further enhancements:
Improvements to Schema Retrieval:
Hybrid Retrieval (BM25 + Embedding Similarity)
Instead of pure TF-IDF → add BM25 lexical search + semantic similarity (e.g., BGE, OpenAI embeddings). Then fuse results (RRF / weighted sum). It helps when query uses synonyms or abbreviations are not in column names.
Business Rule Augmentation
Automatically append relevant business rule snippets before retrieval.
Improvements to the Planner:
Reasoning-Led Planner (Chain-of-Thought LLM)
Before outputting JSON, have the planner explain reasoning → which filters/metrics map to business rules → then compress to JSON. This additionally reduces hallucination of columns.
Query Rewriting Layer
Normalize ambiguous queries into canonical forms before planning.
Example: “highest demand” → “MAX(quantity_value)”
Structured Verification
Auto-check:
Did every metric alias exist in schema snippet?
Did group_by align with projections?
Did planner obey level_type rules?
Improvements to SQL Writer
SQL Repair Post-Check:
Run a lightweight parser (e.g., sqlglot) after SQL generation. If invalid, automatically repair before execution.
Join Inference Assistant:
When multiple tables are in scope, reason explicitly about foreign keys and relationships (FK graph traversal).
Improvements to LLM Judge (Evaluation)
Execution-Aware Evaluation:
Run queries against sample DB (or synthetic subsets) and compare rowsets/aggregations. Combine execution difference with LLM-J judgment.
Confidence Scoring:
Instead of binary verdict, output an uncertainty/confidence score (e.g., if filters differ slightly but intent is 80% aligned).
Explainability:
LLM-J generates a short rationale why SQL is partially correct, useful for debugging/training the next iteration.
Workflow Enhancements
Active Learning Loop:
Wrong SQLs (flagged by judge) → added as new few-shot examples → retrain/reinforce planner.
2. Query Similarity Caching:
If a query is ~80% similar to a past one (BM25+embedding hybrid), reuse the SQL plan directly instead of regenerating.
3. Ambiguity in User Queries (Underspecified Queries):
Users often under-specify intent (e.g., “Show me sales last quarter” — missing columns, metrics, or aggregation). Below are some possible approaches to add to for addressing ambiguity:
Schema + Business Rules Grounding: Inject domain-specific defaults (e.g., “sales” → SUM(revenue) by rule).
Clarification Strategy: Either prompt user for clarification (“Do you want daily sales or aggregated sales?”) or apply defaults.
Query Templates & Few-Shot Examples: Train/fine-tune with prior ambiguous cases, showing how defaults are applied.
4. Handling Complex Queries with CTEs/Subqueries:
Decompose into Steps (Reasoning Planner): Instead of writing SQL in one go, generate a logical plan: Step 1 → intermediate aggregation, Step 2 → final aggregation.
CTE Fallback Rule: If SQL becomes long with repeated subqueries, force use of CTE for clarity.
LLM Hints: Adding some in-system prompt instructions like: “If intermediate filtering/aggregation is needed, use WITH (CTE) instead of nesting.”
Intent Detector: If query contains “top-k per group”, “growth over time”, “ranked” etc. → trigger a window function template.
