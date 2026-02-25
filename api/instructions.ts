// api/instructions.ts
import type { VercelRequest, VercelResponse } from "@vercel/node";
import Airtable from "airtable";
import OpenAI from "openai";
import { put, head } from "@vercel/blob";

const {
  AIRTABLE_TOKEN,
  AIRTABLE_BASE_ID,
  OPENAI_API_KEY,
  BLOB_READ_WRITE_TOKEN,
  RUN_SECRET,
  MAX_CASE_ID,
  INSTRUCTIONS_TEXT_MODEL,
} = process.env;

if (!AIRTABLE_TOKEN || !AIRTABLE_BASE_ID || !OPENAI_API_KEY || !BLOB_READ_WRITE_TOKEN || !RUN_SECRET) {
  throw new Error(
    "Missing env vars: AIRTABLE_TOKEN, AIRTABLE_BASE_ID, OPENAI_API_KEY, BLOB_READ_WRITE_TOKEN, RUN_SECRET"
  );
}

const maxCaseDefault = 355;

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
const airtable = new Airtable({ apiKey: AIRTABLE_TOKEN }).base(AIRTABLE_BASE_ID);

// Default to GPT-5.2 unless overridden
const TEXT_MODEL_USED = INSTRUCTIONS_TEXT_MODEL || "gpt-5.2";

// Keep low for stable formatting + less nonsense
const TEMPERATURE = 0.2;

/* -----------------------------
   Auth + retry helpers
----------------------------- */

function getHeader(req: VercelRequest, name: string): string | undefined {
  const v = req.headers[name.toLowerCase()];
  if (Array.isArray(v)) return v[0];
  if (typeof v === "string") return v;
  return undefined;
}

function requireSecret(req: VercelRequest) {
  const provided = getHeader(req, "x-run-secret");
  if (!provided || provided !== RUN_SECRET) {
    const err: any = new Error("AUTH_FAILED_RUN_SECRET");
    err.status = 401;
    throw err;
  }
}

async function sleep(ms: number) {
  await new Promise((r) => setTimeout(r, ms));
}

function isRetryableStatus(status?: number) {
  return status === 429 || (status != null && status >= 500);
}

async function withRetry<T>(fn: () => Promise<T>, tries = 3): Promise<T> {
  let lastErr: any;
  for (let i = 0; i < tries; i++) {
    try {
      return await fn();
    } catch (e: any) {
      lastErr = e;
      const status = e?.status || e?.statusCode;
      if (!isRetryableStatus(status) || i === tries - 1) break;
      await sleep(800 * (i + 1));
    }
  }
  throw lastErr;
}

function extractErr(e: any) {
  return {
    status: e?.status || e?.statusCode || 500,
    message: e?.message || String(e),
  };
}

function pad4(n: number) {
  return String(n).padStart(4, "0");
}

/* -----------------------------
   Airtable: full case text + aggregated fields
----------------------------- */

const CASE_FIELDS = [
  "Name",
  "Age",
  "PMHx Record",
  "DHx",
  "Medical Notes",
  "Medical Notes Content",
  "Notes Photo",
  "Results",
  "Results Content",
  "Instructions",
  "Opening Sentence",
  "Divulge Freely",
  "Divulge Asked",
  "PMHx RP",
  "Social History",
  "Family History",
  "ICE",
  "Reaction",
] as const;

function normalizeFieldValue(v: any): string {
  if (v == null) return "";
  if (typeof v === "string") return v;
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}

function buildRecordText(fields: Record<string, any>): string {
  const parts: string[] = [];
  for (const k of CASE_FIELDS) {
    const v = (fields as any)[k];
    if (v == null || v === "") continue;
    parts.push(`${k}: ${normalizeFieldValue(v)}`);
  }
  for (const [k, v] of Object.entries(fields)) {
    if ((CASE_FIELDS as readonly string[]).includes(k)) continue;
    if (v == null || v === "") continue;
    parts.push(`${k}: ${normalizeFieldValue(v)}`);
  }
  return parts.join("\n");
}

async function getCaseText(caseId: number): Promise<{ tableName: string; caseText: string; recordCount: number }> {
  const tableName = `Case ${caseId}`;
  const table = airtable(tableName);

  try {
    const records = await table.select({ maxRecords: 100 }).firstPage();
    if (!records.length) return { tableName, caseText: "", recordCount: 0 };

    const combined = records
      .map((r) => buildRecordText(r.fields as any))
      .filter(Boolean)
      .join("\n\n---\n\n");

    return { tableName, caseText: combined, recordCount: records.length };
  } catch (err: any) {
    const e: any = new Error(`AIRTABLE_READ_FAILED table="${tableName}" msg="${err?.message || String(err)}"`);
    e.status = err?.statusCode || err?.status || 500;
    e.details = err;
    throw e;
  }
}

const FIELDS = [
  "Name",
  "Age",
  "Opening Sentence",
  "Divulge Freely",
  "Divulge Asked",
  "PMHx RP",
  "Social History",
  "Family History",
  "ICE",
  "Reaction",
  "Instructions",
] as const;

type FieldName = (typeof FIELDS)[number];
type InstructionInput = Record<FieldName, string>;

function clean(v: any): string {
  if (v == null) return "";
  if (typeof v === "string") return v.trim();
  try {
    return JSON.stringify(v).trim();
  } catch {
    return String(v).trim();
  }
}

function uniqPreserve(values: string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const s of values.map((x) => String(x || "").trim()).filter(Boolean)) {
    if (seen.has(s)) continue;
    seen.add(s);
    out.push(s);
  }
  return out;
}

/**
 * Explicitly index snippets to reduce confusion.
 */
function joinManyIndexed(values: string[]): string {
  const arr = uniqPreserve(values);
  if (!arr.length) return "";
  return arr.map((v, i) => `[${i + 1}] ${v}`).join("\n\n---\n\n");
}

async function getInstructionFields(caseId: number): Promise<InstructionInput> {
  const tableName = `Case ${caseId}`;
  const table = airtable(tableName);
  const records = await table.select({ maxRecords: 100 }).firstPage();

  const buckets: Record<string, string[]> = {};
  for (const k of FIELDS) buckets[k] = [];

  for (const r of records) {
    const f: any = (r.fields || {}) as any;
    const instr = f["Instructions"] ?? f["Instruction"]; // tolerate both

    buckets["Name"].push(clean(f["Name"]));
    buckets["Age"].push(clean(f["Age"]));
    buckets["Opening Sentence"].push(clean(f["Opening Sentence"]));
    buckets["Divulge Freely"].push(clean(f["Divulge Freely"]));
    buckets["Divulge Asked"].push(clean(f["Divulge Asked"]));
    buckets["PMHx RP"].push(clean(f["PMHx RP"]));
    buckets["Social History"].push(clean(f["Social History"]));
    buckets["Family History"].push(clean(f["Family History"]));
    buckets["ICE"].push(clean(f["ICE"]));
    buckets["Reaction"].push(clean(f["Reaction"]));
    buckets["Instructions"].push(clean(instr));
  }

  return {
    Name: joinManyIndexed(buckets["Name"]),
    Age: joinManyIndexed(buckets["Age"]),
    "Opening Sentence": joinManyIndexed(buckets["Opening Sentence"]),
    "Divulge Freely": joinManyIndexed(buckets["Divulge Freely"]),
    "Divulge Asked": joinManyIndexed(buckets["Divulge Asked"]),
    "PMHx RP": joinManyIndexed(buckets["PMHx RP"]),
    "Social History": joinManyIndexed(buckets["Social History"]),
    "Family History": joinManyIndexed(buckets["Family History"]),
    ICE: joinManyIndexed(buckets["ICE"]),
    Reaction: joinManyIndexed(buckets["Reaction"]),
    Instructions: joinManyIndexed(buckets["Instructions"]),
  };
}

/* -----------------------------
   Blob bundling (groups of N)
----------------------------- */

async function blobExists(pathname: string): Promise<boolean> {
  try {
    await head(pathname);
    return true;
  } catch {
    return false;
  }
}

async function uploadBundleJson(
  startCaseId: number,
  endCaseId: number,
  bundleObj: any,
  overwrite: boolean
): Promise<string> {
  const pathname = `case-instructions/batch-${pad4(startCaseId)}-${pad4(endCaseId)}.json`;

  if (!overwrite) {
    const exists = await blobExists(pathname);
    if (exists) return (await head(pathname)).url;
  }

  const bytes = Buffer.from(JSON.stringify(bundleObj, null, 2), "utf-8");
  const res = await put(pathname, bytes, {
    access: "public",
    contentType: "application/json",
    addRandomSuffix: false,
  });

  return res.url;
}

/* -----------------------------
   Undiluted instruction block (verbatim)
----------------------------- */

const FINAL_POLISHED_INSTRUCTION_BLOCK = `
Final Polished Instruction (Optimised for Your Use Case)  

Please read the case details below and produce two outputs: 

Instructions: Write in UK English, in second person, present tense, starting exactly with “You are…”. Make it clear this is a video or telephone consultation, and include the name and specific age of the person speaking; if the speaker is calling about or attending with someone else, also include the name and specific age of the other person. If the case specifies multiple speaking roles (e.g., patient plus partner/parent/paramedic), explicitly state that there are multiple voices in the consultation, name each speaker, and specify who leads the history and when the other person speaks (e.g., answers only when prompted, interrupts to disagree, corrects details, etc.). 
Assume the bot will have access to the full case details, so do not restate any facts already provided anywhere in the case (including symptoms, timeline, examination findings, medications, past history, social history, and ideas/concerns/expectations); instead, write only what is needed to guide roleplay: each speaker’s tone, emotional state, communication style, level of health anxiety, how cooperative or challenging they are, what triggers escalation, what reassures them, and how they interact with each other. 

Exception: Some cases include an “instruction field” that will be replaced/overwritten in the final bot, meaning the bot will not be able to see that field afterwards. If important roleplay information appears in that instruction field, you must carry it over into the new 2-sentence character brief. 

Opening line (1 sentence): 
Write the patient’s first spoken sentence in first person, phrased in a natural, conversational way that fits how this individual would actually speak (matching their age, confidence, education, and personality). Avoid overly formal or clinical wording unless the case clearly suggests the patient speaks that way. It’s important that the opening sentence doesn’t reveal any plans or worries. 

3. Patient Cues 

This section is critical. 

Include no more than 2 cues. 

Each cue must be a neutral observation, not a disclosure. 

Cues must not reveal key clinical facts, red flags, diagnoses, exact timelines, risks, or underlying causes. 

Do not include interpretation, conclusions, or emotional reasoning (avoid “because…”, “so I think…”, “it must be…”). 

A cue should never give the answer — it should only prompt the clinician to ask the next question. 

Cues are only used if the clinician has not already explored that area. 

Keep both cues within one short paragraph, with each cue written as a single sentence. 

A good cue: 

Hints at a missing domain. 

Sounds natural and spontaneous. 

Could safely be said even if the clinician does not follow up. 

Opens a door without stepping through it. 

A bad cue: 

States a red-flag symptom directly. 

Provides a diagnosis or label. 

Gives an exact timeframe. 

Reveals a safeguarding, risk, or safety-critical detail outright. 

Answers a question the clinician has not yet asked. 

Think: 
A cue should create curiosity, not provide clarity. 

Important: Treat every case as independent and do not carry over information from previous cases. 
`.trim();

/* -----------------------------
   LLM JSON helper
----------------------------- */

function extractJsonObject(raw: string) {
  const firstBrace = raw.indexOf("{");
  const lastBrace = raw.lastIndexOf("}");
  if (firstBrace === -1 || lastBrace === -1 || lastBrace <= firstBrace) return null;
  return raw.slice(firstBrace, lastBrace + 1);
}

async function llmJson(prompt: string): Promise<any> {
  const resp = await withRetry(() =>
    openai.responses.create({
      model: TEXT_MODEL_USED,
      input: prompt,
      temperature: TEMPERATURE,
    } as any)
  );

  const raw = (resp as any).output_text?.trim?.() || "";
  const jsonStr = extractJsonObject(raw);
  if (!jsonStr) throw new Error(`LLM_JSON_PARSE_FAILED: ${raw.slice(0, 240)}`);
  return JSON.parse(jsonStr);
}

/* -----------------------------
   Output types
----------------------------- */

type MainOutput = {
  instructions: string;
  opening_line: string;
};

type CuePlanItem = {
  domain: string; // e.g. "safety netting / worsening"
  if_not_mentioned: string; // generic condition
  then_say: string; // neutral hint, no disclosures
};

type CuePlan = {
  plan: CuePlanItem[]; // 3-6 items ideally
  selected_indexes: number[]; // choose up to 2
};

type InstructionOutput = {
  instructions: string;
  opening_line: string;
  patient_cues: string; // paragraph with 1-2 sentences, conditional format
};

/* -----------------------------
   Validation
----------------------------- */

function mustStartWithYouAre(s: string) {
  return /^you are\b/i.test(String(s || "").trim());
}

function sentenceCount(s: string): number {
  const t = String(s || "").trim();
  if (!t) return 0;
  const m = t.match(/[.!?]+/g);
  return m ? m.length : 1;
}

function hasDisallowedCuesContent(s: string): string | null {
  const t = String(s || "").toLowerCase();

  // reasoning / interpretation
  if (/\bbecause\b/.test(t)) return "contains 'because'";
  if (/\bso i think\b/.test(t)) return "contains 'so I think'";
  if (/\bit must\b/.test(t)) return "contains 'it must'";
  if (/\bi think\b/.test(t)) return "contains 'I think'";

  // digits often leak timeframes/ages/timelines; keep strict
  if (/\b\d+\b/.test(t)) return "contains a number/digit";

  // avoid explicit diagnostic / red-flag language
  if (/\bdiagnos/.test(t)) return "mentions diagnosis";
  if (/\bred flag\b/.test(t)) return "mentions red flag";

  return null;
}

function validateCueSentenceFormat(s: string): { ok: boolean; reason?: string } {
  const c = String(s || "").trim();
  if (!c) return { ok: false, reason: "empty cue" };
  if (!/^if\b/i.test(c)) return { ok: false, reason: "cue does not start with 'If'" };
  if (!/\bthen\b/i.test(c)) return { ok: false, reason: "cue missing 'then'" };
  if (sentenceCount(c) !== 1) return { ok: false, reason: "cue is not exactly one sentence" };
  const bad = hasDisallowedCuesContent(c);
  if (bad) return { ok: false, reason: bad };
  if (c.length > 260) return { ok: false, reason: "cue too long" };
  return { ok: true };
}

function validateCues(cues: string[]): { ok: boolean; reason?: string } {
  const arr = cues.map((x) => String(x || "").trim()).filter(Boolean);
  if (arr.length > 2) return { ok: false, reason: "more than 2 cues" };
  for (const c of arr) {
    const v = validateCueSentenceFormat(c);
    if (!v.ok) return v;
  }
  return { ok: true };
}

function joinCuesToParagraph(cues: string[]): string {
  const arr = cues.map((x) => String(x || "").trim()).filter(Boolean);
  if (!arr.length) return "";
  return arr.join(" ");
}

/* -----------------------------
   Generation: main + cue planning + cue composing + repair
----------------------------- */

async function generateMain(caseText: string, input: InstructionInput, caseId: number): Promise<MainOutput> {
  const prompt = `
${FINAL_POLISHED_INSTRUCTION_BLOCK}

IMPORTANT OUTPUT FORMAT:
Return ONLY valid JSON with EXACTLY these keys:
{
  "instructions": "...",
  "opening_line": "..."
}

CRITICAL CLARIFICATION (DO NOT IGNORE):
In THIS system, the existing Airtable "Instructions" field WILL be overwritten by your new "instructions" output, and the final bot will NOT have access to the old Airtable Instructions text. Therefore, you MUST carry over ANY roleplay-relevant details from the existing Airtable Instructions field into your new "instructions" output (integrated naturally). Do NOT rely on the old Instructions field being visible later.

Additional constraints:
- "instructions" MUST start exactly with "You are".
- Do NOT include patient cues in "instructions" (those are generated separately).
- Treat every case as independent.

Deterministic key (do not output): CASE_ID=${caseId}

FULL CASE TEXT (bot can see this; do not restate facts):
${caseText}

SELECTED FIELDS (may contain multiple snippets separated by --- and indexed):
Name:
${input["Name"]}

Age:
${input["Age"]}

Reaction:
${input["Reaction"]}

ICE:
${input["ICE"]}

Instructions field (THIS WILL BE OVERWRITTEN; MUST CARRY OVER ROLEPLAY-RELEVANT CONTENT):
${input["Instructions"]}
`.trim();

  const parsed = await llmJson(prompt);

  const instructions = String(parsed.instructions ?? "").trim();
  const opening_line = String(parsed.opening_line ?? "").trim();

  if (!mustStartWithYouAre(instructions)) throw new Error(`INSTRUCTION_BAD_START: must start with "You are..."`);
  if (sentenceCount(opening_line) !== 1) throw new Error(`OPENING_LINE_BAD: must be exactly 1 sentence`);

  return { instructions, opening_line };
}

/**
 * Step 1: plan cues by prioritising IMPORTANT follow-up domains.
 * Must NOT disclose case facts; must create safe "If X isn't mentioned, then Y" scaffolds.
 */
async function planCues(caseText: string, input: InstructionInput, caseId: number): Promise<CuePlan> {
  const prompt = `
You are creating a CUE PLAN for a medical roleplay case.

Goal:
- Prioritise what is most IMPORTANT for a clinician to ask next (high-yield follow-up domains),
  and design safe cues that nudge those domains WITHOUT disclosing key clinical facts.

Return ONLY valid JSON with EXACTLY these keys:
{
  "plan": [
    { "domain": "...", "if_not_mentioned": "...", "then_say": "..." }
  ],
  "selected_indexes": [0, 1]
}

Rules (CRITICAL):
- You MUST first decide (internally) what the highest-priority follow-up domains are for this case.
- Choose 4 to 6 domains in "plan".
- selected_indexes must pick the BEST 1 or 2 cues (max 2) that are most important for this specific case.
- You must not disclose clinical facts, red flags, diagnoses, exact timelines, risks, or underlying causes in then_say.
- then_say MUST be a neutral observation (not a disclosure), and should sound like something the person might casually add.
- Avoid digits entirely (no numbers).
- Avoid reasoning language ("because", "I think", "it must", "so I think").
- Each then_say should be short (<= 160 chars).

Important constraint:
- The final output cues MUST be in the format "If <xyz isn't mentioned>, then <patient says abc>."
- This plan is just components: if_not_mentioned and then_say will be combined later.

Deterministic key (do not output): CASE_ID=${caseId}

FULL CASE TEXT (use it to prioritise; do not copy facts into the cue text):
${caseText}

Tone/personality hints (use only for style, not for adding facts):
Reaction:
${input["Reaction"]}

ICE:
${input["ICE"]}
`.trim();

  const parsed = await llmJson(prompt);

  const planArr = Array.isArray(parsed.plan) ? parsed.plan : [];
  const idxArr = Array.isArray(parsed.selected_indexes) ? parsed.selected_indexes : [];

  const plan: CuePlanItem[] = planArr
    .map((x: any) => ({
      domain: String(x?.domain ?? "").trim().slice(0, 80),
      if_not_mentioned: String(x?.if_not_mentioned ?? "").trim().slice(0, 140),
      then_say: String(x?.then_say ?? "").trim().slice(0, 180),
    }))
    .filter((x: CuePlanItem) => x.domain && x.if_not_mentioned && x.then_say)
    .slice(0, 6);

  const selected_indexes: number[] = idxArr
    .map((n: any) => Number(n))
    .filter((n: number) => Number.isFinite(n) && n >= 0 && n < plan.length)
    .slice(0, 2);

  // fallback selection
  const finalSel = selected_indexes.length ? selected_indexes : plan.length ? [0] : [];

  return { plan, selected_indexes: finalSel };
}

/**
 * Step 2: compose final cues in the required conditional format.
 */
async function composeCues(caseText: string, plan: CuePlan, caseId: number): Promise<string[]> {
  const prompt = `
You are writing the FINAL Patient Cues from a cue plan.

Return ONLY valid JSON with EXACTLY these keys:
{
  "patient_cues": [
    "If ... then ... .",
    "If ... then ... ."
  ]
}

Rules (CRITICAL):
- Use ONLY the cue plan items provided.
- Use selected_indexes to choose 1 or 2 cues (max 2).
- Each cue MUST be exactly ONE sentence.
- Each cue MUST start with "If" and MUST contain the word "then".
- Each cue MUST be a neutral observation, not a disclosure.
- Do NOT reveal key clinical facts, red flags, diagnoses, exact timelines, risks, or underlying causes.
- Avoid digits entirely.
- Avoid reasoning language ("because", "I think", "it must", "so I think").
- Keep cues short and natural.

Deterministic key (do not output): CASE_ID=${caseId}

CUE PLAN JSON:
${JSON.stringify(plan, null, 2)}

FULL CASE TEXT (for safety checking only; do not add facts):
${caseText}
`.trim();

  const parsed = await llmJson(prompt);
  const cues = Array.isArray(parsed.patient_cues) ? parsed.patient_cues : [];
  return cues.map((x: any) => String(x || "").trim()).filter(Boolean).slice(0, 2);
}

async function repairCues(caseText: string, badCues: string[], caseId: number): Promise<string[]> {
  const prompt = `
You are correcting Patient Cues that failed strict rules.

Return ONLY valid JSON with EXACTLY these keys:
{
  "patient_cues": ["If ... then ... .", "If ... then ... ."]
}

Rules (must pass):
- Max 2 cues.
- Each cue exactly ONE sentence.
- Each cue MUST start with "If" and MUST contain the word "then".
- Neutral observation, not a disclosure.
- Must NOT reveal key clinical facts, red flags, diagnoses, exact timelines, risks, underlying causes.
- Must NOT contain any digits.
- Must NOT include reasoning language (no “because”, “I think”, “it must”, “so I think”).
- If safer, output only 1 cue (or []).

Deterministic key (do not output): CASE_ID=${caseId}

FULL CASE TEXT (for safety; do not leak facts in cues):
${caseText}

BAD CUES TO FIX:
${badCues.map((c, i) => `- [${i + 1}] ${c}`).join("\n")}
`.trim();

  const parsed = await llmJson(prompt);
  const cues = Array.isArray(parsed.patient_cues) ? parsed.patient_cues : [];
  return cues.map((x: any) => String(x || "").trim()).filter(Boolean).slice(0, 2);
}

async function generateCuesPrioritised(caseText: string, input: InstructionInput, caseId: number): Promise<string[]> {
  let last: string[] = [];

  for (let attempt = 1; attempt <= 3; attempt++) {
    const plan = await planCues(caseText, input, caseId);
    const composed = await composeCues(caseText, plan, caseId);
    last = composed;

    const v = validateCues(last);
    if (v.ok) return last;

    const repaired = await repairCues(caseText, last, caseId);
    last = repaired;

    const v2 = validateCues(last);
    if (v2.ok) return last;

    await sleep(120);
  }

  // best-effort
  return last.slice(0, 2);
}

/* -----------------------------
   One-shot wrapper per case
----------------------------- */

async function generateInstructionOutput(caseText: string, input: InstructionInput, caseId: number): Promise<InstructionOutput> {
  const main = await generateMain(caseText, input, caseId);
  const cuesArr = await generateCuesPrioritised(caseText, input, caseId);
  return {
    instructions: main.instructions,
    opening_line: main.opening_line,
    patient_cues: joinCuesToParagraph(cuesArr),
  };
}

/* -----------------------------
   Handler (bundles of 10)
----------------------------- */

export default async function handler(req: VercelRequest, res: VercelResponse) {
  try {
    requireSecret(req);

    const startFrom = Number(req.query.startFrom ?? 1);
    const maxCase = Number(MAX_CASE_ID ?? maxCaseDefault);
    const endAt = Number(req.query.endAt ?? maxCase);

    const bundleSize = Math.max(1, Math.min(50, Number(req.query.bundleSize ?? 10)));
    const limit = Math.max(1, Number(req.query.limit ?? bundleSize));

    const dryRun = String(req.query.dryRun ?? "0") === "1";
    const overwrite = String(req.query.overwrite ?? "0") === "1";
    const debug = String(req.query.debug ?? "0") === "1";

    const processed: any[] = [];
    const bundles: any[] = [];

    let currentBundleStart: number | null = null;
    let currentItems: any[] = [];

    const flushBundle = async () => {
      if (currentBundleStart == null) return;
      if (!currentItems.length) return;

      const startId = currentBundleStart;
      const endId = currentItems[currentItems.length - 1]?.caseId ?? startId;

      const bundleObj = {
        createdAt: new Date().toISOString(),
        model: TEXT_MODEL_USED,
        temperature: TEMPERATURE,
        range: { start: startId, end: endId },
        count: currentItems.length,
        items: currentItems,
      };

      if (dryRun) {
        bundles.push({ range: `${pad4(startId)}-${pad4(endId)}`, status: "dryrun-skip-upload" });
      } else {
        const url = await uploadBundleJson(startId, endId, bundleObj, overwrite);
        bundles.push({ range: `${pad4(startId)}-${pad4(endId)}`, status: "uploaded", url });
      }

      currentBundleStart = null;
      currentItems = [];
    };

    for (let caseId = startFrom; caseId <= endAt; caseId++) {
      if (processed.length >= limit) break;

      try {
        const { tableName, caseText, recordCount } = await getCaseText(caseId);

        if (!caseText.trim()) {
          const item = { caseId, tableName, recordCount, status: "no-text" };
          processed.push(item);
          if (currentBundleStart == null) currentBundleStart = caseId;
          currentItems.push(item);
          if (currentItems.length >= bundleSize) await flushBundle();
          continue;
        }

        if (dryRun) {
          const item = { caseId, tableName, recordCount, status: "dryrun-ok" };
          processed.push(item);
          if (currentBundleStart == null) currentBundleStart = caseId;
          currentItems.push(item);
          if (currentItems.length >= bundleSize) await flushBundle();
          continue;
        }

        const input = await getInstructionFields(caseId);
        const output = await generateInstructionOutput(caseText, input, caseId);

        const item = {
          caseId,
          tableName,
          recordCount,
          input,   // keep for traceability; remove if you want smaller files
          output,  // { instructions, opening_line, patient_cues }
        };

        processed.push({ caseId, status: "done", tableName, recordCount });

        if (debug) {
          return res.status(200).json({ ok: true, debug: true, item, model: TEXT_MODEL_USED });
        }

        if (currentBundleStart == null) currentBundleStart = caseId;
        currentItems.push(item);
        if (currentItems.length >= bundleSize) await flushBundle();

        await sleep(150);
      } catch (e: any) {
        const item = { caseId, status: "error", error: extractErr(e) };
        processed.push(item);

        if (currentBundleStart == null) currentBundleStart = caseId;
        currentItems.push(item);
        if (currentItems.length >= bundleSize) await flushBundle();
      }
    }

    await flushBundle();

    return res.status(200).json({
      ok: true,
      startFrom,
      endAt,
      limit,
      bundleSize,
      dryRun,
      overwrite,
      model: TEXT_MODEL_USED,
      temperature: TEMPERATURE,
      processedCount: processed.length,
      processed,
      bundles,
    });
  } catch (e: any) {
    const status = e?.status || 500;
    return res.status(status).json({ ok: false, error: e?.message || "Unknown error", status });
  }
}
