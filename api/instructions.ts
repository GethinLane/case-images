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

function pad3(n: number) {
  return String(n).padStart(3, "0");
}
function pad4(n: number) {
  return String(n).padStart(4, "0");
}

/* -----------------------------
   Airtable aggregation (multi-record)
----------------------------- */

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

function normalizeFieldValue(v: any): string {
  if (v == null) return "";
  if (typeof v === "string") return v.trim();
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}

// Collect ALL values across up to ~8 records (or more), de-dupe exact duplicates, preserve order.
// Join with --- separators to keep distinct snippets visible.
function joinMany(values: string[]): string {
  const cleaned = values.map((s) => String(s || "").trim()).filter(Boolean);
  if (!cleaned.length) return "";
  const out: string[] = [];
  const seen = new Set<string>();
  for (const s of cleaned) {
    if (seen.has(s)) continue;
    seen.add(s);
    out.push(s);
  }
  return out.join("\n\n---\n\n");
}

async function getInstructionFields(
  caseId: number
): Promise<{ tableName: string; input: InstructionInput; recordCount: number }> {
  const tableName = `Case ${caseId}`;
  const table = airtable(tableName);

  try {
    const records = await table.select({ maxRecords: 100 }).firstPage();

    const buckets: Record<string, string[]> = {};
    for (const k of FIELDS) buckets[k] = [];

    for (const r of records) {
      const f: any = (r.fields || {}) as any;
      const instr = f["Instructions"] ?? f["Instruction"]; // tolerate either column name

      buckets["Name"].push(normalizeFieldValue(f["Name"]));
      buckets["Age"].push(normalizeFieldValue(f["Age"]));
      buckets["Opening Sentence"].push(normalizeFieldValue(f["Opening Sentence"]));
      buckets["Divulge Freely"].push(normalizeFieldValue(f["Divulge Freely"]));
      buckets["Divulge Asked"].push(normalizeFieldValue(f["Divulge Asked"]));
      buckets["PMHx RP"].push(normalizeFieldValue(f["PMHx RP"]));
      buckets["Social History"].push(normalizeFieldValue(f["Social History"]));
      buckets["Family History"].push(normalizeFieldValue(f["Family History"]));
      buckets["ICE"].push(normalizeFieldValue(f["ICE"]));
      buckets["Reaction"].push(normalizeFieldValue(f["Reaction"]));
      buckets["Instructions"].push(normalizeFieldValue(instr));
    }

    const input: InstructionInput = {
      Name: joinMany(buckets["Name"]),
      Age: joinMany(buckets["Age"]),
      "Opening Sentence": joinMany(buckets["Opening Sentence"]),
      "Divulge Freely": joinMany(buckets["Divulge Freely"]),
      "Divulge Asked": joinMany(buckets["Divulge Asked"]),
      "PMHx RP": joinMany(buckets["PMHx RP"]),
      "Social History": joinMany(buckets["Social History"]),
      "Family History": joinMany(buckets["Family History"]),
      ICE: joinMany(buckets["ICE"]),
      Reaction: joinMany(buckets["Reaction"]),
      Instructions: joinMany(buckets["Instructions"]),
    };

    return { tableName, input, recordCount: records.length };
  } catch (err: any) {
    const e: any = new Error(`AIRTABLE_READ_FAILED table="${tableName}" msg="${err?.message || String(err)}"`);
    e.status = err?.statusCode || err?.status || 500;
    e.details = err;
    throw e;
  }
}

/* -----------------------------
   LLM generation (no dilution of your instruction text)
----------------------------- */

type InstructionOutput = {
  instructions: string;
  opening_line: string;
  patient_cues: string;
};

function mustStartWithYouAre(s: string) {
  return /^you are\b/i.test(String(s || "").trim());
}

async function generateInstructionOutput(input: InstructionInput, caseId: number): Promise<InstructionOutput> {
  // NOTE: The user's instruction block is kept essentially verbatim below.
  // Only additions:
  // - strict JSON output requirement
  // - explicit hard constraint about old Instructions field not being visible later
  const prompt = `
Final Polished Instruction (Optimised for Your Use Case)  

Please read the case details below and produce two outputs: 

IMPORTANT OUTPUT FORMAT:
Return ONLY valid JSON with EXACTLY these keys:
{
  "instructions": "...",
  "opening_line": "...",
  "patient_cues": "..."
}

Instructions: Write in UK English, in second person, present tense, starting exactly with “You are…”. Make it clear this is a video or telephone consultation, and include the name and specific age of the person speaking; if the speaker is calling about or attending with someone else, also include the name and specific age of the other person. If the case specifies multiple speaking roles (e.g., patient plus partner/parent/paramedic), explicitly state that there are multiple voices in the consultation, name each speaker, and specify who leads the history and when the other person speaks (e.g., answers only when prompted, interrupts to disagree, corrects details, etc.). 
Assume the bot will have access to the full case details, so do not restate any facts already provided anywhere in the case (including symptoms, timeline, examination findings, medications, past history, social history, and ideas/concerns/expectations); instead, write only what is needed to guide roleplay: each speaker’s tone, emotional state, communication style, level of health anxiety, how cooperative or challenging they are, what triggers escalation, what reassures them, and how they interact with each other. 

Exception: Some cases include an “instruction field” that will be replaced/overwritten in the final bot, meaning the bot will not be able to see that field afterwards. If important roleplay information appears in that instruction field, you must carry it over into the new 2-sentence character brief. 

CRITICAL CLARIFICATION (DO NOT IGNORE):
In THIS system, the existing Airtable "Instructions" field WILL be overwritten by your new "instructions" output, and the final bot will NOT have access to the old Airtable Instructions text. Therefore, you MUST carry over ANY roleplay-relevant details from the existing Airtable Instructions field into your new "instructions" output (integrated naturally). Do NOT rely on the old Instructions field being visible later.

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

Deterministic key (do not output): CASE_ID=${caseId}

CASE DETAILS (Airtable fields; may contain multiple snippets separated by ---):

Name:
${input["Name"]}

Age:
${input["Age"]}

Opening Sentence:
${input["Opening Sentence"]}

Divulge Freely:
${input["Divulge Freely"]}

Divulge Asked:
${input["Divulge Asked"]}

PMHx RP:
${input["PMHx RP"]}

Social History:
${input["Social History"]}

Family History:
${input["Family History"]}

ICE:
${input["ICE"]}

Reaction:
${input["Reaction"]}

Instructions field (THIS WILL BE OVERWRITTEN; MUST CARRY OVER ROLEPLAY-RELEVANT CONTENT INTO NEW OUTPUT):
${input["Instructions"]}
`.trim();

  const resp = await withRetry(() =>
    openai.responses.create({
      model: TEXT_MODEL_USED,
      input: prompt,
    } as any)
  );

  const raw = (resp as any).output_text?.trim?.() || "";
  const firstBrace = raw.indexOf("{");
  const lastBrace = raw.lastIndexOf("}");
  if (firstBrace === -1 || lastBrace === -1 || lastBrace <= firstBrace) {
    throw new Error(`INSTRUCTION_JSON_PARSE_FAILED: ${raw.slice(0, 240)}`);
  }

  const parsed: any = JSON.parse(raw.slice(firstBrace, lastBrace + 1));

  parsed.instructions = String(parsed.instructions ?? "").trim();
  parsed.opening_line = String(parsed.opening_line ?? "").trim();
  parsed.patient_cues = String(parsed.patient_cues ?? "").trim();

  if (!mustStartWithYouAre(parsed.instructions)) {
    throw new Error(`INSTRUCTION_BAD_START: "instructions" must start with "You are..."`);
  }

  return parsed as InstructionOutput;
}

/* -----------------------------
   Bundle upload (groups of N)
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
   Handler
----------------------------- */

export default async function handler(req: VercelRequest, res: VercelResponse) {
  try {
    requireSecret(req);

    const startFrom = Number(req.query.startFrom ?? 1);
    const maxCase = Number(MAX_CASE_ID ?? maxCaseDefault);
    const endAt = Number(req.query.endAt ?? maxCase);

    // bundleSize = how many cases per uploaded JSON file (default 10)
    const bundleSize = Math.max(1, Math.min(50, Number(req.query.bundleSize ?? 10)));

    // limit = max cases processed per invocation (default = bundleSize, but you can set larger)
    const limit = Math.max(1, Number(req.query.limit ?? bundleSize));

    const dryRun = String(req.query.dryRun ?? "0") === "1";
    const overwrite = String(req.query.overwrite ?? "0") === "1";
    const debug = String(req.query.debug ?? "0") === "1";

    const processed: any[] = [];
    const bundleUploads: any[] = [];

    let currentBundleStart: number | null = null;
    let currentBundleItems: any[] = [];

    const flushBundle = async () => {
      if (currentBundleStart == null) return;
      if (!currentBundleItems.length) return;

      const bundleStart = currentBundleStart;
      const bundleEnd = currentBundleItems[currentBundleItems.length - 1]?.caseId ?? bundleStart;

      const bundleObj = {
        createdAt: new Date().toISOString(),
        model: TEXT_MODEL_USED,
        range: { start: bundleStart, end: bundleEnd },
        count: currentBundleItems.length,
        items: currentBundleItems,
      };

      if (dryRun) {
        bundleUploads.push({ range: `${pad4(bundleStart)}-${pad4(bundleEnd)}`, status: "dryrun-skip-upload" });
      } else {
        const url = await uploadBundleJson(bundleStart, bundleEnd, bundleObj, overwrite);
        bundleUploads.push({ range: `${pad4(bundleStart)}-${pad4(bundleEnd)}`, status: "uploaded", url });
      }

      currentBundleStart = null;
      currentBundleItems = [];
    };

    for (let caseId = startFrom; caseId <= endAt; caseId++) {
      if (processed.length >= limit) break;

      try {
        const { tableName, input, recordCount } = await getInstructionFields(caseId);

        const hasAny = Object.values(input).some((v) => String(v || "").trim().length > 0);
        if (!hasAny) {
          const item = { caseId, tableName, recordCount, status: "no-fields" };
          processed.push(item);

          // still include in bundle (optional). If you prefer to omit, remove next lines.
          if (currentBundleStart == null) currentBundleStart = caseId;
          currentBundleItems.push(item);

          if (currentBundleItems.length >= bundleSize) await flushBundle();
          continue;
        }

        if (dryRun) {
          const item = { caseId, tableName, recordCount, status: "dryrun-ok" };
          processed.push(item);

          if (currentBundleStart == null) currentBundleStart = caseId;
          currentBundleItems.push(item);

          if (currentBundleItems.length >= bundleSize) await flushBundle();
          continue;
        }

        const output = await generateInstructionOutput(input, caseId);

        const item = {
          caseId,
          tableName,
          recordCount,
          input,   // keep input for traceability; remove if you want smaller files
          output,  // { instructions, opening_line, patient_cues }
        };

        processed.push({ caseId, status: "done", tableName, recordCount });

        if (debug) {
          // In debug mode, show one case immediately
          return res.status(200).json({ ok: true, debug: true, item, model: TEXT_MODEL_USED });
        }

        if (currentBundleStart == null) currentBundleStart = caseId;
        currentBundleItems.push(item);

        if (currentBundleItems.length >= bundleSize) await flushBundle();

        await sleep(120);
      } catch (e: any) {
        const item = { caseId, status: "error", error: extractErr(e) };
        processed.push(item);

        if (currentBundleStart == null) currentBundleStart = caseId;
        currentBundleItems.push(item);

        if (currentBundleItems.length >= bundleSize) await flushBundle();
      }
    }

    // flush any remaining partial bundle
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
      processedCount: processed.length,
      processed,
      bundles: bundleUploads,
    });
  } catch (e: any) {
    const status = e?.status || 500;
    return res.status(status).json({ ok: false, error: e?.message || "Unknown error", status });
  }
}
