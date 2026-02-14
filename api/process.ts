// api/process.ts
import type { VercelRequest, VercelResponse } from "@vercel/node";
import Airtable from "airtable";
import OpenAI from "openai";
import { put, head } from "@vercel/blob";

const {
  AIRTABLE_TOKEN,
  AIRTABLE_BASE_ID,
  OPENAI_API_KEY,
  BLOB_READ_WRITE_TOKEN, // required by Vercel Blob (presence check)
  RUN_SECRET,
  MAX_CASE_ID,
  IMAGE_MODEL,
  IMAGE_QUALITY,
  IMAGE_SIZE
} = process.env;

if (!AIRTABLE_TOKEN || !AIRTABLE_BASE_ID || !OPENAI_API_KEY || !BLOB_READ_WRITE_TOKEN || !RUN_SECRET) {
  throw new Error(
    "Missing env vars: AIRTABLE_TOKEN, AIRTABLE_BASE_ID, OPENAI_API_KEY, BLOB_READ_WRITE_TOKEN, RUN_SECRET"
  );
}

const maxCaseDefault = 355;
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
const airtable = new Airtable({ apiKey: AIRTABLE_TOKEN }).base(AIRTABLE_BASE_ID);

const IMAGE_MODEL_USED = IMAGE_MODEL || "gpt-image-1";
const IMAGE_QUALITY_USED = (IMAGE_QUALITY || "medium") as any;
const IMAGE_SIZE_USED = (IMAGE_SIZE || "1024x1024") as any;

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

// Fields you listed (we include these first, but also include any extras present)
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
  "Reaction"
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

  // Preferred known fields first
  for (const k of CASE_FIELDS) {
    const v = fields[k];
    if (v == null || v === "") continue;
    parts.push(`${k}: ${normalizeFieldValue(v)}`);
  }

  // Any extra fields too (in case your base has more)
  for (const [k, v] of Object.entries(fields)) {
    if ((CASE_FIELDS as readonly string[]).includes(k)) continue;
    if (v == null || v === "") continue;
    parts.push(`${k}: ${normalizeFieldValue(v)}`);
  }

  return parts.join("\n");
}

/**
 * Pull ALL records from "Case N" and concatenate their fields into one text blob.
 * This avoids the "maxRecords:1" problem and ensures we feed OpenAI all content.
 */
async function getCaseText(caseId: number): Promise<{ tableName: string; caseText: string }> {
  const tableName = `Case ${caseId}`;
  const table = airtable(tableName);

  try {
    const records = await table.select({ maxRecords: 100 }).firstPage();
    if (!records.length) return { tableName, caseText: "" };

    const combined = records
      .map((r) => buildRecordText(r.fields as any))
      .filter(Boolean)
      .join("\n\n---\n\n");

    return { tableName, caseText: combined };
  } catch (err: any) {
    const e: any = new Error(
      `AIRTABLE_READ_FAILED table="${tableName}" msg="${err?.message || String(err)}"`
    );
    e.status = err?.statusCode || err?.status || 500;
    e.details = err;
    throw e;
  }
}

function pad3(n: number) {
  return String(n).padStart(3, "0");
}

async function blobExists(pathname: string): Promise<boolean> {
  try {
    await head(pathname);
    return true;
  } catch {
    return false;
  }
}

async function uploadJson(caseId: number, obj: any, overwrite: boolean): Promise<string> {
  const pathname = `case-profiles/${pad3(caseId)}.json`;

  if (!overwrite) {
    const exists = await blobExists(pathname);
    if (exists) return (await head(pathname)).url;
  }

  const bytes = Buffer.from(JSON.stringify(obj, null, 2), "utf-8");
  const res = await put(pathname, bytes, {
    access: "public",
    contentType: "application/json",
    addRandomSuffix: false
  });

  return res.url;
}

async function uploadPng(caseId: number, b64: string, overwrite: boolean): Promise<string> {
  const pathname = `case-avatars/${pad3(caseId)}.png`;

  if (!overwrite) {
    const exists = await blobExists(pathname);
    if (exists) return (await head(pathname)).url;
  }

  const bytes = Buffer.from(b64, "base64");
  const res = await put(pathname, bytes, {
    access: "public",
    contentType: "image/png",
    addRandomSuffix: false
  });

  return res.url;
}

/**
 * JSON profile extraction: forces the model to populate clothing from occupation/social history if possible,
 * while filling other visual defaults when missing.
 *
 * NOTE: We do NOT infer ethnicity/country from name. If explicitly mentioned in case text, it can appear in notes only.
 */
type VisualProfile = {
  age: string;
  build: "slim" | "average" | "stocky";
  hair: string;
  eyes: string;
  facial_features: string;
  clothing: string;
  notes: string;
};

async function makeProfile(caseText: string, caseId: number): Promise<VisualProfile> {
  const prompt = `
You are creating a visual profile for a SYNTHETIC patient headshot.

Return ONLY valid JSON with EXACTLY these keys:
{
  "age": "...",
  "build": "slim|average|stocky",
  "hair": "...",
  "eyes": "...",
  "facial_features": "...",
  "clothing": "...",
  "notes": "short reasoning using occupation/social history, if any"
}

Rules:
- Use explicit details from the text when present.
- If missing, infer realistic defaults and state "inferred" inside the value (e.g., "inferred 30–40", "inferred medium-length brown hair").
- Clothing: if the person’s occupation/social history implies attire, infer headshot-appropriate clothing:
  - healthcare → scrubs or clinical attire
  - office/finance/law/admin → business casual (button-down/blouse), optional blazer
  - construction/trades → subtle workwear (sturdy jacket), no hi-vis
  - retail/hospitality → smart casual (polo/shirt/blouse)
  - student → casual hoodie/t-shirt
  - fitness/outdoors → athleisure
- Avoid default beige. Prefer navy/grey/white/blue/black/green tones.
- BMI: do NOT output numeric BMI unless height AND weight are explicitly present and you compute it; otherwise ignore BMI and just set build.
- Do NOT infer country/ethnicity from the name. If explicitly mentioned, put it in "notes" only.
- Do NOT include the person’s name.

Deterministic key (do not output): CASE_ID=${caseId}

TEXT:
${caseText}
`.trim();

  const resp = await withRetry(() =>
    openai.responses.create({
      model: "gpt-4.1-mini",
      input: prompt
    })
  );

  const raw = (resp.output_text || "").trim();

  // Robust parse: sometimes models add stray text; try to extract JSON object.
  const firstBrace = raw.indexOf("{");
  const lastBrace = raw.lastIndexOf("}");
  if (firstBrace === -1 || lastBrace === -1 || lastBrace <= firstBrace) {
    throw new Error(`PROFILE_JSON_PARSE_FAILED: ${raw.slice(0, 200)}`);
  }
  const jsonText = raw.slice(firstBrace, lastBrace + 1);

  const parsed = JSON.parse(jsonText);

  // Minimal validation
  const requiredKeys = ["age", "build", "hair", "eyes", "facial_features", "clothing", "notes"];
  for (const k of requiredKeys) {
    if (!(k in parsed)) throw new Error(`PROFILE_JSON_MISSING_KEY:${k}`);
  }
  if (!["slim", "average", "stocky"].includes(parsed.build)) {
    throw new Error(`PROFILE_JSON_BAD_BUILD:${String(parsed.build)}`);
  }

  return parsed as VisualProfile;
}

function buildImagePromptFromProfile(profile: VisualProfile, caseId: number): string {
  return `
Photorealistic studio headshot, facing camera, mild happy expression.
Plain light background (blue/white/grey only), studio side lighting, DSLR 80mm full-frame look.
Single person centered, shoulders and head in frame.
No text, no logos, no watermark.

IMPORTANT: Clothing and styling must match the profile. Use headshot-appropriate attire.
Avoid repeated generic beige defaults unless explicitly specified by the profile.

Subject profile:
- Age: ${profile.age}
- Build: ${profile.build}
- Hair: ${profile.hair}
- Eyes: ${profile.eyes}
- Facial features: ${profile.facial_features}
- Clothing: ${profile.clothing}

Variation tag (do not render): V-${caseId}
`.trim().replace(/\s+/g, " ");
}

async function generateHeadshotPngBase64(prompt: string): Promise<string> {
  const img = await withRetry(() =>
    openai.images.generate({
      model: IMAGE_MODEL_USED,
      prompt,
      size: IMAGE_SIZE_USED,
      quality: IMAGE_QUALITY_USED
    } as any)
  );

  const first: any = (img as any).data?.[0];
  if (!first?.b64_json) throw new Error("OPENAI_IMAGE_NO_B64_JSON");
  return first.b64_json as string;
}

function extractErr(e: any) {
  return {
    status: e?.status || e?.statusCode || 500,
    message: e?.message || String(e)
  };
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  try {
    requireSecret(req);

    const startFrom = Number(req.query.startFrom ?? 1);
    const limit = Number(req.query.limit ?? 2);
    const maxCase = Number(MAX_CASE_ID ?? maxCaseDefault);

    // dryRun=1 => no OpenAI calls, no blob writes (just Airtable reads)
    const dryRun = String(req.query.dryRun ?? "0") === "1";

    // overwrite=1 => regenerate even if blob files exist
    const overwrite = String(req.query.overwrite ?? "0") === "1";

    // debug=1 => return caseText preview + profile JSON for the first case only (no image)
    const debug = String(req.query.debug ?? "0") === "1";

    const processed: any[] = [];

    for (let caseId = startFrom; caseId <= maxCase; caseId++) {
      if (processed.length >= limit) break;

      try {
        const { tableName, caseText } = await getCaseText(caseId);

        if (!caseText.trim()) {
          processed.push({ caseId, status: "no-text", tableName });
          continue;
        }

        if (dryRun) {
          processed.push({ caseId, status: "dryrun-ok", tableName });
          continue;
        }

        const profile = await makeProfile(caseText, caseId);

        if (debug) {
          return res.status(200).json({
            ok: true,
            caseId,
            tableName,
            caseTextPreview: caseText.slice(0, 2500),
            profile
          });
        }

        const imagePrompt = buildImagePromptFromProfile(profile, caseId);
        const b64 = await generateHeadshotPngBase64(imagePrompt);

        const imageUrl = await uploadPng(caseId, b64, overwrite);

        const profileDoc = {
          caseId,
          createdAt: new Date().toISOString(),
          profile,
          imagePrompt,
          imageUrl,
          source: { tableName }
        };

        const profileUrl = await uploadJson(caseId, profileDoc, overwrite);

        processed.push({ caseId, status: "done", imageUrl, profileUrl });

        await sleep(250);
      } catch (e: any) {
        processed.push({ caseId, status: "error", error: extractErr(e) });
      }
    }

    res.status(200).json({
      ok: true,
      startFrom,
      limit,
      maxCase,
      dryRun,
      overwrite,
      model: IMAGE_MODEL_USED,
      quality: IMAGE_QUALITY_USED,
      size: IMAGE_SIZE_USED,
      processedCount: processed.length,
      processed
    });
  } catch (e: any) {
    const status = e?.status || 500;
    res.status(status).json({ ok: false, error: e?.message || "Unknown error", status });
  }
}
