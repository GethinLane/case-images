// api/process.ts
import type { VercelRequest, VercelResponse } from "@vercel/node";
import Airtable from "airtable";
import OpenAI from "openai";
import { put, head } from "@vercel/blob";
import crypto from "crypto";

const {
  AIRTABLE_TOKEN,
  AIRTABLE_BASE_ID,
  OPENAI_API_KEY,
  BLOB_READ_WRITE_TOKEN,
  RUN_SECRET,
  MAX_CASE_ID,
  IMAGE_MODEL,
  IMAGE_QUALITY,
  IMAGE_SIZE,
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

// Airtable fields you listed; we include these first, then include any extra fields present too.
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
    const v = fields[k];
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

/**
 * Pull ALL records from "Case N" (up to 100) and concatenate.
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
    addRandomSuffix: false,
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
    addRandomSuffix: false,
  });

  return res.url;
}

// Deterministic selection so clothing colors don’t collapse to one option.
function seedFromText(caseId: number, text: string) {
  const h = crypto.createHash("sha256").update(`${caseId}::${text}`, "utf8").digest();
  return h.readUInt32BE(0);
}
function pick<T>(arr: readonly T[], seed: number): T {
  return arr[seed % arr.length];
}
const CLOTHING_COLORS = [
  "white",
  "light grey",
  "charcoal",
  "black",
  "light blue",
  "medium blue",
  "teal",
  "dark green",
  "navy",
] as const;

type GenderPresentation = "female-presenting" | "male-presenting";
type Socioeconomic = "affluent" | "average" | "struggling" | "homeless" | "unknown";

type VisualProfile = {
  gender_presentation: GenderPresentation;
  age: string;

  // MUST be one of slim/average/stocky (code will normalize if model adds extra words)
  build: "slim" | "average" | "stocky";

  hair: string;
  eyes: string;
  facial_features: string;

  socioeconomic: Socioeconomic;
  clothing_type: string;
  clothing_color: "auto" | string; // model returns "auto"; code assigns a real color
  accessories: string;
  grooming: string;
  style_context: string;

  notes: string;
};

// Normalizers to prevent “PROFILE_JSON_BAD_BUILD: average inferred”
function normalizeBuild(raw: any): "slim" | "average" | "stocky" | null {
  if (!raw) return null;
  const s = String(raw).toLowerCase();
  if (s.includes("slim")) return "slim";
  if (s.includes("stocky")) return "stocky";
  if (s.includes("average")) return "average";
  return null;
}

function normalizeGender(raw: any): GenderPresentation | null {
  if (!raw) return null;
  const s = String(raw).toLowerCase();
  if (s.includes("female")) return "female-presenting";
  if (s.includes("male")) return "male-presenting";
  return null;
}

function normalizeSocio(raw: any): Socioeconomic {
  const s = String(raw || "").toLowerCase();
  if (s.includes("homeless")) return "homeless";
  if (s.includes("struggling")) return "struggling";
  if (s.includes("affluent")) return "affluent";
  if (s.includes("average")) return "average";
  return "unknown";
}

async function makeProfile(caseText: string, caseId: number): Promise<VisualProfile> {
  const prompt = `
You are creating a visual profile for a SYNTHETIC patient headshot.

Return ONLY valid JSON with EXACTLY these keys:
{
  "gender_presentation": "female-presenting|male-presenting",
  "age": "...",
  "build": "slim|average|stocky",
  "hair": "...",
  "eyes": "...",
  "facial_features": "...",
  "socioeconomic": "affluent|average|struggling|homeless|unknown",
  "clothing_type": "...",
  "clothing_color": "auto",
  "accessories": "...",
  "grooming": "...",
  "style_context": "...",
  "notes": "short reasoning / cues used"
}

Rules:
- Use explicit details from the text when present.
- If missing, infer realistic defaults.
- IMPORTANT: For enum fields (gender_presentation, build, socioeconomic) output ONLY one of the allowed values (no extra words).
  Put any "inferred" wording into notes/style_context instead.
- gender_presentation:
  - If text explicitly indicates girl/woman/female or boy/man/male, follow that.
  - Else infer from pronouns/context; if unclear choose one and say "inferred" in notes.
- BMI: do NOT output a numeric BMI unless height AND weight are explicitly present and you compute it; otherwise ignore BMI and just set build.

Socioeconomic inference:
- affluent: executive roles, expensive hobbies, private care, large house, luxury car, etc.
- struggling: unemployment, financial stress, unstable housing, poor access.
- homeless: explicit mention of homelessness/shelter/rough sleeping.
- otherwise average/unknown.

Clothing inference (headshot appropriate; subtle cues, not costume):
- Prefer clothing that fits occupation AND lifestyle AND socioeconomic context.
- Follow these examples (very important):
  - knitting/cross-stitch hobby → "knitted jumper" or "cozy cardigan"
  - cashier/checkout worker → "store-uniform polo" or "work shirt" (NO logos, no text)
  - police officer → "uniform-style shirt" or "smart uniform shirt" (NO badges, no insignia, no text)
  - healthcare worker → "scrubs" or "clinical tunic"
  - office/law/finance/admin → "button-down shirt/blouse", optional blazer
  - hospitality → "smart casual shirt/blouse"
  - construction/trades → "sturdy zip-up" or "workwear jacket" (no hi-vis)
  - student → "hoodie" or "casual sweatshirt"
  - fitness/outdoors → "athleisure top"
- Avoid defaulting to a plain t-shirt if hobbies/occupation suggest a better choice.

Accessories:
- affluent: include 1 subtle accessory (simple necklace OR small earrings OR a watch).
- average: optional simple accessory, otherwise "none".
- struggling/homeless: usually "none" unless text suggests otherwise.

Grooming:
- affluent/average: generally "neat".
- struggling: "casual, slightly tired" (subtle).
- homeless: "tired, slightly unkempt" (subtle, not extreme).

Safety constraints:
- No logos, no badges, no insignia, no readable text.
- Do NOT include the person’s name.
- Do NOT infer country/ethnicity from the name. If explicitly mentioned, put it in notes only.
- For clothing_color ALWAYS output exactly "auto" (do not choose a color).

Deterministic key (do not output): CASE_ID=${caseId}

TEXT:
${caseText}
`.trim();

  const resp = await withRetry(() =>
    openai.responses.create({
      model: "gpt-4.1-mini",
      input: prompt,
    })
  );

  const raw = (resp.output_text || "").trim();

  const firstBrace = raw.indexOf("{");
  const lastBrace = raw.lastIndexOf("}");
  if (firstBrace === -1 || lastBrace === -1 || lastBrace <= firstBrace) {
    throw new Error(`PROFILE_JSON_PARSE_FAILED: ${raw.slice(0, 240)}`);
  }

  const jsonText = raw.slice(firstBrace, lastBrace + 1);
  const parsed: any = JSON.parse(jsonText);

  const requiredKeys = [
    "gender_presentation",
    "age",
    "build",
    "hair",
    "eyes",
    "facial_features",
    "socioeconomic",
    "clothing_type",
    "clothing_color",
    "accessories",
    "grooming",
    "style_context",
    "notes",
  ];
  for (const k of requiredKeys) {
    if (!(k in parsed)) throw new Error(`PROFILE_JSON_MISSING_KEY:${k}`);
  }

  // Normalize enums (prevents failures when model adds "inferred" etc.)
  const ng = normalizeGender(parsed.gender_presentation);
  if (!ng) throw new Error(`PROFILE_JSON_BAD_GENDER:${String(parsed.gender_presentation)}`);
  parsed.gender_presentation = ng;

  const nb = normalizeBuild(parsed.build);
  if (!nb) throw new Error(`PROFILE_JSON_BAD_BUILD:${String(parsed.build)}`);
  parsed.build = nb;

  parsed.socioeconomic = normalizeSocio(parsed.socioeconomic);

  // enforce contract: model must not pick a color
  if (String(parsed.clothing_color).toLowerCase() !== "auto") {
    parsed.clothing_color = "auto";
  }

  return parsed as VisualProfile;
}

function buildImagePromptFromProfile(profile: VisualProfile, caseId: number): string {
  return `
Photorealistic studio headshot, facing camera, mild happy expression.
Plain light background (blue/white/grey only), studio side lighting, DSLR 80mm full-frame look.
Single person centered, shoulders and head in frame.
No text, no logos, no watermark.

CRITICAL CONSTRAINT:
- The subject MUST be ${profile.gender_presentation}. Do not generate the opposite.

IMPORTANT:
- Clothing and accessories must match the profile below.
- No logos, no badges, no insignia, no readable text on clothing.
- Headshot-appropriate attire only; subtle cues, not costume.
- Grooming should reflect the profile subtly.

Subject profile:
- Gender presentation: ${profile.gender_presentation}
- Age: ${profile.age}
- Build: ${profile.build}
- Hair: ${profile.hair}
- Eyes: ${profile.eyes}
- Facial features: ${profile.facial_features}
- Socioeconomic vibe: ${profile.socioeconomic}
- Clothing: ${profile.clothing_type} in ${profile.clothing_color}
- Accessories: ${profile.accessories}
- Grooming: ${profile.grooming}
- Style context: ${profile.style_context}

Variation tag (do not render): V-${caseId}
`.trim().replace(/\s+/g, " ");
}

async function generateHeadshotPngBase64(prompt: string): Promise<string> {
  const img = await withRetry(() =>
    openai.images.generate({
      model: IMAGE_MODEL_USED,
      prompt,
      size: IMAGE_SIZE_USED,
      quality: IMAGE_QUALITY_USED,
    } as any)
  );

  const first: any = (img as any).data?.[0];
  if (!first?.b64_json) throw new Error("OPENAI_IMAGE_NO_B64_JSON");
  return first.b64_json as string;
}

async function verifyGenderPresentation(imageB64: string, expected: GenderPresentation): Promise<boolean> {
  const checkPrompt = `
Look at this headshot image and answer ONLY "yes" or "no".
Question: Is the subject clearly ${expected}?
If ambiguous, answer "no".
`.trim();

  const resp = await withRetry(() =>
    openai.responses.create({
      model: "gpt-4.1-mini",
      input: [
        {
          role: "user",
          content: [
            { type: "input_text", text: checkPrompt },
            { type: "input_image", image_url: `data:image/png;base64,${imageB64}` },
          ],
        },
      ],
    } as any)
  );

  const out = (resp.output_text || "").trim().toLowerCase();
  return out.startsWith("yes");
}

async function generateVerifiedHeadshot(prompt: string, expected: GenderPresentation): Promise<string> {
  let lastB64 = "";
  for (let attempt = 1; attempt <= 3; attempt++) {
    const b64 = await generateHeadshotPngBase64(prompt);
    lastB64 = b64;

    await sleep(150);

    const ok = await verifyGenderPresentation(b64, expected);
    if (ok) return b64;
  }
  return lastB64;
}

function extractErr(e: any) {
  return {
    status: e?.status || e?.statusCode || 500,
    message: e?.message || String(e),
  };
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  try {
    requireSecret(req);

    const startFrom = Number(req.query.startFrom ?? 1);
    const limit = Number(req.query.limit ?? 2);
    const maxCase = Number(MAX_CASE_ID ?? maxCaseDefault);

    const dryRun = String(req.query.dryRun ?? "0") === "1";
    const overwrite = String(req.query.overwrite ?? "0") === "1";
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

        // deterministic clothing color chosen from palette
        const seed = seedFromText(caseId, caseText);
        profile.clothing_color = pick(CLOTHING_COLORS, seed);

        if (debug) {
          return res.status(200).json({
            ok: true,
            caseId,
            tableName,
            caseTextPreview: caseText.slice(0, 2500),
            profile,
          });
        }

        const imagePrompt = buildImagePromptFromProfile(profile, caseId);

        const b64 = await generateVerifiedHeadshot(imagePrompt, profile.gender_presentation);

        const imageUrl = await uploadPng(caseId, b64, overwrite);

        const profileDoc = {
          caseId,
          createdAt: new Date().toISOString(),
          profile,
          imagePrompt,
          imageUrl,
          source: { tableName },
          image: { model: IMAGE_MODEL_USED, quality: IMAGE_QUALITY_USED, size: IMAGE_SIZE_USED },
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
      processed,
    });
  } catch (e: any) {
    const status = e?.status || 500;
    res.status(status).json({ ok: false, error: e?.message || "Unknown error", status });
  }
}
