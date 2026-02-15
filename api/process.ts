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

const IMAGE_MODEL_USED = IMAGE_MODEL || "gpt-image-1.5";
const IMAGE_QUALITY_USED = (IMAGE_QUALITY || "high") as any;
const IMAGE_SIZE_USED = (IMAGE_SIZE || "1024x1024") as any;

const CASE_ORIGIN_OVERRIDES: Record<number, string> = {
  // Iberian / Hispanic
  9: "Iberian (Spanish) / Hispanic appearance cues",
  52: "Iberian (Spanish) / Hispanic appearance cues",
  69: "Iberian (Spanish) / Hispanic appearance cues",
  36: "Iberian (Spanish) / Hispanic appearance cues",
  126: "Iberian (Spanish) / Hispanic appearance cues",
  340: "Iberian (Portuguese/Spanish) appearance cues",
  341: "Iberian-language origin appearance cues (Portuguese/Spanish leaning)",
  151: "Iberian (Spanish/Portuguese) appearance cues",

  // South Asian
  323: "South Asian appearance cues (India, Gujarati)",
  325: "South Asian appearance cues (India, Gujarati)",
  333: "South Asian appearance cues (India, Gujarati)",
  337: "South Asian appearance cues (India, Gujarati)",
  349: "South Asian appearance cues (India)",
  112: "South Asian appearance cues (Pakistan/South Asia)",
  220: "South/Central Asian appearance cues",
  332: "South/Central Asian appearance cues",
  170: "South/Central Asian appearance cues",
  300: "South Asian / MENA-adjacent Muslim-community appearance cues",
  129: "South Asian / MENA-adjacent Muslim-community appearance cues",
  354: "South Asian / Muslim-community appearance cues",

  // MENA / Maghreb
  297: "Middle Eastern (Arab) appearance cues",
  148: "North African / Maghreb appearance cues",
  301: "Middle Eastern (Arab) appearance cues",

  // West Africa (Nigeria)
  272: "West African appearance cues (Nigeria)",
  316: "West African appearance cues (Nigeria)",
  350: "West African appearance cues (Nigeria)",

  // Slavic / Eastern Europe
  183: "Slavic / Eastern European appearance cues (South Slavic leaning)",
  246: "Slavic / Eastern European appearance cues",
  113: "Slavic/Russian-leaning appearance cues",

  // East Asian
  256: "East Asian appearance cues (Chinese-origin)",

  // Italian
  204: "Southern European appearance cues (Italian)",
  190: "Southern European appearance cues (Italian)",
  156: "Southern European appearance cues (Italian/Portuguese/Spanish leaning)",

  // Germanic / Scandinavian
  43: "Northern/Central European appearance cues (Germanic)",
  110: "Northern/Central European appearance cues (Germanic)",
  56: "Northern/Central European appearance cues (German/Jewish diaspora leaning)",
  79: "Northern European appearance cues (Scandinavian)",
  211: "Northern/Central European appearance cues (Germanic)",

  // French / Francophone
  299: "Francophone appearance cues (French/Caribbean possible)",
  146: "Francophone appearance cues (French/Francophone)",
};

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

/* -----------------------------
   Airtable input aggregation
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
    const v = fields[k];
    if (v == null || v === "") continue;
    parts.push(`${k}: ${normalizeFieldValue(v)}`);
  }

  // include any extra fields too (helps if your base has more columns)
  for (const [k, v] of Object.entries(fields)) {
    if ((CASE_FIELDS as readonly string[]).includes(k)) continue;
    if (v == null || v === "") continue;
    parts.push(`${k}: ${normalizeFieldValue(v)}`);
  }

  return parts.join("\n");
}

/**
 * Pull all records from "Case N" and concatenate
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
    const e: any = new Error(`AIRTABLE_READ_FAILED table="${tableName}" msg="${err?.message || String(err)}"`);
    e.status = err?.statusCode || err?.status || 500;
    e.details = err;
    throw e;
  }
}

/* -----------------------------
   Vercel Blob storage
----------------------------- */

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

/* -----------------------------
   Deterministic variety controls
----------------------------- */

function seedFromText(caseId: number, text: string) {
  const h = crypto.createHash("sha256").update(`${caseId}::${text}`, "utf8").digest();
  return h.readUInt32BE(0);
}
function pick<T>(arr: readonly T[], seed: number): T {
  return arr[seed % arr.length];
}

// Background: ALWAYS very light grey variants only (never medium/dark; no blue tint)
const BACKGROUNDS = [
  "very light neutral grey (#f7f7f7) seamless studio background with a subtle gradient",
  "very light neutral grey (#f6f6f6) paper sweep background, evenly lit",
  "very pale grey (#f8f8f8) soft vignette, still very light",
  "off-white grey (#fafafa) seamless background, no color tint",
] as const;

// Clothing colors: remove ALL grey, allow more colour, avoid yellow/red, avoid background match
const CLOTHING_COLORS = [
  "navy",
  "cobalt blue",
  "royal blue",
  "deep teal",
  "forest green",
  "emerald green",
  "burgundy",
  "plum",
  "black",
  "white",
  "cream",
  "soft pastel blue",
  "soft pastel green",
  "soft lavender",
] as const;

function ensureNoBackgroundClothingClash(clothingColor: string, background: string) {
  // Since backgrounds are light grey/off-white, avoid white/cream if it risks blending.
  const c = clothingColor.toLowerCase();
  const b = background.toLowerCase();
  if ((c.includes("white") || c.includes("cream")) && (b.includes("off-white") || b.includes("#fafafa"))) {
    return "navy";
  }
  return clothingColor;
}

function knitHint(clothingType: string) {
  const s = clothingType.toLowerCase();
  if (s.includes("knit") || s.includes("jumper") || s.includes("cardigan") || s.includes("sweater")) {
    return "The jumper/cardigan must show clearly visible knitted texture (ribbed or cable-knit) around neckline and shoulders.";
  }
  return "";
}

/* -----------------------------
   Profile + prompt building
----------------------------- */

type GenderPresentation = "female-presenting" | "male-presenting";
type Socioeconomic = "affluent" | "average" | "struggling" | "homeless" | "unknown";
type GlamLevel = "low" | "medium" | "high";
type Retouching = "none" | "light";

type SkinTone = "very_light" | "light" | "medium" | "dark" | "very_dark" | "unspecified";
type HairTexture = "straight" | "wavy" | "curly" | "coily" | "unspecified";

/** NEW: composition support */
type Composition = "single" | "pair";
type CompanionProfile = {
  role: string; // mother/father/guardian/carer/partner/interpreter/etc
  gender_presentation: GenderPresentation;
  age: string; // "adult" etc
  notes: string;
};

/** NOTE: everything else kept as-is; new fields appended only */
type VisualProfile = {
  gender_presentation: GenderPresentation;
  age: string;
  build: "slim" | "average" | "stocky";
  hair: string;
  eyes: string;
  facial_features: string;
  appearance_findings: string;

  // Explicit-only cultural fields (from text only; do not guess)
  cultural_origin: string; // "unspecified" unless explicitly stated
  ethnic_background: string; // "unspecified" unless explicitly stated
  skin_tone: SkinTone; // "unspecified" unless explicitly stated
  hair_texture: HairTexture; // "unspecified" unless explicitly stated

  socioeconomic: Socioeconomic;
  glam_level: GlamLevel;
  retouching: Retouching;

  origin_cues: string; // "none" or the override string

  clothing_type: string;
  clothing_color: "auto" | string;
  accessories: string;
  grooming: string;
  makeup: string;

  style_context: string;
  notes: string;

  background: "auto" | string;

  /** NEW */
  composition?: Composition;
  companion?: CompanionProfile | null;
};

// Normalizers
function normalizeBuild(raw: any): "slim" | "average" | "stocky" | null {
  const s = String(raw || "").toLowerCase();
  if (s.includes("slim")) return "slim";
  if (s.includes("stocky")) return "stocky";
  if (s.includes("average")) return "average";
  return null;
}
function normalizeGender(raw: any): GenderPresentation | null {
  const s = String(raw || "").toLowerCase();
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
function normalizeGlam(raw: any): GlamLevel {
  const s = String(raw || "").toLowerCase();
  if (s.includes("high")) return "high";
  if (s.includes("low")) return "low";
  return "medium";
}
function normalizeRetouching(raw: any): Retouching {
  const s = String(raw || "").toLowerCase();
  if (s.includes("light")) return "light";
  return "none";
}
function normalizeSkinTone(raw: any): SkinTone {
  const s = String(raw || "").toLowerCase();
  if (s.includes("very_dark")) return "very_dark";
  if (s.includes("dark")) return "dark";
  if (s.includes("medium")) return "medium";
  if (s.includes("very_light")) return "very_light";
  if (s.includes("light")) return "light";
  return "unspecified";
}
function normalizeHairTexture(raw: any): HairTexture {
  const s = String(raw || "").toLowerCase();
  if (s.includes("coily")) return "coily";
  if (s.includes("curly")) return "curly";
  if (s.includes("wavy")) return "wavy";
  if (s.includes("straight")) return "straight";
  return "unspecified";
}
function normalizeExplicitString(raw: any): string {
  const s = String(raw ?? "").trim();
  return s.length ? s : "unspecified";
}

/** NEW: simple age parsing to enforce child <6 => pair */
function parseAgeNumber(ageStr: string): number | null {
  const m = String(ageStr || "").match(/(\d{1,3})/);
  if (!m) return null;
  const n = Number(m[1]);
  return Number.isFinite(n) ? n : null;
}

function glamStyleBlock(glam: GlamLevel, retouching: Retouching) {
  if (glam === "high") {
    return `
Style: well-presented and stylish (higher social presentation), but still a real person.
- No fashion/editorial vibe; no "model" look.
- Retouching: ${retouching} (never heavy; never airbrushed).
- Natural skin texture must remain visible (pores OK).`.trim();
  }
  if (glam === "low") {
    return `
Style: ordinary everyday realism.
- Practical, minimally styled appearance.
- Retouching: none.
- Natural skin texture, small imperfections allowed.`.trim();
  }
  return `
Style: typical everyday “put together” realism.
- Light/normal styling, not fashion/editorial.
- Retouching: ${retouching} (light at most).
- Natural skin texture visible.`.trim();
}

async function makeProfile(caseText: string, caseId: number): Promise<VisualProfile> {
  const prompt = `
You are creating a visual profile for a SYNTHETIC patient headshot for a medical training product.

Return ONLY valid JSON with EXACTLY these keys:
{
  "gender_presentation": "female-presenting|male-presenting",
  "age": "...",
  "build": "slim|average|stocky",
  "hair": "...",
  "eyes": "...",
  "facial_features": "...",
  "appearance_findings": "...",

  "cultural_origin": "...",
  "ethnic_background": "...",
  "skin_tone": "very_light|light|medium|dark|very_dark|unspecified",
  "hair_texture": "straight|wavy|curly|coily|unspecified",

  "socioeconomic": "affluent|average|struggling|homeless|unknown",
  "glam_level": "low|medium|high",
  "retouching": "none|light",
  "clothing_type": "...",
  "clothing_color": "auto",
  "accessories": "...",
  "origin_cues": "...",
  "skin_tone": "...",
  "grooming": "...",
  "makeup": "...",
  "style_context": "...",
  "notes": "...",
  "background": "auto"
}

Rules:
- Use explicit details from the text when present.
- If missing, infer realistic defaults.
- For enum fields output ONLY allowed values.
- clothing_color MUST be exactly "auto".
- background MUST be exactly "auto".
- Do NOT include the person’s name.
- you CAN infer country/ethnicity from the name.

Cultural/ethnic representation (CRITICAL):
- set cultural_origin / ethnic_background / skin_tone / hair_texture if stated in TEXT.
- Please guess origin/ethnicity.
- Do NOT add traditional clothing unless explicitly stated.

Appearance findings (CRITICAL):
- Only include features visible in a headshot AND explicitly stated in the text.
- If none explicitly stated, output "none".
- Do NOT infer visible signs from diagnoses alone.
- Do make clothing and appearance relevant to situation (such as homelessness, or extremely wealthy)

Clothing inference (headshot appropriate; subtle cues, not costume):
- Prefer clothing that fits occupation AND hobbies/lifestyle AND socioeconomic context.
- Avoid defaulting to a plain t-shirt if occupation/hobbies suggest a better choice.
- No logos/text.

Accessories/Grooming/Makeup:
- Keep appropriate to socioeconomic + age; natural realism.

Origin / appearance cues:
- origin_cues should be "none" unless an override is provided by the system.
- If origin_cues is provided, use it to guide plausible facial features + skin tone range.
- Do not mention ethnicity labels in the final image (no text), this is only guidance.

Skin tone:
- Choose a plausible skin_tone description consistent with origin_cues if provided.
- Otherwise infer from case context only; do not infer from name.


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

  const parsed: any = JSON.parse(raw.slice(firstBrace, lastBrace + 1));

  const requiredKeys = [
    "gender_presentation",
    "age",
    "build",
    "hair",
    "eyes",
    "facial_features",
    "appearance_findings",

    "cultural_origin",
    "ethnic_background",
    "skin_tone",
    "hair_texture",

    "socioeconomic",
    "glam_level",
    "retouching",
    "clothing_type",
    "clothing_color",
    "accessories",
    "grooming",
    "makeup",
    "style_context",
    "notes",
    "background",
  ];
  for (const k of requiredKeys) {
    if (!(k in parsed)) throw new Error(`PROFILE_JSON_MISSING_KEY:${k}`);
  }

  const ng = normalizeGender(parsed.gender_presentation);
  if (!ng) throw new Error(`PROFILE_JSON_BAD_GENDER:${String(parsed.gender_presentation)}`);
  parsed.gender_presentation = ng;

  const nb = normalizeBuild(parsed.build);
  if (!nb) throw new Error(`PROFILE_JSON_BAD_BUILD:${String(parsed.build)}`);
  parsed.build = nb;

  parsed.socioeconomic = normalizeSocio(parsed.socioeconomic);
  parsed.glam_level = normalizeGlam(parsed.glam_level);
  parsed.retouching = normalizeRetouching(parsed.retouching);

  // enforce contracts for deterministic choices
  parsed.clothing_color = "auto";
  parsed.background = "auto";

  // appearance findings contract
  if (!parsed.appearance_findings || typeof parsed.appearance_findings !== "string") {
    parsed.appearance_findings = "none";
  }
  if (String(parsed.appearance_findings).trim() === "") parsed.appearance_findings = "none";

  // cultural explicit-only fields
  parsed.cultural_origin = normalizeExplicitString(parsed.cultural_origin);
  parsed.ethnic_background = normalizeExplicitString(parsed.ethnic_background);
  parsed.skin_tone = normalizeSkinTone(parsed.skin_tone);
  parsed.hair_texture = normalizeHairTexture(parsed.hair_texture);

  return parsed as VisualProfile;
}

/* -----------------------------
   NEW: Scan + decision for TWO-person headshots
   - child under ~6 => must be "pair" with adult guardian
   - someone present / speaking on behalf => "pair"
   (AI-based, not keyword search)
----------------------------- */

type CompositionDecision = {
  composition: "single" | "pair";
  reason: string;
  evidence: string;
  confidence: "high" | "medium" | "low";
  companion: null | {
    role: string;
    gender_presentation: GenderPresentation;
    age: string;
    notes: string;
  };
};

async function decideComposition(caseText: string, caseId: number): Promise<CompositionDecision> {
  const prompt = `
You are deciding whether a synthetic patient headshot should include ONE person or TWO people.

Return ONLY valid JSON with EXACTLY these keys:
{
  "composition": "single|pair",
  "reason": "...",
  "evidence": "...",
  "confidence": "high|medium|low",
  "companion": null OR {
    "role": "...",
    "gender_presentation": "female-presenting|male-presenting",
    "age": "...",
    "notes": "..."
  }
}

Rules (CRITICAL):
- Use AI understanding, NOT keyword matching.
- Base decisions ONLY on what is stated or clearly implied by the TEXT.
- If the case subject is a child under ~6 years old, composition MUST be "pair" with an adult guardian present.
- If the TEXT indicates someone is present with them OR speaking on their behalf (e.g., parent, guardian, carer, partner, interpreter), composition MUST be "pair".
- If uncertain, prefer "single" unless it's a small child where you must choose "pair".
- "evidence" should be a short pointer (max 180 chars) referencing the relevant statement(s).
- Companion:
  - role should be plain (mother/father/guardian/carer/partner/interpreter/etc).
  - gender_presentation: choose based on text if possible; if not explicit, choose the most plausible but lower confidence.
  - age: if not explicit, use "adult".

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
    throw new Error(`COMPOSITION_JSON_PARSE_FAILED: ${raw.slice(0, 240)}`);
  }

  const parsed: any = JSON.parse(raw.slice(firstBrace, lastBrace + 1));

  const allowedComp = new Set(["single", "pair"]);
  const allowedConf = new Set(["high", "medium", "low"]);

  if (!allowedComp.has(parsed.composition)) parsed.composition = "single";
  if (!allowedConf.has(parsed.confidence)) parsed.confidence = "low";

  parsed.reason = String(parsed.reason || "").trim().slice(0, 240);
  parsed.evidence = String(parsed.evidence || "").trim().slice(0, 180);

  if (parsed.composition === "pair") {
    if (!parsed.companion || typeof parsed.companion !== "object") {
      parsed.companion = {
        role: "guardian",
        gender_presentation: "female-presenting",
        age: "adult",
        notes: "present with patient",
      };
      if (parsed.confidence === "high") parsed.confidence = "medium";
    } else {
      parsed.companion.role = String(parsed.companion.role || "guardian").trim().slice(0, 60);
      parsed.companion.gender_presentation =
        normalizeGender(parsed.companion.gender_presentation) || "female-presenting";
      parsed.companion.age = String(parsed.companion.age || "adult").trim().slice(0, 40);
      parsed.companion.notes = String(parsed.companion.notes || "").trim().slice(0, 160);
    }
  } else {
    parsed.companion = null;
  }

  return parsed as CompositionDecision;
}

function forceChildPairIfNeeded(profile: VisualProfile) {
  const ageN = parseAgeNumber(profile.age);
  if (ageN != null && ageN < 6) {
    profile.composition = "pair";
    if (!profile.companion) {
      profile.companion = {
        role: "parent/guardian",
        gender_presentation: "female-presenting",
        age: "adult",
        notes: "child under 6 must be accompanied",
      };
    }
  }
}

function buildImagePromptFromProfile(profile: VisualProfile, caseId: number): string {
  const background = String(profile.background);
  const knit = knitHint(profile.clothing_type);
  const glamBlock = glamStyleBlock(profile.glam_level, profile.retouching);

  const cultureLine =
    profile.cultural_origin !== "unspecified" || profile.ethnic_background !== "unspecified"
      ? `Cultural context (explicit from case text): ${profile.cultural_origin}${
          profile.ethnic_background !== "unspecified" ? `; ${profile.ethnic_background}` : ""
        }.`
      : `Cultural context: unspecified (do not guess).`;

  const originOverride = CASE_ORIGIN_OVERRIDES[caseId];
  const originLine = originOverride
    ? `Origin/appearance guidance: ${originOverride}. Reflect this through plausible skin tone range, facial features, and hair traits.`
    : `Origin/appearance guidance: none.`;

  const phenotypeParts: string[] = [];
  if (profile.skin_tone !== "unspecified") phenotypeParts.push(`Skin tone: ${profile.skin_tone}`);
  if (profile.hair_texture !== "unspecified") phenotypeParts.push(`Hair texture: ${profile.hair_texture}`);
  const phenotypeLine = phenotypeParts.length
    ? `Phenotype constraints (explicit only): ${phenotypeParts.join(", ")}.`
    : `Phenotype constraints: unspecified (do not guess).`;

  /** NEW: composition block (single vs pair) */
  const isPair = profile.composition === "pair" && !!profile.companion;

  const compositionBlock = isPair
    ? `COMPOSITION (CRITICAL): Render EXACTLY TWO people. Person A is the PRIMARY patient (described below). Person B is a COMPANION: ${
        profile.companion!.role
      }, ${profile.companion!.gender_presentation}, age ${profile.companion!.age}. Both faces fully visible, head-and-shoulders framing for both. PRIMARY is slightly more prominent. No extra people.`
    : `COMPOSITION (CRITICAL): Render EXACTLY ONE person only (the PRIMARY patient). No other people.`;

  return `
Photorealistic studio headshot, facing camera, mildly happy expression.
Studio side lighting, DSLR 80mm full-frame look. Single scene centered, shoulders and head in frame.

${compositionBlock}

No text, no logos, no watermark.

Background MUST be: ${background}.
- It must be VERY LIGHT neutral grey only (never medium grey or dark).
- No blue tint. No colored backdrop.

CRITICAL CONSTRAINT:
- The PRIMARY subject MUST be ${profile.gender_presentation}. Do not generate the opposite.

${glamBlock}

CULTURAL APPROPRIATENESS (IMPORTANT):
- ${cultureLine}
- ${originLine}
- ${phenotypeLine}
- Do NOT caricature or exaggerate features.
- Keep styling contemporary and natural.
- Do NOT add traditional clothing unless explicitly stated.

IMPORTANT:
- Clothing and accessories must match the PRIMARY profile below.
- NO grey clothing. Do not match clothing color to background.
- No logos, no badges, no insignia, no readable text on clothing.
- Headshot-appropriate attire only; subtle cues, not costume.
- ${knit || "Ensure clothing looks correct for the described type."}
- Makeup must match the PRIMARY profile below.
- If appearance findings are not "none", they MUST be visible and match exactly. Do not invent additional lesions.

PRIMARY subject profile:
- Gender presentation: ${profile.gender_presentation}
- Age: ${profile.age}
- Build: ${profile.build}
- Hair: ${profile.hair}
- Eyes: ${profile.eyes}
- Facial features: ${profile.facial_features}
- Visible facial findings: ${profile.appearance_findings}

- Cultural origin: ${profile.cultural_origin}
- Ethnic background: ${profile.ethnic_background}
- Skin tone: ${profile.skin_tone}
- Hair texture: ${profile.hair_texture}

- Socioeconomic vibe: ${profile.socioeconomic}
- Glam level: ${profile.glam_level}
- Retouching: ${profile.retouching}
- Clothing: ${profile.clothing_type} in ${profile.clothing_color}
- Accessories: ${profile.accessories}
- Grooming: ${profile.grooming}
- Makeup: ${profile.makeup}
- Style context: ${profile.style_context}

${isPair ? `COMPANION notes: ${profile.companion!.notes}` : ""}

Variation tag (do not render): V-${caseId}
`.trim().replace(/\s+/g, " ");
}

/* -----------------------------
   Scan-only mode (NO images)
   Find cases with explicit non-UK / mixed origin mentioned in text
----------------------------- */

type OriginScanResult = {
  origin_status: "explicit-non-uk" | "explicit-uk" | "explicit-mixed" | "not-stated";
  origins_mentioned: string[];
  evidence: string;
  confidence: "high" | "medium" | "low";
};

async function extractOriginFlag(caseText: string, caseId: number): Promise<OriginScanResult> {
  const prompt = `
You are scanning a synthetic patient case for cultural/origin info.

Return ONLY valid JSON with EXACTLY these keys:
{
  "origin_status": "explicit-non-uk|explicit-uk|explicit-mixed|not-stated",
  "origins_mentioned": ["..."],
  "evidence": "...",
  "confidence": "high|medium|low"
}

Rules (CRITICAL):
- ONLY use information explicitly stated in the TEXT. Do NOT guess.
- you CAN infer nationality/ethnicity from the patient's name.
- Do NOT infer from diagnosis alone (e.g. FGM does NOT automatically mean Nigeria).
- If the text explicitly mentions any country/region/cultural origin, list it in origins_mentioned.
- origin_status meanings:
  - explicit-non-uk: text explicitly states origin is non-UK (e.g. "from Nigeria", "born in Pakistan", etc.)
  - explicit-uk: text explicitly states UK origin only and no other origins
  - explicit-mixed: text explicitly includes UK + at least one other origin (or multiple non-UK origins)
  - not-stated: no explicit origin/cultural origin mentioned
- evidence should be a short pointer (max 180 chars) quoting or clearly referencing the statement.
- confidence: high when clear statement; medium if somewhat indirect but still explicit; low if weak/ambiguous but still explicit.

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
    throw new Error(`ORIGIN_SCAN_JSON_PARSE_FAILED: ${raw.slice(0, 240)}`);
  }

  const parsed: any = JSON.parse(raw.slice(firstBrace, lastBrace + 1));

  const allowedStatus = new Set(["explicit-non-uk", "explicit-uk", "explicit-mixed", "not-stated"]);
  const allowedConf = new Set(["high", "medium", "low"]);

  if (!allowedStatus.has(parsed.origin_status)) parsed.origin_status = "not-stated";
  if (!Array.isArray(parsed.origins_mentioned)) parsed.origins_mentioned = [];
  parsed.origins_mentioned = parsed.origins_mentioned.map((x: any) => String(x).trim()).filter(Boolean).slice(0, 10);
  parsed.evidence = String(parsed.evidence || "").trim().slice(0, 180);
  if (!allowedConf.has(parsed.confidence)) parsed.confidence = "low";

  return parsed as OriginScanResult;
}

/* -----------------------------
   Image generation + verification (gender + clothing only)
----------------------------- */

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

async function visionYesNo(imageB64: string, question: string): Promise<boolean> {
  const resp = await withRetry(() =>
    openai.responses.create({
      model: "gpt-4.1-mini",
      input: [
        {
          role: "user",
          content: [
            { type: "input_text", text: `Answer ONLY "yes" or "no". ${question} If ambiguous, answer "no".` },
            { type: "input_image", image_url: `data:image/png;base64,${imageB64}` },
          ],
        },
      ],
    } as any)
  );

  const out = (resp.output_text || "").trim().toLowerCase();
  return out.startsWith("yes");
}

async function verifyGender(imageB64: string, expected: GenderPresentation) {
  return visionYesNo(imageB64, `Is the PRIMARY subject clearly ${expected}?`);
}

function clothingCheckQuestion(clothingType: string) {
  const ct = clothingType.toLowerCase();
  if (ct.includes("knit") || ct.includes("jumper") || ct.includes("cardigan") || ct.includes("sweater")) {
    return `Is the PRIMARY subject wearing a knitted jumper/cardigan/sweater with visible knit texture?`;
  }
  if (ct.includes("scrubs")) return `Is the PRIMARY subject wearing scrubs (medical work attire)?`;
  if (ct.includes("uniform")) return `Is the PRIMARY subject wearing a uniform-style shirt (without visible badges/logos)?`;
  if (ct.includes("blazer")) return `Is the PRIMARY subject wearing a blazer or smart jacket?`;
  if (ct.includes("button-down") || ct.includes("blouse")) return `Is the PRIMARY subject wearing a button-down shirt or blouse?`;
  if (ct.includes("polo")) return `Is the PRIMARY subject wearing a plain polo shirt (no logo)?`;
  if (ct.includes("hoodie") || ct.includes("sweatshirt")) return `Is the PRIMARY subject wearing a hoodie or sweatshirt?`;
  return `Does the PRIMARY subject's clothing match this description: "${clothingType}"?`;
}

/** NEW: checks for person count + child/adult pairing */
async function verifyPersonCount(imageB64: string, expected: 1 | 2) {
  return visionYesNo(imageB64, `Are there exactly ${expected} people visible in the image?`);
}
async function verifyChildAndAdult(imageB64: string) {
  return visionYesNo(imageB64, `Is there one young child (toddler/preschool age) and one adult visible?`);
}

async function generateVerifiedHeadshot(
  prompt: string,
  expectedGender: GenderPresentation,
  clothingType: string,
  expectedPeople: 1 | 2,
  requireChildAdult: boolean
): Promise<{
  b64: string;
  attempts: number;
  peopleOk: boolean;
  genderOk: boolean;
  clothingOk: boolean;
  childAdultOk: boolean;
}> {
  let last = "";
  let lastPeopleOk = false;
  let lastGenderOk = false;
  let lastClothingOk = false;
  let lastChildAdultOk = !requireChildAdult;

  for (let attempt = 1; attempt <= 3; attempt++) {
    const b64 = await generateHeadshotPngBase64(prompt);
    last = b64;

    await sleep(150);

    const peopleOk = await verifyPersonCount(b64, expectedPeople);
    const genderOk = peopleOk ? await verifyGender(b64, expectedGender) : false;
    const clothingOk = peopleOk ? await visionYesNo(b64, clothingCheckQuestion(clothingType)) : false;
    const childAdultOk = requireChildAdult ? (peopleOk ? await verifyChildAndAdult(b64) : false) : true;

    lastPeopleOk = peopleOk;
    lastGenderOk = genderOk;
    lastClothingOk = clothingOk;
    lastChildAdultOk = childAdultOk;

    if (peopleOk && genderOk && clothingOk && childAdultOk) {
      return { b64, attempts: attempt, peopleOk, genderOk, clothingOk, childAdultOk };
    }
  }

  return {
    b64: last,
    attempts: 3,
    peopleOk: lastPeopleOk,
    genderOk: lastGenderOk,
    clothingOk: lastClothingOk,
    childAdultOk: lastChildAdultOk,
  };
}

/* -----------------------------
   Handler
----------------------------- */

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

    // NEW: scan mode (no images). Finds explicit non-UK/mixed origin mentioned in case text.
    const scanOrigin = String(req.query.scanOrigin ?? "0") === "1";

    // NEW: scan mode (no images). Finds cases that require TWO-person headshot.
    const scanPair = String(req.query.scanPair ?? "0") === "1";

    const endAt = Number(req.query.endAt ?? maxCase);

    if (scanOrigin) {
      const flagged: any[] = [];
      const ukOnly: any[] = [];
      const notStated: any[] = [];
      const errors: any[] = [];

      for (let caseId = startFrom; caseId <= endAt; caseId++) {
        try {
          const { tableName, caseText } = await getCaseText(caseId);

          if (!caseText.trim()) {
            notStated.push({
              caseId,
              tableName,
              origin_status: "not-stated",
              origins_mentioned: [],
              evidence: "",
              confidence: "low",
            });
            continue;
          }

          const r = await extractOriginFlag(caseText, caseId);

          const row = {
            caseId,
            tableName,
            origin_status: r.origin_status,
            origins_mentioned: r.origins_mentioned,
            evidence: r.evidence,
            confidence: r.confidence,
          };

          if (r.origin_status === "explicit-non-uk" || r.origin_status === "explicit-mixed") {
            flagged.push(row);
          } else if (r.origin_status === "explicit-uk") {
            ukOnly.push(row);
          } else {
            notStated.push(row);
          }

          await sleep(80);
        } catch (e: any) {
          errors.push({ caseId, error: extractErr(e) });
        }
      }

      return res.status(200).json({
        ok: true,
        scanOrigin: true,
        startFrom,
        endAt,
        counts: {
          flagged: flagged.length,
          ukOnly: ukOnly.length,
          notStated: notStated.length,
          errors: errors.length,
        },
        flagged,
        errors,
      });
    }

    /** NEW: scanPair mode (NO images / NO blob writes) */
    if (scanPair) {
      const pair: any[] = [];
      const single: any[] = [];
      const errors: any[] = [];

      for (let caseId = startFrom; caseId <= endAt; caseId++) {
        try {
          const { tableName, caseText } = await getCaseText(caseId);

          if (!caseText.trim()) {
            single.push({ caseId, tableName, composition: "single", reason: "no-text", confidence: "low", evidence: "" });
            continue;
          }

          // use profile age as the deterministic child<6 rule enforcement
          const profile = await makeProfile(caseText, caseId);
          const comp = await decideComposition(caseText, caseId);

          let composition: "single" | "pair" = comp.composition;
          const ageN = parseAgeNumber(profile.age);
          if (ageN != null && ageN < 6) composition = "pair";

          const row = {
            caseId,
            tableName,
            age: profile.age,
            composition,
            reason: comp.reason,
            evidence: comp.evidence,
            confidence: comp.confidence,
            companion: composition === "pair" ? comp.companion : null,
          };

          if (composition === "pair") pair.push(row);
          else single.push(row);

          await sleep(80);
        } catch (e: any) {
          errors.push({ caseId, error: extractErr(e) });
        }
      }

      return res.status(200).json({
        ok: true,
        scanPair: true,
        startFrom,
        endAt,
        counts: {
          pair: pair.length,
          single: single.length,
          errors: errors.length,
        },
        pair,
        single,
        errors,
      });
    }

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

        // NEW: decide composition + enforce child<6 => pair
        const comp = await decideComposition(caseText, caseId);
        profile.composition = comp.composition;
        profile.companion = comp.companion;
        forceChildPairIfNeeded(profile);

        // Deterministic background + clothing color; ensure no clash
        const seed = seedFromText(caseId, caseText);
        const background = pick(BACKGROUNDS, seed);
        let clothingColor = pick(CLOTHING_COLORS, seed + 17);
        clothingColor = ensureNoBackgroundClothingClash(clothingColor, background);

        profile.background = background;
        profile.clothing_color = clothingColor;

        if (debug) {
          return res.status(200).json({
            ok: true,
            caseId,
            tableName,
            caseTextPreview: caseText.slice(0, 2500),
            profile,
            compositionDecision: comp,
          });
        }

        const imagePrompt = buildImagePromptFromProfile(profile, caseId);

        // NEW: expected people (1 or 2) + child/adult check for <6
        const isPair = profile.composition === "pair" && !!profile.companion;
        const expectedPeople: 1 | 2 = isPair ? 2 : 1;

        const ageN = parseAgeNumber(profile.age);
        const requireChildAdult = ageN != null && ageN < 6;

        const gen = await generateVerifiedHeadshot(
          imagePrompt,
          profile.gender_presentation,
          profile.clothing_type,
          expectedPeople,
          requireChildAdult
        );

        const imageUrl = await uploadPng(caseId, gen.b64, overwrite);

        const profileDoc = {
          caseId,
          createdAt: new Date().toISOString(),
          profile,
          imagePrompt,
          imageUrl,
          generationChecks: {
            attempts: gen.attempts,
            peopleOk: gen.peopleOk,
            genderOk: gen.genderOk,
            clothingOk: gen.clothingOk,
            childAdultOk: gen.childAdultOk,
          },
          source: { tableName },
          image: { model: IMAGE_MODEL_USED, quality: IMAGE_QUALITY_USED, size: IMAGE_SIZE_USED },
        };

        const profileUrl = await uploadJson(caseId, profileDoc, overwrite);

        processed.push({ caseId, status: "done", imageUrl, profileUrl, checks: profileDoc.generationChecks });

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
