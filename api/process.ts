import type { VercelRequest, VercelResponse } from "@vercel/node";
import Airtable from "airtable";
import OpenAI from "openai";
import { put } from "@vercel/blob";

const {
  AIRTABLE_TOKEN,
  AIRTABLE_BASE_ID,
  OPENAI_API_KEY,
  RUN_SECRET,
  MAX_CASE_ID,
  CASE_PROFILES_TABLE
} = process.env;

if (!AIRTABLE_TOKEN || !AIRTABLE_BASE_ID || !OPENAI_API_KEY || !CASE_PROFILES_TABLE) {
  throw new Error("Missing required env vars.");
}

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
const airtable = new Airtable({ apiKey: AIRTABLE_TOKEN }).base(AIRTABLE_BASE_ID);

function requireSecret(req: VercelRequest) {
  const secret = req.headers["x-run-secret"];
  if (!RUN_SECRET || secret !== RUN_SECRET) {
    const err: any = new Error("Unauthorized");
    err.status = 401;
    throw err;
  }
}

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
  // Airtable attachments/arrays/objects
  try { return JSON.stringify(v); } catch { return String(v); }
}

function buildCaseText(fields: Record<string, any>): string {
  const parts: string[] = [];
  for (const k of CASE_FIELDS) {
    const v = fields[k];
    if (v == null || v === "") continue;
    parts.push(`${k}: ${normalizeFieldValue(v)}`);
  }
  return parts.join("\n");
}

async function sleep(ms: number) {
  await new Promise(r => setTimeout(r, ms));
}

async function withRetry<T>(fn: () => Promise<T>, tries = 3): Promise<T> {
  let lastErr: any;
  for (let i = 0; i < tries; i++) {
    try { return await fn(); }
    catch (e: any) {
      lastErr = e;
      // simple backoff for 429/5xx
      await sleep(800 * (i + 1));
    }
  }
  throw lastErr;
}

async function makePhotoDescription(caseText: string): Promise<string> {
  const prompt = `
From the text below, write EXACTLY 2 sentences describing the person’s appearance for a headshot.

Include:
- age (only if explicitly stated),
- BMI only if explicitly stated or directly computable from explicit height and weight in the text,
- any country / nationality / ethnicity ONLY if explicitly stated in the text,
- any particular facial features (only if explicitly stated),
- clothing they may wear (only if explicitly stated).

If any item is missing, write "not specified" for that detail.
Do NOT guess country/nationality/ethnicity from the name.
Do NOT include the person’s name.
Do NOT include medical diagnoses unless they explicitly describe visible appearance traits.

TEXT:
${caseText}
`.trim();

  const resp = await withRetry(() =>
    openai.responses.create({
      model: "gpt-4.1-mini",
      input: prompt
    })
  );

  const out = (resp.output_text || "").trim().replace(/\s+/g, " ");
  // Enforce “2 sentences” lightly (don’t overthink; just keep it short)
  return out;
}

function buildImagePrompt(photoDesc: string): string {
  return `
I need a headshot photo - facing the camera. I want it against a plain background.
The facial expression should be mildly happy always, regardless of what’s mentioned below.
There should be no names, and the background should be light in color and can be any color between blue and white and grey,
but nothing dark and nothing yellow or red.
Studio side lighting like you would get in a photo shoot.
Photorealistic DSLR 80mm lens full frame equivalent.
No text, no watermark, no logo. Single person, centered framing.

Description:
${photoDesc}
`.trim().replace(/\s+/g, " ");
}

async function generateHeadshotPngBase64(prompt: string): Promise<string> {
  // GPT image models return base64 by default :contentReference[oaicite:1]{index=1}
  const img = await withRetry(() =>
    openai.images.generate({
      model: "gpt-image-1",
      prompt,
      size: "1024x1024"
    })
  );

  const first: any = img.data?.[0];
  if (!first?.b64_json) throw new Error("No b64_json returned from images API.");
  return first.b64_json as string;
}

async function uploadToBlob(caseId: number, b64: string): Promise<string> {
  const bytes = Buffer.from(b64, "base64");
  const path = `case-avatars/${String(caseId).padStart(3, "0")}.png`;

  const res = await withRetry(() =>
    put(path, bytes, {
      access: "public",
      contentType: "image/png",
      addRandomSuffix: false
    })
  );

  return res.url;
}

async function upsertCaseProfile(caseId: number, updateFields: Record<string, any>) {
  const table = airtable(CASE_PROFILES_TABLE!);

  const existing = await table
    .select({ filterByFormula: `{CaseId} = ${caseId}`, maxRecords: 1 })
    .firstPage();

  if (existing.length) {
    await table.update(existing[0].id, updateFields);
  } else {
    await table.create({ CaseId: caseId, ...updateFields });
  }
}

async function getCaseRecord(caseId: number) {
  const tableName = `Case ${caseId}`;
  const records = await airtable(tableName).select({ maxRecords: 1 }).firstPage();
  return records[0] || null;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  try {
    requireSecret(req);

    const startFrom = Number(req.query.startFrom ?? 1);
    const limit = Number(req.query.limit ?? 10);
    const maxCase = Number(MAX_CASE_ID ?? 355);

    const processed: any[] = [];

    for (let caseId = startFrom; caseId <= maxCase; caseId++) {
      if (processed.length >= limit) break;

      try {
        // Optional: skip if already done
        // (we check CaseProfiles for Status=done)
        // For speed, you can remove this and rely on startFrom batching.
        await upsertCaseProfile(caseId, { Status: "processing", LastError: "" });

        const rec = await getCaseRecord(caseId);
        if (!rec) {
          await upsertCaseProfile(caseId, { Status: "error", LastError: "No record found in Case table." });
          processed.push({ caseId, status: "no-record" });
          continue;
        }

        const caseText = buildCaseText(rec.fields as any);
        const photoDesc = await makePhotoDescription(caseText);

        await upsertCaseProfile(caseId, {
          PhotoDescription: photoDesc,
          Status: "described"
        });

        const imagePrompt = buildImagePrompt(photoDesc);
        const b64 = await generateHeadshotPngBase64(imagePrompt);
        const url = await uploadToBlob(caseId, b64);

        await upsertCaseProfile(caseId, {
          ImageUrl: url,
          Status: "done"
        });

        processed.push({ caseId, status: "done" });

        // gentle throttle
        await sleep(250);
      } catch (err: any) {
        await upsertCaseProfile(caseId, {
          Status: "error",
          LastError: err?.message || String(err)
        });
        processed.push({ caseId, status: "error" });
      }
    }

    res.status(200).json({
      ok: true,
      startFrom,
      limit,
      processedCount: processed.length,
      processed
    });
  } catch (e: any) {
    res.status(e?.status || 500).json({ ok: false, error: e?.message || "Unknown error" });
  }
}
