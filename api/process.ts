import type { VercelRequest, VercelResponse } from "@vercel/node";
import Airtable from "airtable";
import OpenAI from "openai";
import { put, head } from "@vercel/blob";

const {
  AIRTABLE_TOKEN,
  AIRTABLE_BASE_ID,
  OPENAI_API_KEY,
  RUN_SECRET,
  MAX_CASE_ID,
} = process.env;

if (!AIRTABLE_TOKEN || !AIRTABLE_BASE_ID || !OPENAI_API_KEY || !RUN_SECRET) {
  throw new Error(
    "Missing env vars: AIRTABLE_TOKEN, AIRTABLE_BASE_ID, OPENAI_API_KEY, RUN_SECRET"
  );
}

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
const airtable = new Airtable({ apiKey: AIRTABLE_TOKEN }).base(AIRTABLE_BASE_ID);

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

function buildCaseText(fields: Record<string, any>): string {
  const parts: string[] = [];
  for (const k of CASE_FIELDS) {
    const v = fields[k];
    if (v == null || v === "") continue;
    parts.push(`${k}: ${normalizeFieldValue(v)}`);
  }
  return parts.join("\n");
}

async function getCaseRecord(caseId: number) {
  const tableName = `Case ${caseId}`;
  try {
    const records = await airtable(tableName).select({ maxRecords: 1 }).firstPage();
    return { tableName, record: records[0] || null };
  } catch (err: any) {
    const e: any = new Error(
      `AIRTABLE_READ_FAILED table="${tableName}" msg="${err?.message || String(err)}"`
    );
    e.status = err?.statusCode || err?.status || 500;
    throw e;
  }
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
      input: prompt,
    })
  );

  return (resp.output_text || "").trim().replace(/\s+/g, " ");
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
  const img = await withRetry(() =>
    openai.images.generate({
      model: "gpt-image-1",
      prompt,
      size: "1024x1024",
    })
  );

  const first: any = img.data?.[0];
  if (!first?.b64_json) throw new Error("OPENAI_IMAGE_NO_B64_JSON");
  return first.b64_json as string;
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

async function uploadProfileJson(
  caseId: number,
  obj: any,
  overwrite: boolean
): Promise<string> {
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
    const limit = Number(req.query.limit ?? 10);
    const maxCase = Number(MAX_CASE_ID ?? 355);

    // dryRun=1 => does NOT call OpenAI or Blob puts; just checks Airtable reads.
    const dryRun = String(req.query.dryRun ?? "0") === "1";

    // overwrite=1 => re-generate and re-upload even if files exist
    const overwrite = String(req.query.overwrite ?? "0") === "1";

    const processed: any[] = [];

    for (let caseId = startFrom; caseId <= maxCase; caseId++) {
      if (processed.length >= limit) break;

      try {
        const { tableName, record } = await getCaseRecord(caseId);
        if (!record) {
          processed.push({ caseId, status: "no-record", tableName });
          continue;
        }

        const caseText = buildCaseText(record.fields as any);

        if (dryRun) {
          processed.push({ caseId, status: "dryrun-ok" });
          continue;
        }

        const photoDescription = await makePhotoDescription(caseText);
        const imagePrompt = buildImagePrompt(photoDescription);

        const b64 = await generateHeadshotPngBase64(imagePrompt);
        const imageUrl = await uploadPng(caseId, b64, overwrite);

        const profile = {
          caseId,
          createdAt: new Date().toISOString(),
          photoDescription,
          imagePrompt,
          imageUrl,
          source: {
            tableName,
            fieldsIncluded: CASE_FIELDS,
          },
        };

        const profileUrl = await uploadProfileJson(caseId, profile, overwrite);

        processed.push({
          caseId,
          status: "done",
          imageUrl,
          profileUrl,
        });

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
      processedCount: processed.length,
      processed,
    });
  } catch (e: any) {
    const status = e?.status || 500;
    res.status(status).json({ ok: false, error: e?.message || "Unknown error", status });
  }
}
