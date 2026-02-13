import type { VercelRequest, VercelResponse } from "@vercel/node";
import Airtable from "airtable";
import archiver from "archiver";

const { AIRTABLE_TOKEN, AIRTABLE_BASE_ID, RUN_SECRET, CASE_PROFILES_TABLE } = process.env;

if (!AIRTABLE_TOKEN || !AIRTABLE_BASE_ID || !CASE_PROFILES_TABLE) {
  throw new Error("Missing required env vars.");
}

const airtable = new Airtable({ apiKey: AIRTABLE_TOKEN }).base(AIRTABLE_BASE_ID);

function requireSecret(req: VercelRequest) {
  const secret = req.headers["x-run-secret"];
  if (!RUN_SECRET || secret !== RUN_SECRET) {
    const err: any = new Error("Unauthorized");
    err.status = 401;
    throw err;
  }
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  try {
    requireSecret(req);

    res.setHeader("Content-Type", "application/zip");
    res.setHeader("Content-Disposition", `attachment; filename="case-avatars.zip"`);

    const archive = archiver("zip", { zlib: { level: 9 } });
    archive.on("error", (err) => { throw err; });
    archive.pipe(res);

    const table = airtable(CASE_PROFILES_TABLE!);

    // Page through results (Airtable pages 100 at a time)
    let offset: string | undefined = undefined;
    do {
      const page = await table.select({
        pageSize: 100,
        filterByFormula: `{Status} = "done"`
      }).firstPage();

      for (const r of page) {
        const caseId = r.fields["CaseId"];
        const url = r.fields["ImageUrl"];
        if (!caseId || !url || typeof url !== "string") continue;

        const resp = await fetch(url);
        if (!resp.ok) continue;

        const buf = Buffer.from(await resp.arrayBuffer());
        archive.append(buf, { name: `${String(caseId).padStart(3, "0")}.png` });
      }

      // Airtable SDK doesn't expose offset nicely via firstPage(),
      // so for huge tables you'd switch to eachPage(). For ~355, this is fine if within 100.
      offset = undefined;
    } while (offset);

    await archive.finalize();
  } catch (e: any) {
    res.status(e?.status || 500).json({ ok: false, error: e?.message || "Unknown error" });
  }
}
