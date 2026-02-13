import type { VercelRequest, VercelResponse } from "@vercel/node";
import archiver from "archiver";
import { list } from "@vercel/blob";

const { RUN_SECRET } = process.env;

if (!RUN_SECRET) throw new Error("Missing RUN_SECRET env var");

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

async function fetchToBuffer(url: string): Promise<Buffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Failed to fetch blob: ${r.status}`);
  return Buffer.from(await r.arrayBuffer());
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  try {
    requireSecret(req);

    const includeProfiles = String(req.query.includeProfiles ?? "0") === "1";

    res.setHeader("Content-Type", "application/zip");
    res.setHeader(
      "Content-Disposition",
      `attachment; filename="case-blob-export.zip"`
    );

    const archive = archiver("zip", { zlib: { level: 9 } });
    archive.on("error", (err) => {
      throw err;
    });
    archive.pipe(res);

    // 1) Avatars
    {
      let cursor: string | undefined = undefined;
      do {
        const page = await list({
          prefix: "case-avatars/",
          cursor,
          limit: 1000
        });

        for (const b of page.blobs) {
          const buf = await fetchToBuffer(b.url);
          // Store inside zip without the prefix
          const name = b.pathname.replace(/^case-avatars\//, "avatars/");
          archive.append(buf, { name });
        }

        cursor = page.cursor ?? undefined;
      } while (cursor);
    }

    // 2) Profiles (optional)
    if (includeProfiles) {
      let cursor: string | undefined = undefined;
      do {
        const page = await list({
          prefix: "case-profiles/",
          cursor,
          limit: 1000
        });

        for (const b of page.blobs) {
          const buf = await fetchToBuffer(b.url);
          const name = b.pathname.replace(/^case-profiles\//, "profiles/");
          archive.append(buf, { name });
        }

        cursor = page.cursor ?? undefined;
      } while (cursor);
    }

    await archive.finalize();
  } catch (e: any) {
    const status = e?.status || 500;
    res.status(status).json({ ok: false, error: e?.message || "Unknown error", status });
  }
}
