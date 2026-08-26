/**
 * Local Ollama routing for read-only queries.
 * Responses are tagged with LOCAL_SIGIL so the user knows which model answered.
 */

import { existsSync, readFileSync } from "fs";
import { join } from "path";

const OLLAMA_BASE = "http://localhost:11434";
export const LOCAL_SIGIL = "◉";

const CLASSIFIER_PROMPT = (msg: string) => `Classify this message as READ or WRITE. Answer with exactly one word.

READ = looking up info, asking questions, requesting summaries or reports
WRITE = logging, adding, updating, deleting, deploying, creating, changing

Message: "${msg}"

Answer:`;

const HOME = process.env.USERPROFILE ?? process.env.HOME ?? "";
const PROMPTS_DIR = join(import.meta.dir, "..", "prompts");

const CONTEXT_MAP: Array<{ pattern: RegExp; file: string }> = [
  { pattern: /whiskey|whisky/i, file: `${HOME}/Dev/whiskey-dashboard/public/whiskeys.json` },
  { pattern: /beer/i, file: `${HOME}/Dev/beer-dashboard/public/beers.json` },
  { pattern: /todo|task/i, file: `${HOME}/Dev/todos-dashboard/public/tasks.json` },
  { pattern: /birthday/i, file: `${HOME}/Dev/birthday-tracker/src/data/birthdays.json` },
  { pattern: /banjo|song/i, file: `${HOME}/Dev/banjo-dashboard/songs.json` },
];

function safeRead(path: string): string {
  try { return readFileSync(path, "utf8").trim(); } catch { return ""; }
}

function gatherContextFiles(prompt: string): string[] {
  return CONTEXT_MAP
    .filter(({ pattern }) => pattern.test(prompt))
    .map(({ file }) => file)
    .filter(existsSync);
}

function buildDataContext(files: string[]): string {
  return files
    .map(f => { try { return `[${f}]\n${readFileSync(f, "utf8")}`; } catch { return ""; } })
    .filter(Boolean)
    .join("\n\n");
}

/**
 * Build the system prompt for Ollama: SOUL.md + CLAUDE.md.
 * SOUL.md provides the persona/vibe; CLAUDE.md has Adam's full context.
 */
function buildSystemPrompt(): string {
  const parts: string[] = [];

  const soul = safeRead(join(PROMPTS_DIR, "SOUL.md"));
  if (soul) parts.push(soul);

  // Project CLAUDE.md has the richest context (workspace, apps, key facts)
  const claudeMd = safeRead(join(process.cwd(), "CLAUDE.md"));
  if (claudeMd) parts.push(claudeMd);

  return parts.join("\n\n---\n\n");
}

/** Returns true if Ollama is reachable. */
export async function isOllamaAvailable(): Promise<boolean> {
  try {
    const res = await fetch(`${OLLAMA_BASE}/api/tags`, { signal: AbortSignal.timeout(2000) });
    return res.ok;
  } catch {
    return false;
  }
}

/** Classifies a prompt as read-only or write. Throws if Ollama is unreachable. */
export async function classifyReadOnly(prompt: string, classifierModel: string): Promise<boolean> {
  const res = await fetch(`${OLLAMA_BASE}/api/generate`, {
    method: "POST",
    body: JSON.stringify({ model: classifierModel, prompt: CLASSIFIER_PROMPT(prompt), stream: false }),
    signal: AbortSignal.timeout(8000),
  });
  if (!res.ok) throw new Error(`Ollama classify failed: ${res.status}`);
  const { response } = await res.json() as { response: string };
  return response.trim().toUpperCase().startsWith("READ");
}

/**
 * Run a read-only query through a local Ollama model.
 * Injects SOUL.md + CLAUDE.md as system prompt, data files as context.
 * Appends LOCAL_SIGIL to the response.
 * Throws if Ollama is unreachable or returns an error.
 */
export async function queryOllama(prompt: string, readerModel: string): Promise<string> {
  const files = gatherContextFiles(prompt);
  const dataContext = buildDataContext(files);
  const system = buildSystemPrompt();

  const userPrompt = dataContext
    ? `Here is the relevant data:\n\n${dataContext}\n\nUser query: ${prompt}\n\nAnswer concisely:`
    : prompt;

  const res = await fetch(`${OLLAMA_BASE}/api/generate`, {
    method: "POST",
    body: JSON.stringify({ model: readerModel, system, prompt: userPrompt, stream: false }),
    signal: AbortSignal.timeout(30000),
  });
  if (!res.ok) throw new Error(`Ollama query failed: ${res.status}`);
  const { response } = await res.json() as { response: string };
  return `${response.trim()} ${LOCAL_SIGIL}`;
}

// --- Claude-outage emergency fallback (Muse Glimmer) ---
// runner.ts routes user-facing messages here when the Claude INSTALL itself
// is broken (stub exe from a bad npm auto-update, missing binary). Distinct
// from the disabled read-only routing above: this is a last-resort "stay
// responsive" path, not a quality-parity one. The model is told explicitly
// that Claude is down so it doesn't claim capabilities it lacks, and the
// caller prepends a user-visible outage banner.
const GLIMMER_MODEL = "muse-glimmer:30b";
const GLIMMER_TIMEOUT_MS = 8 * 60 * 1000; // 30B on the 3060 Ti takes 2-4+ min, longer on a cold load
const GLIMMER_CONTEXT_CAP = 12_000; // keep injected data inside the model's context window

export async function queryGlimmerOutage(prompt: string): Promise<string> {
  const files = gatherContextFiles(prompt);
  let dataContext = buildDataContext(files);
  if (dataContext.length > GLIMMER_CONTEXT_CAP) {
    dataContext = dataContext.slice(0, GLIMMER_CONTEXT_CAP) + "\n...[truncated]";
  }
  const system = [
    "You are Glimmer, Adam's local backup assistant (covering for PunkAss). Sharp, warm, brief — Discord-length replies.",
    "EMERGENCY BACKUP MODE: Claude, the primary assistant, is currently DOWN (broken installation). You are covering until the automatic repair completes.",
    "You have NO tools in this mode — never claim to have edited files, run commands, deployed, logged, or scheduled anything. If the request needs an action, say it'll be picked up once Claude is repaired.",
    "The user already sees a banner saying Claude is down — don't re-explain it, just answer.",
  ].join("\n\n");
  const userPrompt = dataContext
    ? `Relevant local data:\n\n${dataContext}\n\nUser message: ${prompt}`
    : prompt;
  const res = await fetch(`${OLLAMA_BASE}/api/chat`, {
    method: "POST",
    body: JSON.stringify({
      model: GLIMMER_MODEL,
      messages: [
        { role: "system", content: system },
        { role: "user", content: userPrompt },
      ],
      stream: false,
      think: false,
      keep_alive: "10m",
    }),
    signal: AbortSignal.timeout(GLIMMER_TIMEOUT_MS),
  });
  if (!res.ok) throw new Error(`Ollama outage fallback failed: ${res.status}`);
  const data = await res.json() as { message?: { content?: string } };
  const text = (data.message?.content ?? "").trim();
  if (!text) throw new Error("Ollama outage fallback returned an empty response");
  return `${text} ✨`;
}
