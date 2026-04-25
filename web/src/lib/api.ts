const API_ROOT = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');
const BASE = `${API_ROOT}/api`;

export async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

export interface QueryResult {
  photo_id: string;
  photo_path: string;
  relevance_score: number;
  evidence: string;
  image_type: string;
  matched_entity?: string | null;
}

export interface QueryResponse {
  query: string;
  mode: string;
  query_type_detected: string;
  results: QueryResult[];
  confidence_grade: string;
  confidence_score: number;
  answer_summary: string;
  source_photos: string[];
  warning: string | null;
  latency_s: number;
  routing_source: string | null;
}

export interface Photo {
  id: string;
  file_path: string;
  filename: string;
  image_type: string;
  ocr_text: string;
  description: string;
  entities: { type: string; value: string }[];
  confidence: number;
}

export interface KBStats {
  total_photos: number;
  type_distribution: Record<string, number>;
  entity_type_distribution: Record<string, number>;
  total_entities: number;
  has_ocr: number;
  avg_entities_per_photo: number;
}

export interface HealthStatus {
  status: string;
  knowledge_base_photos: number;
  has_eval_results: boolean;
  has_ablation_results: boolean;
  has_rl_models: boolean;
}

/**
 * Lightweight health probe used by BackendGate while Render free-tier
 * instances cold-start. Never throws — returns `{ ok: false }` on any
 * network error, timeout, or non-2xx response.
 */
export async function pingHealth(timeoutMs = 4000): Promise<{ ok: boolean; data?: HealthStatus }> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${BASE}/health`, { signal: ctrl.signal, cache: 'no-store' });
    if (!res.ok) return { ok: false };
    const data = (await res.json()) as HealthStatus;
    return { ok: true, data };
  } catch {
    return { ok: false };
  } finally {
    clearTimeout(timer);
  }
}

/** True when the frontend is configured to talk to a remote backend. */
export const HAS_REMOTE_BACKEND = API_ROOT.length > 0;
