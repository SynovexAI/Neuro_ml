// Upstream LLM errors arrive as a raw provider JSON body, which is useless to a
// student staring at a lab. Map the common ones to one actionable sentence.
export function explainProviderError(msg: string): string {
  const m = msg || "Unknown error";
  if (/\b402\b|payment_required|billing|insufficient_quota|credit/i.test(m))
    return "The LLM provider rejected the request — this account is out of credit or quota (HTTP 402). Add a free provider (Groq, Cerebras, Gemini) or top up billing under Admin → Providers.";
  if (/\b401\b|invalid_api_key|unauthorized|authentication/i.test(m))
    return "The provider rejected the API key (HTTP 401). Check the key under Admin → Providers.";
  if (/\b403\b|permission|forbidden/i.test(m))
    return "The provider refused this model (HTTP 403) — the key may not have access to it. Pick a different model.";
  if (/\b404\b|model_not_found|does not exist/i.test(m))
    return "That model does not exist on this provider (HTTP 404). Pick another from the model list.";
  if (/\b429\b|rate.?limit/i.test(m))
    return "Rate limited by the provider (HTTP 429). Wait a few seconds and retry.";
  if (/monthly token limit/i.test(m))
    return "This workspace hit its monthly token quota. Ask an admin to raise it.";
  if (/failed to fetch|network|ECONNREFUSED/i.test(m))
    return "The request never reached the provider — check the dev server and your connection.";
  if (/no (llm )?provider|not configured/i.test(m))
    return "No LLM provider is configured. Add one under Admin → Providers — Groq and Cerebras have free tiers.";
  return m.slice(0, 240);
}
