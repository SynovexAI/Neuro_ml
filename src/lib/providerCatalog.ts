// Shared provider presets — OpenAI-compatible endpoints. Client-safe (no server-only import)
// so both the admin UI and server code use the same base URLs. `free` = a free API key/tier
// exists for testing; `embeddings` = the provider serves an /embeddings endpoint (needed for
// RAG neural/semantic vectors). `keyHint` points to where to get a key.
export type ProviderPreset = {
  label: string;
  baseUrl: string;
  free: boolean;
  embeddings: boolean;
  keyHint?: string;
};

export const PROVIDER_CATALOG: Record<string, ProviderPreset> = {
  groq:       { label: "Groq",                    baseUrl: "https://api.groq.com/openai/v1",                    free: true,  embeddings: false, keyHint: "console.groq.com/keys" },
  gemini:     { label: "Google Gemini",           baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai", free: true,  embeddings: true,  keyHint: "aistudio.google.com/apikey" },
  cerebras:   { label: "Cerebras",                baseUrl: "https://api.cerebras.ai/v1",                        free: true,  embeddings: false, keyHint: "cloud.cerebras.ai" },
  openrouter: { label: "OpenRouter",              baseUrl: "https://openrouter.ai/api/v1",                      free: true,  embeddings: false, keyHint: "openrouter.ai/keys · has free models" },
  mistral:    { label: "Mistral",                 baseUrl: "https://api.mistral.ai/v1",                         free: true,  embeddings: true,  keyHint: "console.mistral.ai · free tier" },
  nvidia:     { label: "NVIDIA NIM",              baseUrl: "https://integrate.api.nvidia.com/v1",               free: true,  embeddings: true,  keyHint: "build.nvidia.com · free credits" },
  github:     { label: "GitHub Models",           baseUrl: "https://models.github.ai/inference",                free: true,  embeddings: true,  keyHint: "github.com/settings/tokens (PAT) · alt base: models.inference.ai.azure.com" },
  sambanova:  { label: "SambaNova",               baseUrl: "https://api.sambanova.ai/v1",                       free: true,  embeddings: true,  keyHint: "cloud.sambanova.ai · fast, free tier" },
  huggingface:{ label: "Hugging Face",            baseUrl: "https://router.huggingface.co/v1",                  free: true,  embeddings: true,  keyHint: "huggingface.co/settings/tokens · monthly free credits" },
  cohere:     { label: "Cohere",                  baseUrl: "https://api.cohere.ai/compatibility/v1",            free: true,  embeddings: true,  keyHint: "dashboard.cohere.com · free trial keys" },
  qwen:       { label: "Alibaba Qwen (DashScope)", baseUrl: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", free: true, embeddings: true, keyHint: "dashscope.console.aliyun.com · free quota" },
  cloudflare: { label: "Cloudflare Workers AI",   baseUrl: "https://api.cloudflare.com/client/v4/accounts/ACCOUNT_ID/ai/v1", free: true, embeddings: true, keyHint: "replace ACCOUNT_ID in the URL · dash.cloudflare.com" },
  openai:     { label: "OpenAI",                  baseUrl: "https://api.openai.com/v1",                         free: false, embeddings: true,  keyHint: "platform.openai.com/api-keys" },
  deepseek:   { label: "DeepSeek",                baseUrl: "https://api.deepseek.com/v1",                       free: false, embeddings: false, keyHint: "platform.deepseek.com" },
  together:   { label: "Together AI",             baseUrl: "https://api.together.xyz/v1",                       free: false, embeddings: true,  keyHint: "api.together.ai" },
  fireworks:  { label: "Fireworks AI",            baseUrl: "https://api.fireworks.ai/inference/v1",             free: false, embeddings: true,  keyHint: "fireworks.ai" },
  xai:        { label: "xAI (Grok)",              baseUrl: "https://api.x.ai/v1",                               free: false, embeddings: false, keyHint: "console.x.ai" },
  ollama:     { label: "Ollama (local)",          baseUrl: "http://localhost:11434/v1",                         free: true,  embeddings: true,  keyHint: "no key — runs locally" },
  custom:     { label: "Custom (OpenAI-compatible)", baseUrl: "",                                               free: false, embeddings: false },
};
