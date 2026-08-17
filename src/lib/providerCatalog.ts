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
  defaultModels: string[];
};

export const PROVIDER_CATALOG: Record<string, ProviderPreset> = {
  groq:       { label: "Groq",                    baseUrl: "https://api.groq.com/openai/v1",                    free: true,  embeddings: false, keyHint: "console.groq.com/keys", defaultModels: ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it", "deepseek-r1-distill-llama-70b"] },
  gemini:     { label: "Google Gemini",           baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai", free: true,  embeddings: true,  keyHint: "aistudio.google.com/apikey", defaultModels: ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "text-embedding-004"] },
  cerebras:   { label: "Cerebras",                baseUrl: "https://api.cerebras.ai/v1",                        free: true,  embeddings: false, keyHint: "cloud.cerebras.ai", defaultModels: ["llama3.3-70b", "llama3.1-8b"] },
  openrouter: { label: "OpenRouter",              baseUrl: "https://openrouter.ai/api/v1",                      free: true,  embeddings: false, keyHint: "openrouter.ai/keys · has free models", defaultModels: ["deepseek/deepseek-r1", "meta-llama/llama-3.3-70b-instruct", "google/gemini-2.0-flash-001", "qwen/qwen-2.5-72b-instruct"] },
  mistral:    { label: "Mistral",                 baseUrl: "https://api.mistral.ai/v1",                         free: true,  embeddings: true,  keyHint: "console.mistral.ai · free tier", defaultModels: ["mistral-large-latest", "mistral-small-latest", "open-mixtral-8x7b", "mistral-embed"] },
  nvidia:     { label: "NVIDIA NIM",              baseUrl: "https://integrate.api.nvidia.com/v1",               free: true,  embeddings: true,  keyHint: "build.nvidia.com · free credits", defaultModels: ["meta/llama-3.3-70b-instruct", "deepseek-ai/deepseek-r1", "nvidia/llama-3.1-nemotron-70b-instruct"] },
  github:     { label: "GitHub Models (preview)", baseUrl: "https://models.inference.ai.azure.com", free: true, embeddings: true, keyHint: "preview: Azure host needs classic ghp_ token", defaultModels: ["gpt-4o", "gpt-4o-mini", "Meta-Llama-3.3-70B-Instruct", "Phi-3.5-mini-instruct"] },
  sambanova:  { label: "SambaNova",               baseUrl: "https://api.sambanova.ai/v1",                       free: true,  embeddings: true,  keyHint: "cloud.sambanova.ai · fast, free tier", defaultModels: ["Meta-Llama-3.3-70B-Instruct", "DeepSeek-R1-Distill-Llama-70B", "Qwen2.5-72B-Instruct"] },
  huggingface:{ label: "Hugging Face",            baseUrl: "https://router.huggingface.co/v1",                  free: true,  embeddings: true,  keyHint: "huggingface.co/settings/tokens · monthly free credits", defaultModels: ["meta-llama/Llama-3.3-70B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "Qwen/Qwen2.5-72B-Instruct"] },
  cohere:     { label: "Cohere",                  baseUrl: "https://api.cohere.ai/compatibility/v1",            free: true,  embeddings: true,  keyHint: "dashboard.cohere.com · free trial keys", defaultModels: ["command-r-plus", "command-r", "embed-english-v3.0", "embed-multilingual-v3.0"] },
  jina:       { label: "Jina AI (embeddings)",     baseUrl: "https://api.jina.ai/v1",                            free: true,  embeddings: true,  keyHint: "jina.ai · 1M free tokens", defaultModels: ["jina-embeddings-v3", "jina-embeddings-v2-base-en"] },
  qwen:       { label: "Alibaba Qwen (DashScope)", baseUrl: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", free: true, embeddings: true, keyHint: "dashscope.console.aliyun.com · free quota", defaultModels: ["qwen-max", "qwen-plus", "qwen-turbo", "text-embedding-v2"] },
  cloudflare: { label: "Cloudflare Workers AI",   baseUrl: "https://api.cloudflare.com/client/v4/accounts/ACCOUNT_ID/ai/v1", free: true, embeddings: true, keyHint: "dash.cloudflare.com", defaultModels: ["@cf/meta/llama-3.3-70b-instruct", "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b", "@cf/baai/bge-small-en-v1.5"] },
  openai:     { label: "OpenAI",                  baseUrl: "https://api.openai.com/v1",                         free: false, embeddings: true,  keyHint: "platform.openai.com/api-keys", defaultModels: ["gpt-4o", "gpt-4o-mini", "o3-mini", "gpt-4-turbo", "text-embedding-3-small"] },
  deepseek:   { label: "DeepSeek",                baseUrl: "https://api.deepseek.com/v1",                       free: false, embeddings: false, keyHint: "platform.deepseek.com", defaultModels: ["deepseek-chat", "deepseek-reasoner"] },
  together:   { label: "Together AI",             baseUrl: "https://api.together.xyz/v1",                       free: false, embeddings: true,  keyHint: "api.together.ai", defaultModels: ["meta-llama/Llama-3.3-70B-Instruct-Turbo", "deepseek-ai/DeepSeek-R1", "Qwen/Qwen2.5-72B-Instruct-Turbo"] },
  fireworks:  { label: "Fireworks AI",            baseUrl: "https://api.fireworks.ai/inference/v1",             free: false, embeddings: true,  keyHint: "fireworks.ai", defaultModels: ["accounts/fireworks/models/llama-v3p3-70b-instruct", "accounts/fireworks/models/deepseek-r1"] },
  xai:        { label: "xAI (Grok)",              baseUrl: "https://api.x.ai/v1",                               free: false, embeddings: false, keyHint: "console.x.ai", defaultModels: ["grok-2-latest", "grok-2-vision-latest", "grok-beta"] },
  ollama:     { label: "Ollama (local)",          baseUrl: "http://localhost:11434/v1",                         free: true,  embeddings: true,  keyHint: "no key — runs locally", defaultModels: ["llama3.3:latest", "llama3.1:latest", "qwen2.5:latest", "mistral:latest", "deepseek-r1:8b", "nomic-embed-text"] },
  custom:     { label: "Custom (OpenAI-compatible)", baseUrl: "",                                               free: false, embeddings: false, defaultModels: ["gpt-4o-mini", "llama-3.3-70b"] },
};
