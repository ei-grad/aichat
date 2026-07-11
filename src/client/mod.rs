mod access_token;
mod common;
mod message;
#[macro_use]
mod macros;
mod model;
mod stream;

pub use crate::function::ToolCall;
pub use common::*;
pub use message::*;
pub use model::*;
pub use stream::*;

register_client!(
    (openai, "openai", OpenAIConfig, OpenAIClient),
    (
        openai_compatible,
        "openai-compatible",
        OpenAICompatibleConfig,
        OpenAICompatibleClient
    ),
    (gemini, "gemini", GeminiConfig, GeminiClient),
    (claude, "claude", ClaudeConfig, ClaudeClient),
    (cohere, "cohere", CohereConfig, CohereClient),
    (
        azure_openai,
        "azure-openai",
        AzureOpenAIConfig,
        AzureOpenAIClient
    ),
    (vertexai, "vertexai", VertexAIConfig, VertexAIClient),
    (bedrock, "bedrock", BedrockConfig, BedrockClient),
);

pub const OPENAI_COMPATIBLE_PROVIDERS: [(&str, &str); 19] = [
    ("ai21", "https://api.ai21.com/studio/v1"),
    (
        "cloudflare",
        "https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1",
    ),
    ("deepinfra", "https://api.deepinfra.com/v1/openai"),
    ("deepseek", "https://api.deepseek.com"),
    ("ernie", "https://qianfan.baidubce.com/v2"),
    ("github", "https://models.inference.ai.azure.com"),
    ("groq", "https://api.groq.com/openai/v1"),
    ("hunyuan", "https://api.hunyuan.cloud.tencent.com/v1"),
    ("minimax", "https://api.minimax.io/v1"),
    ("mistral", "https://api.mistral.ai/v1"),
    ("moonshot", "https://api.moonshot.cn/v1"),
    ("moonshot_intl", "https://api.moonshot.ai/v1"),
    ("openrouter", "https://openrouter.ai/api/v1"),
    ("perplexity", "https://api.perplexity.ai"),
    (
        "qianwen",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    ("xai", "https://api.x.ai/v1"),
    ("zhipuai", "https://open.bigmodel.cn/api/paas/v4"),
    // RAG-dedicated
    ("jina", "https://api.jina.ai/v1"),
    ("voyageai", "https://api.voyageai.com/v1"),
];

#[cfg(test)]
mod catalog_tests {
    use super::*;
    use serde_json::{json, Value};

    fn catalog() -> Vec<ProviderModels> {
        serde_yaml::from_str(include_str!("../../models.yaml")).expect("invalid models.yaml")
    }

    fn provider<'a>(catalog: &'a [ProviderModels], name: &str) -> &'a ProviderModels {
        catalog
            .iter()
            .find(|provider| provider.provider == name)
            .unwrap_or_else(|| panic!("missing provider {name}"))
    }

    fn model<'a>(provider: &'a ProviderModels, name: &str) -> &'a ModelData {
        provider
            .models
            .iter()
            .find(|model| model.name == name)
            .unwrap_or_else(|| panic!("missing model {}:{name}", provider.provider))
    }

    fn patched_body(model: &ModelData, body: Value) -> Value {
        let mut request = RequestData::new("https://example.invalid", body);
        request.apply_patch(model.patch.clone().expect("missing request patch"));
        request.body
    }

    #[test]
    fn regional_provider_endpoints_are_explicit() {
        assert!(OPENAI_COMPATIBLE_PROVIDERS.contains(&("moonshot", "https://api.moonshot.cn/v1")));
        assert!(
            OPENAI_COMPATIBLE_PROVIDERS.contains(&("moonshot_intl", "https://api.moonshot.ai/v1"))
        );
        assert!(OPENAI_COMPATIBLE_PROVIDERS.contains(&("minimax", "https://api.minimax.io/v1")));
    }

    #[test]
    fn refreshed_catalog_tracks_gemini_and_minimax_lifecycle() {
        let catalog = catalog();
        let gemini = provider(&catalog, "gemini");
        for name in [
            "gemini-3-pro-preview",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "text-embedding-004",
        ] {
            assert!(!gemini.models.iter().any(|model| model.name == name));
        }

        let minimax = provider(&catalog, "minimax");
        assert_eq!(
            model(minimax, "MiniMax-M2.7").max_input_tokens,
            Some(204800)
        );
    }

    #[test]
    fn claude_opus_4_7_uses_adaptive_thinking_without_sampling() {
        let catalog = catalog();
        for (provider_name, base_name, thinking_name, thinking_pointer) in [
            (
                "claude",
                "claude-opus-4-7",
                "claude-opus-4-7:thinking",
                "/body/thinking/type",
            ),
            (
                "vertexai",
                "claude-opus-4-7",
                "claude-opus-4-7:thinking",
                "/body/thinking/type",
            ),
            (
                "bedrock",
                "us.anthropic.claude-opus-4-7",
                "us.anthropic.claude-opus-4-7:thinking",
                "/body/additionalModelRequestFields/thinking/type",
            ),
        ] {
            let provider = provider(&catalog, provider_name);
            let base = model(provider, base_name);
            assert_eq!(base.max_input_tokens, Some(1_000_000));
            assert_eq!(base.max_output_tokens, Some(128_000));
            let base_patch = base.patch.as_ref().expect("missing base request patch");
            let sampling_prefix = if provider_name == "bedrock" {
                "/body/inferenceConfig"
            } else {
                "/body"
            };
            assert!(base_patch
                .pointer(&format!("{sampling_prefix}/temperature"))
                .is_some_and(serde_json::Value::is_null));
            let top_p = if provider_name == "bedrock" {
                "topP"
            } else {
                "top_p"
            };
            assert!(base_patch
                .pointer(&format!("{sampling_prefix}/{top_p}"))
                .is_some_and(serde_json::Value::is_null));
            let request_body = if provider_name == "bedrock" {
                json!({"inferenceConfig": {"temperature": 0.2, "topP": 0.9}})
            } else {
                json!({"temperature": 0.2, "top_p": 0.9})
            };
            let body = patched_body(base, request_body.clone());
            let body_sampling_prefix = sampling_prefix.strip_prefix("/body").unwrap();
            assert!(body
                .pointer(&format!("{body_sampling_prefix}/temperature"))
                .is_none());
            assert!(body
                .pointer(&format!("{body_sampling_prefix}/{top_p}"))
                .is_none());

            let thinking = model(provider, thinking_name);
            let patch = thinking.patch.as_ref().expect("missing thinking patch");
            assert_eq!(
                patch
                    .pointer(thinking_pointer)
                    .and_then(|value| value.as_str()),
                Some("adaptive")
            );
            assert!(patch.pointer("/body/thinking/budget_tokens").is_none());
            assert!(patch
                .pointer("/body/additionalModelRequestFields/thinking/budget_tokens")
                .is_none());
            let body = patched_body(thinking, request_body);
            assert_eq!(
                body.pointer(thinking_pointer.strip_prefix("/body").unwrap())
                    .and_then(|value| value.as_str()),
                Some("adaptive")
            );
        }

        let openrouter = provider(&catalog, "openrouter");
        let thinking = model(openrouter, "anthropic/claude-opus-4.7:thinking");
        assert_eq!(thinking.max_input_tokens, Some(1_000_000));
        assert_eq!(thinking.max_output_tokens, Some(128_000));
        let patch = thinking.patch.as_ref().expect("missing thinking patch");
        assert_eq!(
            patch
                .pointer("/body/reasoning/enabled")
                .and_then(|value| value.as_bool()),
            Some(true)
        );
        assert!(patch.pointer("/body/reasoning/max_tokens").is_none());
        assert!(patch
            .pointer("/body/temperature")
            .is_some_and(serde_json::Value::is_null));
        assert!(patch
            .pointer("/body/top_p")
            .is_some_and(serde_json::Value::is_null));
        let body = patched_body(thinking, json!({"temperature": 0.2, "top_p": 0.9}));
        assert!(body.pointer("/temperature").is_none());
        assert!(body.pointer("/top_p").is_none());
        assert_eq!(
            body.pointer("/reasoning/enabled")
                .and_then(|value| value.as_bool()),
            Some(true)
        );
    }
}
