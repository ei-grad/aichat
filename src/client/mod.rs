#[macro_use]
mod macros;

mod access_token;
mod azure_openai;
mod bedrock;
mod claude;
mod cohere;
mod common;
mod error;
mod gemini;
mod message;
mod model;
mod openai;
mod openai_compatible;
pub mod openai_responses;
mod registry;
mod retry;
mod stream;
mod vertexai;
#[cfg(test)]
mod wire_tests;

pub use crate::function::ToolCall;
pub use common::*;
pub use error::*;
pub use message::*;
pub use model::*;
pub use registry::*;
pub(crate) use retry::*;
pub use stream::*;

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
        let catalog = catalog();
        assert_eq!(
            provider(&catalog, "moonshot").api_base.as_deref(),
            Some("https://api.moonshot.cn/v1")
        );
        assert_eq!(
            provider(&catalog, "moonshot_intl").api_base.as_deref(),
            Some("https://api.moonshot.ai/v1")
        );
        assert_eq!(
            provider(&catalog, "minimax").api_base.as_deref(),
            Some("https://api.minimax.io/v1")
        );
    }

    #[test]
    fn openai_catalog_exposes_current_models_and_reasoning_efforts() {
        let catalog = catalog();
        let openai = provider(&catalog, "openai");
        let expanded = Model::from_config("openai", "openai", &openai.models);

        for (name, input_price, output_price, cached_price, write_price) in [
            ("gpt-5.6", 5.0, 30.0, 0.5, 6.25),
            ("gpt-5.6-sol", 5.0, 30.0, 0.5, 6.25),
            ("gpt-5.6-terra", 2.5, 15.0, 0.25, 3.125),
            ("gpt-5.6-luna", 1.0, 6.0, 0.1, 1.25),
        ] {
            let catalog_model = model(openai, name);
            assert_eq!(catalog_model.max_input_tokens, Some(1_050_000));
            assert_eq!(catalog_model.max_output_tokens, Some(128_000));
            assert_eq!(catalog_model.input_price, Some(input_price));
            assert_eq!(catalog_model.output_price, Some(output_price));
            let response_pricing = catalog_model
                .response_pricing
                .as_ref()
                .expect("missing Responses pricing");
            assert_eq!(response_pricing.cached_input_price, cached_price);
            assert_eq!(response_pricing.cache_write_input_price, write_price);
            assert_eq!(response_pricing.long_context_threshold, 272_000);
            assert_eq!(response_pricing.long_context_input_multiplier, 2.0);
            assert_eq!(response_pricing.long_context_output_multiplier, 1.5);
            assert_eq!(
                response_pricing.service_tier_multipliers.get("default"),
                Some(&1.0)
            );
            assert_eq!(
                response_pricing.service_tier_multipliers.get("flex"),
                Some(&0.5)
            );
            assert_eq!(
                response_pricing.service_tier_multipliers.get("priority"),
                Some(&2.0)
            );
            assert!(catalog_model.supports_vision);
            assert!(catalog_model.supports_function_calling);
            assert_eq!(
                catalog_model.reasoning_efforts,
                ["none", "low", "medium", "high", "xhigh", "max"]
            );

            let listed: Vec<_> = expanded
                .iter()
                .filter(|model| model.real_name() == name)
                .collect();
            assert_eq!(listed.len(), 7, "unexpected variants for openai:{name}");
            assert!(listed
                .iter()
                .all(|model| { model.data().response_pricing == catalog_model.response_pricing }));

            let base = listed
                .iter()
                .find(|model| model.name() == name)
                .unwrap_or_else(|| panic!("missing openai:{name}"));
            assert_eq!(base.real_name(), name);
            assert_eq!(base.reasoning_effort(), None);
            assert!(base.data().patch.is_none());

            for effort in ["none", "low", "medium", "high", "xhigh", "max"] {
                let variant = listed
                    .iter()
                    .find(|model| model.name() == format!("{name}:{effort}"))
                    .unwrap_or_else(|| panic!("missing openai:{name}:{effort}"));
                assert_eq!(variant.real_name(), name);
                assert_eq!(variant.reasoning_effort(), Some(effort));
                let body = patched_body(variant.data(), json!({"temperature": 0.4, "top_p": 0.8}));
                assert_eq!(body["reasoning_effort"], effort);
                assert!(body.get("temperature").is_none());
                assert!(body.get("top_p").is_none());
            }
        }
    }

    #[test]
    fn reasoning_efforts_use_provider_specific_request_shapes() {
        let mut data = ModelData::new("reasoning-model");
        data.reasoning_efforts = vec!["medium".into()];

        for provider_name in ["claude", "vertexai"] {
            let models =
                Model::from_config(provider_name, provider_name, std::slice::from_ref(&data));
            let variant = models
                .iter()
                .find(|model| model.name() == "reasoning-model:medium")
                .expect("missing reasoning effort variant");
            let body = patched_body(variant.data(), json!({"temperature": 0.4, "top_p": 0.8}));
            assert_eq!(body["thinking"]["type"], "adaptive");
            assert_eq!(body["output_config"]["effort"], "medium");
            assert!(body.get("temperature").is_none());
            assert!(body.get("top_p").is_none());
        }

        let models = Model::from_config("bedrock", "bedrock", std::slice::from_ref(&data));
        let variant = models
            .iter()
            .find(|model| model.name() == "reasoning-model:medium")
            .expect("missing Bedrock reasoning effort variant");
        let body = patched_body(
            variant.data(),
            json!({"inferenceConfig": {"temperature": 0.4, "topP": 0.8}}),
        );
        assert_eq!(
            body["additionalModelRequestFields"]["thinking"]["type"],
            "adaptive"
        );
        assert_eq!(
            body["additionalModelRequestFields"]["output_config"]["effort"],
            "medium"
        );
        assert!(body["inferenceConfig"].get("temperature").is_none());
        assert!(body["inferenceConfig"].get("topP").is_none());

        for provider_name in ["gemini", "vertexai"] {
            let mut gemini = data.clone();
            gemini.name = "gemini-reasoning-model".into();
            let models = Model::from_config(provider_name, provider_name, &[gemini]);
            let variant = models
                .iter()
                .find(|model| model.name() == "gemini-reasoning-model:medium")
                .expect("missing Gemini reasoning effort variant");
            let body = patched_body(variant.data(), json!({"generationConfig": {}}));
            assert_eq!(
                body["generationConfig"]["thinkingConfig"]["thinkingLevel"],
                "medium"
            );
        }

        let named = Model::from_config("work-claude", "claude", std::slice::from_ref(&data));
        assert_eq!(named.len(), 2);
        assert_eq!(named[1].client_name(), "work-claude");
        assert_eq!(named[1].name(), "reasoning-model:medium");

        assert_eq!(Model::from_config("moonshot", "moonshot", &[data]).len(), 1);
    }

    #[test]
    fn catalog_effort_variants_preserve_sampling_deletion_patches() {
        let catalog = catalog();
        for provider_name in ["claude", "vertexai"] {
            let provider = provider(&catalog, provider_name);
            let models = Model::from_config(provider_name, provider_name, &provider.models);
            let variant = models
                .iter()
                .find(|model| model.name() == "claude-opus-4-7:medium")
                .expect("missing catalog effort variant");
            let body = patched_body(variant.data(), json!({"temperature": 0.4, "top_p": 0.8}));
            assert!(body.get("temperature").is_none());
            assert!(body.get("top_p").is_none());
            assert_eq!(body["thinking"]["type"], "adaptive");
            assert_eq!(body["output_config"]["effort"], "medium");

            let fable = models
                .iter()
                .find(|model| model.name() == "claude-fable-5:medium")
                .expect("missing Fable effort variant");
            let body = patched_body(fable.data(), json!({}));
            assert!(body.get("thinking").is_none());
            assert_eq!(body["output_config"]["effort"], "medium");
        }

        let bedrock = provider(&catalog, "bedrock");
        let models = Model::from_config("bedrock", "bedrock", &bedrock.models);
        let variant = models
            .iter()
            .find(|model| model.name() == "us.anthropic.claude-opus-4-7:medium")
            .expect("missing Bedrock catalog effort variant");
        let body = patched_body(
            variant.data(),
            json!({"inferenceConfig": {"temperature": 0.4, "topP": 0.8}}),
        );
        assert!(body["inferenceConfig"].get("temperature").is_none());
        assert!(body["inferenceConfig"].get("topP").is_none());
        assert_eq!(
            body["additionalModelRequestFields"]["output_config"]["effort"],
            "medium"
        );
    }

    #[test]
    fn unsupported_catalog_effort_fails_before_provider_request() {
        let catalog = catalog();
        let claude = provider(&catalog, "claude");
        let expanded = Model::from_config("claude", "claude", &claude.models);
        let models: Vec<_> = expanded.iter().collect();

        validate_reasoning_effort(&models, "claude", "claude-fable-5:medium")
            .expect("supported effort must pass");
        let err = validate_reasoning_effort(&models, "claude", "claude-fable-5:none")
            .expect_err("unsupported effort must fail locally");
        assert_eq!(
            err.to_string(),
            "Model 'claude:claude-fable-5' does not support reasoning effort 'none'. Supported efforts: low, medium, high, xhigh, max"
        );
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
