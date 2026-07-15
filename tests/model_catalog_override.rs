use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderModels {
    provider: String,
    api_base: Option<String>,
    wire_format: Option<String>,
    models: Vec<ModelRecord>,
}

#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ModelRecord {
    name: String,
    #[serde(rename = "type")]
    model_type: Option<String>,
    real_name: Option<String>,
    max_input_tokens: Option<usize>,
    input_price: Option<f64>,
    output_price: Option<f64>,
    response_pricing: Option<ResponsePricingRecord>,
    patch: Option<Value>,
    reasoning_efforts: Option<Vec<String>>,
    max_output_tokens: Option<isize>,
    require_max_tokens: Option<bool>,
    supports_vision: Option<bool>,
    supports_function_calling: Option<bool>,
    no_stream: Option<bool>,
    no_system_message: Option<bool>,
    system_prompt_prefix: Option<String>,
    max_tokens_per_chunk: Option<usize>,
    default_chunk_size: Option<usize>,
    max_batch_size: Option<usize>,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ResponsePricingRecord {
    cached_input_price: f64,
    cache_write_input_price: f64,
    web_search_call_price: Option<f64>,
    long_context_threshold: u64,
    long_context_input_multiplier: f64,
    long_context_output_multiplier: f64,
    service_tier_multipliers: BTreeMap<String, f64>,
}

fn catalog() -> Vec<ProviderModels> {
    serde_yaml::from_str(include_str!("../models.yaml")).expect("models.yaml must match schema")
}

fn provider<'a>(catalog: &'a [ProviderModels], name: &str) -> &'a ProviderModels {
    catalog
        .iter()
        .find(|provider| provider.provider == name)
        .unwrap_or_else(|| panic!("missing provider {name}"))
}

fn model<'a>(provider: &'a ProviderModels, name: &str) -> &'a ModelRecord {
    provider
        .models
        .iter()
        .find(|model| model.name == name)
        .unwrap_or_else(|| panic!("missing model {}:{name}", provider.provider))
}

fn chat(name: &str) -> ModelRecord {
    ModelRecord {
        name: name.to_string(),
        model_type: Some("chat".to_string()),
        ..Default::default()
    }
}

fn names(provider: &ProviderModels) -> Vec<&str> {
    provider
        .models
        .iter()
        .map(|model| model.name.as_str())
        .collect()
}

#[test]
fn openai_gpt_5_6_models_include_response_pricing() {
    let catalog = catalog();
    let openai = provider(&catalog, "openai");

    for (name, cached_price, write_price) in [
        ("gpt-5.6", 0.5, 6.25),
        ("gpt-5.6-sol", 0.5, 6.25),
        ("gpt-5.6-terra", 0.25, 3.125),
        ("gpt-5.6-luna", 0.1, 1.25),
    ] {
        let pricing = model(openai, name)
            .response_pricing
            .as_ref()
            .unwrap_or_else(|| panic!("missing response pricing for openai:{name}"));

        assert_eq!(pricing.cached_input_price, cached_price);
        assert_eq!(pricing.cache_write_input_price, write_price);
        assert_eq!(pricing.web_search_call_price, Some(0.01));
        assert_eq!(pricing.long_context_threshold, 272_000);
        assert_eq!(pricing.long_context_input_multiplier, 2.0);
        assert_eq!(pricing.long_context_output_multiplier, 1.5);
        assert_eq!(
            pricing.service_tier_multipliers,
            BTreeMap::from([
                ("default".to_string(), 1.0),
                ("flex".to_string(), 0.5),
                ("priority".to_string(), 2.0),
            ])
        );
    }
}

fn sampling_patch() -> Value {
    json!({"body": {"temperature": null, "top_p": null}})
}

fn adaptive_patch() -> Value {
    json!({
        "body": {
            "temperature": null,
            "top_p": null,
            "thinking": {"type": "adaptive"}
        }
    })
}

#[test]
fn accepted_overlay_records_match_verified_allowlist_exactly() {
    let catalog = catalog();

    assert_eq!(
        model(provider(&catalog, "gemini"), "gemini-3-flash-preview"),
        &ModelRecord {
            max_input_tokens: Some(1_048_576),
            max_output_tokens: Some(65_536),
            supports_vision: Some(true),
            reasoning_efforts: Some(
                ["minimal", "low", "medium", "high"]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
            ),
            ..chat("gemini-3-flash-preview")
        }
    );

    let claude = provider(&catalog, "claude");
    assert_eq!(
        model(claude, "claude-opus-4-8"),
        &ModelRecord {
            max_input_tokens: Some(1_000_000),
            input_price: Some(5.0),
            output_price: Some(25.0),
            patch: Some(sampling_patch()),
            reasoning_efforts: Some(
                ["low", "medium", "high", "xhigh", "max"]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
            ),
            max_output_tokens: Some(128_000),
            require_max_tokens: Some(true),
            supports_vision: Some(true),
            supports_function_calling: Some(true),
            ..chat("claude-opus-4-8")
        }
    );
    assert_eq!(
        model(claude, "claude-opus-4-8:thinking"),
        &ModelRecord {
            real_name: Some("claude-opus-4-8".to_string()),
            max_input_tokens: Some(1_000_000),
            input_price: Some(5.0),
            output_price: Some(25.0),
            patch: Some(adaptive_patch()),
            max_output_tokens: Some(128_000),
            require_max_tokens: Some(true),
            supports_vision: Some(true),
            ..chat("claude-opus-4-8:thinking")
        }
    );
    assert_eq!(
        model(claude, "claude-sonnet-5"),
        &ModelRecord {
            max_input_tokens: Some(1_000_000),
            input_price: Some(2.0),
            output_price: Some(10.0),
            patch: Some(sampling_patch()),
            max_output_tokens: Some(128_000),
            require_max_tokens: Some(true),
            supports_vision: Some(true),
            reasoning_efforts: Some(
                ["low", "medium", "high", "xhigh", "max"]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
            ),
            ..chat("claude-sonnet-5")
        }
    );
    assert_eq!(
        model(claude, "claude-fable-5"),
        &ModelRecord {
            max_input_tokens: Some(1_000_000),
            input_price: Some(10.0),
            output_price: Some(50.0),
            patch: Some(sampling_patch()),
            max_output_tokens: Some(128_000),
            require_max_tokens: Some(true),
            supports_vision: Some(true),
            reasoning_efforts: Some(
                ["low", "medium", "high", "xhigh", "max"]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
            ),
            ..chat("claude-fable-5")
        }
    );

    let vertexai = provider(&catalog, "vertexai");
    assert_eq!(
        model(vertexai, "gemini-3-flash-preview"),
        &ModelRecord {
            max_input_tokens: Some(1_048_576),
            input_price: Some(0.5),
            output_price: Some(3.0),
            max_output_tokens: Some(65_536),
            supports_vision: Some(true),
            reasoning_efforts: Some(
                ["minimal", "low", "medium", "high"]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
            ),
            ..chat("gemini-3-flash-preview")
        }
    );
    assert_eq!(
        model(vertexai, "claude-opus-4-8"),
        &ModelRecord {
            max_input_tokens: Some(1_000_000),
            patch: Some(sampling_patch()),
            reasoning_efforts: Some(
                ["low", "medium", "high", "xhigh", "max"]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
            ),
            max_output_tokens: Some(128_000),
            require_max_tokens: Some(true),
            supports_vision: Some(true),
            supports_function_calling: Some(true),
            ..chat("claude-opus-4-8")
        }
    );
    assert_eq!(
        model(vertexai, "claude-opus-4-8:thinking"),
        &ModelRecord {
            real_name: Some("claude-opus-4-8".to_string()),
            max_input_tokens: Some(1_000_000),
            patch: Some(adaptive_patch()),
            max_output_tokens: Some(128_000),
            require_max_tokens: Some(true),
            supports_vision: Some(true),
            ..chat("claude-opus-4-8:thinking")
        }
    );
    assert_eq!(
        model(vertexai, "claude-sonnet-5"),
        &ModelRecord {
            max_input_tokens: Some(1_000_000),
            patch: Some(sampling_patch()),
            reasoning_efforts: Some(
                ["low", "medium", "high", "xhigh", "max"]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
            ),
            max_output_tokens: Some(128_000),
            require_max_tokens: Some(true),
            supports_vision: Some(true),
            ..chat("claude-sonnet-5")
        }
    );
    assert_eq!(
        model(vertexai, "claude-fable-5"),
        &ModelRecord {
            max_input_tokens: Some(1_000_000),
            patch: Some(sampling_patch()),
            max_output_tokens: Some(128_000),
            require_max_tokens: Some(true),
            supports_vision: Some(true),
            reasoning_efforts: Some(
                ["low", "medium", "high", "xhigh", "max"]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
            ),
            ..chat("claude-fable-5")
        }
    );

    for provider in [claude, vertexai] {
        let fable = model(provider, "claude-fable-5");
        assert_eq!(fable.supports_function_calling, None);
        assert_eq!(fable.real_name, None);
        assert!(provider
            .models
            .iter()
            .all(|model| model.name != "claude-fable-5:thinking"));
    }
}

#[test]
fn rejected_override_additions_remain_absent() {
    let catalog = catalog();

    for (provider_name, model_name) in [
        ("gemini", "gemini-3-pro-preview"),
        ("gemini", "gemini-2.0-flash"),
        ("gemini", "gemini-2.0-flash-lite"),
        ("gemini", "text-embedding-004"),
        ("vertexai", "gemini-3-pro-preview"),
        ("vertexai", "gemini-2.0-flash-001"),
        ("vertexai", "gemini-2.0-flash-lite-001"),
        ("moonshot", "kimi-k2-turbo-preview"),
        ("moonshot", "kimi-k2-0905-preview"),
        ("moonshot", "kimi-k2-thinking-turbo"),
        ("moonshot", "kimi-k2-thinking"),
        ("minimax", "minimax-m2.5"),
        ("minimax", "minimax-m2.5-highspeed"),
        ("minimax", "minimax-m2.1"),
        ("minimax", "minimax-m2.1-highspeed"),
    ] {
        assert!(
            provider(&catalog, provider_name)
                .models
                .iter()
                .all(|model| model.name != model_name),
            "rejected override entry present: {provider_name}:{model_name}"
        );
    }
}

#[test]
fn overlay_is_append_only_and_preserves_reviewed_catalog_entries() {
    let catalog = catalog();
    let provider_order: Vec<_> = catalog
        .iter()
        .map(|provider| provider.provider.as_str())
        .collect();
    assert_eq!(
        provider_order,
        [
            "openai",
            "gemini",
            "claude",
            "mistral",
            "ai21",
            "cohere",
            "xai",
            "perplexity",
            "groq",
            "vertexai",
            "bedrock",
            "cloudflare",
            "ernie",
            "qianwen",
            "hunyuan",
            "moonshot_intl",
            "moonshot",
            "deepseek",
            "zhipuai",
            "minimax",
            "openrouter",
            "github",
            "deepinfra",
            "jina",
            "voyageai",
        ]
    );

    assert_eq!(
        names(provider(&catalog, "gemini")),
        [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash-lite",
            "gemini-3.1-pro-preview",
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
            "gemma-3-27b-it",
            "gemini-3-flash-preview",
        ]
    );
    assert_eq!(
        names(provider(&catalog, "claude")),
        [
            "claude-opus-4-7",
            "claude-opus-4-7:thinking",
            "claude-opus-4-6",
            "claude-opus-4-6:thinking",
            "claude-sonnet-4-6",
            "claude-sonnet-4-6:thinking",
            "claude-opus-4-5-20251101",
            "claude-opus-4-5-20251101:thinking",
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4-5-20250929:thinking",
            "claude-haiku-4-5-20251001",
            "claude-haiku-4-5-20251001:thinking",
            "claude-opus-4-8",
            "claude-opus-4-8:thinking",
            "claude-sonnet-5",
            "claude-fable-5",
        ]
    );
    assert_eq!(
        names(provider(&catalog, "vertexai")),
        [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash-lite",
            "gemini-3.1-pro-preview",
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
            "claude-opus-4-7",
            "claude-opus-4-7:thinking",
            "claude-opus-4-6",
            "claude-opus-4-6:thinking",
            "claude-sonnet-4-6",
            "claude-sonnet-4-6:thinking",
            "claude-opus-4-5@20251101",
            "claude-opus-4-5@20251101:thinking",
            "claude-sonnet-4-5@20250929",
            "claude-sonnet-4-5@20250929:thinking",
            "claude-haiku-4-5@20251001",
            "claude-haiku-4-5@20251001:thinking",
            "text-embedding-005",
            "text-multilingual-embedding-002",
            "gemini-3-flash-preview",
            "claude-opus-4-8",
            "claude-opus-4-8:thinking",
            "claude-sonnet-5",
            "claude-fable-5",
        ]
    );

    for provider_name in ["gemini", "vertexai"] {
        let gemini = model(provider(&catalog, provider_name), "gemini-3.1-pro-preview");
        assert_eq!(gemini.max_input_tokens, Some(1_048_576));
        assert_eq!(gemini.max_output_tokens, Some(65_536));
        assert_eq!(gemini.supports_vision, Some(true));
        assert_eq!(gemini.supports_function_calling, Some(true));
    }

    for provider_name in ["claude", "vertexai"] {
        let provider = provider(&catalog, provider_name);
        assert_eq!(
            model(provider, "claude-opus-4-7").patch,
            Some(sampling_patch())
        );
        assert_eq!(
            model(provider, "claude-opus-4-7:thinking").patch,
            Some(adaptive_patch())
        );
    }

    let deepseek = provider(&catalog, "deepseek");
    for name in ["deepseek-chat", "deepseek-reasoner", "deepseek-v4-flash"] {
        let model = model(deepseek, name);
        assert_eq!(model.max_input_tokens, Some(1_000_000));
        assert_eq!(model.max_output_tokens, Some(384_000));
        assert_eq!(model.input_price, Some(0.14));
        assert_eq!(model.output_price, Some(0.28));
        assert_eq!(model.supports_function_calling, Some(true));
    }
    let deepseek_v4_pro = model(deepseek, "deepseek-v4-pro");
    assert_eq!(deepseek_v4_pro.max_input_tokens, Some(1_000_000));
    assert_eq!(deepseek_v4_pro.max_output_tokens, Some(384_000));
    assert_eq!(deepseek_v4_pro.input_price, Some(0.435));
    assert_eq!(deepseek_v4_pro.output_price, Some(0.87));
    assert_eq!(deepseek_v4_pro.supports_function_calling, Some(true));

    let minimax = provider(&catalog, "minimax");
    for (name, input_price, output_price) in [
        ("MiniMax-M2.7", 0.3, 1.2),
        ("MiniMax-M2.7-highspeed", 0.6, 2.4),
    ] {
        let model = model(minimax, name);
        assert_eq!(model.max_input_tokens, Some(204_800));
        assert_eq!(model.input_price, Some(input_price));
        assert_eq!(model.output_price, Some(output_price));
        assert_eq!(model.supports_function_calling, Some(true));
    }

    let moonshot_intl = provider(&catalog, "moonshot_intl");
    let moonshot = provider(&catalog, "moonshot");
    assert_eq!(names(moonshot_intl).first(), Some(&"kimi-k2.7-code"));
    assert_eq!(names(moonshot).first(), Some(&"kimi-k2.6"));
    assert_eq!(
        model(moonshot_intl, "kimi-k2.5").patch,
        Some(sampling_patch())
    );
    assert_eq!(model(moonshot, "kimi-k2.5").patch, Some(sampling_patch()));
}

#[test]
fn catalog_schema_is_typed_and_only_preexisting_collision_remains() {
    let catalog = catalog();
    let mut counts = BTreeMap::new();

    for provider in &catalog {
        assert!(!provider.provider.is_empty());
        assert!(!provider.models.is_empty());
        if let Some(api_base) = &provider.api_base {
            assert!(
                api_base.starts_with("https://"),
                "non-https api_base for {}",
                provider.provider
            );
        }
        assert!(matches!(
            provider.wire_format.as_deref(),
            None | Some("openai" | "claude" | "gemini" | "cohere")
        ));
        for model in &provider.models {
            assert!(!model.name.is_empty());
            assert!(matches!(
                model.model_type.as_deref(),
                None | Some("chat" | "embedding" | "reranker")
            ));
            *counts
                .entry((provider.provider.as_str(), model.name.as_str()))
                .or_insert(0usize) += 1;
        }
    }

    let collisions: Vec<_> = counts.into_iter().filter(|(_, count)| *count > 1).collect();
    assert_eq!(collisions, vec![(("jina", "jina-colbert-v2"), 2)]);
}
