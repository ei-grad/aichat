//! Explicit client registry: the `ClientConfig` enum users deserialize their
//! `clients:` section into, the per-provider client structs, and the dispatch
//! functions over them. The serde representation (`type: openai` etc.) is a
//! public contract — changing tags or variant shapes breaks user configs.

pub use super::azure_openai::AzureOpenAIConfig;
pub use super::bedrock::BedrockConfig;
pub use super::claude::ClaudeConfig;
pub use super::cohere::CohereConfig;
pub use super::gemini::GeminiConfig;
pub use super::openai::OpenAIConfig;
pub use super::openai_compatible::OpenAICompatibleConfig;
pub use super::vertexai::VertexAIConfig;
use super::{
    create_config, create_openai_compatible_client_config, optional_config_field,
    resolve_config_field, Client, Model, ModelData, ModelType, ALL_PROVIDER_MODELS,
};
use crate::config::{Config, GlobalConfig};

use anyhow::{anyhow, bail, Result};
use serde_json::Value;
use std::collections::HashSet;

const CANONICAL_OPENAI_API_BASE: &str = "https://api.openai.com/v1";

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "type")]
pub enum ClientConfig {
    #[serde(rename = "openai")]
    OpenAIConfig(OpenAIConfig),
    #[serde(rename = "openai-compatible")]
    OpenAICompatibleConfig(OpenAICompatibleConfig),
    #[serde(rename = "gemini")]
    GeminiConfig(GeminiConfig),
    #[serde(rename = "claude")]
    ClaudeConfig(ClaudeConfig),
    #[serde(rename = "cohere")]
    CohereConfig(CohereConfig),
    #[serde(rename = "azure-openai")]
    AzureOpenAIConfig(AzureOpenAIConfig),
    #[serde(rename = "vertexai")]
    VertexAIConfig(VertexAIConfig),
    #[serde(rename = "bedrock")]
    BedrockConfig(BedrockConfig),
    #[serde(other)]
    Unknown,
}

macro_rules! define_client_struct {
    ($client:ident, $config:ident, $name:literal) => {
        #[derive(Debug)]
        pub struct $client {
            pub(super) global_config: GlobalConfig,
            pub(super) config: $config,
            pub(super) model: Model,
        }

        impl $client {
            pub const NAME: &'static str = $name;

            pub fn name(config: &$config) -> &str {
                config.name.as_deref().unwrap_or(Self::NAME)
            }
        }
    };
}

define_client_struct!(OpenAIClient, OpenAIConfig, "openai");
define_client_struct!(
    OpenAICompatibleClient,
    OpenAICompatibleConfig,
    "openai-compatible"
);
define_client_struct!(GeminiClient, GeminiConfig, "gemini");
define_client_struct!(ClaudeClient, ClaudeConfig, "claude");
define_client_struct!(CohereClient, CohereConfig, "cohere");
define_client_struct!(AzureOpenAIClient, AzureOpenAIConfig, "azure-openai");
define_client_struct!(VertexAIClient, VertexAIConfig, "vertexai");
define_client_struct!(BedrockClient, BedrockConfig, "bedrock");

/// Effective client name of a config entry (explicit `name:` or the type's
/// default), or `None` for unrecognized entries.
fn client_config_name(client_config: &ClientConfig) -> Option<&str> {
    match client_config {
        ClientConfig::OpenAIConfig(c) => Some(OpenAIClient::name(c)),
        ClientConfig::OpenAICompatibleConfig(c) => Some(OpenAICompatibleClient::name(c)),
        ClientConfig::GeminiConfig(c) => Some(GeminiClient::name(c)),
        ClientConfig::ClaudeConfig(c) => Some(ClaudeClient::name(c)),
        ClientConfig::CohereConfig(c) => Some(CohereClient::name(c)),
        ClientConfig::AzureOpenAIConfig(c) => Some(AzureOpenAIClient::name(c)),
        ClientConfig::VertexAIConfig(c) => Some(VertexAIClient::name(c)),
        ClientConfig::BedrockConfig(c) => Some(BedrockClient::name(c)),
        ClientConfig::Unknown => None,
    }
}

fn is_canonical_openai_api_base(api_base: &str) -> bool {
    api_base.trim_end_matches('/') == CANONICAL_OPENAI_API_BASE
}

fn uses_public_openai_catalog(config: &OpenAIConfig) -> bool {
    match optional_config_field(resolve_config_field(
        OpenAIClient::name(config),
        "api_base",
        config.api_base.as_deref(),
        &[],
    )) {
        Ok(None) => true,
        Ok(Some(api_base)) => is_canonical_openai_api_base(&api_base),
        Err(_) => false,
    }
}

pub(crate) fn client_allows_raw_reasoning_suffix(config: &Config, client_name: &str) -> bool {
    config.clients.iter().any(|client_config| {
        matches!(
            client_config,
            ClientConfig::OpenAICompatibleConfig(client)
                if OpenAICompatibleClient::name(client) == client_name
        )
    })
}

pub fn init_client(config: &GlobalConfig, model: Option<Model>) -> Result<Box<dyn Client>> {
    let model = model.unwrap_or_else(|| config.read().model.clone());
    let global_config = config.clone();
    let client = config.read().clients.iter().find_map(|client_config| {
        if client_config_name(client_config) != Some(model.client_name()) {
            return None;
        }
        let global_config = global_config.clone();
        let model = model.clone();
        let client: Box<dyn Client> = match client_config {
            ClientConfig::OpenAIConfig(c) => Box::new(OpenAIClient {
                global_config,
                config: c.clone(),
                model,
            }),
            ClientConfig::OpenAICompatibleConfig(c) => Box::new(OpenAICompatibleClient {
                global_config,
                config: c.clone(),
                model,
            }),
            ClientConfig::GeminiConfig(c) => Box::new(GeminiClient {
                global_config,
                config: c.clone(),
                model,
            }),
            ClientConfig::ClaudeConfig(c) => Box::new(ClaudeClient {
                global_config,
                config: c.clone(),
                model,
            }),
            ClientConfig::CohereConfig(c) => Box::new(CohereClient {
                global_config,
                config: c.clone(),
                model,
            }),
            ClientConfig::AzureOpenAIConfig(c) => Box::new(AzureOpenAIClient {
                global_config,
                config: c.clone(),
                model,
            }),
            ClientConfig::VertexAIConfig(c) => Box::new(VertexAIClient {
                global_config,
                config: c.clone(),
                model,
            }),
            ClientConfig::BedrockConfig(c) => Box::new(BedrockClient {
                global_config,
                config: c.clone(),
                model,
            }),
            ClientConfig::Unknown => return None,
        };
        Some(client)
    });
    client.ok_or_else(|| anyhow!("Invalid model '{}'", model.id()))
}

pub(super) fn init_openai_client(config: &GlobalConfig, model: &Model) -> Result<OpenAIClient> {
    let client_config = config
        .read()
        .clients
        .iter()
        .find(|client_config| client_config_name(client_config) == Some(model.client_name()))
        .cloned();

    match client_config {
        Some(ClientConfig::OpenAIConfig(client_config)) => Ok(OpenAIClient {
            global_config: config.clone(),
            config: client_config,
            model: model.clone(),
        }),
        Some(_) => bail!(
            "OpenAI Responses multi-agent requires a native OpenAI client; model '{}' uses client '{}'",
            model.id(),
            model.client_name()
        ),
        None => bail!("Invalid model '{}'", model.id()),
    }
}

pub fn list_client_types() -> Vec<&'static str> {
    let mut client_types = vec![
        OpenAIClient::NAME,
        OpenAICompatibleClient::NAME,
        GeminiClient::NAME,
        ClaudeClient::NAME,
        CohereClient::NAME,
        AzureOpenAIClient::NAME,
        VertexAIClient::NAME,
        BedrockClient::NAME,
    ];
    client_types.extend(
        ALL_PROVIDER_MODELS
            .iter()
            .filter(|v| v.api_base.is_some())
            .map(|v| v.provider.as_str()),
    );
    client_types
}

pub async fn create_client_config(client: &str) -> Result<(String, Value)> {
    let prompts: Option<&[super::PromptAction<'static>]> = match client {
        OpenAIClient::NAME => Some(&OpenAIClient::PROMPTS),
        GeminiClient::NAME => Some(&GeminiClient::PROMPTS),
        ClaudeClient::NAME => Some(&ClaudeClient::PROMPTS),
        CohereClient::NAME => Some(&CohereClient::PROMPTS),
        AzureOpenAIClient::NAME => Some(&AzureOpenAIClient::PROMPTS),
        VertexAIClient::NAME => Some(&VertexAIClient::PROMPTS),
        BedrockClient::NAME => Some(&BedrockClient::PROMPTS),
        _ => None,
    };
    if let Some(prompts) = prompts {
        return create_config(prompts, client).await;
    }
    if let Some(ret) = create_openai_compatible_client_config(client).await? {
        return Ok(ret);
    }
    bail!("Unknown client '{}'", client)
}

pub fn list_client_names(config: &Config) -> Vec<&String> {
    config
        .client_names_cache
        .get_or_init(|| {
            config
                .clients
                .iter()
                .filter_map(|v| client_config_name(v).map(str::to_string))
                .collect()
        })
        .iter()
        .collect()
}

pub fn list_all_models(config: &Config) -> Vec<&Model> {
    config
        .all_models_cache
        .get_or_init(|| {
            config
                .clients
                .iter()
                .flat_map(|client_config| match client_config {
                    ClientConfig::OpenAIConfig(c) => provider_models(
                        OpenAIClient::name(c),
                        OpenAIClient::NAME,
                        c.name.as_deref(),
                        &c.models,
                        uses_public_openai_catalog(c),
                    ),
                    ClientConfig::OpenAICompatibleConfig(c) => provider_models(
                        OpenAICompatibleClient::name(c),
                        OpenAICompatibleClient::NAME,
                        c.name.as_deref(),
                        &c.models,
                        true,
                    ),
                    ClientConfig::GeminiConfig(c) => provider_models(
                        GeminiClient::name(c),
                        GeminiClient::NAME,
                        c.name.as_deref(),
                        &c.models,
                        true,
                    ),
                    ClientConfig::ClaudeConfig(c) => provider_models(
                        ClaudeClient::name(c),
                        ClaudeClient::NAME,
                        c.name.as_deref(),
                        &c.models,
                        true,
                    ),
                    ClientConfig::CohereConfig(c) => provider_models(
                        CohereClient::name(c),
                        CohereClient::NAME,
                        c.name.as_deref(),
                        &c.models,
                        true,
                    ),
                    ClientConfig::AzureOpenAIConfig(c) => provider_models(
                        AzureOpenAIClient::name(c),
                        AzureOpenAIClient::NAME,
                        c.name.as_deref(),
                        &c.models,
                        true,
                    ),
                    ClientConfig::VertexAIConfig(c) => provider_models(
                        VertexAIClient::name(c),
                        VertexAIClient::NAME,
                        c.name.as_deref(),
                        &c.models,
                        true,
                    ),
                    ClientConfig::BedrockConfig(c) => provider_models(
                        BedrockClient::name(c),
                        BedrockClient::NAME,
                        c.name.as_deref(),
                        &c.models,
                        true,
                    ),
                    ClientConfig::Unknown => vec![],
                })
                .collect()
        })
        .iter()
        .collect()
}

pub fn list_models(config: &Config, model_type: ModelType) -> Vec<&Model> {
    list_all_models(config)
        .into_iter()
        .filter(|v| v.model_type() == model_type)
        .collect()
}

/// Models for one configured client. Native OpenAI entries overlay the bundled
/// catalog only at the canonical public endpoint. Local overrides win without
/// hiding newly cataloged models or reasoning variants there; custom endpoints
/// and other clients keep replacement semantics for explicit model lists.
fn provider_models(
    client_name: &str,
    client_kind: &str,
    custom_name: Option<&str>,
    models: &[ModelData],
    use_public_openai_catalog: bool,
) -> Vec<Model> {
    let provider = use_public_openai_catalog.then(|| {
        ALL_PROVIDER_MODELS.iter().find(|v| {
            v.provider == client_kind
                || (client_kind == OpenAICompatibleClient::NAME
                    && custom_name
                        .map(|name| name.starts_with(&v.provider))
                        .unwrap_or_default())
        })
    });
    let provider = provider.flatten();
    if models.is_empty() {
        return provider
            .map(|provider| Model::from_config(client_name, client_kind, &provider.models))
            .unwrap_or_default();
    }

    if client_kind != OpenAIClient::NAME {
        return Model::from_config(client_name, client_kind, models);
    }
    let Some(provider) = provider else {
        return Model::from_config(client_name, client_kind, models);
    };
    let effective_models = models
        .iter()
        .map(|configured| {
            provider
                .models
                .iter()
                .find(|catalog| catalog.name == configured.name)
                .map(|catalog| configured.with_catalog_capabilities(catalog))
                .unwrap_or_else(|| configured.clone())
        })
        .collect::<Vec<_>>();
    let mut configured = Model::from_config(client_name, client_kind, &effective_models);
    let mut configured_ids = configured
        .iter()
        .map(|model| model.id())
        .collect::<HashSet<_>>();
    for model in Model::from_config(client_name, client_kind, &provider.models) {
        if configured_ids.insert(model.id()) {
            configured.push(model);
        }
    }
    configured
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_openai_catalog_exposes_sol_high_without_explicit_models() {
        let config = Config {
            clients: vec![ClientConfig::OpenAIConfig(OpenAIConfig::default())],
            ..Default::default()
        };
        let models = list_models(&config, ModelType::Chat);

        assert!(models
            .iter()
            .any(|model| model.id() == "openai:gpt-5.6-sol:high"));
    }

    #[test]
    fn native_openai_name_only_entry_inherits_catalog_pricing_and_efforts() {
        let configured = ModelData::new("gpt-5.6-sol");
        let models = provider_models(
            "openai",
            OpenAIClient::NAME,
            None,
            std::slice::from_ref(&configured),
            true,
        );
        let high = models
            .iter()
            .find(|model| model.id() == "openai:gpt-5.6-sol:high")
            .unwrap();

        assert_eq!(high.data().input_price, Some(5.0));
        assert_eq!(high.data().output_price, Some(30.0));
        assert_eq!(
            high.data()
                .response_pricing
                .as_ref()
                .and_then(|pricing| pricing.web_search_call_price),
            Some(0.01)
        );
    }

    #[test]
    fn native_openai_explicit_models_overlay_instead_of_hiding_catalog_variants() {
        let mut configured = ModelData::new("gpt-5.6-sol");
        configured.input_price = Some(42.0);
        configured.patch = Some(serde_json::json!({"body": {"custom": true}}));
        let models = provider_models(
            "openai",
            OpenAIClient::NAME,
            None,
            std::slice::from_ref(&configured),
            true,
        );

        assert_eq!(
            models
                .iter()
                .filter(|model| model.id() == "openai:gpt-5.6-sol")
                .count(),
            1
        );
        assert!(models
            .iter()
            .any(|model| model.id() == "openai:gpt-5.6-sol:high"));
        let base = models
            .iter()
            .find(|model| model.id() == "openai:gpt-5.6-sol")
            .unwrap();
        assert_eq!(base.data().input_price, configured.input_price);
        let high = models
            .iter()
            .find(|model| model.id() == "openai:gpt-5.6-sol:high")
            .unwrap();
        assert_eq!(high.data().input_price, configured.input_price);
        assert_eq!(high.data().output_price, Some(30.0));
        assert_eq!(
            high.data()
                .response_pricing
                .as_ref()
                .and_then(|pricing| pricing.web_search_call_price),
            Some(0.01)
        );
        assert_eq!(high.real_name(), "gpt-5.6-sol");
        assert_eq!(high.data().patch.as_ref().unwrap()["body"]["custom"], true);
        assert_eq!(
            high.data().patch.as_ref().unwrap()["body"]["reasoning_effort"],
            "high"
        );
    }

    #[test]
    fn native_openai_old_response_pricing_inherits_web_search_price() {
        let defaults = ALL_PROVIDER_MODELS
            .iter()
            .find(|provider| provider.provider == "openai")
            .and_then(|provider| {
                provider
                    .models
                    .iter()
                    .find(|model| model.name == "gpt-5.6-sol")
            })
            .unwrap();
        let mut configured = ModelData::new("gpt-5.6-sol");
        let mut old_pricing = defaults.response_pricing.clone().unwrap();
        old_pricing.cached_input_price = 99.0;
        old_pricing.web_search_call_price = None;
        configured.response_pricing = Some(old_pricing);

        let merged = configured.with_catalog_defaults(defaults);
        let pricing = merged.response_pricing.unwrap();

        assert_eq!(pricing.cached_input_price, 99.0);
        assert_eq!(pricing.web_search_call_price, Some(0.01));
    }

    #[test]
    fn openai_compatible_explicit_models_keep_replacement_semantics() {
        let configured = ModelData::new("private-model");
        let models = provider_models(
            "openai-proxy",
            OpenAICompatibleClient::NAME,
            Some("openai-proxy"),
            std::slice::from_ref(&configured),
            true,
        );

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id(), "openai-proxy:private-model");
    }

    #[test]
    fn canonical_openai_endpoint_normalizes_trailing_slashes() {
        assert!(is_canonical_openai_api_base("https://api.openai.com/v1///"));
        assert!(!is_canonical_openai_api_base(
            "https://region.example.test/openai/v1"
        ));
    }

    #[test]
    fn custom_native_openai_endpoint_does_not_inherit_public_catalog() {
        let configured = ModelData::new("gpt-5.6-sol");
        let config = Config {
            clients: vec![ClientConfig::OpenAIConfig(OpenAIConfig {
                name: Some("regional-openai-catalog-test".into()),
                api_key: None,
                api_base: Some("https://region.example.test/openai/v1".into()),
                organization_id: None,
                models: vec![configured],
                patch: None,
                extra: None,
            })],
            ..Default::default()
        };

        let models = list_models(&config, ModelType::Chat);

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id(), "regional-openai-catalog-test:gpt-5.6-sol");
        assert_eq!(models[0].data().input_price, None);
        assert_eq!(models[0].data().output_price, None);
        assert_eq!(models[0].data().response_pricing, None);
        assert!(models[0].data().reasoning_efforts.is_empty());
    }

    #[test]
    fn openai_compatible_uncataloged_effort_suffix_remains_literal() {
        let config = Config {
            clients: vec![ClientConfig::OpenAICompatibleConfig(
                OpenAICompatibleConfig {
                    name: Some("raw-reasoning-suffix-test".into()),
                    api_base: Some("https://compatible.example.test/v1".into()),
                    api_key: None,
                    wire_format: None,
                    models: vec![],
                    patch: None,
                    extra: None,
                },
            )],
            ..Default::default()
        };

        let model = Model::retrieve_model(
            &config,
            "raw-reasoning-suffix-test:private-model:high",
            ModelType::Chat,
        )
        .unwrap();

        assert_eq!(model.id(), "raw-reasoning-suffix-test:private-model:high");
        assert_eq!(model.real_name(), "private-model:high");
        assert_eq!(model.reasoning_effort(), None);
    }
}
