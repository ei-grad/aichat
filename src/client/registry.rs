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
    create_config, create_openai_compatible_client_config, Client, Model, ModelData, ModelType,
    ALL_PROVIDER_MODELS,
};
use crate::config::{Config, GlobalConfig};

use anyhow::{anyhow, bail, Result};
use serde_json::Value;

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
                    ),
                    ClientConfig::OpenAICompatibleConfig(c) => provider_models(
                        OpenAICompatibleClient::name(c),
                        OpenAICompatibleClient::NAME,
                        c.name.as_deref(),
                        &c.models,
                    ),
                    ClientConfig::GeminiConfig(c) => provider_models(
                        GeminiClient::name(c),
                        GeminiClient::NAME,
                        c.name.as_deref(),
                        &c.models,
                    ),
                    ClientConfig::ClaudeConfig(c) => provider_models(
                        ClaudeClient::name(c),
                        ClaudeClient::NAME,
                        c.name.as_deref(),
                        &c.models,
                    ),
                    ClientConfig::CohereConfig(c) => provider_models(
                        CohereClient::name(c),
                        CohereClient::NAME,
                        c.name.as_deref(),
                        &c.models,
                    ),
                    ClientConfig::AzureOpenAIConfig(c) => provider_models(
                        AzureOpenAIClient::name(c),
                        AzureOpenAIClient::NAME,
                        c.name.as_deref(),
                        &c.models,
                    ),
                    ClientConfig::VertexAIConfig(c) => provider_models(
                        VertexAIClient::name(c),
                        VertexAIClient::NAME,
                        c.name.as_deref(),
                        &c.models,
                    ),
                    ClientConfig::BedrockConfig(c) => provider_models(
                        BedrockClient::name(c),
                        BedrockClient::NAME,
                        c.name.as_deref(),
                        &c.models,
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

/// Models for one configured client: explicitly configured models win,
/// otherwise the bundled catalog is consulted — by client type, and for
/// `openai-compatible` entries by the custom name's provider prefix.
fn provider_models(
    client_name: &str,
    client_kind: &str,
    custom_name: Option<&str>,
    models: &[ModelData],
) -> Vec<Model> {
    if !models.is_empty() {
        return Model::from_config(client_name, client_kind, models);
    }
    if let Some(provider) = ALL_PROVIDER_MODELS.iter().find(|v| {
        v.provider == client_kind
            || (client_kind == OpenAICompatibleClient::NAME
                && custom_name
                    .map(|name| name.starts_with(&v.provider))
                    .unwrap_or_default())
    }) {
        return Model::from_config(client_name, client_kind, &provider.models);
    }
    vec![]
}
