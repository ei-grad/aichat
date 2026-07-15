use super::*;

use crate::{
    config::{Config, GlobalConfig, Input},
    function::{eval_tool_calls, FunctionDeclaration, ToolCall, ToolResult},
    render::render_stream,
    utils::*,
};

use anyhow::{bail, Context, Result};
use fancy_regex::Regex;
use indexmap::IndexMap;
use inquire::{
    list_option::ListOption, required, validator::Validation, MultiSelect, Select, Text,
};
use reqwest::{Client as ReqwestClient, RequestBuilder};
use serde::Deserialize;
use serde_json::{json, Value};
use std::borrow::Cow;
use std::sync::LazyLock;
use std::time::Duration;
use tokio::sync::mpsc::unbounded_channel;

const MODELS_YAML: &str = include_str!("../../models.yaml");

pub static ALL_PROVIDER_MODELS: LazyLock<Vec<ProviderModels>> = LazyLock::new(|| {
    Config::loal_models_override()
        .ok()
        .unwrap_or_else(|| serde_yaml::from_str(MODELS_YAML).unwrap())
});

/// Catalog endpoint of a provider served by the generic `openai-compatible`
/// client, or `None` for unknown/native providers.
pub(crate) fn compat_provider_api_base(client_name: &str) -> Option<&'static str> {
    ALL_PROVIDER_MODELS
        .iter()
        .find(|v| v.provider == client_name)
        .and_then(|v| v.api_base.as_deref())
}

static EMBEDDING_MODEL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"((^|/)(bge-|e5-|uae-|gte-|text-)|embed|multilingual|minilm)").unwrap()
});

static ESCAPE_SLASH_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?<!\\)/").unwrap());

enum ConfigFieldValue<'a> {
    Env(&'a str),
    Literal(Cow<'a, str>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConfigFieldErrorKind {
    Absent,
    InvalidReference,
    UnresolvedReference,
}

#[derive(Debug)]
pub(crate) struct ConfigFieldError {
    field_name: String,
    kind: ConfigFieldErrorKind,
}

impl ConfigFieldError {
    fn new(field_name: &str, kind: ConfigFieldErrorKind) -> Self {
        Self {
            field_name: field_name.to_string(),
            kind,
        }
    }
}

impl std::fmt::Display for ConfigFieldError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            ConfigFieldErrorKind::Absent => write!(formatter, "Miss '{}'", self.field_name),
            ConfigFieldErrorKind::InvalidReference => {
                write!(
                    formatter,
                    "Invalid environment reference for '{}'",
                    self.field_name
                )
            }
            ConfigFieldErrorKind::UnresolvedReference => write!(
                formatter,
                "Environment variable for '{}' is missing or empty",
                self.field_name
            ),
        }
    }
}

impl std::error::Error for ConfigFieldError {}

pub(crate) type ConfigFieldResult<T> = std::result::Result<T, ConfigFieldError>;

fn parse_config_field_value<'a>(
    field_name: &str,
    value: &'a str,
) -> ConfigFieldResult<ConfigFieldValue<'a>> {
    if let Some(value) = value.strip_prefix("$$") {
        return Ok(ConfigFieldValue::Literal(Cow::Owned(format!("${value}"))));
    }
    if !value.starts_with('$') {
        return Ok(ConfigFieldValue::Literal(Cow::Borrowed(value)));
    }

    let env_name = if let Some(value) = value.strip_prefix("${") {
        value.strip_suffix('}')
    } else {
        value.strip_prefix('$')
    };
    let Some(env_name) = env_name.filter(|name| is_valid_env_name(name)) else {
        return Err(ConfigFieldError::new(
            field_name,
            ConfigFieldErrorKind::InvalidReference,
        ));
    };
    Ok(ConfigFieldValue::Env(env_name))
}

fn is_valid_env_name(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some('_' | 'a'..='z' | 'A'..='Z'))
        && chars.all(|ch| matches!(ch, '_' | 'a'..='z' | 'A'..='Z' | '0'..='9'))
}

pub(crate) fn resolve_config_field(
    client_name: &str,
    field_name: &str,
    value: Option<&str>,
    env_aliases: &[&str],
) -> ConfigFieldResult<String> {
    resolve_config_field_with(client_name, field_name, value, env_aliases, |name| {
        std::env::var(name).ok()
    })
}

pub(crate) fn optional_config_field(
    value: ConfigFieldResult<String>,
) -> ConfigFieldResult<Option<String>> {
    match value {
        Ok(value) => Ok(Some(value)),
        Err(err) if err.kind == ConfigFieldErrorKind::Absent => Ok(None),
        Err(err) => Err(err),
    }
}

fn resolve_config_field_with<F>(
    client_name: &str,
    field_name: &str,
    value: Option<&str>,
    env_aliases: &[&str],
    env_lookup: F,
) -> ConfigFieldResult<String>
where
    F: Fn(&str) -> Option<String>,
{
    let value = value
        .map(|value| parse_config_field_value(field_name, value))
        .transpose()?;

    if let Some(ConfigFieldValue::Env(env_name)) = value.as_ref() {
        return env_lookup(env_name)
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                ConfigFieldError::new(field_name, ConfigFieldErrorKind::UnresolvedReference)
            });
    }

    let primary_env = format!("{client_name}_{field_name}").to_ascii_uppercase();
    for env_name in std::iter::once(primary_env.as_str()).chain(env_aliases.iter().copied()) {
        if let Some(value) = env_lookup(env_name)
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
        {
            return Ok(value);
        }
    }

    if let Some(ConfigFieldValue::Literal(value)) = value {
        return Ok(value.into_owned());
    }
    Err(ConfigFieldError::new(
        field_name,
        ConfigFieldErrorKind::Absent,
    ))
}

#[async_trait::async_trait]
pub trait Client: Sync + Send {
    fn global_config(&self) -> &GlobalConfig;

    fn extra_config(&self) -> Option<&ExtraConfig>;

    fn patch_config(&self) -> Option<&RequestPatch>;

    fn name(&self) -> &str;

    fn model(&self) -> &Model;

    fn model_mut(&mut self) -> &mut Model;

    fn build_client(&self) -> Result<ReqwestClient> {
        let mut builder = ReqwestClient::builder();
        let extra = self.extra_config();
        let timeout = extra.and_then(|v| v.connect_timeout).unwrap_or(10);
        if let Some(proxy) = extra.and_then(|v| v.proxy.as_deref()) {
            builder = set_proxy(builder, proxy)?;
        }
        if let Some(user_agent) = self.global_config().read().user_agent.as_ref() {
            builder = builder.user_agent(user_agent);
        }
        let client = builder
            .connect_timeout(Duration::from_secs(timeout))
            .build()
            .with_context(|| "Failed to build client")?;
        Ok(client)
    }

    async fn chat_completions(&self, input: Input) -> Result<ChatCompletionsOutput> {
        if self.global_config().read().dry_run {
            let content = input.echo_messages();
            return Ok(ChatCompletionsOutput::new(&content));
        }
        let client = self.build_client()?;
        let data = input.prepare_completion_data(self.model(), false)?;
        retry_request(|| self.chat_completions_inner(&client, data.clone()))
            .await
            .with_context(|| "Failed to call chat-completions api")
    }

    async fn chat_completions_streaming(
        &self,
        input: &Input,
        handler: &mut SseHandler,
    ) -> Result<()> {
        let abort_signal = handler.abort();
        let input = input.clone();
        tokio::select! {
            ret = async {
                if self.global_config().read().dry_run {
                    let content = input.echo_messages();
                    handler.text(&content)?;
                    return Ok(());
                }
                let client = self.build_client()?;
                let data = input.prepare_completion_data(self.model(), true)?;
                let stream =
                    retry_chat_events(|| self.chat_events_inner(&client, data.clone())).await?;
                drive_chat_events(stream, handler).await
            } => {
                handler.done();
                ret.with_context(|| "Failed to call chat-completions api")
            }
            _ = wait_abort_signal(&abort_signal) => {
                handler.done();
                Ok(())
            },
        }
    }

    async fn embeddings(&self, data: &EmbeddingsData) -> Result<Vec<Vec<f32>>> {
        let client = self.build_client()?;
        retry_request(|| self.embeddings_inner(&client, data))
            .await
            .context("Failed to call embeddings api")
    }

    async fn rerank(&self, data: &RerankData) -> Result<RerankOutput> {
        let client = self.build_client()?;
        retry_request(|| self.rerank_inner(&client, data))
            .await
            .context("Failed to call rerank api")
    }

    /// Non-streaming completion. Providers with a dedicated non-streaming
    /// wire format override this; the default collects the event stream, so
    /// a provider only has to implement `chat_events_inner`.
    async fn chat_completions_inner(
        &self,
        client: &ReqwestClient,
        mut data: ChatCompletionsData,
    ) -> Result<ChatCompletionsOutput> {
        data.stream = true;
        let stream = self.chat_events_inner(client, data).await?;
        let (text, tool_calls, usage) = collect_chat_events(stream).await?;
        Ok(ChatCompletionsOutput {
            text,
            tool_calls,
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            ..Default::default()
        })
    }

    async fn chat_events_inner(
        &self,
        client: &ReqwestClient,
        data: ChatCompletionsData,
    ) -> Result<ChatEventStream>;

    async fn embeddings_inner(
        &self,
        _client: &ReqwestClient,
        _data: &EmbeddingsData,
    ) -> Result<EmbeddingsOutput> {
        bail!("The client doesn't support embeddings api")
    }

    async fn rerank_inner(
        &self,
        _client: &ReqwestClient,
        _data: &RerankData,
    ) -> Result<RerankOutput> {
        bail!("The client doesn't support rerank api")
    }

    fn request_builder(
        &self,
        client: &reqwest::Client,
        mut request_data: RequestData,
    ) -> RequestBuilder {
        self.patch_request_data(&mut request_data);
        request_data.into_builder(client)
    }

    fn patch_request_data(&self, request_data: &mut RequestData) {
        let model_type = self.model().model_type();
        if let Some(patch) = self.model().patch() {
            request_data.apply_patch(patch.clone());
        }

        let patch_map = std::env::var(get_env_name(&format!(
            "patch_{}_{}",
            self.model().client_name(),
            model_type.api_name(),
        )))
        .ok()
        .and_then(|v| serde_json::from_str(&v).ok())
        .or_else(|| {
            self.patch_config()
                .and_then(|v| model_type.extract_patch(v))
                .cloned()
        });
        let patch_map = match patch_map {
            Some(v) => v,
            _ => return,
        };
        for (key, patch) in patch_map {
            let key = ESCAPE_SLASH_RE.replace_all(&key, r"\/");
            if let Ok(regex) = Regex::new(&format!("^({key})$")) {
                if let Ok(true) = regex.is_match(self.model().name()) {
                    request_data.apply_patch(patch);
                    return;
                }
            }
        }
    }
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self::OpenAIConfig(OpenAIConfig::default())
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ExtraConfig {
    pub proxy: Option<String>,
    pub connect_timeout: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RequestPatch {
    pub chat_completions: Option<ApiPatch>,
    pub embeddings: Option<ApiPatch>,
    pub rerank: Option<ApiPatch>,
}

pub type ApiPatch = IndexMap<String, Value>;

pub struct RequestData {
    pub url: String,
    pub headers: IndexMap<String, String>,
    pub body: Value,
}

impl RequestData {
    pub fn new<T>(url: T, body: Value) -> Self
    where
        T: std::fmt::Display,
    {
        Self {
            url: url.to_string(),
            headers: Default::default(),
            body,
        }
    }

    pub fn bearer_auth<T>(&mut self, auth: T)
    where
        T: std::fmt::Display,
    {
        self.headers
            .insert("authorization".into(), format!("Bearer {auth}"));
    }

    pub fn header<K, V>(&mut self, key: K, value: V)
    where
        K: std::fmt::Display,
        V: std::fmt::Display,
    {
        self.headers.insert(key.to_string(), value.to_string());
    }

    pub fn into_builder(self, client: &ReqwestClient) -> RequestBuilder {
        let RequestData { url, headers, body } = self;
        debug!("Request {url} {body}");

        let mut builder = client.post(url);
        for (key, value) in headers {
            builder = builder.header(key, value);
        }
        builder = builder.json(&body);
        builder
    }

    pub fn apply_patch(&mut self, patch: Value) {
        if let Some(patch_url) = patch["url"].as_str() {
            self.url = patch_url.into();
        }
        if let Some(patch_body) = patch.get("body") {
            json_patch::merge(&mut self.body, patch_body)
        }
        if let Some(patch_headers) = patch["headers"].as_object() {
            for (key, value) in patch_headers {
                if let Some(value) = value.as_str() {
                    self.header(key, value)
                } else if value.is_null() {
                    self.headers.swap_remove(key);
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatCompletionsData {
    pub messages: Vec<Message>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub functions: Option<Vec<FunctionDeclaration>>,
    pub stream: bool,
    pub include_usage: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ChatCompletionsOutput {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub id: Option<String>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
}

pub fn format_usage_cost(model: &Model, usage: TokenUsage) -> String {
    let input = usage
        .input_tokens
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unavailable".into());
    let output = usage
        .output_tokens
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unavailable".into());
    let cost = model
        .usage_cost(usage)
        .map(|value| format!("${value:.6}"))
        .unwrap_or_else(|| "unavailable".into());
    format!("Tokens: {input} input + {output} output | Estimated cost: {cost}")
}

impl ChatCompletionsOutput {
    pub fn new(text: &str) -> Self {
        Self {
            text: text.to_string(),
            ..Default::default()
        }
    }

    pub fn usage(&self) -> TokenUsage {
        TokenUsage::new(self.input_tokens, self.output_tokens)
    }
}

#[derive(Debug)]
pub struct EmbeddingsData {
    pub texts: Vec<String>,
    pub query: bool,
}

impl EmbeddingsData {
    pub fn new(texts: Vec<String>, query: bool) -> Self {
        Self { texts, query }
    }
}

pub type EmbeddingsOutput = Vec<Vec<f32>>;

#[derive(Debug)]
pub struct RerankData {
    pub query: String,
    pub documents: Vec<String>,
    pub top_n: usize,
}

impl RerankData {
    pub fn new(query: String, documents: Vec<String>, top_n: usize) -> Self {
        Self {
            query,
            documents,
            top_n,
        }
    }
}

pub type RerankOutput = Vec<RerankResult>;

#[derive(Debug, Deserialize)]
pub struct RerankResult {
    pub index: usize,
    pub relevance_score: f64,
}

pub type PromptAction<'a> = (&'a str, &'a str, Option<&'a str>);

pub async fn create_config(
    prompts: &[PromptAction<'static>],
    client: &str,
) -> Result<(String, Value)> {
    let mut config = json!({
        "type": client,
    });
    for (key, desc, help_message) in prompts {
        let env_name = format!("{client}_{key}").to_ascii_uppercase();
        let required = std::env::var(&env_name).is_err();
        let value = prompt_input_string(desc, required, *help_message)?;
        if !value.is_empty() {
            config[key] = value.into();
        }
    }
    let model = set_client_models_config(&mut config, client).await?;
    let clients = json!(vec![config]);
    Ok((model, clients))
}

pub async fn create_openai_compatible_client_config(
    client: &str,
) -> Result<Option<(String, Value)>> {
    let api_base = compat_provider_api_base(client).unwrap_or("http(s)://{API_ADDR}/v1");

    let name = if client == OpenAICompatibleClient::NAME {
        let value = prompt_input_string("Provider Name", true, None)?;
        value.replace(' ', "-")
    } else {
        client.to_string()
    };

    let mut config = json!({
        "type": OpenAICompatibleClient::NAME,
        "name": &name,
    });

    let api_base = if api_base.contains('{') {
        prompt_input_string("API Base", true, Some(&format!("e.g. {api_base}")))?
    } else {
        api_base.to_string()
    };
    config["api_base"] = api_base.into();

    let api_key = prompt_input_string("API Key", false, None)?;
    if !api_key.is_empty() {
        config["api_key"] = api_key.into();
    }

    let model = set_client_models_config(&mut config, &name).await?;
    let clients = json!(vec![config]);
    Ok(Some((model, clients)))
}

pub async fn call_chat_completions(
    input: &Input,
    print: bool,
    extract_code: bool,
    client: &dyn Client,
    abort_signal: AbortSignal,
) -> Result<(ChatCompletionsOutput, Vec<ToolResult>)> {
    let ret = abortable_run_with_spinner(
        client.chat_completions(input.clone()),
        "Generating",
        abort_signal,
    )
    .await;

    match ret {
        Ok(ret) => {
            let mut output = ret;
            let mut text = std::mem::take(&mut output.text);
            let tool_calls = std::mem::take(&mut output.tool_calls);
            if !text.is_empty() {
                if extract_code {
                    text = extract_code_block(&strip_think_tag(&text)).to_string();
                }
                if print {
                    client.global_config().read().print_markdown(&text)?;
                }
            }
            output.text = text;
            Ok((output, eval_tool_calls(client.global_config(), tool_calls)?))
        }
        Err(err) => Err(err),
    }
}

pub async fn call_chat_completions_streaming(
    input: &Input,
    client: &dyn Client,
    abort_signal: AbortSignal,
) -> Result<(ChatCompletionsOutput, Vec<ToolResult>)> {
    let (tx, rx) = unbounded_channel();
    let mut handler = SseHandler::new(tx, abort_signal.clone());

    let (send_ret, render_ret) = tokio::join!(
        client.chat_completions_streaming(input, &mut handler),
        render_stream(rx, client.global_config(), abort_signal.clone()),
    );

    if handler.abort().aborted() {
        bail!("Aborted.");
    }

    render_ret?;

    let (text, tool_calls, usage) = handler.take();
    match send_ret {
        Ok(_) => {
            if !text.is_empty() && !text.ends_with('\n') {
                println!();
            }
            Ok((
                ChatCompletionsOutput {
                    text,
                    tool_calls: Vec::new(),
                    input_tokens: usage.input_tokens,
                    output_tokens: usage.output_tokens,
                    ..Default::default()
                },
                eval_tool_calls(client.global_config(), tool_calls)?,
            ))
        }
        Err(err) => {
            if !text.is_empty() {
                println!();
            }
            Err(err)
        }
    }
}

pub fn noop_prepare_embeddings<T>(_client: &T, _data: &EmbeddingsData) -> Result<RequestData> {
    bail!("The client doesn't support embeddings api")
}

pub async fn noop_embeddings(_builder: RequestBuilder, _model: &Model) -> Result<EmbeddingsOutput> {
    bail!("The client doesn't support embeddings api")
}

pub fn noop_prepare_rerank<T>(_client: &T, _data: &RerankData) -> Result<RequestData> {
    bail!("The client doesn't support rerank api")
}

pub async fn noop_rerank(_builder: RequestBuilder, _model: &Model) -> Result<RerankOutput> {
    bail!("The client doesn't support rerank api")
}

pub fn catch_error(data: &Value, status: u16) -> Result<()> {
    if (200..300).contains(&status) {
        return Ok(());
    }
    debug!("Invalid response, status: {status}, data: {data}");
    let (message, hint) = extract_error_message(data, status);
    let kind = classify_provider_error(status, hint.as_deref());
    Err(ProviderError::new(kind, message, Some(status)).into())
}

/// Probe the known provider error response shapes. Returns the user-facing
/// message (formats preserved from the pre-`ProviderError` string errors) and
/// the provider-reported error type/code usable as a classification hint.
fn extract_error_message(data: &Value, status: u16) -> (String, Option<String>) {
    if let Some(error) = data["error"].as_object() {
        if let (Some(typ), Some(message)) = (
            json_str_from_map(error, "type"),
            json_str_from_map(error, "message"),
        ) {
            return (format!("{message} (type: {typ})"), Some(typ.to_string()));
        } else if let (Some(code), Some(message)) = (
            json_str_from_map(error, "code"),
            json_str_from_map(error, "message"),
        ) {
            return (format!("{message} (code: {code})"), Some(code.to_string()));
        }
    } else if let Some(error) = data["errors"][0].as_object() {
        if let (Some(code), Some(message)) = (
            error.get("code").and_then(|v| v.as_u64()),
            json_str_from_map(error, "message"),
        ) {
            return (format!("{message} (status: {code})"), None);
        }
    } else if let Some(error) = data[0]["error"].as_object() {
        if let (Some(error_status), Some(message)) = (
            json_str_from_map(error, "status"),
            json_str_from_map(error, "message"),
        ) {
            return (
                format!("{message} (status: {error_status})"),
                Some(error_status.to_string()),
            );
        }
    } else if let (Some(detail), Some(status)) = (data["detail"].as_str(), data["status"].as_i64())
    {
        return (format!("{detail} (status: {status})"), None);
    } else if let (Some(detail), Some(code)) = (data["detail"].as_str(), data["code"].as_i64()) {
        return (format!("{detail} (status: {code})"), None);
    } else if let Some(error) = data["error"].as_str() {
        return (error.to_string(), None);
    } else if let Some(message) = data["message"].as_str() {
        return (message.to_string(), None);
    }
    (
        format!("Invalid response data: {data} (status: {status})"),
        None,
    )
}

pub fn json_str_from_map<'a>(
    map: &'a serde_json::Map<String, Value>,
    field_name: &str,
) -> Option<&'a str> {
    map.get(field_name).and_then(|v| v.as_str())
}

async fn set_client_models_config(client_config: &mut Value, client: &str) -> Result<String> {
    if let Some(provider) = ALL_PROVIDER_MODELS.iter().find(|v| v.provider == client) {
        let models: Vec<String> = provider
            .models
            .iter()
            .filter(|v| v.model_type == "chat")
            .map(|v| v.name.clone())
            .collect();
        let model_name = select_model(models)?;
        return Ok(format!("{client}:{model_name}"));
    }
    let mut model_names = vec![];
    if let (Some(true), Some(api_base), api_key) = (
        client_config["type"]
            .as_str()
            .map(|v| v == OpenAICompatibleClient::NAME),
        client_config["api_base"].as_str(),
        client_config["api_key"]
            .as_str()
            .map(|v| v.to_string())
            .or_else(|| {
                let env_name = format!("{client}_api_key").to_ascii_uppercase();
                std::env::var(&env_name).ok()
            }),
    ) {
        match abortable_run_with_spinner(
            fetch_models(api_base, api_key.as_deref()),
            "Fetching models",
            create_abort_signal(),
        )
        .await
        {
            Ok(fetched_models) => {
                model_names = MultiSelect::new("LLMs to include (required):", fetched_models)
                    .with_validator(|list: &[ListOption<&String>]| {
                        if list.is_empty() {
                            Ok(Validation::Invalid(
                                "At least one item must be selected".into(),
                            ))
                        } else {
                            Ok(Validation::Valid)
                        }
                    })
                    .prompt()?;
            }
            Err(err) => {
                eprintln!("✗ Fetch models failed: {err}");
            }
        }
    }
    if model_names.is_empty() {
        model_names = prompt_input_string(
            "LLMs to add",
            true,
            Some("Separated by commas, e.g. llama3.3,qwen2.5"),
        )?
        .split(',')
        .filter_map(|v| {
            let v = v.trim();
            if v.is_empty() {
                None
            } else {
                Some(v.to_string())
            }
        })
        .collect::<Vec<_>>();
    }
    if model_names.is_empty() {
        bail!("No models");
    }
    let models: Vec<Value> = model_names
        .iter()
        .map(|v| {
            let l = v.to_lowercase();
            if l.contains("rank") {
                json!({
                    "name": v,
                    "type": "reranker",
                })
            } else if let Ok(true) = EMBEDDING_MODEL_RE.is_match(&l) {
                json!({
                    "name": v,
                    "type": "embedding",
                    "default_chunk_size": 1000,
                    "max_batch_size": 100
                })
            } else if v.contains("vision") {
                json!({
                    "name": v,
                    "supports_vision": true
                })
            } else {
                json!({
                    "name": v,
                })
            }
        })
        .collect();
    client_config["models"] = models.into();
    let model_name = select_model(model_names)?;
    Ok(format!("{client}:{model_name}"))
}

fn select_model(model_names: Vec<String>) -> Result<String> {
    if model_names.is_empty() {
        bail!("No models");
    }
    let model = if model_names.len() == 1 {
        model_names[0].clone()
    } else {
        Select::new("Default Model (required):", model_names).prompt()?
    };
    Ok(model)
}

fn prompt_input_string(
    desc: &str,
    required: bool,
    help_message: Option<&str>,
) -> anyhow::Result<String> {
    let desc = if required {
        format!("{desc} (required):")
    } else {
        format!("{desc} (optional):")
    };
    let mut text = Text::new(&desc);
    if required {
        text = text.with_validator(required!("This field is required"))
    }
    if let Some(help_message) = help_message {
        text = text.with_help_message(help_message);
    }
    let text = text.prompt()?;
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn resolve_with_env(
        client_name: &str,
        field_name: &str,
        value: Option<&str>,
        aliases: &[&str],
        env: &[(&str, &str)],
    ) -> ConfigFieldResult<String> {
        let env: HashMap<_, _> = env
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect();
        resolve_config_field_with(client_name, field_name, value, aliases, |name| {
            env.get(name).cloned()
        })
    }

    #[test]
    fn config_field_resolves_exact_environment_references() {
        let env = [("DIRECT_KEY", "  secret\r\n")];

        assert_eq!(
            resolve_with_env("claude", "api_key", Some("$DIRECT_KEY"), &[], &env).unwrap(),
            "secret"
        );
        assert_eq!(
            resolve_with_env("claude", "api_key", Some("${DIRECT_KEY}"), &[], &env).unwrap(),
            "secret"
        );
    }

    #[test]
    fn config_field_rejects_missing_empty_and_malformed_references() {
        for value in ["$", "${}", "${KEY", "$KEY}", "$1KEY", "${KEY}-suffix"] {
            let err = resolve_with_env("claude", "api_key", Some(value), &[], &[])
                .expect_err("malformed reference must fail");
            assert!(!err.to_string().contains(value));
        }

        for env in [&[][..], &[("DIRECT_KEY", " \r\n")][..]] {
            let err = resolve_with_env("claude", "api_key", Some("$DIRECT_KEY"), &[], env)
                .expect_err("missing or empty explicit reference must fail");
            assert!(!err.to_string().contains("DIRECT_KEY"));
        }
    }

    #[test]
    fn config_field_precedence_and_literal_escape_are_deterministic() {
        let env = [
            ("DIRECT_KEY", "direct"),
            ("CUSTOM_API_KEY", "primary"),
            ("ANTHROPIC_API_KEY", "alias"),
        ];

        assert_eq!(
            resolve_with_env(
                "custom",
                "api_key",
                Some("$DIRECT_KEY"),
                &["ANTHROPIC_API_KEY"],
                &env,
            )
            .unwrap(),
            "direct"
        );
        assert_eq!(
            resolve_with_env(
                "custom",
                "api_key",
                Some("literal"),
                &["ANTHROPIC_API_KEY"],
                &env,
            )
            .unwrap(),
            "primary"
        );
        assert_eq!(
            resolve_with_env("custom", "api_key", Some("$$DIRECT_KEY"), &[], &[]).unwrap(),
            "$DIRECT_KEY"
        );
    }

    #[test]
    fn anthropic_alias_is_limited_to_claude_api_keys() {
        let env = [("ANTHROPIC_API_KEY", " anthropic-key\n")];

        assert_eq!(
            resolve_with_env(
                "custom-claude",
                "api_key",
                None,
                &["ANTHROPIC_API_KEY"],
                &env,
            )
            .unwrap(),
            "anthropic-key"
        );
        assert!(resolve_with_env("custom-claude", "api_base", None, &[], &env,).is_err());
        assert!(resolve_with_env("claude-openai-compatible", "api_key", None, &[], &env,).is_err());
    }

    #[test]
    fn empty_conventional_environment_value_falls_back_to_literal() {
        assert_eq!(
            resolve_with_env(
                "custom",
                "api_key",
                Some("literal"),
                &[],
                &[("CUSTOM_API_KEY", " \r\n")],
            )
            .unwrap(),
            "literal"
        );
    }

    #[test]
    fn optional_config_field_only_converts_true_absence() {
        let absent = resolve_with_env("optional", "api_key", None, &[], &[]);
        assert_eq!(optional_config_field(absent).unwrap(), None);

        let literal = resolve_with_env("optional", "api_key", Some("literal"), &[], &[]);
        assert_eq!(
            optional_config_field(literal).unwrap(),
            Some("literal".into())
        );

        for resolved in [
            resolve_with_env("optional", "api_key", Some("$"), &[], &[]),
            resolve_with_env("optional", "api_key", Some("$MISSING_KEY"), &[], &[]),
            resolve_with_env(
                "optional",
                "api_key",
                Some("$EMPTY_KEY"),
                &[],
                &[("EMPTY_KEY", " \r\n")],
            ),
        ] {
            assert!(optional_config_field(resolved).is_err());
        }
    }

    #[test]
    fn catch_error_recognizes_numeric_code_and_detail() {
        let err = catch_error(&json!({"code":529,"detail":"Overloaded"}), 529)
            .expect_err("non-success response must fail");

        assert_eq!(err.to_string(), "Overloaded (status: 529)");
    }

    #[test]
    fn catch_error_keeps_successful_response_as_control() {
        assert!(catch_error(&json!({"code":529,"detail":"Overloaded"}), 200).is_ok());
    }

    #[test]
    fn formats_provider_usage_and_catalog_cost() {
        let mut model = Model::new("test", "priced");
        model.data_mut().input_price = Some(2.0);
        model.data_mut().output_price = Some(8.0);

        assert_eq!(
            format_usage_cost(&model, TokenUsage::new(Some(1_000), Some(250))),
            "Tokens: 1000 input + 250 output | Estimated cost: $0.004000"
        );
    }

    #[test]
    fn reports_missing_usage_without_estimating_cost() {
        let model = Model::new("test", "unpriced");
        assert_eq!(
            format_usage_cost(&model, TokenUsage::default()),
            "Tokens: unavailable input + unavailable output | Estimated cost: unavailable"
        );
    }
}
