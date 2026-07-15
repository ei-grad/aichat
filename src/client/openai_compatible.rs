use super::claude::{
    claude_build_chat_completions_body, claude_chat_completions, claude_chat_events,
    CLAUDE_API_VERSION,
};
use super::cohere::{
    cohere_build_chat_completions_body, cohere_chat_completions, cohere_chat_events,
};
use super::openai::*;
use super::vertexai::{
    gemini_build_chat_completions_body, gemini_chat_completions, gemini_chat_events,
};
use super::*;

use anyhow::{Context, Result};
use reqwest::RequestBuilder;
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAICompatibleConfig {
    pub name: Option<String>,
    pub api_base: Option<String>,
    pub api_key: Option<String>,
    /// Wire protocol the endpoint speaks; defaults to the catalog entry
    /// matching the client name, then to the OpenAI dialect.
    pub wire_format: Option<WireFormat>,
    #[serde(default)]
    pub models: Vec<ModelData>,
    pub patch: Option<RequestPatch>,
    pub extra: Option<ExtraConfig>,
}

impl OpenAICompatibleClient {
    config_get_fn!(api_base, get_api_base);
    config_get_fn!(api_key, get_api_key);

    fn wire_format(&self) -> WireFormat {
        self.config.wire_format.unwrap_or_else(|| {
            ALL_PROVIDER_MODELS
                .iter()
                .find(|v| self.model.client_name().starts_with(&v.provider))
                .map(|v| v.wire_format)
                .unwrap_or_default()
        })
    }
}

#[async_trait::async_trait]
impl Client for OpenAICompatibleClient {
    client_common_fns!();

    async fn chat_completions_inner(
        &self,
        client: &reqwest::Client,
        data: ChatCompletionsData,
    ) -> Result<ChatCompletionsOutput> {
        let request_data = prepare_chat_completions(self, data)?;
        let builder = self.request_builder(client, request_data);
        match self.wire_format() {
            WireFormat::Openai => openai_chat_completions(builder, self.model()).await,
            WireFormat::Claude => claude_chat_completions(builder, self.model()).await,
            WireFormat::Gemini => gemini_chat_completions(builder, self.model()).await,
            WireFormat::Cohere => cohere_chat_completions(builder, self.model()).await,
        }
    }

    async fn chat_events_inner(
        &self,
        client: &reqwest::Client,
        data: ChatCompletionsData,
    ) -> Result<ChatEventStream> {
        let request_data = prepare_chat_completions(self, data)?;
        let builder = self.request_builder(client, request_data);
        match self.wire_format() {
            WireFormat::Openai => openai_chat_events(builder, self.model()).await,
            WireFormat::Claude => claude_chat_events(builder, self.model()).await,
            WireFormat::Gemini => gemini_chat_events(builder, self.model()).await,
            WireFormat::Cohere => cohere_chat_events(builder, self.model()).await,
        }
    }

    async fn embeddings_inner(
        &self,
        client: &reqwest::Client,
        data: &EmbeddingsData,
    ) -> Result<EmbeddingsOutput> {
        let request_data = prepare_embeddings(self, data)?;
        let builder = self.request_builder(client, request_data);
        openai_embeddings(builder, self.model()).await
    }

    async fn rerank_inner(
        &self,
        client: &reqwest::Client,
        data: &RerankData,
    ) -> Result<RerankOutput> {
        let request_data = prepare_rerank(self, data)?;
        let builder = self.request_builder(client, request_data);
        generic_rerank(builder, self.model()).await
    }
}

fn prepare_chat_completions(
    self_: &OpenAICompatibleClient,
    data: ChatCompletionsData,
) -> Result<RequestData> {
    let api_key = optional_config_field(self_.get_api_key())?;
    let api_base = get_api_base_ext(self_)?;
    let wire_format = self_.wire_format();

    let mut request_data = match wire_format {
        WireFormat::Openai => RequestData::new(
            format!("{api_base}/chat/completions"),
            openai_build_chat_completions_body(data, &self_.model),
        ),
        WireFormat::Claude => {
            let mut request_data = RequestData::new(
                format!("{api_base}/messages"),
                claude_build_chat_completions_body(data, &self_.model)?,
            );
            request_data.header("anthropic-version", CLAUDE_API_VERSION);
            request_data
        }
        WireFormat::Gemini => {
            let func = match data.stream {
                true => "streamGenerateContent",
                false => "generateContent",
            };
            RequestData::new(
                format!("{api_base}/models/{}:{func}", self_.model.real_name()),
                gemini_build_chat_completions_body(data, &self_.model)?,
            )
        }
        WireFormat::Cohere => RequestData::new(
            format!("{api_base}/chat"),
            cohere_build_chat_completions_body(data, &self_.model),
        ),
    };

    if let Some(api_key) = api_key {
        match wire_format {
            WireFormat::Claude => request_data.header("x-api-key", api_key),
            WireFormat::Gemini => request_data.header("x-goog-api-key", api_key),
            WireFormat::Openai | WireFormat::Cohere => request_data.bearer_auth(api_key),
        }
    }

    Ok(request_data)
}

fn prepare_embeddings(
    self_: &OpenAICompatibleClient,
    data: &EmbeddingsData,
) -> Result<RequestData> {
    let api_key = optional_config_field(self_.get_api_key())?;
    let api_base = get_api_base_ext(self_)?;

    let url = format!("{api_base}/embeddings");

    let body = openai_build_embeddings_body(data, &self_.model);

    let mut request_data = RequestData::new(url, body);

    if let Some(api_key) = api_key {
        request_data.bearer_auth(api_key);
    }

    Ok(request_data)
}

fn prepare_rerank(self_: &OpenAICompatibleClient, data: &RerankData) -> Result<RequestData> {
    let api_key = optional_config_field(self_.get_api_key())?;
    let api_base = get_api_base_ext(self_)?;

    let url = if self_.name().starts_with("ernie") {
        format!("{api_base}/rerankers")
    } else {
        format!("{api_base}/rerank")
    };

    let body = generic_build_rerank_body(data, &self_.model);

    let mut request_data = RequestData::new(url, body);

    if let Some(api_key) = api_key {
        request_data.bearer_auth(api_key);
    }

    Ok(request_data)
}

fn get_api_base_ext(self_: &OpenAICompatibleClient) -> Result<String> {
    let api_base = match optional_config_field(self_.get_api_base())? {
        Some(api_base) => api_base,
        None => compat_provider_api_base(self_.model.client_name())
            .ok_or_else(|| anyhow::anyhow!("Miss 'api_base'"))?
            .to_string(),
    };
    Ok(api_base.trim_end_matches('/').to_string())
}

pub async fn generic_rerank(builder: RequestBuilder, _model: &Model) -> Result<RerankOutput> {
    let res = builder.send().await?;
    let status = res.status();
    let mut data: Value = res.json().await?;
    if !status.is_success() {
        catch_error(&data, status.as_u16())?;
    }
    if data.get("results").is_none() && data.get("data").is_some() {
        if let Some(data_obj) = data.as_object_mut() {
            if let Some(value) = data_obj.remove("data") {
                data_obj.insert("results".to_string(), value);
            }
        }
    }
    let res_body: GenericRerankResBody =
        serde_json::from_value(data).context("Invalid rerank data")?;
    Ok(res_body.results)
}

#[derive(Deserialize)]
pub struct GenericRerankResBody {
    pub results: RerankOutput,
}

pub fn generic_build_rerank_body(data: &RerankData, model: &Model) -> Value {
    let RerankData {
        query,
        documents,
        top_n,
    } = data;

    let mut body = json!({
        "model": model.real_name(),
        "query": query,
        "documents": documents,
    });
    if model.client_name().starts_with("voyageai") {
        body["top_k"] = (*top_n).into()
    } else {
        body["top_n"] = (*top_n).into()
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;

    const MISSING_API_KEY_ENV: &str = "AICHAT_TEST_MISSING_OPENAI_COMPATIBLE_API_KEY_B07D8216";
    const MISSING_API_BASE_ENV: &str = "AICHAT_TEST_MISSING_OPENAI_COMPATIBLE_API_BASE_F5813573";

    fn request_client(
        name: &str,
        api_base: Option<String>,
        api_key: Option<String>,
    ) -> OpenAICompatibleClient {
        OpenAICompatibleClient {
            global_config: Default::default(),
            config: OpenAICompatibleConfig {
                name: Some(name.to_string()),
                api_base,
                api_key,
                wire_format: None,
                models: vec![],
                patch: None,
                extra: None,
            },
            model: Model::new(name, "test-model"),
        }
    }

    fn wire_client(wire_format: WireFormat) -> OpenAICompatibleClient {
        let mut client = request_client(
            "wire-format-test",
            Some("https://local.invalid/v1".into()),
            Some("test-key".into()),
        );
        client.config.wire_format = Some(wire_format);
        client
    }

    fn completion_data() -> ChatCompletionsData {
        ChatCompletionsData {
            messages: vec![Message::new(
                MessageRole::User,
                MessageContent::Text("hello".into()),
            )],
            temperature: None,
            top_p: None,
            functions: None,
            stream: false,
            include_usage: false,
        }
    }

    fn preparation_results(client: &OpenAICompatibleClient) -> [Result<RequestData>; 3] {
        [
            prepare_chat_completions(client, completion_data()),
            prepare_embeddings(client, &EmbeddingsData::new(vec!["hello".into()], false)),
            prepare_rerank(
                client,
                &RerankData::new("query".into(), vec!["document".into()], 1),
            ),
        ]
    }

    fn assert_reference_errors(
        results: [Result<RequestData>; 3],
        field_name: &str,
        env_name: &str,
    ) {
        for result in results {
            let err = result
                .err()
                .expect("missing explicit reference must fail before request preparation");
            assert_eq!(
                err.to_string(),
                format!("Environment variable for '{field_name}' is missing or empty")
            );
            assert!(!err.to_string().contains(env_name));
        }
    }

    #[test]
    fn missing_explicit_api_key_is_not_converted_to_no_auth() {
        assert!(std::env::var_os(MISSING_API_KEY_ENV).is_none());
        let client = request_client(
            "openai-compatible-remediation-test",
            Some("https://local.invalid/v1".into()),
            Some(format!("${MISSING_API_KEY_ENV}")),
        );

        assert_reference_errors(preparation_results(&client), "api_key", MISSING_API_KEY_ENV);
    }

    #[test]
    fn absent_api_key_keeps_optional_no_auth_requests() {
        let client = request_client(
            "openai-compatible-remediation-test",
            Some("https://local.invalid/v1".into()),
            None,
        );

        for request in preparation_results(&client).map(Result::unwrap) {
            assert!(!request.headers.contains_key("authorization"));
        }
    }

    #[test]
    fn missing_explicit_api_base_is_not_replaced_by_provider_default() {
        assert!(std::env::var_os(MISSING_API_BASE_ENV).is_none());
        let client = request_client("openrouter", Some(format!("${MISSING_API_BASE_ENV}")), None);

        assert_reference_errors(
            preparation_results(&client),
            "api_base",
            MISSING_API_BASE_ENV,
        );
    }

    #[test]
    fn claude_wire_format_uses_messages_path_and_api_key_header() {
        let request =
            prepare_chat_completions(&wire_client(WireFormat::Claude), completion_data()).unwrap();
        assert_eq!(request.url, "https://local.invalid/v1/messages");
        assert_eq!(
            request.headers.get("x-api-key").map(String::as_str),
            Some("test-key")
        );
        assert_eq!(
            request.headers.get("anthropic-version").map(String::as_str),
            Some(CLAUDE_API_VERSION)
        );
        assert!(!request.headers.contains_key("authorization"));
        assert!(request.body.get("messages").is_some());
    }

    #[test]
    fn gemini_wire_format_uses_model_path_and_goog_header() {
        let request =
            prepare_chat_completions(&wire_client(WireFormat::Gemini), completion_data()).unwrap();
        assert_eq!(
            request.url,
            "https://local.invalid/v1/models/test-model:generateContent"
        );
        assert_eq!(
            request.headers.get("x-goog-api-key").map(String::as_str),
            Some("test-key")
        );
        assert!(request.body.get("contents").is_some());
    }

    #[test]
    fn cohere_wire_format_uses_chat_path_and_renamed_sampling() {
        let mut data = completion_data();
        data.top_p = Some(0.9);
        let request = prepare_chat_completions(&wire_client(WireFormat::Cohere), data).unwrap();
        assert_eq!(request.url, "https://local.invalid/v1/chat");
        assert_eq!(
            request.headers.get("authorization").map(String::as_str),
            Some("Bearer test-key")
        );
        assert!(request.body.get("top_p").is_none());
        assert_eq!(request.body["p"], serde_json::json!(0.9));
    }

    #[test]
    fn wire_format_defaults_to_openai_for_unknown_providers() {
        let client = request_client(
            "wire-format-test",
            Some("https://local.invalid/v1".into()),
            None,
        );
        assert_eq!(client.wire_format(), WireFormat::Openai);
        let request = prepare_chat_completions(&client, completion_data()).unwrap();
        assert_eq!(request.url, "https://local.invalid/v1/chat/completions");
    }

    #[test]
    fn absent_api_base_keeps_known_provider_default() {
        assert!(std::env::var_os("VOYAGEAI_API_BASE").is_none());
        let [chat, embeddings, rerank] =
            preparation_results(&request_client("voyageai", None, None));

        assert_eq!(
            chat.unwrap().url,
            "https://api.voyageai.com/v1/chat/completions"
        );
        assert_eq!(
            embeddings.unwrap().url,
            "https://api.voyageai.com/v1/embeddings"
        );
        assert_eq!(rerank.unwrap().url, "https://api.voyageai.com/v1/rerank");
    }
}
