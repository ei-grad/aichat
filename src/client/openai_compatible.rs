use super::openai::*;
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
    #[serde(default)]
    pub models: Vec<ModelData>,
    pub patch: Option<RequestPatch>,
    pub extra: Option<ExtraConfig>,
}

impl OpenAICompatibleClient {
    config_get_fn!(api_base, get_api_base);
    config_get_fn!(api_key, get_api_key);

    pub const PROMPTS: [PromptAction<'static>; 0] = [];
}

impl_client_trait!(
    OpenAICompatibleClient,
    (
        prepare_chat_completions,
        openai_chat_completions,
        openai_chat_completions_streaming
    ),
    (prepare_embeddings, openai_embeddings),
    (prepare_rerank, generic_rerank),
);

fn prepare_chat_completions(
    self_: &OpenAICompatibleClient,
    data: ChatCompletionsData,
) -> Result<RequestData> {
    let api_key = optional_config_field(self_.get_api_key())?;
    let api_base = get_api_base_ext(self_)?;

    let url = format!("{api_base}/chat/completions");

    let body = openai_build_chat_completions_body(data, &self_.model);

    let mut request_data = RequestData::new(url, body);

    if let Some(api_key) = api_key {
        request_data.bearer_auth(api_key);
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
        None => OPENAI_COMPATIBLE_PROVIDERS
            .into_iter()
            .find_map(|(name, api_base)| {
                if name == self_.model.client_name() {
                    Some(api_base.to_string())
                } else {
                    None
                }
            })
            .ok_or_else(|| anyhow::anyhow!("Miss 'api_base'"))?,
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

    const MISSING_API_KEY_ENV: &str =
        "AICHAT_TEST_MISSING_OPENAI_COMPATIBLE_API_KEY_B07D8216";
    const MISSING_API_BASE_ENV: &str =
        "AICHAT_TEST_MISSING_OPENAI_COMPATIBLE_API_BASE_F5813573";

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
                models: vec![],
                patch: None,
                extra: None,
            },
            model: Model::new(name, "test-model"),
        }
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
        }
    }

    fn preparation_results(client: &OpenAICompatibleClient) -> [Result<RequestData>; 3] {
        [
            prepare_chat_completions(client, completion_data()),
            prepare_embeddings(
                client,
                &EmbeddingsData::new(vec!["hello".into()], false),
            ),
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

        assert_reference_errors(
            preparation_results(&client),
            "api_key",
            MISSING_API_KEY_ENV,
        );
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
        let client = request_client(
            "openrouter",
            Some(format!("${MISSING_API_BASE_ENV}")),
            None,
        );

        assert_reference_errors(
            preparation_results(&client),
            "api_base",
            MISSING_API_BASE_ENV,
        );
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
