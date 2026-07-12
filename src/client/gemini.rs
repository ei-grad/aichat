use super::vertexai::*;
use super::*;

use anyhow::{Context, Result};
use reqwest::RequestBuilder;
use serde::Deserialize;
use serde_json::{json, Value};

const API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta";

#[derive(Debug, Clone, Deserialize, Default)]
pub struct GeminiConfig {
    pub name: Option<String>,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    #[serde(default)]
    pub models: Vec<ModelData>,
    pub patch: Option<RequestPatch>,
    pub extra: Option<ExtraConfig>,
}

impl GeminiClient {
    config_get_fn!(api_key, get_api_key);
    config_get_fn!(api_base, get_api_base);

    pub const PROMPTS: [PromptAction<'static>; 1] = [("api_key", "API Key", None)];
}

impl_client_trait!(
    GeminiClient,
    (
        prepare_chat_completions,
        gemini_chat_completions,
        gemini_chat_events
    ),
    (prepare_embeddings, embeddings),
    (noop_prepare_rerank, noop_rerank),
);

fn prepare_chat_completions(
    self_: &GeminiClient,
    data: ChatCompletionsData,
) -> Result<RequestData> {
    let api_key = self_.get_api_key()?;
    let api_base =
        optional_config_field(self_.get_api_base())?.unwrap_or_else(|| API_BASE.to_string());

    let func = match data.stream {
        true => "streamGenerateContent",
        false => "generateContent",
    };

    let url = format!(
        "{}/models/{}:{}",
        api_base.trim_end_matches('/'),
        self_.model.real_name(),
        func
    );

    let body = gemini_build_chat_completions_body(data, &self_.model)?;

    let mut request_data = RequestData::new(url, body);

    request_data.header("x-goog-api-key", api_key);

    Ok(request_data)
}

fn prepare_embeddings(self_: &GeminiClient, data: &EmbeddingsData) -> Result<RequestData> {
    let api_key = self_.get_api_key()?;
    let api_base =
        optional_config_field(self_.get_api_base())?.unwrap_or_else(|| API_BASE.to_string());

    let url = format!(
        "{}/models/{}:batchEmbedContents?key={}",
        api_base.trim_end_matches('/'),
        self_.model.real_name(),
        api_key
    );

    let model_id = format!("models/{}", self_.model.real_name());

    let requests: Vec<_> = data
        .texts
        .iter()
        .map(|text| {
            json!({
                "model": model_id,
                "content": {
                    "parts": [
                        {
                            "text": text
                        }
                    ]
                },
            })
        })
        .collect();

    let body = json!({
        "requests": requests,
    });

    let request_data = RequestData::new(url, body);

    Ok(request_data)
}

async fn embeddings(builder: RequestBuilder, _model: &Model) -> Result<EmbeddingsOutput> {
    let res = builder.send().await?;
    let status = res.status();
    let data: Value = res.json().await?;
    if !status.is_success() {
        catch_error(&data, status.as_u16())?;
    }
    let res_body: EmbeddingsResBody =
        serde_json::from_value(data).context("Invalid embeddings data")?;
    let output = res_body
        .embeddings
        .into_iter()
        .map(|embedding| embedding.values)
        .collect();
    Ok(output)
}

#[derive(Deserialize)]
struct EmbeddingsResBody {
    embeddings: Vec<EmbeddingsResBodyEmbedding>,
}

#[derive(Deserialize)]
struct EmbeddingsResBodyEmbedding {
    values: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    const MISSING_API_BASE_ENV: &str = "AICHAT_TEST_MISSING_GEMINI_API_BASE_8F65F4CE";

    fn request_client(api_base: Option<String>) -> GeminiClient {
        GeminiClient {
            global_config: Default::default(),
            config: GeminiConfig {
                name: Some("gemini-remediation-test".into()),
                api_key: Some("test-key".into()),
                api_base,
                ..Default::default()
            },
            model: Model::new("gemini-remediation-test", "test-model"),
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

    #[test]
    fn missing_explicit_api_base_does_not_fall_back_to_public_gemini() {
        assert!(std::env::var_os(MISSING_API_BASE_ENV).is_none());
        let client = request_client(Some(format!("${MISSING_API_BASE_ENV}")));

        for result in [
            prepare_chat_completions(&client, completion_data()),
            prepare_embeddings(&client, &EmbeddingsData::new(vec!["hello".into()], false)),
        ] {
            let err = result
                .err()
                .expect("missing explicit reference must fail before request preparation");
            assert_eq!(
                err.to_string(),
                "Environment variable for 'api_base' is missing or empty"
            );
            assert!(!err.to_string().contains(MISSING_API_BASE_ENV));
        }
    }

    #[test]
    fn absent_api_base_keeps_public_gemini_fallbacks() {
        let client = request_client(None);
        let chat = prepare_chat_completions(&client, completion_data()).unwrap();
        let embeddings =
            prepare_embeddings(&client, &EmbeddingsData::new(vec!["hello".into()], false)).unwrap();

        assert_eq!(
            chat.url,
            "https://generativelanguage.googleapis.com/v1beta/models/test-model:generateContent"
        );
        assert_eq!(
            embeddings.url,
            "https://generativelanguage.googleapis.com/v1beta/models/test-model:batchEmbedContents?key=test-key"
        );
    }
}
