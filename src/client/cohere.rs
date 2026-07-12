use super::openai::*;
use super::openai_compatible::*;
use super::*;

use anyhow::{bail, Context, Result};
use reqwest::RequestBuilder;
use serde::Deserialize;
use serde_json::{json, Value};

const API_BASE: &str = "https://api.cohere.ai/v2";

#[derive(Debug, Clone, Deserialize, Default)]
pub struct CohereConfig {
    pub name: Option<String>,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    #[serde(default)]
    pub models: Vec<ModelData>,
    pub patch: Option<RequestPatch>,
    pub extra: Option<ExtraConfig>,
}

impl CohereClient {
    config_get_fn!(api_key, get_api_key);
    config_get_fn!(api_base, get_api_base);

    pub const PROMPTS: [PromptAction<'static>; 1] = [("api_key", "API Key", None)];
}

impl_client_trait!(
    CohereClient,
    (
        prepare_chat_completions,
        cohere_chat_completions,
        cohere_chat_events
    ),
    (prepare_embeddings, embeddings),
    (prepare_rerank, generic_rerank),
);

fn prepare_chat_completions(
    self_: &CohereClient,
    data: ChatCompletionsData,
) -> Result<RequestData> {
    let api_key = self_.get_api_key()?;
    let api_base =
        optional_config_field(self_.get_api_base())?.unwrap_or_else(|| API_BASE.to_string());

    let url = format!("{}/chat", api_base.trim_end_matches('/'));
    let body = cohere_build_chat_completions_body(data, &self_.model);

    let mut request_data = RequestData::new(url, body);

    request_data.bearer_auth(api_key);

    Ok(request_data)
}

fn prepare_embeddings(self_: &CohereClient, data: &EmbeddingsData) -> Result<RequestData> {
    let api_key = self_.get_api_key()?;
    let api_base =
        optional_config_field(self_.get_api_base())?.unwrap_or_else(|| API_BASE.to_string());

    let url = format!("{}/embed", api_base.trim_end_matches('/'));

    let input_type = match data.query {
        true => "search_query",
        false => "search_document",
    };

    let body = json!({
        "model": self_.model.real_name(),
        "texts": data.texts,
        "input_type": input_type,
        "embedding_types": ["float"],
    });

    let mut request_data = RequestData::new(url, body);

    request_data.bearer_auth(api_key);

    Ok(request_data)
}

fn prepare_rerank(self_: &CohereClient, data: &RerankData) -> Result<RequestData> {
    let api_key = self_.get_api_key()?;
    let api_base =
        optional_config_field(self_.get_api_base())?.unwrap_or_else(|| API_BASE.to_string());

    let url = format!("{}/rerank", api_base.trim_end_matches('/'));
    let body = generic_build_rerank_body(data, &self_.model);

    let mut request_data = RequestData::new(url, body);

    request_data.bearer_auth(api_key);

    Ok(request_data)
}

pub fn cohere_build_chat_completions_body(data: ChatCompletionsData, model: &Model) -> Value {
    let mut body = openai_build_chat_completions_body(data, model);
    if let Some(obj) = body.as_object_mut() {
        if let Some(top_p) = obj.remove("top_p") {
            obj.insert("p".to_string(), top_p);
        }
    }
    body
}

pub async fn cohere_chat_completions(
    builder: RequestBuilder,
    _model: &Model,
) -> Result<ChatCompletionsOutput> {
    let res = builder.send().await?;
    let status = res.status();
    let data: Value = res.json().await?;
    if !status.is_success() {
        catch_error(&data, status.as_u16())?;
    }

    debug!("non-stream-data: {data}");
    extract_chat_completions(&data)
}

#[derive(Debug, Default)]
struct CohereStreamState {
    tool_call: Option<PendingCohereToolCall>,
    completed_tool_calls: Vec<ToolCall>,
}

impl CohereStreamState {
    fn finish_tool_call(&mut self, index: u64) -> Result<()> {
        let tool_call = self
            .tool_call
            .take()
            .context("Cohere tool-call-end arrived without tool-call-start")?;
        if tool_call.index != index {
            bail!(
                "Cohere tool call {} ended by index {index}",
                tool_call.index
            );
        }
        self.completed_tool_calls.push(tool_call.into_tool_call()?);
        Ok(())
    }

    fn finish_message(&mut self, events: &mut Vec<ChatEvent>, terminal: &str) -> Result<()> {
        if let Some(tool_call) = self.tool_call.as_ref() {
            bail!(
                "Cohere {terminal} arrived before tool-call-end for index {}",
                tool_call.index
            );
        }
        for tool_call in std::mem::take(&mut self.completed_tool_calls) {
            events.push(ChatEvent::ToolCall(tool_call));
        }
        Ok(())
    }
}

#[derive(Debug)]
struct PendingCohereToolCall {
    index: u64,
    id: String,
    name: String,
    arguments: String,
}

impl PendingCohereToolCall {
    fn into_tool_call(self) -> Result<ToolCall> {
        let arguments: Value = self.arguments.parse().with_context(|| {
            format!(
                "Tool call '{}' have non-JSON arguments '{}'",
                self.name, self.arguments
            )
        })?;
        Ok(ToolCall::new(self.name, arguments, Some(self.id)))
    }
}

fn handle_cohere_stream_message(
    state: &mut CohereStreamState,
    events: &mut Vec<ChatEvent>,
    message: &str,
) -> Result<bool> {
    if message == "[DONE]" {
        state.finish_message(events, "[DONE]")?;
        return Ok(true);
    }

    let data: Value = serde_json::from_str(message).context("Invalid Cohere streaming response")?;
    debug!("stream-data: {data}");
    let Some(typ) = data["type"].as_str() else {
        return Ok(false);
    };

    match typ {
        "content-delta" => {
            if let Some(text) = data["delta"]["message"]["content"]["text"].as_str() {
                events.push(ChatEvent::Text(text.to_string()));
            }
        }
        "tool-plan-delta" => {
            if let Some(text) = data["delta"]["message"]["tool_plan"].as_str() {
                events.push(ChatEvent::Text(text.to_string()));
            }
        }
        "tool-call-start" => {
            if let Some(tool_call) = state.tool_call.as_ref() {
                bail!(
                    "Cohere started a new tool call before tool-call-end for index {}",
                    tool_call.index
                );
            }
            let index = data["index"]
                .as_u64()
                .context("Cohere tool-call-start is missing an index")?;
            let call = &data["delta"]["message"]["tool_calls"];
            let id = call["id"]
                .as_str()
                .context("Cohere tool-call-start is missing an id")?;
            let name = call["function"]["name"]
                .as_str()
                .context("Cohere tool-call-start is missing a function name")?;
            state.tool_call = Some(PendingCohereToolCall {
                index,
                id: id.to_string(),
                name: name.to_string(),
                arguments: String::new(),
            });
        }
        "tool-call-delta" => {
            let index = data["index"]
                .as_u64()
                .context("Cohere tool-call-delta is missing an index")?;
            let tool_call = state
                .tool_call
                .as_mut()
                .context("Cohere tool-call-delta arrived without tool-call-start")?;
            if tool_call.index != index {
                bail!(
                    "Cohere tool call {} received a delta for index {index}",
                    tool_call.index
                );
            }
            let arguments = data["delta"]["message"]["tool_calls"]["function"]["arguments"]
                .as_str()
                .context("Cohere tool-call-delta is missing arguments")?;
            tool_call.arguments.push_str(arguments);
        }
        "tool-call-end" => {
            let index = data["index"]
                .as_u64()
                .context("Cohere tool-call-end is missing an index")?;
            state.finish_tool_call(index)?;
        }
        "message-end" => {
            let finish_reason = data["delta"]["finish_reason"]
                .as_str()
                .context("Cohere message-end is missing a finish_reason")?;
            if matches!(finish_reason, "ERROR" | "TIMEOUT") {
                bail!("Cohere streaming request ended with {finish_reason}");
            }
            if !state.completed_tool_calls.is_empty() && finish_reason != "TOOL_CALL" {
                bail!(
                    "Cohere message-end used finish_reason {finish_reason} for completed tool calls"
                );
            }
            state.finish_message(events, "message-end")?;
            return Ok(true);
        }
        _ => {}
    }
    Ok(false)
}

pub async fn cohere_chat_events(
    builder: RequestBuilder,
    _model: &Model,
) -> Result<ChatEventStream> {
    let mut state = CohereStreamState::default();
    Ok(sse_chat_events(builder, move |message, events| {
        handle_cohere_stream_message(&mut state, events, &message.data)
    }))
}

#[cfg(test)]
async fn stream_into_handler(
    builder: RequestBuilder,
    handler: &mut SseHandler,
    model: &Model,
) -> Result<()> {
    drive_chat_events(cohere_chat_events(builder, model).await?, handler).await
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
    Ok(res_body.embeddings.float)
}

#[derive(Deserialize)]
struct EmbeddingsResBody {
    embeddings: EmbeddingsResBodyEmbeddings,
}

#[derive(Deserialize)]
struct EmbeddingsResBodyEmbeddings {
    float: Vec<Vec<f32>>,
}

fn extract_chat_completions(data: &Value) -> Result<ChatCompletionsOutput> {
    let mut text = data["message"]["content"][0]["text"]
        .as_str()
        .unwrap_or_default()
        .to_string();

    let mut tool_calls = vec![];
    if let Some(calls) = data["message"]["tool_calls"].as_array() {
        if text.is_empty() {
            if let Some(tool_plain) = data["message"]["tool_plan"].as_str() {
                text = tool_plain.to_string();
            }
        }
        for call in calls {
            if let (Some(name), Some(arguments), Some(id)) = (
                call["function"]["name"].as_str(),
                call["function"]["arguments"].as_str(),
                call["id"].as_str(),
            ) {
                let arguments: Value = arguments.parse().with_context(|| {
                    format!("Tool call '{name}' have non-JSON arguments '{arguments}'")
                })?;
                tool_calls.push(ToolCall::new(
                    name.to_string(),
                    arguments,
                    Some(id.to_string()),
                ));
            }
        }
    }

    if text.is_empty() && tool_calls.is_empty() {
        bail!("Invalid response data: {data}");
    }
    let output = ChatCompletionsOutput {
        text,
        tool_calls,
        id: data["id"].as_str().map(|v| v.to_string()),
        input_tokens: data["usage"]["billed_units"]["input_tokens"].as_u64(),
        output_tokens: data["usage"]["billed_units"]["output_tokens"].as_u64(),
    };
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::utils::create_abort_signal;
    use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};

    const MISSING_API_BASE_ENV: &str = "AICHAT_TEST_MISSING_COHERE_API_BASE_04BB6F78";

    fn test_handler() -> (SseHandler, UnboundedReceiver<SseEvent>) {
        let (tx, rx) = unbounded_channel();
        (SseHandler::new(tx, create_abort_signal()), rx)
    }

    fn official_sse_body(events: &[Value]) -> String {
        events
            .iter()
            .map(|event| {
                let typ = event["type"].as_str().expect("fixture event type");
                format!("event: {typ}\ndata: {event}\n\n")
            })
            .collect()
    }

    fn tool_call_events() -> [Value; 4] {
        [
            json!({
                "type":"message-start",
                "id":"message-tool",
                "delta":{"message":{"role":"assistant","content":[],"tool_plan":"","tool_calls":[],"citations":[]}}
            }),
            json!({
                "type":"tool-call-start",
                "index":0,
                "delta":{"message":{"tool_calls":{
                    "id":"call-side-effect",
                    "type":"function",
                    "function":{"name":"side_effect","arguments":""}
                }}}
            }),
            json!({
                "type":"tool-call-delta",
                "index":0,
                "delta":{"message":{"tool_calls":{"function":{"arguments":"{}"}}}}
            }),
            json!({"type":"tool-call-end","index":0}),
        ]
    }

    fn request_client(api_base: Option<String>) -> CohereClient {
        CohereClient {
            global_config: Default::default(),
            config: CohereConfig {
                name: Some("cohere-remediation-test".into()),
                api_key: Some("test-key".into()),
                api_base,
                ..Default::default()
            },
            model: Model::new("cohere-remediation-test", "test-model"),
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

    fn preparation_results(client: &CohereClient) -> [Result<RequestData>; 3] {
        [
            prepare_chat_completions(client, completion_data()),
            prepare_embeddings(client, &EmbeddingsData::new(vec!["hello".into()], false)),
            prepare_rerank(
                client,
                &RerankData::new("query".into(), vec!["document".into()], 1),
            ),
        ]
    }

    #[test]
    fn missing_explicit_api_base_does_not_fall_back_to_public_cohere() {
        assert!(std::env::var_os(MISSING_API_BASE_ENV).is_none());
        let client = request_client(Some(format!("${MISSING_API_BASE_ENV}")));

        for result in preparation_results(&client) {
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
    fn absent_api_base_keeps_public_cohere_fallbacks() {
        let [chat, embeddings, rerank] = preparation_results(&request_client(None));

        assert_eq!(chat.unwrap().url, "https://api.cohere.ai/v2/chat");
        assert_eq!(embeddings.unwrap().url, "https://api.cohere.ai/v2/embed");
        assert_eq!(rerank.unwrap().url, "https://api.cohere.ai/v2/rerank");
    }

    #[test]
    fn content_and_tool_plan_deltas_become_text_events() -> Result<()> {
        let mut events = Vec::new();
        let mut state = CohereStreamState::default();

        handle_cohere_stream_message(
            &mut state,
            &mut events,
            &json!({
                "type":"tool-plan-delta",
                "delta":{"message":{"tool_plan":"plan "}}
            })
            .to_string(),
        )?;
        handle_cohere_stream_message(
            &mut state,
            &mut events,
            &json!({
                "type":"content-delta",
                "index":0,
                "delta":{"message":{"content":{"text":"hello"}}}
            })
            .to_string(),
        )?;
        let done = handle_cohere_stream_message(
            &mut state,
            &mut events,
            &json!({
                "type":"message-end",
                "delta":{"finish_reason":"COMPLETE"}
            })
            .to_string(),
        )?;

        assert!(done);
        assert_eq!(
            events,
            [
                ChatEvent::Text("plan ".into()),
                ChatEvent::Text("hello".into()),
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn official_content_stream_completes_on_message_end() -> Result<()> {
        let body = official_sse_body(&[
            json!({
                "type":"message-start",
                "id":"message-content",
                "delta":{"message":{"role":"assistant"}}
            }),
            json!({
                "type":"content-start",
                "index":0,
                "delta":{"message":{"content":{"text":"","type":"text"}}}
            }),
            json!({
                "type":"content-delta",
                "index":0,
                "delta":{"message":{"content":{"text":"hello"}}}
            }),
            json!({"type":"content-end","index":0}),
            json!({
                "type":"message-end",
                "delta":{
                    "finish_reason":"COMPLETE",
                    "usage":{"billed_units":{"input_tokens":1,"output_tokens":1}}
                }
            }),
        ]);
        let builder = sse_fixture_builder(&body).await?;
        let (mut handler, _rx) = test_handler();

        stream_into_handler(builder, &mut handler, &Model::new("cohere", "test-model")).await?;

        let (text, calls) = handler.take();
        assert_eq!(text, "hello");
        assert!(calls.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn official_tool_stream_dispatches_once_at_message_end() -> Result<()> {
        let mut events = tool_call_events().to_vec();
        events.push(json!({
            "type":"message-end",
            "delta":{
                "finish_reason":"TOOL_CALL",
                "usage":{"billed_units":{"input_tokens":1,"output_tokens":1}}
            }
        }));
        let builder = sse_fixture_builder(&official_sse_body(&events)).await?;
        let (mut handler, _rx) = test_handler();

        stream_into_handler(builder, &mut handler, &Model::new("cohere", "test-model")).await?;

        let (text, calls) = handler.take();
        assert!(text.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "side_effect");
        assert_eq!(calls[0].arguments, json!({}));
        Ok(())
    }

    #[tokio::test]
    async fn eof_before_message_end_does_not_dispatch_stopped_tool_call() -> Result<()> {
        let builder = sse_fixture_builder(&official_sse_body(&tool_call_events())).await?;
        let (mut handler, mut rx) = test_handler();

        let err = stream_into_handler(builder, &mut handler, &Model::new("cohere", "test-model"))
            .await
            .expect_err("EOF before message-end must fail");

        assert_eq!(
            err.to_string(),
            "SSE stream ended before protocol completion"
        );
        assert!(handler.tool_calls().is_empty());
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        let (text, calls) = handler.take();
        assert!(text.is_empty());
        assert!(calls.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn message_end_before_tool_call_end_fails_closed() -> Result<()> {
        let mut events = tool_call_events()[..3].to_vec();
        events.push(json!({
            "type":"message-end",
            "delta":{"finish_reason":"TOOL_CALL"}
        }));
        let builder = sse_fixture_builder(&official_sse_body(&events)).await?;
        let (mut handler, _rx) = test_handler();

        let err = stream_into_handler(builder, &mut handler, &Model::new("cohere", "test-model"))
            .await
            .expect_err("message-end before tool-call-end must fail");

        assert_eq!(
            err.to_string(),
            "Cohere message-end arrived before tool-call-end for index 0"
        );
        assert!(handler.tool_calls().is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn legacy_done_marker_remains_a_completion_signal() -> Result<()> {
        let builder = sse_fixture_builder("data: [DONE]\n\n").await?;
        let (mut handler, _rx) = test_handler();

        stream_into_handler(builder, &mut handler, &Model::new("cohere", "test-model")).await?;

        let (text, calls) = handler.take();
        assert!(text.is_empty());
        assert!(calls.is_empty());
        Ok(())
    }
}
