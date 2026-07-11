use super::*;

use crate::utils::strip_think_tag;

use anyhow::{bail, Context, Result};
use indexmap::IndexMap;
use reqwest::RequestBuilder;
use serde::Deserialize;
use serde_json::{json, Value};

const API_BASE: &str = "https://api.openai.com/v1";

#[derive(Debug, Clone, Deserialize, Default)]
pub struct OpenAIConfig {
    pub name: Option<String>,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub organization_id: Option<String>,
    #[serde(default)]
    pub models: Vec<ModelData>,
    pub patch: Option<RequestPatch>,
    pub extra: Option<ExtraConfig>,
}

impl OpenAIClient {
    config_get_fn!(api_key, get_api_key);
    config_get_fn!(api_base, get_api_base);

    pub const PROMPTS: [PromptAction<'static>; 1] = [("api_key", "API Key", None)];
}

impl_client_trait!(
    OpenAIClient,
    (
        prepare_chat_completions,
        openai_chat_completions,
        openai_chat_completions_streaming
    ),
    (prepare_embeddings, openai_embeddings),
    (noop_prepare_rerank, noop_rerank),
);

fn prepare_chat_completions(
    self_: &OpenAIClient,
    data: ChatCompletionsData,
) -> Result<RequestData> {
    let api_key = self_.get_api_key()?;
    let api_base =
        optional_config_field(self_.get_api_base())?.unwrap_or_else(|| API_BASE.to_string());

    let url = format!("{}/chat/completions", api_base.trim_end_matches('/'));

    let body = openai_build_chat_completions_body(data, &self_.model);

    let mut request_data = RequestData::new(url, body);

    request_data.bearer_auth(api_key);
    if let Some(organization_id) = &self_.config.organization_id {
        request_data.header("OpenAI-Organization", organization_id);
    }

    Ok(request_data)
}

fn prepare_embeddings(self_: &OpenAIClient, data: &EmbeddingsData) -> Result<RequestData> {
    let api_key = self_.get_api_key()?;
    let api_base =
        optional_config_field(self_.get_api_base())?.unwrap_or_else(|| API_BASE.to_string());

    let url = format!("{api_base}/embeddings");

    let body = openai_build_embeddings_body(data, &self_.model);

    let mut request_data = RequestData::new(url, body);

    request_data.bearer_auth(api_key);
    if let Some(organization_id) = &self_.config.organization_id {
        request_data.header("OpenAI-Organization", organization_id);
    }

    Ok(request_data)
}

pub async fn openai_chat_completions(
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
    openai_extract_chat_completions(&data)
}

#[derive(Debug, Default)]
struct OpenAiStreamState {
    tool_calls: OpenAiToolCallAccumulator,
    reasoning: bool,
}

impl OpenAiStreamState {
    fn finish_tool_calls(&mut self, handler: &mut SseHandler) -> Result<()> {
        let calls = std::mem::take(&mut self.tool_calls).finish()?;
        for call in calls {
            handler.tool_call(call)?;
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
struct OpenAiToolCallAccumulator {
    completed: Vec<PendingOpenAiToolCall>,
    active: IndexMap<u64, PendingOpenAiToolCall>,
}

impl OpenAiToolCallAccumulator {
    fn push_delta(&mut self, index: u64, id: Option<&str>, function: Option<&serde_json::Map<String, Value>>) {
        // OpenAI sends an id only in the first delta. A different non-empty id on the
        // same index is the only reliable boundary for providers that reuse indexes.
        let starts_new_call = self.active.get(&index).is_some_and(|pending| {
            id.is_some_and(|id| !pending.id.is_empty() && pending.id != id)
        });
        if starts_new_call {
            if let Some(previous) = self.active.shift_remove(&index) {
                self.completed.push(previous);
            }
        }

        let pending = self.active.entry(index).or_default();
        if pending.id.is_empty() {
            if let Some(id) = id {
                pending.id = id.to_string();
            }
        }
        if let Some(function) = function {
            pending.push_function_delta(function);
        }
    }

    fn finish(mut self) -> Result<Vec<ToolCall>> {
        self.completed.extend(self.active.into_values());

        let mut calls = Vec::with_capacity(self.completed.len());
        for pending in self.completed {
            if let Some(call) = pending.into_tool_call()? {
                calls.push(call);
            }
        }
        Ok(calls)
    }
}

#[derive(Debug, Default)]
struct PendingOpenAiToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl PendingOpenAiToolCall {
    fn push_function_delta(&mut self, function: &serde_json::Map<String, Value>) {
        if let Some(name) = function.get("name").and_then(Value::as_str) {
            if name.starts_with(&self.name) {
                self.name = name.to_string();
            } else {
                self.name.push_str(name);
            }
        }
        if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
            self.arguments.push_str(arguments);
        }
    }

    fn into_tool_call(self) -> Result<Option<ToolCall>> {
        if self.name.is_empty() {
            return Ok(None);
        }
        let arguments = if self.arguments.is_empty() {
            json!({})
        } else {
            self.arguments.parse().with_context(|| {
                format!(
                    "Tool call '{}' has non-JSON arguments '{}'",
                    self.name, self.arguments
                )
            })?
        };
        Ok(Some(ToolCall::new(
            self.name,
            arguments,
            normalize_function_id(&self.id),
        )))
    }
}

fn handle_openai_stream_message(
    state: &mut OpenAiStreamState,
    handler: &mut SseHandler,
    message: &str,
) -> Result<bool> {
    if message == "[DONE]" {
        state.finish_tool_calls(handler)?;
        return Ok(true);
    }

    let data: Value = serde_json::from_str(message).context("Invalid OpenAI streaming response")?;
    debug!("stream-data: {data}");
    if let Some(text) = data["choices"][0]["delta"]["content"]
        .as_str()
        .filter(|v| !v.is_empty())
    {
        if state.reasoning {
            handler.text("\n</think>\n\n")?;
            state.reasoning = false;
        }
        handler.text(text)?;
    } else if let Some(text) = data["choices"][0]["delta"]["reasoning_content"]
        .as_str()
        .or_else(|| data["choices"][0]["delta"]["reasoning"].as_str())
        .filter(|v| !v.is_empty())
    {
        if !state.reasoning {
            handler.text("<think>\n")?;
            state.reasoning = true;
        }
        handler.text(text)?;
    }

    if let Some(tool_calls) = data["choices"][0]["delta"]["tool_calls"].as_array() {
        if !tool_calls.is_empty() && state.reasoning {
            handler.text("\n</think>\n\n")?;
            state.reasoning = false;
        }
        for (position, call) in tool_calls.iter().enumerate() {
            let index = call["index"].as_u64().unwrap_or(position as u64);
            let id = call["id"].as_str().filter(|v| !v.is_empty());
            state
                .tool_calls
                .push_delta(index, id, call["function"].as_object());
        }
    }
    Ok(false)
}

pub async fn openai_chat_completions_streaming(
    builder: RequestBuilder,
    handler: &mut SseHandler,
    _model: &Model,
) -> Result<()> {
    let mut state = OpenAiStreamState::default();
    sse_stream(builder, |message: SseMmessage| {
        handle_openai_stream_message(&mut state, handler, &message.data)
    })
    .await
}

pub async fn openai_embeddings(
    builder: RequestBuilder,
    _model: &Model,
) -> Result<EmbeddingsOutput> {
    let res = builder.send().await?;
    let status = res.status();
    let data: Value = res.json().await?;
    if !status.is_success() {
        catch_error(&data, status.as_u16())?;
    }
    let res_body: EmbeddingsResBody =
        serde_json::from_value(data).context("Invalid embeddings data")?;
    let output = res_body.data.into_iter().map(|v| v.embedding).collect();
    Ok(output)
}

#[derive(Deserialize)]
struct EmbeddingsResBody {
    data: Vec<EmbeddingsResBodyEmbedding>,
}

#[derive(Deserialize)]
struct EmbeddingsResBodyEmbedding {
    embedding: Vec<f32>,
}

pub fn openai_build_chat_completions_body(data: ChatCompletionsData, model: &Model) -> Value {
    let ChatCompletionsData {
        messages,
        temperature,
        top_p,
        functions,
        stream,
    } = data;

    let messages_len = messages.len();
    let messages: Vec<Value> = messages
        .into_iter()
        .enumerate()
        .flat_map(|(i, message)| {
            let Message { role, content } = message;
            match content {
                MessageContent::ToolCalls(MessageContentToolCalls {
                    tool_results,
                    text: _,
                    sequence,
                }) => {
                    if !sequence {
                        let tool_calls: Vec<_> = tool_results
                            .iter()
                            .map(|tool_result| {
                                json!({
                                    "id": tool_result.call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_result.call.name,
                                        "arguments": tool_result.call.arguments.to_string(),
                                    },
                                })
                            })
                            .collect();
                        let mut messages = vec![
                            json!({ "role": MessageRole::Assistant, "tool_calls": tool_calls }),
                        ];
                        for tool_result in tool_results {
                            messages.push(json!({
                                "role": "tool",
                                "content": tool_result.output.to_string(),
                                "tool_call_id": tool_result.call.id,
                            }));
                        }
                        messages
                    } else {
                        tool_results.into_iter().flat_map(|tool_result| {
                            vec![
                                json!({
                                    "role": MessageRole::Assistant,
                                    "tool_calls": [
                                        {
                                            "id": tool_result.call.id,
                                            "type": "function",
                                            "function": {
                                                "name": tool_result.call.name,
                                                "arguments": tool_result.call.arguments.to_string(),
                                            },
                                        }
                                    ]
                                }),
                                json!({
                                    "role": "tool",
                                    "content": tool_result.output.to_string(),
                                    "tool_call_id": tool_result.call.id,
                                })
                            ]

                        }).collect()
                    }
                }
                MessageContent::Text(text) if role.is_assistant() && i != messages_len - 1 => {
                    vec![json!({ "role": role, "content": strip_think_tag(&text) }
                    )]
                }
                _ => vec![json!({ "role": role, "content": content })],
            }
        })
        .collect();

    let mut body = json!({
        "model": &model.real_name(),
        "messages": messages,
    });

    if let Some(v) = model.max_tokens_param() {
        if model
            .patch()
            .and_then(|v| v.get("body").and_then(|v| v.get("max_tokens")))
            == Some(&Value::Null)
        {
            body["max_completion_tokens"] = v.into();
        } else {
            body["max_tokens"] = v.into();
        }
    }
    if let Some(v) = temperature {
        body["temperature"] = v.into();
    }
    if let Some(v) = top_p {
        body["top_p"] = v.into();
    }
    body["stream"] = stream.into();
    if let Some(functions) = functions {
        body["tools"] = functions
            .iter()
            .map(|v| {
                json!({
                    "type": "function",
                    "function": v,
                })
            })
            .collect();
    }
    body
}

pub fn openai_build_embeddings_body(data: &EmbeddingsData, model: &Model) -> Value {
    json!({
        "input": data.texts,
        "model": model.real_name()
    })
}

pub fn openai_extract_chat_completions(data: &Value) -> Result<ChatCompletionsOutput> {
    let text = data["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();

    let reasoning = data["choices"][0]["message"]["reasoning_content"]
        .as_str()
        .or_else(|| data["choices"][0]["message"]["reasoning"].as_str())
        .unwrap_or_default()
        .trim();

    let mut tool_calls = vec![];
    if let Some(calls) = data["choices"][0]["message"]["tool_calls"].as_array() {
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
    };

    if text.is_empty() && tool_calls.is_empty() {
        bail!("Invalid response data: {data}");
    }
    let text = if !reasoning.is_empty() {
        format!("<think>\n{reasoning}\n</think>\n\n{text}")
    } else {
        text.to_string()
    };
    let output = ChatCompletionsOutput {
        text,
        tool_calls,
        id: data["id"].as_str().map(|v| v.to_string()),
        input_tokens: data["usage"]["prompt_tokens"].as_u64(),
        output_tokens: data["usage"]["completion_tokens"].as_u64(),
    };
    Ok(output)
}

fn normalize_function_id(value: &str) -> Option<String> {
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::utils::create_abort_signal;
    use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};

    const MISSING_API_BASE_ENV: &str = "AICHAT_TEST_MISSING_OPENAI_API_BASE_8CDB63A4";

    fn test_handler() -> (SseHandler, UnboundedReceiver<SseEvent>) {
        let (tx, rx) = unbounded_channel();
        (SseHandler::new(tx, create_abort_signal()), rx)
    }

    fn handle_value(
        state: &mut OpenAiStreamState,
        handler: &mut SseHandler,
        value: Value,
    ) -> Result<bool> {
        handle_openai_stream_message(state, handler, &value.to_string())
    }

    fn completion_data(stream: bool) -> ChatCompletionsData {
        ChatCompletionsData {
            messages: vec![Message::new(
                MessageRole::User,
                MessageContent::Text("hello".to_string()),
            )],
            temperature: None,
            top_p: None,
            functions: None,
            stream,
        }
    }

    fn request_client(api_base: Option<String>) -> OpenAIClient {
        OpenAIClient {
            global_config: Default::default(),
            config: OpenAIConfig {
                name: Some("openai-remediation-test".into()),
                api_key: Some("test-key".into()),
                api_base,
                ..Default::default()
            },
            model: Model::new("openai-remediation-test", "test-model"),
        }
    }

    fn assert_api_base_reference_error(result: Result<RequestData>) {
        let err = result
            .err()
            .expect("missing explicit reference must fail before request preparation");
        assert_eq!(
            err.to_string(),
            "Environment variable for 'api_base' is missing or empty"
        );
        assert!(!err.to_string().contains(MISSING_API_BASE_ENV));
    }

    #[test]
    fn missing_explicit_api_base_does_not_fall_back_to_public_openai() {
        assert!(std::env::var_os(MISSING_API_BASE_ENV).is_none());
        let client = request_client(Some(format!("${MISSING_API_BASE_ENV}")));

        assert_api_base_reference_error(prepare_chat_completions(
            &client,
            completion_data(false),
        ));
        assert_api_base_reference_error(prepare_embeddings(
            &client,
            &EmbeddingsData::new(vec!["hello".into()], false),
        ));
    }

    #[test]
    fn absent_api_base_keeps_public_openai_fallbacks() {
        let client = request_client(None);

        let chat = prepare_chat_completions(&client, completion_data(false)).unwrap();
        let embeddings = prepare_embeddings(
            &client,
            &EmbeddingsData::new(vec!["hello".into()], false),
        )
        .unwrap();

        assert_eq!(chat.url, "https://api.openai.com/v1/chat/completions");
        assert_eq!(
            embeddings.url,
            "https://api.openai.com/v1/embeddings"
        );
    }

    #[test]
    fn chat_body_always_serializes_stream_boolean() {
        let model = Model::new("openai", "test-model");

        let non_streaming = openai_build_chat_completions_body(completion_data(false), &model);
        let streaming = openai_build_chat_completions_body(completion_data(true), &model);

        assert_eq!(non_streaming["stream"], json!(false));
        assert_eq!(streaming["stream"], json!(true));
    }

    #[test]
    fn tool_call_keeps_arguments_when_continuations_omit_id() -> Result<()> {
        let (mut handler, _rx) = test_handler();
        let mut state = OpenAiStreamState::default();

        handle_value(
            &mut state,
            &mut handler,
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_long_identifier","function":{"name":"web_search","arguments":""}}]}}]}),
        )?;
        handle_value(
            &mut state,
            &mut handler,
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"query\":\""}}]}}]}),
        )?;
        handle_value(
            &mut state,
            &mut handler,
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Rust\"}"}}]}}]}),
        )?;
        assert!(handle_openai_stream_message(
            &mut state,
            &mut handler,
            "[DONE]"
        )?);

        let (_, calls) = handler.take();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[0].id.as_deref(), Some("call_long_identifier"));
        assert_eq!(calls[0].arguments, json!({"query":"Rust"}));
        Ok(())
    }

    #[tokio::test]
    async fn truncated_sse_does_not_dispatch_pending_tool_call() -> Result<()> {
        let tool_delta = json!({
            "choices":[{"delta":{"tool_calls":[{
                "index":0,
                "id":"call_side_effect",
                "function":{"name":"side_effect","arguments":"{}"}
            }]}}]
        });
        let builder = sse_fixture_builder(&format!("data: {tool_delta}\n\n")).await?;
        let (mut handler, mut rx) = test_handler();

        let err = openai_chat_completions_streaming(
            builder,
            &mut handler,
            &Model::new("openai", "test-model"),
        )
        .await
        .expect_err("EOF without [DONE] must fail");

        assert_eq!(err.to_string(), "SSE stream ended before protocol completion");
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
    async fn done_marker_dispatches_completed_tool_call_once() -> Result<()> {
        let tool_delta = json!({
            "choices":[{"delta":{"tool_calls":[{
                "index":0,
                "id":"call_complete",
                "function":{"name":"side_effect","arguments":"{}"}
            }]}}]
        });
        let builder = sse_fixture_builder(&format!(
            "data: {tool_delta}\n\ndata: [DONE]\n\n"
        ))
        .await?;
        let (mut handler, _rx) = test_handler();

        openai_chat_completions_streaming(
            builder,
            &mut handler,
            &Model::new("openai", "test-model"),
        )
        .await?;

        let (text, calls) = handler.take();
        assert!(text.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "side_effect");
        assert_eq!(calls[0].arguments, json!({}));
        Ok(())
    }

    #[test]
    fn tool_call_boundaries_support_shorter_ids_and_reused_or_incrementing_indexes() -> Result<()> {
        let (mut handler, _rx) = test_handler();
        let mut state = OpenAiStreamState::default();

        for value in [
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_long_identifier","function":{"name":"first","arguments":"{\"value\":1}"}}]}}]}),
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"x","function":{"name":"second","arguments":"{\"value\":2}"}}]}}]}),
            json!({"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_three","function":{"name":"third","arguments":"{\"value\":3}"}}]}}]}),
        ] {
            handle_value(&mut state, &mut handler, value)?;
        }
        handle_openai_stream_message(&mut state, &mut handler, "[DONE]")?;

        let (_, calls) = handler.take();
        assert_eq!(
            calls.iter().map(|call| call.name.as_str()).collect::<Vec<_>>(),
            ["first", "second", "third"]
        );
        assert_eq!(
            calls
                .iter()
                .map(|call| call.id.as_deref())
                .collect::<Vec<_>>(),
            [Some("call_long_identifier"), Some("x"), Some("call_three")]
        );
        assert_eq!(calls[1].arguments, json!({"value":2}));
        Ok(())
    }

    #[test]
    fn tool_call_accumulator_handles_multiple_calls_per_delta() -> Result<()> {
        let (mut handler, _rx) = test_handler();
        let mut state = OpenAiStreamState::default();

        handle_value(
            &mut state,
            &mut handler,
            json!({"choices":[{"delta":{"tool_calls":[
                {"index":0,"id":"call_a","function":{"name":"alpha","arguments":"{\"a\":"}},
                {"index":1,"id":"call_b","function":{"name":"beta","arguments":"{\"b\":"}}
            ]}}]}),
        )?;
        handle_value(
            &mut state,
            &mut handler,
            json!({"choices":[{"delta":{"tool_calls":[
                {"index":0,"function":{"arguments":"1}"}},
                {"index":1,"function":{"arguments":"2}"}}
            ]}}]}),
        )?;
        handle_openai_stream_message(&mut state, &mut handler, "[DONE]")?;

        let (_, calls) = handler.take();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments, json!({"a":1}));
        assert_eq!(calls[1].arguments, json!({"b":2}));
        Ok(())
    }

    #[test]
    fn invalid_tool_arguments_fail_without_dispatching_empty_object() -> Result<()> {
        let (mut handler, _rx) = test_handler();
        let mut state = OpenAiStreamState::default();

        handle_value(
            &mut state,
            &mut handler,
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_bad","function":{"name":"broken","arguments":"{\"value\":"}}]}}]}),
        )?;
        let err = handle_openai_stream_message(&mut state, &mut handler, "[DONE]")
            .expect_err("incomplete JSON must fail");

        assert!(err
            .to_string()
            .contains("Tool call 'broken' has non-JSON arguments"));
        let (_, calls) = handler.take();
        assert!(calls.is_empty());
        Ok(())
    }

    #[test]
    fn successful_text_and_reasoning_stream_is_unchanged() -> Result<()> {
        let (mut handler, _rx) = test_handler();
        let mut state = OpenAiStreamState::default();

        handle_value(
            &mut state,
            &mut handler,
            json!({"choices":[{"delta":{"reasoning_content":"reason"}}]}),
        )?;
        handle_value(
            &mut state,
            &mut handler,
            json!({"choices":[{"delta":{"content":"answer"}}]}),
        )?;
        handle_openai_stream_message(&mut state, &mut handler, "[DONE]")?;

        let (text, calls) = handler.take();
        assert_eq!(text, "<think>\nreason\n</think>\n\nanswer");
        assert!(calls.is_empty());
        Ok(())
    }
}
