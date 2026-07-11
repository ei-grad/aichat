use super::*;

use crate::utils::strip_think_tag;

use anyhow::{bail, Context, Result};
use reqwest::RequestBuilder;
use serde::Deserialize;
use serde_json::{json, Value};

const API_BASE: &str = "https://api.anthropic.com/v1";

#[derive(Debug, Clone, Deserialize)]
pub struct ClaudeConfig {
    pub name: Option<String>,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    #[serde(default)]
    pub models: Vec<ModelData>,
    pub patch: Option<RequestPatch>,
    pub extra: Option<ExtraConfig>,
}

impl ClaudeClient {
    config_get_fn!(api_key, get_api_key, ["ANTHROPIC_API_KEY"]);
    config_get_fn!(api_base, get_api_base);

    pub const PROMPTS: [PromptAction<'static>; 1] = [("api_key", "API Key", None)];
}

impl_client_trait!(
    ClaudeClient,
    (
        prepare_chat_completions,
        claude_chat_completions,
        claude_chat_completions_streaming
    ),
    (noop_prepare_embeddings, noop_embeddings),
    (noop_prepare_rerank, noop_rerank),
);

fn prepare_chat_completions(
    self_: &ClaudeClient,
    data: ChatCompletionsData,
) -> Result<RequestData> {
    let api_key = self_.get_api_key()?;
    let api_base = self_
        .get_api_base()
        .unwrap_or_else(|_| API_BASE.to_string());

    let url = format!("{}/messages", api_base.trim_end_matches('/'));
    let body = claude_build_chat_completions_body(data, &self_.model)?;

    let mut request_data = RequestData::new(url, body);

    request_data.header("anthropic-version", "2023-06-01");
    request_data.header("x-api-key", api_key);

    Ok(request_data)
}

pub async fn claude_chat_completions(
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
    claude_extract_chat_completions(&data)
}

#[derive(Debug, Default)]
struct ClaudeStreamState {
    tool_call: Option<PendingClaudeToolCall>,
    reasoning: bool,
}

impl ClaudeStreamState {
    fn finish_tool_call(&mut self, handler: &mut SseHandler) -> Result<()> {
        if let Some(tool_call) = self.tool_call.take() {
            handler.tool_call(tool_call.into_tool_call()?)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct PendingClaudeToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl PendingClaudeToolCall {
    fn into_tool_call(self) -> Result<ToolCall> {
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
        Ok(ToolCall::new(self.name, arguments, Some(self.id)))
    }
}

fn handle_claude_stream_message(
    state: &mut ClaudeStreamState,
    handler: &mut SseHandler,
    message: &str,
) -> Result<bool> {
    let data: Value = serde_json::from_str(message).context("Invalid Claude streaming response")?;
    debug!("stream-data: {data}");
    let Some(typ) = data["type"].as_str() else {
        return Ok(false);
    };

    match typ {
        "content_block_start" => {
            if let (Some("tool_use"), Some(name), Some(id)) = (
                data["content_block"]["type"].as_str(),
                data["content_block"]["name"].as_str(),
                data["content_block"]["id"].as_str(),
            ) {
                state.finish_tool_call(handler)?;
                state.tool_call = Some(PendingClaudeToolCall {
                    id: id.to_string(),
                    name: name.to_string(),
                    arguments: String::new(),
                });
            }
        }
        "content_block_delta" => {
            if let Some(text) = data["delta"]["text"].as_str() {
                handler.text(text)?;
            } else if let Some(text) = data["delta"]["thinking"].as_str() {
                if !state.reasoning {
                    handler.text("<think>\n")?;
                    state.reasoning = true;
                }
                handler.text(text)?;
            } else if let (Some(tool_call), Some(partial_json)) = (
                state.tool_call.as_mut(),
                data["delta"]["partial_json"].as_str(),
            ) {
                tool_call.arguments.push_str(partial_json);
            }
        }
        "content_block_stop" => {
            if state.reasoning {
                handler.text("\n</think>\n\n")?;
                state.reasoning = false;
            }
            state.finish_tool_call(handler)?;
        }
        "error" => {
            let error_type = data["error"]["type"].as_str().unwrap_or("unknown_error");
            let message = data["error"]["message"]
                .as_str()
                .unwrap_or("Claude streaming request failed");
            bail!("{message} (type: {error_type})");
        }
        _ => {}
    }
    Ok(false)
}

pub async fn claude_chat_completions_streaming(
    builder: RequestBuilder,
    handler: &mut SseHandler,
    _model: &Model,
) -> Result<()> {
    let mut state = ClaudeStreamState::default();
    sse_stream(builder, |message: SseMmessage| {
        handle_claude_stream_message(&mut state, handler, &message.data)
    })
    .await?;
    state.finish_tool_call(handler)
}

pub fn claude_build_chat_completions_body(
    data: ChatCompletionsData,
    model: &Model,
) -> Result<Value> {
    let ChatCompletionsData {
        mut messages,
        temperature,
        top_p,
        functions,
        stream,
    } = data;

    let system_message = extract_system_message(&mut messages);

    let mut network_image_urls = vec![];

    let messages_len = messages.len();
    let messages: Vec<Value> = messages
        .into_iter()
        .enumerate()
        .flat_map(|(i, message)| {
            let Message { role, content } = message;
            match content {
                MessageContent::Text(text) if role.is_assistant() && i != messages_len - 1 => {
                    vec![json!({ "role": role, "content": strip_think_tag(&text) })]
                }
                MessageContent::Text(text) => vec![json!({
                    "role": role,
                    "content": text,
                })],
                MessageContent::Array(list) => {
                    let content: Vec<_> = list
                        .into_iter()
                        .map(|item| match item {
                            MessageContentPart::Text { text } => {
                                json!({"type": "text", "text": text})
                            }
                            MessageContentPart::ImageUrl {
                                image_url: ImageUrl { url },
                            } => {
                                if let Some((mime_type, data)) = url
                                    .strip_prefix("data:")
                                    .and_then(|v| v.split_once(";base64,"))
                                {
                                    json!({
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": mime_type,
                                            "data": data,
                                        }
                                    })
                                } else {
                                    network_image_urls.push(url.clone());
                                    json!({ "url": url })
                                }
                            }
                        })
                        .collect();
                    vec![json!({
                        "role": role,
                        "content": content,
                    })]
                }
                MessageContent::ToolCalls(MessageContentToolCalls {
                    tool_results, text, ..
                }) => {
                    let mut assistant_parts = vec![];
                    let mut user_parts = vec![];
                    if !text.is_empty() {
                        assistant_parts.push(json!({
                            "type": "text",
                            "text": text,
                        }))
                    }
                    for tool_result in tool_results {
                        assistant_parts.push(json!({
                            "type": "tool_use",
                            "id": tool_result.call.id,
                            "name": tool_result.call.name,
                            "input": tool_result.call.arguments,
                        }));
                        user_parts.push(json!({
                            "type": "tool_result",
                            "tool_use_id": tool_result.call.id,
                            "content": tool_result.output.to_string(),
                        }));
                    }
                    vec![
                        json!({
                            "role": "assistant",
                            "content": assistant_parts,
                        }),
                        json!({
                            "role": "user",
                            "content": user_parts,
                        }),
                    ]
                }
            }
        })
        .collect();

    if !network_image_urls.is_empty() {
        bail!(
            "The model does not support network images: {:?}",
            network_image_urls
        );
    }

    let mut body = json!({
        "model": model.real_name(),
        "messages": messages,
    });
    if let Some(v) = system_message {
        body["system"] = v.into();
    }
    if let Some(v) = model.max_tokens_param() {
        body["max_tokens"] = v.into();
    }
    if let Some(v) = temperature {
        body["temperature"] = v.into();
    }
    if let Some(v) = top_p {
        body["top_p"] = v.into();
    }
    if stream {
        body["stream"] = true.into();
    }
    if let Some(functions) = functions {
        body["tools"] = functions
            .iter()
            .map(|v| {
                json!({
                    "name": v.name,
                    "description": v.description,
                    "input_schema": v.parameters,
                })
            })
            .collect();
    }
    Ok(body)
}

pub fn claude_extract_chat_completions(data: &Value) -> Result<ChatCompletionsOutput> {
    let mut text = String::new();
    let mut reasoning = None;
    let mut tool_calls = vec![];
    if let Some(list) = data["content"].as_array() {
        for item in list {
            match item["type"].as_str() {
                Some("thinking") => {
                    if let Some(v) = item["thinking"].as_str() {
                        reasoning = Some(v.to_string());
                    }
                }
                Some("text") => {
                    if let Some(v) = item["text"].as_str() {
                        if !text.is_empty() {
                            text.push_str("\n\n");
                        }
                        text.push_str(v);
                    }
                }
                Some("tool_use") => {
                    if let (Some(name), Some(input), Some(id)) = (
                        item["name"].as_str(),
                        item.get("input"),
                        item["id"].as_str(),
                    ) {
                        tool_calls.push(ToolCall::new(
                            name.to_string(),
                            input.clone(),
                            Some(id.to_string()),
                        ));
                    }
                }
                _ => {}
            }
        }
    }
    if let Some(reasoning) = reasoning {
        text = format!("<think>\n{reasoning}\n</think>\n\n{text}")
    }

    if text.is_empty() && tool_calls.is_empty() {
        bail!("Invalid response data: {data}");
    }

    let output = ChatCompletionsOutput {
        text: text.to_string(),
        tool_calls,
        id: data["id"].as_str().map(|v| v.to_string()),
        input_tokens: data["usage"]["input_tokens"].as_u64(),
        output_tokens: data["usage"]["output_tokens"].as_u64(),
    };
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::utils::create_abort_signal;
    use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};

    fn test_handler() -> (SseHandler, UnboundedReceiver<SseEvent>) {
        let (tx, rx) = unbounded_channel();
        (SseHandler::new(tx, create_abort_signal()), rx)
    }

    fn handle_value(
        state: &mut ClaudeStreamState,
        handler: &mut SseHandler,
        value: Value,
    ) -> Result<bool> {
        handle_claude_stream_message(state, handler, &value.to_string())
    }

    #[test]
    fn streaming_error_is_readable_and_does_not_dispatch_partial_output() -> Result<()> {
        let (mut handler, _rx) = test_handler();
        let mut state = ClaudeStreamState::default();

        handle_value(
            &mut state,
            &mut handler,
            json!({
                "type":"content_block_start",
                "content_block":{"type":"tool_use","id":"tool_partial","name":"lookup"}
            }),
        )?;
        handle_value(
            &mut state,
            &mut handler,
            json!({"type":"content_block_delta","delta":{"partial_json":"{\"query\":"}}),
        )?;
        let err = handle_value(
            &mut state,
            &mut handler,
            json!({
                "type":"error",
                "error":{"type":"overloaded_error","message":"Overloaded"}
            }),
        )
        .expect_err("provider error must stop the stream");

        assert_eq!(err.to_string(), "Overloaded (type: overloaded_error)");
        let (text, calls) = handler.take();
        assert!(text.is_empty());
        assert!(calls.is_empty());
        Ok(())
    }

    #[test]
    fn successful_content_reasoning_and_tool_stream_is_unchanged() -> Result<()> {
        let (mut handler, _rx) = test_handler();
        let mut state = ClaudeStreamState::default();

        for value in [
            json!({"type":"content_block_delta","delta":{"thinking":"reason"}}),
            json!({"type":"content_block_stop"}),
            json!({"type":"content_block_delta","delta":{"text":"answer"}}),
            json!({
                "type":"content_block_start",
                "content_block":{"type":"tool_use","id":"tool_one","name":"first"}
            }),
            json!({"type":"content_block_delta","delta":{"partial_json":"{\"value\":1}"}}),
            json!({"type":"content_block_stop"}),
            json!({
                "type":"content_block_start",
                "content_block":{"type":"tool_use","id":"tool_two","name":"second"}
            }),
            json!({"type":"content_block_stop"}),
        ] {
            handle_value(&mut state, &mut handler, value)?;
        }
        state.finish_tool_call(&mut handler)?;

        let (text, calls) = handler.take();
        assert_eq!(text, "<think>\nreason\n</think>\n\nanswer");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "first");
        assert_eq!(calls[0].arguments, json!({"value":1}));
        assert_eq!(calls[1].name, "second");
        assert_eq!(calls[1].arguments, json!({}));
        Ok(())
    }
}
