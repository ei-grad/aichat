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
    let api_base =
        optional_config_field(self_.get_api_base())?.unwrap_or_else(|| API_BASE.to_string());

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
    let data: Value = match res.json().await {
        Ok(data) => data,
        Err(_) if !status.is_success() => {
            bail!("Claude request failed (status: {})", status.as_u16());
        }
        Err(err) => return Err(err.into()),
    };
    if !status.is_success() {
        catch_claude_error(&data, status.as_u16())?;
    }
    let output = claude_extract_chat_completions(&data)?;
    debug!("non-stream-data: {data}");
    Ok(output)
}

#[derive(Debug, Default)]
struct ClaudeStreamState {
    active_block: Option<ClaudeContentBlock>,
    completed_tool_calls: Vec<ToolCall>,
    output: String,
    terminal_stop_reason: Option<String>,
}

impl ClaudeStreamState {
    fn start_block(&mut self, block: ClaudeContentBlock) -> Result<()> {
        if let Some(active) = self.active_block.as_ref() {
            bail!(
                "Claude started {} block {} before {} block {} stopped",
                block.kind(),
                block.index(),
                active.kind(),
                active.index()
            );
        }
        self.active_block = Some(block);
        Ok(())
    }

    fn finish_block(&mut self, index: u64) -> Result<()> {
        let active = self
            .active_block
            .take()
            .context("Claude content_block_stop arrived without an active content block")?;
        if active.index() != index {
            bail!(
                "Claude {} block {} stopped by block {index}",
                active.kind(),
                active.index()
            );
        }
        match active {
            ClaudeContentBlock::Thinking { emitted: true, .. } => {
                self.output.push_str("\n</think>\n\n");
            }
            ClaudeContentBlock::ToolUse(tool_call) => {
                self.completed_tool_calls
                    .push(tool_call.into_tool_call()?);
            }
            _ => {}
        }
        Ok(())
    }

    fn set_terminal_stop_reason(&mut self, stop_reason: &str) -> Result<()> {
        if let Some(previous) = self.terminal_stop_reason.as_deref() {
            bail!(
                "Claude sent multiple terminal stop reasons: {previous} and {stop_reason}"
            );
        }
        self.terminal_stop_reason = Some(stop_reason.to_string());
        Ok(())
    }

    fn finish_message(&mut self, handler: &mut SseHandler) -> Result<()> {
        if let Some(active) = self.active_block.as_ref() {
            bail!(
                "Claude message_stop arrived before {} content block {} stopped",
                active.kind(),
                active.index()
            );
        }
        if self.terminal_stop_reason.is_none() {
            bail!("Claude message_stop arrived before a terminal stop_reason");
        }
        handler.text(&std::mem::take(&mut self.output))?;
        for tool_call in std::mem::take(&mut self.completed_tool_calls) {
            handler.tool_call(tool_call)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
enum ClaudeContentBlock {
    Text { index: u64 },
    Thinking { index: u64, emitted: bool },
    ToolUse(PendingClaudeToolCall),
    Unknown { index: u64 },
}

impl ClaudeContentBlock {
    fn index(&self) -> u64 {
        match self {
            Self::Text { index }
            | Self::Thinking { index, .. }
            | Self::Unknown { index } => *index,
            Self::ToolUse(tool_call) => tool_call.index,
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::Text { .. } => "text",
            Self::Thinking { .. } => "thinking",
            Self::ToolUse(_) => "tool_use",
            Self::Unknown { .. } => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClaudeDeltaKind {
    Text,
    Thinking,
    Signature,
    ToolJson,
    Unknown,
}

impl ClaudeDeltaKind {
    fn from_data(data: &Value) -> Self {
        match data["delta"]["type"].as_str() {
            Some("text_delta") => Self::Text,
            Some("thinking_delta") => Self::Thinking,
            Some("signature_delta") => Self::Signature,
            Some("input_json_delta") => Self::ToolJson,
            Some(_) => Self::Unknown,
            None if data["delta"]["text"].is_string() => Self::Text,
            None if data["delta"]["thinking"].is_string() => Self::Thinking,
            None if data["delta"]["signature"].is_string() => Self::Signature,
            None if data["delta"]["partial_json"].is_string() => Self::ToolJson,
            None => Self::Unknown,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Thinking => "thinking",
            Self::Signature => "signature",
            Self::ToolJson => "input_json",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug)]
struct PendingClaudeToolCall {
    index: u64,
    id: String,
    name: String,
    arguments: String,
}

impl PendingClaudeToolCall {
    fn into_tool_call(self) -> Result<ToolCall> {
        let arguments = if self.arguments.is_empty() {
            json!({})
        } else {
            self.arguments
                .parse()
                .context("Claude tool call has invalid JSON arguments")?
        };
        Ok(ToolCall::new(self.name, arguments, Some(self.id)))
    }
}

fn claude_refusal_message(stop_details: &Value) -> String {
    let mut message = "Claude refused the request".to_string();
    if let Some(explanation) = stop_details["explanation"].as_str() {
        message.push_str(": ");
        message.push_str(explanation);
    }
    if let Some(category) = stop_details["category"].as_str() {
        message.push_str(" (category: ");
        message.push_str(category);
        message.push(')');
    }
    message
}

fn vertex_block_reason(data: &Value) -> Option<&str> {
    data["promptFeedback"]["blockReason"].as_str()
}

fn catch_claude_error(data: &Value, status: u16) -> Result<()> {
    if (200..300).contains(&status) {
        return Ok(());
    }
    if let Some(error) = data["error"].as_object() {
        if let Some(error_type) = error.get("type").and_then(Value::as_str) {
            bail!("Claude request failed (status: {status}, type: {error_type})");
        }
        if let Some(error_code) = error.get("code").and_then(Value::as_str) {
            bail!("Claude request failed (status: {status}, code: {error_code})");
        }
    }
    bail!("Claude request failed (status: {status})");
}

fn check_claude_response(data: &Value) -> Result<()> {
    if let Some(reason) = vertex_block_reason(data) {
        bail!("Vertex AI blocked the request (reason: {reason})");
    }
    if data["stop_reason"].as_str() == Some("refusal") {
        bail!("{}", claude_refusal_message(&data["stop_details"]));
    }
    if data["content"]
        .as_array()
        .is_some_and(|content| content.iter().any(|block| block["type"] == "fallback"))
    {
        bail!("Claude server-side fallback responses are not supported");
    }
    Ok(())
}

fn handle_claude_stream_message(
    state: &mut ClaudeStreamState,
    handler: &mut SseHandler,
    event: &str,
    message: &str,
) -> Result<bool> {
    if event == "vertex-block-event" {
        if let Ok(data) = serde_json::from_str::<Value>(message) {
            if let Some(reason) = vertex_block_reason(&data) {
                bail!("Vertex AI blocked the request (reason: {reason})");
            }
        }
        bail!("Vertex AI blocked the request");
    }
    let data: Value = serde_json::from_str(message).context("Invalid Claude streaming response")?;
    if let Some(reason) = vertex_block_reason(&data) {
        bail!("Vertex AI blocked the request (reason: {reason})");
    }
    let Some(typ) = data["type"].as_str() else {
        return Ok(false);
    };

    match typ {
        "content_block_start" => {
            let block_type = data["content_block"]["type"]
                .as_str()
                .context("Claude content_block_start is missing a block type")?;
            if block_type == "fallback" {
                bail!("Claude server-side fallback responses are not supported");
            }
            let index = data["index"]
                .as_u64()
                .context("Claude content_block_start is missing an index")?;
            let block = match block_type {
                "text" => ClaudeContentBlock::Text { index },
                "thinking" => ClaudeContentBlock::Thinking {
                    index,
                    emitted: false,
                },
                "tool_use" => {
                    let name = data["content_block"]["name"]
                        .as_str()
                        .context("Claude tool_use content block is malformed")?;
                    let id = data["content_block"]["id"]
                        .as_str()
                        .context("Claude tool_use content block is malformed")?;
                    ClaudeContentBlock::ToolUse(PendingClaudeToolCall {
                    index,
                    id: id.to_string(),
                    name: name.to_string(),
                    arguments: String::new(),
                    })
                }
                _ => ClaudeContentBlock::Unknown { index },
            };
            state.start_block(block)?;
        }
        "content_block_delta" => {
            let index = data["index"]
                .as_u64()
                .context("Claude content_block_delta is missing an index")?;
            let delta_kind = ClaudeDeltaKind::from_data(&data);
            let active = state
                .active_block
                .as_mut()
                .context("Claude content_block_delta arrived without an active content block")?;
            if active.index() != index {
                bail!(
                    "Claude {} block {} received a delta for block {index}",
                    active.kind(),
                    active.index()
                );
            }
            match (active, delta_kind) {
                (ClaudeContentBlock::Text { .. }, ClaudeDeltaKind::Text) => {
                    let text = data["delta"]["text"]
                        .as_str()
                        .context("Claude text delta is malformed")?;
                    state.output.push_str(text);
                }
                (
                    ClaudeContentBlock::Thinking { emitted, .. },
                    ClaudeDeltaKind::Thinking,
                ) => {
                    let text = data["delta"]["thinking"]
                        .as_str()
                        .context("Claude thinking delta is malformed")?;
                    if !*emitted {
                        state.output.push_str("<think>\n");
                        *emitted = true;
                    }
                    state.output.push_str(text);
                }
                (ClaudeContentBlock::Thinking { .. }, ClaudeDeltaKind::Signature) => {}
                (ClaudeContentBlock::ToolUse(tool_call), ClaudeDeltaKind::ToolJson) => {
                    let partial_json = data["delta"]["partial_json"]
                        .as_str()
                        .context("Claude input_json delta is malformed")?;
                    tool_call.arguments.push_str(partial_json);
                }
                (ClaudeContentBlock::Unknown { .. }, _) | (_, ClaudeDeltaKind::Unknown) => {}
                (active, delta_kind) => {
                    bail!(
                        "Claude {} delta does not match active {} block {}",
                        delta_kind.name(),
                        active.kind(),
                        active.index()
                    );
                }
            }
        }
        "content_block_stop" => {
            let index = data["index"]
                .as_u64()
                .context("Claude content_block_stop is missing an index")?;
            state.finish_block(index)?;
        }
        "message_delta" => {
            if let Some(stop_reason) = data["delta"].get("stop_reason") {
                if stop_reason.is_null() {
                    return Ok(false);
                }
                let stop_reason = stop_reason
                    .as_str()
                    .context("Claude message_delta has an invalid stop_reason")?;
                if stop_reason == "refusal" {
                    bail!("{}", claude_refusal_message(&data["delta"]["stop_details"]));
                }
                state.set_terminal_stop_reason(stop_reason)?;
            }
        }
        "message_stop" => {
            state.finish_message(handler)?;
            return Ok(true);
        }
        "error" => {
            let error_type = data["error"]["type"].as_str().unwrap_or("unknown_error");
            bail!("Claude streaming request failed (type: {error_type})");
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
    sse_stream_sanitized(
        builder,
        |message: SseMmessage| {
            handle_claude_stream_message(&mut state, handler, &message.event, &message.data)
        },
        catch_claude_error,
    )
    .await
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
    check_claude_response(data)?;

    let mut text = String::new();
    let mut reasoning = None;
    let mut tool_calls = vec![];
    if let Some(list) = data["content"].as_array() {
        for item in list {
            match item["type"].as_str() {
                Some("thinking") => {
                    let value = item["thinking"]
                        .as_str()
                        .context("Claude thinking content block is malformed")?;
                    reasoning = Some(value.to_string());
                }
                Some("text") => {
                    let value = item["text"]
                        .as_str()
                        .context("Claude text content block is malformed")?;
                    if !text.is_empty() {
                        text.push_str("\n\n");
                    }
                    text.push_str(value);
                }
                Some("tool_use") => {
                    let name = item["name"]
                        .as_str()
                        .context("Claude tool_use content block is malformed")?;
                    let input = item
                        .get("input")
                        .context("Claude tool_use content block is malformed")?;
                    let id = item["id"]
                        .as_str()
                        .context("Claude tool_use content block is malformed")?;
                    tool_calls.push(ToolCall::new(
                        name.to_string(),
                        input.clone(),
                        Some(id.to_string()),
                    ));
                }
                _ => {}
            }
        }
    }
    if let Some(reasoning) = reasoning {
        text = format!("<think>\n{reasoning}\n</think>\n\n{text}")
    }

    if text.is_empty() && tool_calls.is_empty() {
        bail!("Claude response did not contain supported content");
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

    const MISSING_API_BASE_ENV: &str = "AICHAT_TEST_MISSING_CLAUDE_API_BASE_2B76B42D";

    fn request_client(api_base: Option<String>) -> ClaudeClient {
        ClaudeClient {
            global_config: Default::default(),
            config: ClaudeConfig {
                name: Some("claude-remediation-test".into()),
                api_key: Some("test-key".into()),
                api_base,
                models: vec![],
                patch: None,
                extra: None,
            },
            model: Model::new("claude-remediation-test", "test-model"),
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

    fn test_handler() -> (SseHandler, UnboundedReceiver<SseEvent>) {
        let (tx, rx) = unbounded_channel();
        (SseHandler::new(tx, create_abort_signal()), rx)
    }

    fn handle_value(
        state: &mut ClaudeStreamState,
        handler: &mut SseHandler,
        value: Value,
    ) -> Result<bool> {
        handle_event_value(state, handler, "message", value)
    }

    fn handle_event_value(
        state: &mut ClaudeStreamState,
        handler: &mut SseHandler,
        event: &str,
        value: Value,
    ) -> Result<bool> {
        handle_claude_stream_message(state, handler, event, &value.to_string())
    }

    #[test]
    fn missing_explicit_api_base_does_not_fall_back_to_public_claude() {
        assert!(std::env::var_os(MISSING_API_BASE_ENV).is_none());
        let client = request_client(Some(format!("${MISSING_API_BASE_ENV}")));

        let err = prepare_chat_completions(&client, completion_data())
            .err()
            .expect("missing explicit reference must fail before request preparation");

        assert_eq!(
            err.to_string(),
            "Environment variable for 'api_base' is missing or empty"
        );
        assert!(!err.to_string().contains(MISSING_API_BASE_ENV));
    }

    #[test]
    fn absent_api_base_keeps_public_claude_fallback() {
        let request = prepare_chat_completions(&request_client(None), completion_data()).unwrap();

        assert_eq!(request.url, "https://api.anthropic.com/v1/messages");
    }

    #[test]
    fn nonstream_empty_refusal_without_details_is_explicit() {
        let err = claude_extract_chat_completions(&json!({
            "id": "refusal-empty",
            "stop_reason": "refusal",
            "stop_details": null,
            "content": [],
            "usage": {"input_tokens": 12, "output_tokens": 0}
        }))
        .expect_err("HTTP-200 refusal must not become a successful completion");

        assert_eq!(err.to_string(), "Claude refused the request");
    }

    #[test]
    fn nonstream_mid_output_refusal_discards_text_and_tools() {
        let err = claude_extract_chat_completions(&json!({
            "id": "refusal-partial",
            "stop_reason": "refusal",
            "stop_details": {
                "type": "refusal",
                "explanation": "The request was declined",
                "category": "cyber"
            },
            "content": [
                {"type": "text", "text": "sensitive partial output"},
                {
                    "type": "tool_use",
                    "id": "side-effect-id",
                    "name": "side_effect",
                    "input": {"dangerous": true}
                }
            ],
            "usage": {"input_tokens": 12, "output_tokens": 7}
        }))
        .expect_err("mid-output refusal must discard all partial content");

        assert_eq!(
            err.to_string(),
            "Claude refused the request: The request was declined (category: cyber)"
        );
        assert!(!err.to_string().contains("sensitive partial output"));
        assert!(!err.to_string().contains("side_effect"));
    }

    #[test]
    fn nonstream_fallback_block_is_rejected_without_content() {
        let err = claude_extract_chat_completions(&json!({
            "stop_reason": "end_turn",
            "content": [
                {"type": "text", "text": "pre-fallback partial"},
                {"type": "fallback", "from_model": "claude-primary", "to_model": "claude-backup"},
                {"type": "text", "text": "post-fallback partial"}
            ]
        }))
        .expect_err("fallback state cannot be flattened into text");

        assert_eq!(
            err.to_string(),
            "Claude server-side fallback responses are not supported"
        );
        assert!(!err.to_string().contains("partial"));
    }

    #[test]
    fn vertex_unary_block_is_explicit_and_discards_content() {
        let err = claude_extract_chat_completions(&json!({
            "promptFeedback": {"blockReason": "PROHIBITED_CONTENT"},
            "content": [{"type": "text", "text": "blocked partial"}]
        }))
        .expect_err("Vertex safety response must not be parsed as Claude output");

        assert_eq!(
            err.to_string(),
            "Vertex AI blocked the request (reason: PROHIBITED_CONTENT)"
        );
        assert!(!err.to_string().contains("blocked partial"));
    }

    #[test]
    fn malformed_nonstream_tool_block_does_not_expose_provider_payload() {
        const TOOL_SENTINEL: &str = "NONSTREAM_TOOL_NAME_SENTINEL";
        const INPUT_SENTINEL: &str = "NONSTREAM_TOOL_INPUT_SENTINEL";
        let err = claude_extract_chat_completions(&json!({
            "stop_reason": "tool_use",
            "content": [{
                "type": "tool_use",
                "name": TOOL_SENTINEL,
                "input": {"secret": INPUT_SENTINEL}
            }]
        }))
        .expect_err("tool_use without an id must fail as a malformed shape");

        assert_eq!(
            err.to_string(),
            "Claude tool_use content block is malformed"
        );
        assert!(!err.to_string().contains(TOOL_SENTINEL));
        assert!(!err.to_string().contains(INPUT_SENTINEL));
    }

    #[tokio::test]
    async fn nonstream_http_error_exposes_only_status_and_type() -> Result<()> {
        const BODY_SENTINEL: &str = "NONSTREAM_HTTP_BODY_SENTINEL";
        let body = json!({
            "error": {
                "type": "overloaded_error",
                "message": BODY_SENTINEL
            }
        })
        .to_string();
        let builder = response_fixture_builder("529 Overloaded", "application/json", &body).await?;

        let err = claude_chat_completions(
            builder,
            &Model::new("claude", "test-model"),
        )
        .await
        .expect_err("non-success Claude HTTP response must fail");

        assert_eq!(
            err.to_string(),
            "Claude request failed (status: 529, type: overloaded_error)"
        );
        assert!(!err.to_string().contains(BODY_SENTINEL));
        Ok(())
    }

    #[test]
    fn streaming_error_is_readable_and_does_not_dispatch_partial_output() -> Result<()> {
        let (mut handler, mut rx) = test_handler();
        let mut state = ClaudeStreamState::default();

        handle_value(
            &mut state,
            &mut handler,
            json!({
                "type":"content_block_start",
                "index":10,
                "content_block":{"type":"text","text":""}
            }),
        )?;
        handle_value(
            &mut state,
            &mut handler,
            json!({"type":"content_block_delta","index":10,"delta":{"type":"text_delta","text":"staged answer"}}),
        )?;
        handle_value(
            &mut state,
            &mut handler,
            json!({"type":"content_block_stop","index":10}),
        )?;
        handle_value(
            &mut state,
            &mut handler,
            json!({
                "type":"content_block_start",
                "index":0,
                "content_block":{"type":"tool_use","id":"tool_partial","name":"lookup"}
            }),
        )?;
        handle_value(
            &mut state,
            &mut handler,
            json!({"type":"content_block_delta","index":0,"delta":{"partial_json":"{\"query\":"}}),
        )?;
        let err = handle_value(
            &mut state,
            &mut handler,
            json!({
                "type":"error",
                "error":{"type":"overloaded_error","message":"STREAM_ERROR_SENTINEL"}
            }),
        )
        .expect_err("provider error must stop the stream");

        assert_eq!(
            err.to_string(),
            "Claude streaming request failed (type: overloaded_error)"
        );
        assert!(!err.to_string().contains("STREAM_ERROR_SENTINEL"));
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        let (text, calls) = handler.take();
        assert!(text.is_empty());
        assert!(calls.is_empty());
        Ok(())
    }

    #[test]
    fn streaming_refusal_before_output_without_details_is_transactional() {
        let (mut handler, mut rx) = test_handler();
        let mut state = ClaudeStreamState::default();

        let err = handle_value(
            &mut state,
            &mut handler,
            json!({
                "type": "message_delta",
                "delta": {"stop_reason": "refusal", "stop_details": null}
            }),
        )
        .expect_err("refusal must terminate the stream");

        assert_eq!(err.to_string(), "Claude refused the request");
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        let (text, calls) = handler.take();
        assert!(text.is_empty());
        assert!(calls.is_empty());
    }

    #[test]
    fn streaming_refusal_discards_text_thinking_and_completed_tool() -> Result<()> {
        let (mut handler, mut rx) = test_handler();
        let mut state = ClaudeStreamState::default();

        for value in [
            json!({
                "type":"content_block_start",
                "index":0,
                "content_block":{"type":"thinking","thinking":"","signature":""}
            }),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"private reasoning"}}),
            json!({"type":"content_block_stop","index":0}),
            json!({
                "type":"content_block_start",
                "index":1,
                "content_block":{"type":"text","text":""}
            }),
            json!({"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"partial answer"}}),
            json!({"type":"content_block_stop","index":1}),
            json!({
                "type":"content_block_start",
                "index":2,
                "content_block":{"type":"tool_use","id":"tool-complete","name":"side_effect"}
            }),
            json!({"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"value\":1}"}}),
            json!({"type":"content_block_stop","index":2}),
        ] {
            handle_value(&mut state, &mut handler, value)?;
        }

        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        let err = handle_value(
            &mut state,
            &mut handler,
            json!({
                "type":"message_delta",
                "delta": {
                    "stop_reason":"refusal",
                    "stop_details": {
                        "explanation":"The streamed request was declined",
                        "category":"reasoning_extraction"
                    }
                }
            }),
        )
        .expect_err("refusal must discard completed staged output");

        assert_eq!(
            err.to_string(),
            "Claude refused the request: The streamed request was declined (category: reasoning_extraction)"
        );
        assert!(!err.to_string().contains("partial answer"));
        assert!(!err.to_string().contains("side_effect"));
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        let (text, calls) = handler.take();
        assert!(text.is_empty());
        assert!(calls.is_empty());
        Ok(())
    }

    #[test]
    fn streaming_refusal_discards_pending_tool_with_null_detail_fields() -> Result<()> {
        let (mut handler, mut rx) = test_handler();
        let mut state = ClaudeStreamState::default();

        handle_value(
            &mut state,
            &mut handler,
            json!({
                "type":"content_block_start",
                "index":3,
                "content_block":{"type":"tool_use","id":"tool-pending","name":"side_effect"}
            }),
        )?;
        handle_value(
            &mut state,
            &mut handler,
            json!({"type":"content_block_delta","index":3,"delta":{"partial_json":"{\"pending\":"}}),
        )?;
        let err = handle_value(
            &mut state,
            &mut handler,
            json!({
                "type":"message_delta",
                "delta": {
                    "stop_reason":"refusal",
                    "stop_details":{"explanation":null,"category":null}
                }
            }),
        )
        .expect_err("refusal must discard a pending tool call");

        assert_eq!(err.to_string(), "Claude refused the request");
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        let (text, calls) = handler.take();
        assert!(text.is_empty());
        assert!(calls.is_empty());
        Ok(())
    }

    #[test]
    fn streaming_invalid_json_discards_staged_text() -> Result<()> {
        let (mut handler, mut rx) = test_handler();
        let mut state = ClaudeStreamState::default();

        handle_value(
            &mut state,
            &mut handler,
            json!({
                "type":"content_block_start",
                "index":0,
                "content_block":{"type":"text","text":""}
            }),
        )?;
        handle_value(
            &mut state,
            &mut handler,
            json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"staged answer"}}),
        )?;
        let err = handle_claude_stream_message(
            &mut state,
            &mut handler,
            "message",
            "{not-json",
        )
        .expect_err("invalid JSON must terminate the transaction");

        assert!(err.to_string().starts_with("Invalid Claude streaming response"));
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        let (text, calls) = handler.take();
        assert!(text.is_empty());
        assert!(calls.is_empty());
        Ok(())
    }

    #[test]
    fn streaming_invalid_tool_json_does_not_expose_arguments() -> Result<()> {
        let (mut handler, mut rx) = test_handler();
        let mut state = ClaudeStreamState::default();

        for value in [
            json!({
                "type":"content_block_start",
                "index":0,
                "content_block":{"type":"text","text":""}
            }),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"staged answer"}}),
            json!({"type":"content_block_stop","index":0}),
            json!({
                "type":"content_block_start",
                "index":4,
                "content_block":{"type":"tool_use","id":"tool-invalid","name":"secret_tool"}
            }),
            json!({
                "type":"content_block_delta",
                "index":4,
                "delta":{"type":"input_json_delta","partial_json":"{\"secret\":\"must not escape\""}
            }),
        ] {
            handle_value(&mut state, &mut handler, value)?;
        }
        let err = handle_value(
            &mut state,
            &mut handler,
            json!({"type":"content_block_stop","index":4}),
        )
        .expect_err("invalid tool JSON must abort the transaction");

        assert_eq!(
            err.to_string(),
            "Claude tool call has invalid JSON arguments"
        );
        assert!(!err.to_string().contains("secret_tool"));
        assert!(!err.to_string().contains("must not escape"));
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        let (text, calls) = handler.take();
        assert!(text.is_empty());
        assert!(calls.is_empty());
        Ok(())
    }

    #[test]
    fn streaming_fallback_block_is_rejected_without_staged_text() -> Result<()> {
        let (mut handler, mut rx) = test_handler();
        let mut state = ClaudeStreamState::default();

        handle_value(
            &mut state,
            &mut handler,
            json!({
                "type":"content_block_start",
                "index":0,
                "content_block":{"type":"text","text":""}
            }),
        )?;
        handle_value(
            &mut state,
            &mut handler,
            json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"pre-fallback partial"}}),
        )?;
        let err = handle_value(
            &mut state,
            &mut handler,
            json!({
                "type":"content_block_start",
                "index":1,
                "content_block":{"type":"fallback","from_model":"primary","to_model":"backup"}
            }),
        )
        .expect_err("fallback boundary state cannot be flattened");

        assert_eq!(
            err.to_string(),
            "Claude server-side fallback responses are not supported"
        );
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        let (text, calls) = handler.take();
        assert!(text.is_empty());
        assert!(calls.is_empty());
        Ok(())
    }

    #[test]
    fn vertex_stream_block_event_discards_staged_text() -> Result<()> {
        let (mut handler, mut rx) = test_handler();
        let mut state = ClaudeStreamState::default();

        handle_value(
            &mut state,
            &mut handler,
            json!({
                "type":"content_block_start",
                "index":0,
                "content_block":{"type":"text","text":""}
            }),
        )?;
        handle_value(
            &mut state,
            &mut handler,
            json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"blocked partial"}}),
        )?;
        let err = handle_event_value(
            &mut state,
            &mut handler,
            "vertex-block-event",
            json!({"promptFeedback":{"blockReason":"PROHIBITED_CONTENT"}}),
        )
        .expect_err("Vertex block event must terminate the transaction");

        assert_eq!(
            err.to_string(),
            "Vertex AI blocked the request (reason: PROHIBITED_CONTENT)"
        );
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        let (text, calls) = handler.take();
        assert!(text.is_empty());
        assert!(calls.is_empty());
        Ok(())
    }

    #[test]
    fn malformed_vertex_stream_block_event_is_still_explicit() -> Result<()> {
        let (mut handler, mut rx) = test_handler();
        let mut state = ClaudeStreamState::default();

        handle_value(
            &mut state,
            &mut handler,
            json!({
                "type":"content_block_start",
                "index":0,
                "content_block":{"type":"text","text":""}
            }),
        )?;
        handle_value(
            &mut state,
            &mut handler,
            json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"blocked partial"}}),
        )?;
        let err = handle_claude_stream_message(
            &mut state,
            &mut handler,
            "vertex-block-event",
            "not-json",
        )
        .expect_err("the SSE event name itself identifies a Vertex block");

        assert_eq!(err.to_string(), "Vertex AI blocked the request");
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        let (text, calls) = handler.take();
        assert!(text.is_empty());
        assert!(calls.is_empty());
        Ok(())
    }

    #[test]
    fn terminal_message_does_not_commit_an_unstopped_text_block() -> Result<()> {
        let (mut handler, mut rx) = test_handler();
        let mut state = ClaudeStreamState::default();

        for value in [
            json!({
                "type":"content_block_start",
                "index":0,
                "content_block":{"type":"text","text":""}
            }),
            json!({
                "type":"content_block_delta",
                "index":0,
                "delta":{"type":"text_delta","text":"truncated partial"}
            }),
            json!({"type":"message_delta","delta":{"stop_reason":"end_turn"}}),
        ] {
            handle_value(&mut state, &mut handler, value)?;
        }
        let err = handle_value(
            &mut state,
            &mut handler,
            json!({"type":"message_stop"}),
        )
        .expect_err("message_stop must not commit an active text block");

        assert_eq!(
            err.to_string(),
            "Claude message_stop arrived before text content block 0 stopped"
        );
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        let (text, calls) = handler.take();
        assert!(text.is_empty());
        assert!(calls.is_empty());
        Ok(())
    }

    #[test]
    fn mismatched_content_block_stop_discards_staged_text() -> Result<()> {
        let (mut handler, mut rx) = test_handler();
        let mut state = ClaudeStreamState::default();

        for value in [
            json!({
                "type":"content_block_start",
                "index":0,
                "content_block":{"type":"text","text":""}
            }),
            json!({
                "type":"content_block_delta",
                "index":0,
                "delta":{"type":"text_delta","text":"mismatched partial"}
            }),
        ] {
            handle_value(&mut state, &mut handler, value)?;
        }
        let err = handle_value(
            &mut state,
            &mut handler,
            json!({"type":"content_block_stop","index":1}),
        )
        .expect_err("a different block index must not close the active text block");

        assert_eq!(err.to_string(), "Claude text block 0 stopped by block 1");
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        let (text, calls) = handler.take();
        assert!(text.is_empty());
        assert!(calls.is_empty());
        Ok(())
    }

    #[test]
    fn recognized_content_blocks_reject_overlap_and_delta_kind_mismatch() -> Result<()> {
        for invalid_event in [
            json!({
                "type":"content_block_start",
                "index":1,
                "content_block":{"type":"thinking","thinking":"","signature":""}
            }),
            json!({
                "type":"content_block_delta",
                "index":0,
                "delta":{"type":"thinking_delta","thinking":"wrong kind"}
            }),
        ] {
            let (mut handler, mut rx) = test_handler();
            let mut state = ClaudeStreamState::default();
            handle_value(
                &mut state,
                &mut handler,
                json!({
                    "type":"content_block_start",
                    "index":0,
                    "content_block":{"type":"text","text":""}
                }),
            )?;

            handle_value(&mut state, &mut handler, invalid_event)
                .expect_err("recognized content-block lifecycle mismatch must fail");
            assert!(matches!(
                rx.try_recv(),
                Err(tokio::sync::mpsc::error::TryRecvError::Empty)
            ));
            let (text, calls) = handler.take();
            assert!(text.is_empty());
            assert!(calls.is_empty());
        }
        Ok(())
    }

    #[tokio::test]
    async fn streaming_http_error_exposes_only_status_and_type() -> Result<()> {
        const BODY_SENTINEL: &str = "STREAM_HTTP_BODY_SENTINEL";
        let body = json!({
            "error": {
                "type": "overloaded_error",
                "message": BODY_SENTINEL
            }
        })
        .to_string();
        let builder = response_fixture_builder("529 Overloaded", "application/json", &body).await?;
        let (mut handler, mut rx) = test_handler();

        let err = claude_chat_completions_streaming(
            builder,
            &mut handler,
            &Model::new("claude", "test-model"),
        )
        .await
        .expect_err("non-success Claude SSE response must fail");

        assert_eq!(
            err.to_string(),
            "Claude request failed (status: 529, type: overloaded_error)"
        );
        assert!(!err.to_string().contains(BODY_SENTINEL));
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
    async fn invalid_sse_content_type_does_not_expose_body() -> Result<()> {
        const BODY_SENTINEL: &str = "INVALID_CONTENT_TYPE_BODY_SENTINEL";
        let builder = response_fixture_builder(
            "200 OK",
            "application/json",
            BODY_SENTINEL,
        )
        .await?;
        let (mut handler, mut rx) = test_handler();

        let err = claude_chat_completions_streaming(
            builder,
            &mut handler,
            &Model::new("claude", "test-model"),
        )
        .await
        .expect_err("Claude SSE response with a non-event-stream type must fail");

        assert_eq!(
            err.to_string(),
            "Invalid response event-stream (status: 200, content-type: application/json)"
        );
        assert!(!err.to_string().contains(BODY_SENTINEL));
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
    async fn truncated_sse_does_not_dispatch_pending_tool_call() -> Result<()> {
        let tool_start = json!({
            "type":"content_block_start",
            "index":0,
            "content_block":{
                "type":"tool_use",
                "id":"tool_side_effect",
                "name":"side_effect",
                "input":{}
            }
        });
        let builder = sse_fixture_builder(&format!("data: {tool_start}\n\n")).await?;
        let (mut handler, mut rx) = test_handler();

        let err = claude_chat_completions_streaming(
            builder,
            &mut handler,
            &Model::new("claude", "test-model"),
        )
        .await
        .expect_err("EOF before content_block_stop and message_stop must fail");

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
    async fn truncated_sse_does_not_dispatch_stopped_tool_call() -> Result<()> {
        let events = [
            json!({
                "type":"content_block_start",
                "index":0,
                "content_block":{
                    "type":"tool_use",
                    "id":"tool_stopped",
                    "name":"side_effect",
                    "input":{}
                }
            }),
            json!({"type":"content_block_stop","index":0}),
        ];
        let body = events
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>();
        let builder = sse_fixture_builder(&body).await?;
        let (mut handler, mut rx) = test_handler();

        let err = claude_chat_completions_streaming(
            builder,
            &mut handler,
            &Model::new("claude", "test-model"),
        )
        .await
        .expect_err("EOF after content_block_stop but before message_stop must fail");

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
    async fn truncated_sse_does_not_dispatch_staged_text() -> Result<()> {
        let events = [
            json!({
                "type":"content_block_start",
                "index":0,
                "content_block":{"type":"text","text":""}
            }),
            json!({
                "type":"content_block_delta",
                "index":0,
                "delta":{"type":"text_delta","text":"never externally visible"}
            }),
        ];
        let body = events
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>();
        let builder = sse_fixture_builder(&body).await?;
        let (mut handler, mut rx) = test_handler();

        let err = claude_chat_completions_streaming(
            builder,
            &mut handler,
            &Model::new("claude", "test-model"),
        )
        .await
        .expect_err("EOF before terminal stop_reason and message_stop must fail");

        assert_eq!(err.to_string(), "SSE stream ended before protocol completion");
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        let (text, calls) = handler.take();
        assert!(text.is_empty());
        assert!(calls.is_empty());
        Ok(())
    }

    #[test]
    fn message_stop_without_terminal_reason_does_not_commit() {
        let (mut handler, mut rx) = test_handler();
        let mut state = ClaudeStreamState::default();

        for value in [
            json!({
                "type":"content_block_start",
                "index":0,
                "content_block":{"type":"text","text":""}
            }),
            json!({
                "type":"content_block_delta",
                "index":0,
                "delta":{"type":"text_delta","text":"staged answer"}
            }),
            json!({"type":"content_block_stop","index":0}),
        ] {
            handle_value(&mut state, &mut handler, value).unwrap();
        }
        let err = handle_value(
            &mut state,
            &mut handler,
            json!({"type":"message_stop"}),
        )
        .expect_err("message_stop without message_delta.stop_reason must fail");

        assert_eq!(
            err.to_string(),
            "Claude message_stop arrived before a terminal stop_reason"
        );
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        let (text, calls) = handler.take();
        assert!(text.is_empty());
        assert!(calls.is_empty());
    }

    #[tokio::test]
    async fn message_stop_dispatches_stopped_tool_call_once() -> Result<()> {
        let events = [
            json!({
                "type":"content_block_start",
                "index":0,
                "content_block":{
                    "type":"tool_use",
                    "id":"tool_complete",
                    "name":"side_effect",
                    "input":{}
                }
            }),
            json!({"type":"content_block_stop","index":0}),
            json!({"type":"message_delta","delta":{"stop_reason":"tool_use"}}),
            json!({"type":"message_stop"}),
        ];
        let body = events
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>();
        let builder = sse_fixture_builder(&body).await?;
        let (mut handler, _rx) = test_handler();

        claude_chat_completions_streaming(
            builder,
            &mut handler,
            &Model::new("claude", "test-model"),
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
    fn successful_content_reasoning_and_tool_stream_is_unchanged() -> Result<()> {
        let (mut handler, mut rx) = test_handler();
        let mut state = ClaudeStreamState::default();

        for value in [
            json!({
                "type":"content_block_start",
                "index":0,
                "content_block":{"type":"thinking","thinking":"","signature":""}
            }),
            json!({
                "type":"content_block_delta",
                "index":0,
                "delta":{"type":"thinking_delta","thinking":"reason"}
            }),
            json!({"type":"content_block_stop","index":0}),
            json!({
                "type":"content_block_start",
                "index":1,
                "content_block":{"type":"text","text":""}
            }),
            json!({
                "type":"content_block_delta",
                "index":1,
                "delta":{"type":"text_delta","text":"answer"}
            }),
            json!({"type":"content_block_stop","index":1}),
            json!({
                "type":"content_block_start",
                "index":2,
                "content_block":{"type":"tool_use","id":"tool_one","name":"first"}
            }),
            json!({"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"value\":1}"}}),
            json!({"type":"content_block_stop","index":2}),
            json!({
                "type":"content_block_start",
                "index":3,
                "content_block":{"type":"tool_use","id":"tool_two","name":"second"}
            }),
            json!({"type":"content_block_stop","index":3}),
        ] {
            handle_value(&mut state, &mut handler, value)?;
        }

        assert!(handler.tool_calls().is_empty());
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        handle_value(
            &mut state,
            &mut handler,
            json!({"type":"message_delta","delta":{"stop_reason":"end_turn"}}),
        )?;
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        handle_value(
            &mut state,
            &mut handler,
            json!({"type":"message_stop"}),
        )?;

        match rx.try_recv() {
            Ok(SseEvent::Text(text)) => {
                assert_eq!(text, "<think>\nreason\n</think>\n\nanswer")
            }
            other => panic!("expected one committed text event, got {other:?}"),
        }
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));

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
