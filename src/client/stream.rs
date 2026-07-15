use super::{catch_error, classify_provider_error, ProviderError, ProviderErrorKind, ToolCall};
use crate::utils::AbortSignal;

use anyhow::{anyhow, Context, Result};
use async_stream::try_stream;
use futures_util::{Stream, StreamExt};
use reqwest::{
    header::{HeaderValue, CONTENT_TYPE},
    RequestBuilder, Response,
};
use reqwest_eventsource::{Error as EventSourceError, Event, RequestBuilderExt};
use serde_json::{Map, Value};
use std::pin::Pin;
use tokio::sync::mpsc::UnboundedSender;

/// A single typed unit of provider streaming output. Providers translate
/// their wire format into these events; presentation concerns (think-tag
/// wrapping, rendering) are applied once at the consumption boundary.
#[derive(Debug, Clone, PartialEq)]
pub enum ChatEvent {
    Text(String),
    Reasoning(String),
    ToolCall(ToolCall),
    Usage(TokenUsage),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
}

impl TokenUsage {
    pub fn new(input_tokens: Option<u64>, output_tokens: Option<u64>) -> Self {
        Self {
            input_tokens,
            output_tokens,
        }
    }

    pub fn merge(&mut self, other: Self) {
        if other.input_tokens.is_some() {
            self.input_tokens = other.input_tokens;
        }
        if other.output_tokens.is_some() {
            self.output_tokens = other.output_tokens;
        }
    }

    pub fn is_empty(self) -> bool {
        self.input_tokens.is_none() && self.output_tokens.is_none()
    }

    pub fn add(&mut self, other: Self) {
        self.input_tokens = sum_optional(self.input_tokens, other.input_tokens);
        self.output_tokens = sum_optional(self.output_tokens, other.output_tokens);
    }
}

fn sum_optional(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.saturating_add(right)),
        _ => None,
    }
}

pub type ChatEventStream = Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>;

/// Drain a provider event stream into the render handler, wrapping reasoning
/// deltas in `<think>` tags. Closes an open think tag when the stream ends,
/// so a reasoning-only or interrupted stream never leaves the tag unpaired.
pub async fn drive_chat_events(stream: ChatEventStream, handler: &mut SseHandler) -> Result<()> {
    let mut stream = stream;
    let mut reasoning = false;
    while let Some(event) = stream.next().await {
        let event = match event {
            Ok(event) => event,
            Err(err) => {
                if reasoning {
                    handler.text("\n</think>\n\n")?;
                }
                return Err(err);
            }
        };
        match event {
            ChatEvent::Text(text) => {
                if reasoning {
                    handler.text("\n</think>\n\n")?;
                    reasoning = false;
                }
                handler.text(&text)?;
            }
            ChatEvent::Reasoning(text) => {
                if !reasoning {
                    handler.text("<think>\n")?;
                    reasoning = true;
                }
                handler.text(&text)?;
            }
            ChatEvent::ToolCall(call) => {
                if reasoning {
                    handler.text("\n</think>\n\n")?;
                    reasoning = false;
                }
                handler.tool_call(call)?;
            }
            ChatEvent::Usage(usage) => handler.usage(usage),
        }
    }
    if reasoning {
        handler.text("\n</think>\n\n")?;
    }
    Ok(())
}

/// Collect a provider event stream into final text and tool calls, applying
/// the same think-tag presentation as the streaming pump.
pub async fn collect_chat_events(
    stream: ChatEventStream,
) -> Result<(String, Vec<ToolCall>, TokenUsage)> {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    let mut handler = SseHandler::new(tx, crate::utils::create_abort_signal());
    drive_chat_events(stream, &mut handler).await?;
    Ok(handler.take())
}

pub struct SseHandler {
    sender: UnboundedSender<SseEvent>,
    abort_signal: AbortSignal,
    buffer: String,
    tool_calls: Vec<ToolCall>,
    usage: TokenUsage,
}

impl SseHandler {
    pub fn new(sender: UnboundedSender<SseEvent>, abort_signal: AbortSignal) -> Self {
        Self {
            sender,
            abort_signal,
            buffer: String::new(),
            tool_calls: Vec::new(),
            usage: TokenUsage::default(),
        }
    }

    pub fn text(&mut self, text: &str) -> Result<()> {
        // debug!("HandleText: {}", text);
        if text.is_empty() {
            return Ok(());
        }
        self.buffer.push_str(text);
        let ret = self
            .sender
            .send(SseEvent::Text(text.to_string()))
            .with_context(|| "Failed to send SseEvent:Text");
        if let Err(err) = ret {
            if self.abort_signal.aborted() {
                return Ok(());
            }
            return Err(err);
        }
        Ok(())
    }

    pub fn done(&mut self) {
        // debug!("HandleDone");
        let ret = self.sender.send(SseEvent::Done);
        if ret.is_err() {
            if self.abort_signal.aborted() {
                return;
            }
            warn!("Failed to send SseEvent:Done");
        }
    }

    pub fn tool_call(&mut self, call: ToolCall) -> Result<()> {
        // debug!("HandleCall: {:?}", call);
        self.tool_calls.push(call);
        Ok(())
    }

    pub fn usage(&mut self, usage: TokenUsage) {
        self.usage.merge(usage);
    }

    pub fn abort(&self) -> AbortSignal {
        self.abort_signal.clone()
    }

    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.tool_calls
    }

    pub fn take(self) -> (String, Vec<ToolCall>, TokenUsage) {
        let Self {
            buffer,
            tool_calls,
            usage,
            ..
        } = self;
        (buffer, tool_calls, usage)
    }
}

#[derive(Debug)]
pub enum SseEvent {
    Text(String),
    Done,
}

#[derive(Debug)]
pub struct SseMessage {
    #[allow(unused)]
    pub event: String,
    pub data: String,
}

const MAX_SANITIZED_ERROR_BODY_BYTES: usize = 64 * 1024;
const MAX_SANITIZED_ERROR_MESSAGE_BYTES: usize = 2048;
const MAX_SANITIZED_ERROR_CODE_BYTES: usize = 128;

struct BoundedResponseBody {
    bytes: Vec<u8>,
    observed_len: usize,
    truncated: bool,
}

/// Convert an SSE transport failure into the error to surface. In sanitized
/// mode, provider parsers receive only bounded, allowlisted error fields.
pub(crate) async fn sse_transport_failure<E>(
    err: EventSourceError,
    handle_error: &E,
    sanitize_transport_errors: bool,
    client_request_id: Option<&str>,
) -> anyhow::Error
where
    E: Fn(&Value, u16) -> Result<()>,
{
    match err {
        EventSourceError::StreamEnded if sanitize_transport_errors => {
            let client_request_id = normalized_diagnostic_value(client_request_id, false);
            match client_request_id {
                Some(client_request_id) => anyhow!(
                    "SSE stream ended before protocol completion (x-client-request-id: {client_request_id})"
                ),
                None => anyhow!("SSE stream ended before protocol completion"),
            }
        }
        EventSourceError::StreamEnded => anyhow!("SSE stream ended before protocol completion"),
        EventSourceError::InvalidStatusCode(status, res) => {
            let status = status.as_u16();
            if sanitize_transport_errors {
                let content_type = normalized_header(res.headers().get(CONTENT_TYPE), true);
                let request_id = normalized_header(res.headers().get("x-request-id"), false);
                let client_request_id = normalized_diagnostic_value(client_request_id, false);
                let body = match read_bounded_response_body(res).await {
                    Ok(body) => body,
                    Err(_) => {
                        let diagnostics = transport_diagnostics(
                            status,
                            content_type.as_deref(),
                            None,
                            request_id.as_deref(),
                            client_request_id.as_deref(),
                        );
                        return ProviderError::new(
                            classify_provider_error(status, None),
                            format!("Failed to read streaming error response ({diagnostics})"),
                            Some(status),
                        )
                        .into();
                    }
                };
                let diagnostics = transport_diagnostics(
                    status,
                    content_type.as_deref(),
                    Some(&body),
                    request_id.as_deref(),
                    client_request_id.as_deref(),
                );
                let data = (!body.truncated)
                    .then(|| serde_json::from_slice::<Value>(&body.bytes).ok())
                    .flatten()
                    .and_then(|data| sanitize_provider_error_data(&data));
                if let Some(data) = data.as_ref() {
                    if let Err(err) = handle_error(data, status) {
                        if let Some(provider_error) = err.downcast_ref::<ProviderError>() {
                            return provider_error
                                .with_message_suffix(&format!(" ({diagnostics})"))
                                .into();
                        }
                    }
                }
                return ProviderError::new(
                    classify_provider_error(status, provider_error_hint(data.as_ref())),
                    format!("Streaming request failed ({diagnostics})"),
                    Some(status),
                )
                .into();
            }

            let text = match res.text().await {
                Ok(text) => text,
                Err(err) => return err.into(),
            };
            let data: Value = match text.parse() {
                Ok(data) => data,
                Err(_) => {
                    return ProviderError::new(
                        classify_provider_error(status, None),
                        format!("Invalid response data: {text} (status: {status})"),
                        Some(status),
                    )
                    .into();
                }
            };
            match handle_error(&data, status) {
                Ok(()) => anyhow!("Streaming request failed (status: {status})"),
                Err(err) => err,
            }
        }
        EventSourceError::InvalidContentType(header_value, res) => {
            let status = res.status().as_u16();
            let message = if sanitize_transport_errors {
                let content_type = normalized_header(Some(&header_value), true);
                let request_id = normalized_header(res.headers().get("x-request-id"), false);
                let client_request_id = normalized_diagnostic_value(client_request_id, false);
                let body = match read_bounded_response_body(res).await {
                    Ok(body) => body,
                    Err(_) => {
                        let diagnostics = transport_diagnostics(
                            status,
                            content_type.as_deref(),
                            None,
                            request_id.as_deref(),
                            client_request_id.as_deref(),
                        );
                        return ProviderError::new(
                            ProviderErrorKind::InvalidResponse,
                            format!("Failed to read invalid event-stream response ({diagnostics})"),
                            Some(status),
                        )
                        .into();
                    }
                };
                let diagnostics = transport_diagnostics(
                    status,
                    content_type.as_deref(),
                    Some(&body),
                    request_id.as_deref(),
                    client_request_id.as_deref(),
                );
                format!("Invalid response event-stream ({diagnostics})")
            } else {
                let content_type = header_value.to_str().unwrap_or_default();
                match res.text().await {
                    Ok(text) => format!(
                        "Invalid response event-stream. content-type: {content_type}, data: {text}"
                    ),
                    Err(err) => return err.into(),
                }
            };
            ProviderError::new(ProviderErrorKind::InvalidResponse, message, Some(status)).into()
        }
        _ if sanitize_transport_errors => {
            let client_request_id = normalized_diagnostic_value(client_request_id, false);
            match client_request_id {
                Some(client_request_id) => {
                    anyhow!("SSE transport failed (x-client-request-id: {client_request_id})")
                }
                None => anyhow!("SSE transport failed"),
            }
        }
        _ => anyhow!("{err}"),
    }
}

async fn read_bounded_response_body(response: Response) -> Result<BoundedResponseBody> {
    let mut stream = response.bytes_stream();
    let mut bytes = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let remaining = (MAX_SANITIZED_ERROR_BODY_BYTES + 1).saturating_sub(bytes.len());
        let copy_len = remaining.min(chunk.len());
        bytes.extend_from_slice(&chunk[..copy_len]);
        if bytes.len() > MAX_SANITIZED_ERROR_BODY_BYTES || copy_len < chunk.len() {
            let observed_len = bytes.len();
            bytes.truncate(MAX_SANITIZED_ERROR_BODY_BYTES);
            return Ok(BoundedResponseBody {
                bytes,
                observed_len,
                truncated: true,
            });
        }
    }
    Ok(BoundedResponseBody {
        observed_len: bytes.len(),
        bytes,
        truncated: false,
    })
}

fn sanitize_provider_error_data(data: &Value) -> Option<Value> {
    let error = data.get("error")?.as_object()?;
    let mut sanitized = Map::new();
    for (field, max_bytes) in [
        ("type", MAX_SANITIZED_ERROR_CODE_BYTES),
        ("code", MAX_SANITIZED_ERROR_CODE_BYTES),
        ("message", MAX_SANITIZED_ERROR_MESSAGE_BYTES),
    ] {
        let Some(value) = error
            .get(field)
            .and_then(Value::as_str)
            .and_then(|value| sanitize_error_field(value, max_bytes))
        else {
            continue;
        };
        sanitized.insert(field.to_string(), Value::String(value));
    }
    (!sanitized.is_empty()).then(|| {
        let mut envelope = Map::new();
        envelope.insert("error".to_string(), Value::Object(sanitized));
        Value::Object(envelope)
    })
}

fn provider_error_hint(data: Option<&Value>) -> Option<&str> {
    let error = data?.get("error")?;
    error
        .get("type")
        .and_then(Value::as_str)
        .or_else(|| error.get("code").and_then(Value::as_str))
}

fn sanitize_error_field(value: &str, max_bytes: usize) -> Option<String> {
    let mut sanitized = String::with_capacity(value.len().min(max_bytes));
    let mut pending_space = false;
    for character in value.chars() {
        let character = if character.is_control()
            || character.is_whitespace()
            || is_unicode_format_control(character)
        {
            pending_space = !sanitized.is_empty();
            continue;
        } else {
            character
        };
        if sanitized.len() + usize::from(pending_space) + character.len_utf8() > max_bytes {
            break;
        }
        if pending_space {
            sanitized.push(' ');
            pending_space = false;
        }
        sanitized.push(character);
    }
    let sanitized = sanitized.trim();
    (!sanitized.is_empty()).then(|| sanitized.to_string())
}

fn is_unicode_format_control(character: char) -> bool {
    matches!(
        character,
        '\u{061c}'
            | '\u{200b}'..='\u{200f}'
            | '\u{202a}'..='\u{202e}'
            | '\u{2060}'..='\u{206f}'
            | '\u{feff}'
    )
}

fn normalized_header(value: Option<&HeaderValue>, lowercase: bool) -> Option<String> {
    let value = value?.to_str().ok()?;
    normalized_diagnostic_value(Some(value), lowercase)
}

fn normalized_diagnostic_value(value: Option<&str>, lowercase: bool) -> Option<String> {
    let value = value?;
    let mut normalized = String::with_capacity(value.len().min(256));
    let mut pending_space = false;

    for ch in value.chars() {
        if normalized.len() >= 256 {
            break;
        }
        if ch.is_ascii_whitespace() {
            pending_space = !normalized.is_empty();
            continue;
        }
        if !ch.is_ascii_graphic() {
            continue;
        }
        if normalized.len() + usize::from(pending_space) + 1 > 256 {
            break;
        }
        if pending_space {
            normalized.push(' ');
            pending_space = false;
        }
        normalized.push(if lowercase {
            ch.to_ascii_lowercase()
        } else {
            ch
        });
    }

    (!normalized.is_empty()).then_some(normalized)
}

fn transport_diagnostics(
    status: u16,
    content_type: Option<&str>,
    body: Option<&BoundedResponseBody>,
    request_id: Option<&str>,
    client_request_id: Option<&str>,
) -> String {
    let mut diagnostics = vec![format!("status: {status}")];
    if let Some(content_type) = content_type {
        diagnostics.push(format!("content-type: {content_type}"));
    }
    if let Some(body) = body {
        if body.truncated {
            diagnostics.push(format!("body-bytes: >= {}", body.observed_len));
        } else {
            diagnostics.push(format!("body-bytes: {}", body.observed_len));
        }
    }
    if let Some(request_id) = request_id {
        diagnostics.push(format!("x-request-id: {request_id}"));
    }
    if let Some(client_request_id) = client_request_id {
        diagnostics.push(format!("x-client-request-id: {client_request_id}"));
    }
    diagnostics.join(", ")
}

/// SSE-backed provider event stream. `handle` translates one SSE message
/// into zero or more [`ChatEvent`]s pushed to the scratch buffer and returns
/// `true` once the wire protocol signals completion; ending without that
/// signal is reported as a truncated stream.
pub(crate) fn sse_chat_event_stream<F, E>(
    builder: RequestBuilder,
    mut handle: F,
    handle_error: E,
    sanitize_transport_errors: bool,
) -> ChatEventStream
where
    F: FnMut(SseMessage, &mut Vec<ChatEvent>) -> Result<bool> + Send + 'static,
    E: Fn(&Value, u16) -> Result<()> + Send + Sync + 'static,
{
    Box::pin(try_stream! {
        let mut es = builder.eventsource()?;
        let mut events: Vec<ChatEvent> = Vec::new();
        let mut done = false;
        while let Some(event) = es.next().await {
            match event {
                Ok(Event::Open) => {}
                Ok(Event::Message(message)) => {
                    let message = SseMessage {
                        event: message.event,
                        data: message.data,
                    };
                    done = handle(message, &mut events)?;
                    for event in events.drain(..) {
                        yield event;
                    }
                    if done {
                        es.close();
                        break;
                    }
                }
                Err(err) => {
                    Err(sse_transport_failure(
                        err,
                        &handle_error,
                        sanitize_transport_errors,
                        None,
                    )
                    .await)?;
                }
            }
        }
        if !done {
            Err(anyhow!("SSE stream ended before protocol completion"))?;
        }
    })
}

pub(crate) fn sse_chat_events<F>(builder: RequestBuilder, handle: F) -> ChatEventStream
where
    F: FnMut(SseMessage, &mut Vec<ChatEvent>) -> Result<bool> + Send + 'static,
{
    sse_chat_event_stream(builder, handle, catch_error, false)
}

#[cfg(test)]
pub(crate) async fn sse_fixture_builder(body: &str) -> Result<RequestBuilder> {
    response_fixture_builder("200 OK", "text/event-stream", body).await
}

#[cfg(test)]
pub(crate) async fn response_fixture_builder(
    status: &str,
    content_type: &str,
    body: &str,
) -> Result<RequestBuilder> {
    response_fixture_builder_with_headers(status, content_type, &[], body).await
}

#[cfg(test)]
pub(crate) async fn response_fixture_builder_with_headers(
    status: &str,
    content_type: &str,
    headers: &[(&str, &str)],
    body: &str,
) -> Result<RequestBuilder> {
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let extra_headers = headers
        .iter()
        .map(|(name, value)| format!("{name}: {value}\r\n"))
        .collect::<String>();
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\n{extra_headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("fixture accept failed");
        let mut request = [0; 4096];
        let request_len = socket
            .read(&mut request)
            .await
            .expect("fixture request read failed");
        assert!(request_len > 0, "fixture received an empty request");
        socket
            .write_all(response.as_bytes())
            .await
            .expect("fixture response write failed");
        socket.shutdown().await.expect("fixture shutdown failed");
    });

    Ok(reqwest::Client::new().get(format!("http://{address}")))
}

/// JSON-framed provider event stream (bare JSON/NDJSON bodies rather than
/// SSE). `handle` translates one complete JSON value into zero or more
/// [`ChatEvent`]s. Unlike SSE there is no in-band completion signal; the
/// stream ends with the response body.
pub(crate) fn json_chat_event_stream<S, F, E>(mut stream: S, mut handle: F) -> ChatEventStream
where
    S: Stream<Item = Result<bytes::Bytes, E>> + Unpin + Send + 'static,
    F: FnMut(&str, &mut Vec<ChatEvent>) -> Result<()> + Send + 'static,
    E: std::error::Error + Send + 'static,
{
    Box::pin(try_stream! {
        let mut parser = JsonStreamParser::default();
        let mut events: Vec<ChatEvent> = Vec::new();
        let mut unparsed_bytes = vec![];
        while let Some(chunk_bytes) = stream.next().await {
            let chunk_bytes =
                chunk_bytes.map_err(|err| anyhow!("Failed to read json stream, {err}"))?;
            unparsed_bytes.extend(chunk_bytes);
            match std::str::from_utf8(&unparsed_bytes) {
                Ok(text) => {
                    parser.process(text, &mut |value: &str| handle(value, &mut events))?;
                    unparsed_bytes.clear();
                }
                Err(_) => {
                    continue;
                }
            }
            for event in events.drain(..) {
                yield event;
            }
        }
        if !unparsed_bytes.is_empty() {
            let text = std::str::from_utf8(&unparsed_bytes)?;
            parser.process(text, &mut |value: &str| handle(value, &mut events))?;
            for event in events.drain(..) {
                yield event;
            }
        }
    })
}

#[derive(Debug, Default)]
struct JsonStreamParser {
    buffer: Vec<char>,
    cursor: usize,
    start: Option<usize>,
    balances: Vec<char>,
    quoting: bool,
    escape: bool,
}

impl JsonStreamParser {
    fn process<F>(&mut self, text: &str, handle: &mut F) -> Result<()>
    where
        F: FnMut(&str) -> Result<()>,
    {
        self.buffer.extend(text.chars());

        for i in self.cursor..self.buffer.len() {
            let ch = self.buffer[i];
            if self.quoting {
                if ch == '\\' {
                    self.escape = !self.escape;
                } else {
                    if !self.escape && ch == '"' {
                        self.quoting = false;
                    }
                    self.escape = false;
                }
                continue;
            }
            match ch {
                '"' => {
                    self.quoting = true;
                    self.escape = false;
                }
                '{' => {
                    if self.balances.is_empty() {
                        self.start = Some(i);
                    }
                    self.balances.push(ch);
                }
                '[' => {
                    if self.start.is_some() {
                        self.balances.push(ch);
                    }
                }
                '}' => {
                    self.balances.pop();
                    if self.balances.is_empty() {
                        if let Some(start) = self.start.take() {
                            let value: String = self.buffer[start..=i].iter().collect();
                            handle(&value)?;
                        }
                    }
                }
                ']' => {
                    self.balances.pop();
                }
                _ => {}
            }
        }
        self.cursor = self.buffer.len();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use bytes::Bytes;
    use futures_util::stream;
    use rand::Rng;

    fn split_chunks(text: &str) -> Vec<Vec<u8>> {
        let mut rng = rand::rng();
        let len = text.len();
        let cut1 = rng.random_range(1..len - 1);
        let cut2 = rng.random_range(cut1 + 1..len);
        let chunk1 = text.as_bytes()[..cut1].to_vec();
        let chunk2 = text.as_bytes()[cut1..cut2].to_vec();
        let chunk3 = text.as_bytes()[cut2..].to_vec();
        vec![chunk1, chunk2, chunk3]
    }

    macro_rules! assert_json_stream {
        ($input:expr, $output:expr) => {
            let chunks: Vec<_> = split_chunks($input)
                .into_iter()
                .map(|chunk| Ok::<_, std::convert::Infallible>(Bytes::from(chunk)))
                .collect();
            let stream = stream::iter(chunks);
            let mut events = json_chat_event_stream(stream, |data, events| {
                events.push(ChatEvent::Text(data.to_string()));
                Ok(())
            });
            let mut output = vec![];
            while let Some(event) = events.next().await {
                match event.expect("json event stream must not fail") {
                    ChatEvent::Text(text) => output.push(text),
                    other => panic!("unexpected event: {other:?}"),
                }
            }
            assert_eq!($output.replace("\r\n", "\n"), output.join("\n"))
        };
    }

    #[tokio::test]
    async fn test_json_stream_ndjson() {
        let data = r#"{"key": "value"}
{"key": "value2"}
{"key": "value3"}"#;
        assert_json_stream!(data, data);
    }

    #[tokio::test]
    async fn test_json_stream_array() {
        let input = r#"[
{"key": "value"},
{"key": "value2"},
{"key": "value3"},"#;
        let output = r#"{"key": "value"}
{"key": "value2"}
{"key": "value3"}"#;
        assert_json_stream!(input, output);
    }

    fn event_stream(events: Vec<Result<ChatEvent>>) -> ChatEventStream {
        Box::pin(stream::iter(events))
    }

    fn pump_handler() -> (SseHandler, tokio::sync::mpsc::UnboundedReceiver<SseEvent>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        (SseHandler::new(tx, crate::utils::create_abort_signal()), rx)
    }

    #[tokio::test]
    async fn pump_wraps_reasoning_in_think_tags() -> Result<()> {
        let (mut handler, _rx) = pump_handler();
        drive_chat_events(
            event_stream(vec![
                Ok(ChatEvent::Reasoning("thought".into())),
                Ok(ChatEvent::Text("answer".into())),
            ]),
            &mut handler,
        )
        .await?;
        let (text, calls, _) = handler.take();
        assert_eq!(text, "<think>\nthought\n</think>\n\nanswer");
        assert!(calls.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn pump_closes_think_tag_when_stream_ends_during_reasoning() -> Result<()> {
        let (mut handler, _rx) = pump_handler();
        drive_chat_events(
            event_stream(vec![Ok(ChatEvent::Reasoning("only thought".into()))]),
            &mut handler,
        )
        .await?;
        let (text, _, _) = handler.take();
        assert_eq!(text, "<think>\nonly thought\n</think>\n\n");
        Ok(())
    }

    #[tokio::test]
    async fn pump_closes_think_tag_before_tool_call() -> Result<()> {
        let (mut handler, _rx) = pump_handler();
        drive_chat_events(
            event_stream(vec![
                Ok(ChatEvent::Reasoning("planning".into())),
                Ok(ChatEvent::ToolCall(ToolCall::new(
                    "search".into(),
                    serde_json::json!({}),
                    Some("call_1".into()),
                ))),
            ]),
            &mut handler,
        )
        .await?;
        let (text, calls, _) = handler.take();
        assert_eq!(text, "<think>\nplanning\n</think>\n\n");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        Ok(())
    }

    #[tokio::test]
    async fn pump_closes_think_tag_before_propagating_stream_error() {
        let (mut handler, _rx) = pump_handler();
        let err = drive_chat_events(
            event_stream(vec![
                Ok(ChatEvent::Reasoning("thinking".into())),
                Err(anyhow!("stream broke")),
            ]),
            &mut handler,
        )
        .await
        .expect_err("stream error must propagate");
        assert_eq!(err.to_string(), "stream broke");
        let (text, _, _) = handler.take();
        assert_eq!(text, "<think>\nthinking\n</think>\n\n");
    }

    #[tokio::test]
    async fn pump_propagates_stream_errors_after_partial_text() {
        let (mut handler, _rx) = pump_handler();
        let err = drive_chat_events(
            event_stream(vec![
                Ok(ChatEvent::Text("partial".into())),
                Err(anyhow!("stream broke")),
            ]),
            &mut handler,
        )
        .await
        .expect_err("stream error must propagate");
        assert_eq!(err.to_string(), "stream broke");
        let (text, calls, _) = handler.take();
        assert_eq!(text, "partial");
        assert!(calls.is_empty());
    }

    #[tokio::test]
    async fn pump_merges_split_usage_events() -> Result<()> {
        let (mut handler, _rx) = pump_handler();
        drive_chat_events(
            event_stream(vec![
                Ok(ChatEvent::Usage(TokenUsage::new(Some(120), Some(0)))),
                Ok(ChatEvent::Usage(TokenUsage::new(None, Some(30)))),
            ]),
            &mut handler,
        )
        .await?;
        let (_, _, usage) = handler.take();
        assert_eq!(usage, TokenUsage::new(Some(120), Some(30)));
        Ok(())
    }

    #[test]
    fn usage_adds_multiple_completion_rounds() {
        let mut usage = TokenUsage::new(Some(100), Some(20));
        usage.add(TokenUsage::new(Some(180), Some(40)));
        assert_eq!(usage, TokenUsage::new(Some(280), Some(60)));
    }

    #[test]
    fn usage_total_is_unavailable_when_any_round_omits_usage() {
        let mut usage = TokenUsage::new(Some(100), Some(20));
        usage.add(TokenUsage::default());
        assert_eq!(usage, TokenUsage::default());
    }

    async fn sanitized_fixture_error(
        status: &str,
        content_type: &str,
        request_id: &str,
        body: &str,
    ) -> Result<anyhow::Error> {
        let builder = response_fixture_builder_with_headers(
            status,
            content_type,
            &[("X-Request-Id", request_id)],
            body,
        )
        .await?;
        let mut stream =
            sse_chat_event_stream(builder, |_message, _events| Ok(true), catch_error, true);
        let event = stream
            .next()
            .await
            .expect("fixture stream must produce a transport result");
        Ok(event.expect_err("fixture transport must fail"))
    }

    #[tokio::test]
    async fn sanitized_invalid_status_reports_safe_transport_diagnostics() -> Result<()> {
        const BODY_SENTINEL: &str = "<html>UPSTREAM_GATEWAY_BODY_SENTINEL</html>";
        let cases = [
            (
                "502 Bad Gateway",
                "Text/Plain; Charset=UTF-8",
                "req-empty-502",
                "",
                502,
                "Streaming request failed (status: 502, content-type: text/plain; charset=utf-8, body-bytes: 0, x-request-id: req-empty-502)",
            ),
            (
                "504 Gateway Timeout",
                "Text/HTML; Charset=UTF-8",
                "req-html-504",
                BODY_SENTINEL,
                504,
                "Streaming request failed (status: 504, content-type: text/html; charset=utf-8, body-bytes: 43, x-request-id: req-html-504)",
            ),
        ];

        for (status_line, content_type, request_id, body, status, expected_message) in cases {
            let err = sanitized_fixture_error(status_line, content_type, request_id, body).await?;
            assert_eq!(err.to_string(), expected_message);
            assert!(!err.to_string().contains(BODY_SENTINEL));
            assert_eq!(
                err.downcast_ref::<ProviderError>(),
                Some(&ProviderError::new(
                    ProviderErrorKind::ServerError,
                    expected_message,
                    Some(status),
                ))
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn sanitized_invalid_content_type_does_not_expose_body() -> Result<()> {
        const BODY_SENTINEL: &str = "<html>CONTENT_TYPE_BODY_SENTINEL</html>";
        let err = sanitized_fixture_error(
            "200 OK",
            "Application/JSON; Charset=UTF-8",
            "req-content-type",
            BODY_SENTINEL,
        )
        .await?;
        let expected_message = "Invalid response event-stream (status: 200, content-type: application/json; charset=utf-8, body-bytes: 39, x-request-id: req-content-type)";

        assert_eq!(err.to_string(), expected_message);
        assert!(!err.to_string().contains(BODY_SENTINEL));
        assert_eq!(
            err.downcast_ref::<ProviderError>(),
            Some(&ProviderError::new(
                ProviderErrorKind::InvalidResponse,
                expected_message,
                Some(200),
            ))
        );
        Ok(())
    }

    #[tokio::test]
    async fn sanitized_invalid_status_keeps_provider_json_error() -> Result<()> {
        let body = serde_json::json!({
            "error": {
                "type": "overloaded_error",
                "message": "temporary provider failure"
            }
        })
        .to_string();
        let err = sanitized_fixture_error(
            "502 Bad Gateway",
            "Application/JSON",
            "req-provider-json",
            &body,
        )
        .await?;
        let expected_message = format!(
            "temporary provider failure (type: overloaded_error) (status: 502, content-type: application/json, body-bytes: {}, x-request-id: req-provider-json)",
            body.len()
        );

        assert_eq!(err.to_string(), expected_message);
        assert_eq!(
            err.downcast_ref::<ProviderError>(),
            Some(&ProviderError::new(
                ProviderErrorKind::ServerError,
                expected_message,
                Some(502),
            ))
        );
        Ok(())
    }

    #[tokio::test]
    async fn sanitized_invalid_status_rejects_unknown_json_details() -> Result<()> {
        const SENTINEL: &str = "UNKNOWN_JSON_BODY_SENTINEL";
        let body = serde_json::json!({
            "debug": SENTINEL,
            "request": {"prompt": "must-not-be-rendered"}
        })
        .to_string();
        let err = sanitized_fixture_error(
            "500 Internal Server Error",
            "Application/JSON",
            "req-unknown-json",
            &body,
        )
        .await?;
        let message = err.to_string();

        assert!(message.starts_with("Streaming request failed (status: 500"));
        assert!(message.contains("x-request-id: req-unknown-json"));
        assert!(!message.contains(SENTINEL));
        assert!(!message.contains("must-not-be-rendered"));
        assert!(err.downcast_ref::<ProviderError>().is_some());
        Ok(())
    }

    #[tokio::test]
    async fn sanitized_provider_message_removes_controls_and_is_bounded() -> Result<()> {
        const TAIL_SENTINEL: &str = "MESSAGE_TAIL_SENTINEL";
        let provider_message = format!(
            "visible\n\u{2028}\u{2029}\u{202e}{}{}",
            "x".repeat(MAX_SANITIZED_ERROR_MESSAGE_BYTES + 512),
            TAIL_SENTINEL
        );
        let body = serde_json::json!({
            "error": {
                "type": "overloaded_error",
                "message": provider_message
            }
        })
        .to_string();
        let err = sanitized_fixture_error(
            "503 Service Unavailable",
            "Application/JSON",
            "req-bounded-message",
            &body,
        )
        .await?;
        let message = err.to_string();

        assert!(message.contains("visible x"));
        assert!(!message.contains('\n'));
        assert!(!message.contains('\u{2028}'));
        assert!(!message.contains('\u{2029}'));
        assert!(!message.contains('\u{202e}'));
        assert!(!message.contains(TAIL_SENTINEL));
        assert!(message.len() < MAX_SANITIZED_ERROR_MESSAGE_BYTES + 512);
        Ok(())
    }

    #[tokio::test]
    async fn sanitized_error_body_read_is_capped() -> Result<()> {
        const TAIL_SENTINEL: &str = "OVERSIZED_BODY_TAIL_SENTINEL";
        let body = format!(
            "{}{}",
            "z".repeat(MAX_SANITIZED_ERROR_BODY_BYTES + 2048),
            TAIL_SENTINEL
        );
        let err =
            sanitized_fixture_error("502 Bad Gateway", "Text/Plain", "req-oversized-body", &body)
                .await?;
        let message = err.to_string();

        assert!(message.contains(&format!(
            "body-bytes: >= {}",
            MAX_SANITIZED_ERROR_BODY_BYTES + 1
        )));
        assert!(!message.contains(TAIL_SENTINEL));
        assert!(message.len() < 512);
        Ok(())
    }

    #[test]
    fn diagnostic_headers_are_normalized_and_bounded() {
        let content_type = HeaderValue::from_static("Text/HTML;  Charset=UTF-8");
        assert_eq!(
            normalized_header(Some(&content_type), true).as_deref(),
            Some("text/html; charset=utf-8")
        );

        let request_id = HeaderValue::from_bytes(b"  REQ-ABC\t shard  ")
            .expect("fixture request ID must be a valid header");
        assert_eq!(
            normalized_header(Some(&request_id), false).as_deref(),
            Some("REQ-ABC shard")
        );

        let oversized = HeaderValue::from_str(&"A".repeat(300))
            .expect("fixture request ID must be a valid header");
        assert_eq!(
            normalized_header(Some(&oversized), false)
                .expect("header must remain present")
                .len(),
            256
        );
    }
}
