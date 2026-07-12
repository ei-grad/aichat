use super::{catch_error, classify_provider_error, ProviderError, ProviderErrorKind, ToolCall};
use crate::utils::AbortSignal;

use anyhow::{anyhow, Context, Result};
use async_stream::try_stream;
use futures_util::{Stream, StreamExt};
use reqwest::RequestBuilder;
use reqwest_eventsource::{Error as EventSourceError, Event, RequestBuilderExt};
use serde_json::Value;
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
        }
    }
    if reasoning {
        handler.text("\n</think>\n\n")?;
    }
    Ok(())
}

/// Collect a provider event stream into final text and tool calls, applying
/// the same think-tag presentation as the streaming pump.
pub async fn collect_chat_events(stream: ChatEventStream) -> Result<(String, Vec<ToolCall>)> {
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
}

impl SseHandler {
    pub fn new(sender: UnboundedSender<SseEvent>, abort_signal: AbortSignal) -> Self {
        Self {
            sender,
            abort_signal,
            buffer: String::new(),
            tool_calls: Vec::new(),
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

    pub fn abort(&self) -> AbortSignal {
        self.abort_signal.clone()
    }

    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.tool_calls
    }

    pub fn take(self) -> (String, Vec<ToolCall>) {
        let Self {
            buffer, tool_calls, ..
        } = self;
        (buffer, tool_calls)
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

/// Convert an SSE transport failure into the error to surface, delegating
/// API-level error bodies to the provider's error parser.
async fn sse_transport_failure<E>(
    err: EventSourceError,
    handle_error: &E,
    sanitize_transport_errors: bool,
) -> anyhow::Error
where
    E: Fn(&Value, u16) -> Result<()>,
{
    match err {
        EventSourceError::StreamEnded => anyhow!("SSE stream ended before protocol completion"),
        EventSourceError::InvalidStatusCode(status, res) => {
            let status = status.as_u16();
            let text = match res.text().await {
                Ok(text) => text,
                Err(err) => return err.into(),
            };
            let data: Value = match text.parse() {
                Ok(data) => data,
                Err(_) => {
                    let message = if sanitize_transport_errors {
                        format!("Streaming request failed (status: {status})")
                    } else {
                        format!("Invalid response data: {text} (status: {status})")
                    };
                    return ProviderError::new(
                        classify_provider_error(status, None),
                        message,
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
            let content_type = header_value.to_str().unwrap_or_default();
            let message = if sanitize_transport_errors {
                format!(
                    "Invalid response event-stream (status: {status}, content-type: {content_type})"
                )
            } else {
                match res.text().await {
                    Ok(text) => format!(
                        "Invalid response event-stream. content-type: {content_type}, data: {text}"
                    ),
                    Err(err) => return err.into(),
                }
            };
            ProviderError::new(ProviderErrorKind::InvalidResponse, message, Some(status)).into()
        }
        _ => anyhow!("{err}"),
    }
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
                    Err(sse_transport_failure(err, &handle_error, sanitize_transport_errors)
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
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
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
        let (text, calls) = handler.take();
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
        let (text, _) = handler.take();
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
        let (text, calls) = handler.take();
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
        let (text, _) = handler.take();
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
        let (text, calls) = handler.take();
        assert_eq!(text, "partial");
        assert!(calls.is_empty());
    }
}
