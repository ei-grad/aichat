//! Data-driven wire-format tests: each JSON file under `tests/fixtures/wire/`
//! describes one provider response (SSE or bare-JSON transport) and either
//! the expected `ChatEvent`s or the expected classified error. Adding a
//! provider test case means adding a fixture file, not code.

use super::*;

use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct WireCase {
    transport: Transport,
    #[serde(default = "default_status")]
    status: String,
    content_type: Option<String>,
    body_lines: Vec<String>,
    expect: Expectation,
}

#[derive(Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum Transport {
    Sse,
    Json,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Expectation {
    Events(Vec<Value>),
    Error {
        message: String,
        #[serde(default)]
        transient: bool,
    },
}

fn default_status() -> String {
    "200 OK".into()
}

fn event_to_json(event: &ChatEvent) -> Value {
    match event {
        ChatEvent::Text(text) => json!({ "text": text }),
        ChatEvent::Reasoning(text) => json!({ "reasoning": text }),
        ChatEvent::ToolCall(call) => json!({
            "tool_call": {
                "name": call.name,
                "arguments": call.arguments,
                "id": call.id,
            }
        }),
        ChatEvent::Usage(usage) => json!({
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            }
        }),
    }
}

async fn open_stream(provider: &str, case: &WireCase) -> Result<ChatEventStream> {
    let body = case.body_lines.join("\n");
    let content_type = case.content_type.clone().unwrap_or_else(|| {
        match case.transport {
            Transport::Sse => "text/event-stream",
            Transport::Json => "application/json",
        }
        .to_string()
    });
    let builder = response_fixture_builder(&case.status, &content_type, &body).await?;
    let model = Model::new(provider, "test-model");
    match provider {
        "openai" => super::openai::openai_chat_events(builder, &model).await,
        "claude" => super::claude::claude_chat_events(builder, &model).await,
        "cohere" => super::cohere::cohere_chat_events(builder, &model).await,
        "gemini" => super::vertexai::gemini_chat_events(builder, &model).await,
        _ => bail!("no wire-fixture binding for provider '{provider}'"),
    }
}

async fn run_case(provider: &str, path: &Path) -> Result<()> {
    let case: WireCase = serde_json::from_str(
        &std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("invalid wire case {}", path.display()))?;

    let result = async {
        let mut stream = open_stream(provider, &case).await?;
        let mut events = vec![];
        while let Some(event) = stream.next().await {
            events.push(event?);
        }
        Ok::<_, anyhow::Error>(events)
    }
    .await;

    match (&case.expect, result) {
        (Expectation::Events(expected), Ok(events)) => {
            let actual: Vec<Value> = events.iter().map(event_to_json).collect();
            if &actual != expected {
                bail!(
                    "{}: events mismatch\nexpected: {}\nactual: {}",
                    path.display(),
                    Value::Array(expected.clone()),
                    Value::Array(actual)
                );
            }
        }
        (Expectation::Events(_), Err(err)) => {
            bail!("{}: expected events, got error: {err}", path.display());
        }
        (Expectation::Error { message, transient }, Err(err)) => {
            if err.to_string() != *message {
                bail!(
                    "{}: error message mismatch\nexpected: {message}\nactual: {err}",
                    path.display()
                );
            }
            let actual_transient = err
                .downcast_ref::<ProviderError>()
                .map(|provider_error| provider_error.is_transient())
                .unwrap_or(false);
            if actual_transient != *transient {
                bail!(
                    "{}: transient classification mismatch (expected {transient}, got {actual_transient})",
                    path.display()
                );
            }
        }
        (Expectation::Error { message, .. }, Ok(events)) => {
            bail!(
                "{}: expected error '{message}', got {} events",
                path.display(),
                events.len()
            );
        }
    }
    Ok(())
}

#[tokio::test]
async fn wire_fixtures() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wire");
    let mut ran = 0;
    let mut providers: Vec<_> = std::fs::read_dir(&root)
        .with_context(|| format!("missing fixture root {}", root.display()))?
        .collect::<std::io::Result<Vec<_>>>()?;
    providers.sort_by_key(|entry| entry.file_name());
    for provider_entry in providers {
        if !provider_entry.file_type()?.is_dir() {
            continue;
        }
        let provider = provider_entry.file_name().to_string_lossy().into_owned();
        let mut cases: Vec<_> =
            std::fs::read_dir(provider_entry.path())?.collect::<std::io::Result<Vec<_>>>()?;
        cases.sort_by_key(|entry| entry.file_name());
        for case_entry in cases {
            let path = case_entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }
            run_case(&provider, &path).await?;
            ran += 1;
        }
    }
    assert!(ran > 0, "no wire fixtures found under {}", root.display());
    Ok(())
}
