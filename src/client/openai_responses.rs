use super::registry::init_openai_client;
use super::{
    catch_error, retry_request, ChatCompletionsData, ChatCompletionsOutput, Client, Message,
    MessageContent, MessageContentPart, MessageRole, Model, TokenUsage, ToolCall,
};

use crate::config::{GlobalConfig, Input, RoleLike};
use crate::function::{eval_tool_calls_preserving_results, FunctionDeclaration};
use crate::utils::{strip_think_tag, wait_abort_signal, AbortSignal};

use anyhow::{bail, Context, Result};
use reqwest::{Client as ReqwestClient, RequestBuilder};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::num::NonZeroUsize;

const ROOT_AGENT: &str = "/root";
const MULTI_AGENT_INSTRUCTIONS: &str = "Proactive multi-agent delegation is active. Use subagents when parallel work would materially improve speed or quality.";
const MAX_CONTINUATION_TURNS: usize = 64;

pub async fn run_openai_responses_multi_agent(
    config: &GlobalConfig,
    input: &Input,
    max_concurrent_subagents: Option<NonZeroUsize>,
    abort_signal: AbortSignal,
) -> Result<ChatCompletionsOutput> {
    let model = input.role().model().clone();
    validate_multi_agent_model(&model)?;
    let client = init_openai_client(config, &model)?;
    let data = input.prepare_completion_data(&model, false)?;
    let body = build_openai_responses_multi_agent_body(data, &model, max_concurrent_subagents)?;

    if config.read().dry_run {
        return Ok(ChatCompletionsOutput::new(&serde_json::to_string_pretty(
            &body,
        )?));
    }

    let http = client.build_client()?;
    run_multi_agent_loop(
        body,
        &abort_signal,
        |body| send_openai_responses_turn(&client, &http, body),
        |calls| execute_function_calls(config, calls),
    )
    .await
}

fn validate_multi_agent_model(model: &Model) -> Result<()> {
    let model_name = model.real_name();
    if model_name == "gpt-5.6" || model_name.starts_with("gpt-5.6-") {
        return Ok(());
    }
    bail!(
        "OpenAI Responses multi-agent requires a GPT-5.6 model, but '{}' resolves to '{}'",
        model.id(),
        model_name
    )
}

fn build_openai_responses_multi_agent_body(
    data: ChatCompletionsData,
    model: &Model,
    max_concurrent_subagents: Option<NonZeroUsize>,
) -> Result<Value> {
    let ChatCompletionsData {
        messages,
        temperature,
        top_p,
        functions,
        stream: _,
        include_usage: _,
    } = data;
    let input = messages
        .into_iter()
        .map(build_responses_message)
        .collect::<Result<Vec<_>>>()?;
    let mut multi_agent = json!({"enabled": true});
    if let Some(max_concurrent_subagents) = max_concurrent_subagents {
        multi_agent["max_concurrent_subagents"] = max_concurrent_subagents.get().into();
    }
    let mut body = json!({
        "model": model.real_name(),
        "input": input,
        "instructions": MULTI_AGENT_INSTRUCTIONS,
        "store": false,
        "include": ["reasoning.encrypted_content"],
        "multi_agent": multi_agent,
    });

    if let Some(functions) = functions {
        body["tools"] = Value::Array(
            functions
                .into_iter()
                .map(build_responses_tool)
                .collect::<Vec<_>>(),
        );
    }
    if let Some(effort) = model.reasoning_effort() {
        body["reasoning"] = json!({"effort": effort});
    } else {
        if let Some(temperature) = temperature {
            body["temperature"] = temperature.into();
        }
        if let Some(top_p) = top_p {
            body["top_p"] = top_p.into();
        }
    }
    if let Some(max_output_tokens) = model.max_tokens_param() {
        body["max_output_tokens"] = max_output_tokens.into();
    }

    Ok(body)
}

fn build_responses_message(message: Message) -> Result<Value> {
    let Message { role, content } = message;
    if role == MessageRole::Tool {
        bail!("OpenAI Responses multi-agent does not accept pre-existing tool messages")
    }
    let role_name = match role {
        MessageRole::System => "system",
        MessageRole::Assistant => "assistant",
        MessageRole::User => "user",
        MessageRole::Tool => unreachable!(),
    };
    let text_type = "input_text";
    let content = match content {
        MessageContent::Text(text) => vec![json!({
            "type": text_type,
            "text": responses_message_text(role, &text),
        })],
        MessageContent::Array(parts) => parts
            .into_iter()
            .map(|part| match part {
                MessageContentPart::Text { text } => Ok(json!({
                    "type": text_type,
                    "text": responses_message_text(role, &text),
                })),
                MessageContentPart::ImageUrl { image_url } if role == MessageRole::User => {
                    Ok(json!({"type": "input_image", "image_url": image_url.url}))
                }
                MessageContentPart::ImageUrl { .. } => {
                    bail!("OpenAI Responses multi-agent only supports images in user messages")
                }
            })
            .collect::<Result<Vec<_>>>()?,
        MessageContent::ToolCalls(_) => {
            bail!("OpenAI Responses multi-agent does not accept pre-existing tool-call history")
        }
    };

    Ok(json!({"role": role_name, "content": content}))
}

fn responses_message_text(role: MessageRole, text: &str) -> String {
    if role == MessageRole::Assistant {
        strip_think_tag(text).into_owned()
    } else {
        text.to_string()
    }
}

fn build_responses_tool(function: FunctionDeclaration) -> Value {
    json!({
        "type": "function",
        "name": function.name,
        "description": function.description,
        "parameters": function.parameters,
    })
}

async fn send_openai_responses_turn(
    client: &super::OpenAIClient,
    http: &ReqwestClient,
    body: Value,
) -> Result<Value> {
    retry_request(|| {
        let request = client.prepare_responses_request(body.clone());
        async move {
            let request = request?.into_builder(http);
            send_openai_responses_request(request).await
        }
    })
    .await
    .context("Failed to call OpenAI Responses api")
}

async fn send_openai_responses_request(builder: RequestBuilder) -> Result<Value> {
    let response = builder.send().await?;
    let status = response.status();
    let data: Value = response
        .json()
        .await
        .context("Invalid OpenAI Responses JSON payload")?;
    if !status.is_success() {
        catch_error(&data, status.as_u16())?;
    }
    debug!("openai-responses-data: {data}");
    Ok(data)
}

fn execute_function_calls(
    config: &GlobalConfig,
    calls: Vec<OpenAIResponsesFunctionCall>,
) -> Result<Vec<Value>> {
    let tool_calls = calls
        .iter()
        .map(|call| {
            ToolCall::new(
                call.name.clone(),
                call.arguments.clone(),
                Some(call.call_id.clone()),
            )
        })
        .collect();
    let results = eval_tool_calls_preserving_results(config, tool_calls)?;
    let mut outputs_by_call_id = HashMap::with_capacity(results.len());
    for result in results {
        let call_id = result
            .call
            .id
            .context("A Responses tool result is missing its call_id")?;
        outputs_by_call_id.insert(call_id, result.output);
    }
    calls
        .into_iter()
        .map(|call| {
            outputs_by_call_id.remove(&call.call_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "No tool result was produced for Responses function call '{}'",
                    call.call_id
                )
            })
        })
        .collect()
}

#[derive(Debug, Clone)]
struct CachedFunctionCall {
    name: String,
    arguments: Value,
    output: Value,
}

async fn run_multi_agent_loop<S, SFut, E>(
    mut body: Value,
    abort_signal: &AbortSignal,
    mut send: S,
    mut execute: E,
) -> Result<ChatCompletionsOutput>
where
    S: FnMut(Value) -> SFut,
    SFut: Future<Output = Result<Value>>,
    E: FnMut(Vec<OpenAIResponsesFunctionCall>) -> Result<Vec<Value>>,
{
    let mut input_items = body
        .get("input")
        .and_then(Value::as_array)
        .cloned()
        .context("OpenAI Responses request input must be an array")?;
    let mut cache: HashMap<String, CachedFunctionCall> = HashMap::new();
    let mut cached_only_signatures = HashSet::new();
    let mut total_usage: Option<TokenUsage> = None;

    for _ in 0..MAX_CONTINUATION_TURNS {
        let response = tokio::select! {
            response = send(body.clone()) => response?,
            _ = wait_abort_signal(abort_signal) => bail!("Aborted."),
        };
        let result = parse_openai_responses_multi_agent(&response)?;
        total_usage = Some(match total_usage {
            Some(mut total) => {
                total.add(result.usage);
                total
            }
            None => result.usage,
        });
        input_items.extend(result.output_items.iter().cloned());

        if result.function_calls.is_empty() {
            let text = result.root_final_text.ok_or_else(|| {
                anyhow::anyhow!(
                    "OpenAI Responses request '{}' completed without a root final answer",
                    result.response_id
                )
            })?;
            let usage = total_usage.unwrap_or_default();
            return Ok(ChatCompletionsOutput {
                text,
                id: Some(result.response_id),
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                ..Default::default()
            });
        }
        if abort_signal.aborted() {
            bail!("Aborted.")
        }

        let mut new_calls = Vec::new();
        let mut pending_by_call_id: HashMap<String, usize> = HashMap::new();
        for call in &result.function_calls {
            if let Some(cached) = cache.get(&call.call_id) {
                validate_cached_call(call, cached)?;
                continue;
            }
            if let Some(index) = pending_by_call_id.get(&call.call_id) {
                validate_same_call(call, &new_calls[*index])?;
                continue;
            }
            pending_by_call_id.insert(call.call_id.clone(), new_calls.len());
            new_calls.push(call.clone());
        }

        if new_calls.is_empty() {
            let signature = cached_call_signature(&result.function_calls)?;
            if !cached_only_signatures.insert(signature) {
                bail!(
                    "OpenAI Responses multi-agent repeated the same cached function-call cycle without making progress"
                )
            }
        } else {
            cached_only_signatures.clear();
            let outputs = execute(new_calls.clone())?;
            if outputs.len() != new_calls.len() {
                bail!(
                    "Responses tool executor returned {} results for {} function calls",
                    outputs.len(),
                    new_calls.len()
                )
            }
            for (call, output) in new_calls.into_iter().zip(outputs) {
                cache.insert(
                    call.call_id,
                    CachedFunctionCall {
                        name: call.name,
                        arguments: call.arguments,
                        output,
                    },
                );
            }
        }

        for call in &result.function_calls {
            let cached = cache
                .get(&call.call_id)
                .context("Responses function-call cache lost a completed result")?;
            append_openai_function_call_output(&mut input_items, call, &cached.output);
        }
        body["input"] = Value::Array(input_items.clone());
    }

    bail!(
        "OpenAI Responses multi-agent exceeded the {MAX_CONTINUATION_TURNS}-turn continuation limit"
    )
}

fn validate_cached_call(
    call: &OpenAIResponsesFunctionCall,
    cached: &CachedFunctionCall,
) -> Result<()> {
    if call.name != cached.name || call.arguments != cached.arguments {
        bail!(
            "OpenAI Responses reused function call_id '{}' with different name or arguments",
            call.call_id
        )
    }
    Ok(())
}

fn validate_same_call(
    call: &OpenAIResponsesFunctionCall,
    original: &OpenAIResponsesFunctionCall,
) -> Result<()> {
    if call.name != original.name || call.arguments != original.arguments {
        bail!(
            "OpenAI Responses reused function call_id '{}' with different name or arguments",
            call.call_id
        )
    }
    Ok(())
}

fn cached_call_signature(calls: &[OpenAIResponsesFunctionCall]) -> Result<String> {
    let signature = calls
        .iter()
        .map(|call| {
            json!({
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
            })
        })
        .collect::<Vec<_>>();
    Ok(serde_json::to_string(&signature)?)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenAIResponsesStatus {
    Completed,
    Failed,
    Incomplete,
    Other(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpenAIResponsesFunctionCall {
    pub call_id: String,
    pub name: String,
    pub arguments: Value,
    pub agent_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpenAIResponsesMultiAgentResult {
    pub response_id: String,
    pub status: OpenAIResponsesStatus,
    pub output_items: Vec<Value>,
    pub function_calls: Vec<OpenAIResponsesFunctionCall>,
    pub root_final_text: Option<String>,
    pub usage: TokenUsage,
}

#[derive(Debug)]
pub enum OpenAIResponsesMultiAgentError {
    InvalidResponse(serde_json::Error),
    FailedResponse {
        response_id: String,
        error: Option<Value>,
    },
    IncompleteResponse {
        response_id: String,
        incomplete_details: Option<Value>,
    },
    UnexpectedStatus {
        response_id: String,
        status: String,
    },
    MalformedFunctionCall {
        output_index: usize,
        field: &'static str,
        reason: String,
    },
    MissingRootFinal {
        response_id: String,
    },
}

impl std::fmt::Display for OpenAIResponsesMultiAgentError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidResponse(error) => {
                write!(formatter, "Invalid OpenAI Responses payload: {error}")
            }
            Self::FailedResponse { response_id, error } => write!(
                formatter,
                "OpenAI Responses request '{response_id}' failed: {}",
                display_optional_value(error)
            ),
            Self::IncompleteResponse {
                response_id,
                incomplete_details,
            } => write!(
                formatter,
                "OpenAI Responses request '{response_id}' was incomplete: {}",
                display_optional_value(incomplete_details)
            ),
            Self::UnexpectedStatus {
                response_id,
                status,
            } => write!(
                formatter,
                "OpenAI Responses request '{response_id}' has unsupported status '{status}'"
            ),
            Self::MalformedFunctionCall {
                output_index,
                field,
                reason,
            } => write!(
                formatter,
                "Malformed function_call at output[{output_index}]: field '{field}' {reason}"
            ),
            Self::MissingRootFinal { response_id } => write!(
                formatter,
                "OpenAI Responses request '{response_id}' completed without a root final answer"
            ),
        }
    }
}

impl std::error::Error for OpenAIResponsesMultiAgentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidResponse(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Debug, Deserialize)]
struct ResponseEnvelope {
    id: String,
    status: String,
    output: Vec<Value>,
    usage: Option<ResponseUsage>,
    error: Option<Value>,
    incomplete_details: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct ResponseUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
}

pub fn parse_openai_responses_multi_agent(
    response: &Value,
) -> Result<OpenAIResponsesMultiAgentResult, OpenAIResponsesMultiAgentError> {
    let envelope: ResponseEnvelope = serde_json::from_value(response.clone())
        .map_err(OpenAIResponsesMultiAgentError::InvalidResponse)?;
    let status = match envelope.status.as_str() {
        "completed" => OpenAIResponsesStatus::Completed,
        "failed" => OpenAIResponsesStatus::Failed,
        "incomplete" => OpenAIResponsesStatus::Incomplete,
        _ => OpenAIResponsesStatus::Other(envelope.status.clone()),
    };

    match &status {
        OpenAIResponsesStatus::Failed => {
            return Err(OpenAIResponsesMultiAgentError::FailedResponse {
                response_id: envelope.id,
                error: envelope.error,
            });
        }
        OpenAIResponsesStatus::Incomplete => {
            return Err(OpenAIResponsesMultiAgentError::IncompleteResponse {
                response_id: envelope.id,
                incomplete_details: envelope.incomplete_details,
            });
        }
        OpenAIResponsesStatus::Other(status) => {
            return Err(OpenAIResponsesMultiAgentError::UnexpectedStatus {
                response_id: envelope.id,
                status: status.clone(),
            });
        }
        OpenAIResponsesStatus::Completed => {}
    }

    let function_calls = envelope
        .output
        .iter()
        .enumerate()
        .filter(|(_, item)| item.get("type").and_then(Value::as_str) == Some("function_call"))
        .map(|(output_index, item)| parse_function_call(output_index, item))
        .collect::<Result<Vec<_>, _>>()?;
    let root_final_text = extract_root_final_text(&envelope.output);

    if function_calls.is_empty() && root_final_text.is_none() {
        return Err(OpenAIResponsesMultiAgentError::MissingRootFinal {
            response_id: envelope.id,
        });
    }

    let usage = envelope
        .usage
        .map(|usage| TokenUsage::new(usage.input_tokens, usage.output_tokens))
        .unwrap_or_default();

    Ok(OpenAIResponsesMultiAgentResult {
        response_id: envelope.id,
        status,
        output_items: envelope.output,
        function_calls,
        root_final_text,
        usage,
    })
}

pub fn build_openai_function_call_output(
    call: &OpenAIResponsesFunctionCall,
    output: &Value,
) -> Value {
    let output = match output {
        Value::String(output) => output.clone(),
        _ => output.to_string(),
    };
    json!({
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": output,
    })
}

pub fn append_openai_function_call_output(
    items: &mut Vec<Value>,
    call: &OpenAIResponsesFunctionCall,
    output: &Value,
) {
    items.push(build_openai_function_call_output(call, output));
}

fn parse_function_call(
    output_index: usize,
    item: &Value,
) -> Result<OpenAIResponsesFunctionCall, OpenAIResponsesMultiAgentError> {
    let call_id = required_function_call_string(output_index, item, "call_id")?;
    let name = required_function_call_string(output_index, item, "name")?;
    let arguments = required_function_call_string(output_index, item, "arguments")?;
    let arguments = serde_json::from_str(arguments).map_err(|error| {
        OpenAIResponsesMultiAgentError::MalformedFunctionCall {
            output_index,
            field: "arguments",
            reason: format!("must contain valid JSON: {error}"),
        }
    })?;
    let agent_name = item
        .get("agent")
        .and_then(|agent| agent.get("agent_name"))
        .and_then(Value::as_str)
        .map(str::to_string);

    Ok(OpenAIResponsesFunctionCall {
        call_id: call_id.to_string(),
        name: name.to_string(),
        arguments,
        agent_name,
    })
}

fn required_function_call_string<'a>(
    output_index: usize,
    item: &'a Value,
    field: &'static str,
) -> Result<&'a str, OpenAIResponsesMultiAgentError> {
    item.get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| OpenAIResponsesMultiAgentError::MalformedFunctionCall {
            output_index,
            field,
            reason: "must be a non-empty string".to_string(),
        })
}

fn extract_root_final_text(output: &[Value]) -> Option<String> {
    let mut text = String::new();
    let mut found_output_text = false;

    for item in output {
        if item.get("type").and_then(Value::as_str) != Some("message")
            || item
                .get("agent")
                .and_then(|agent| agent.get("agent_name"))
                .and_then(Value::as_str)
                != Some(ROOT_AGENT)
            || item.get("phase").and_then(Value::as_str) != Some("final_answer")
        {
            continue;
        }
        let Some(content) = item.get("content").and_then(Value::as_array) else {
            continue;
        };
        for part in content {
            if part.get("type").and_then(Value::as_str) == Some("output_text") {
                if let Some(part_text) = part.get("text").and_then(Value::as_str) {
                    found_output_text = true;
                    text.push_str(part_text);
                }
            }
        }
    }

    found_output_text.then_some(text)
}

fn display_optional_value(value: &Option<Value>) -> String {
    value
        .as_ref()
        .map(Value::to_string)
        .unwrap_or_else(|| "no details".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::client::{ImageUrl, ModelData, OpenAIClient, OpenAIConfig};
    use crate::utils::create_abort_signal;
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;
    use std::future::{pending, ready};

    const MIXED_OUTPUT_FIXTURE: &str = r#"
    {
      "id": "resp_mixed",
      "status": "completed",
      "output": [
        {
          "type": "reasoning",
          "id": "rs_1",
          "encrypted_content": "enc_reasoning"
        },
        {
          "type": "multi_agent_call",
          "id": "mac_1",
          "call_id": "call_hosted",
          "name": "not_a_developer_tool",
          "arguments": "not developer JSON",
          "agent": {"agent_name": "/root"}
        },
        {
          "type": "multi_agent_call_output",
          "id": "maco_1",
          "call_id": "call_hosted",
          "output": [{"type": "output_text", "text": "hosted result"}]
        },
        {
          "type": "agent_message",
          "id": "amsg_1",
          "author": "/root/researcher",
          "recipient": "/root",
          "content": [{"type": "encrypted_content", "encrypted_content": "enc_message"}]
        },
        {
          "type": "message",
          "id": "msg_child",
          "agent": {"agent_name": "/root/researcher"},
          "phase": "final_answer",
          "content": [{"type": "output_text", "text": "child-only"}]
        },
        {
          "type": "message",
          "id": "msg_root_commentary",
          "agent": {"agent_name": "/root"},
          "phase": "commentary",
          "content": [{"type": "output_text", "text": "not-final"}]
        },
        {
          "type": "message",
          "id": "msg_root_final",
          "agent": {"agent_name": "/root"},
          "phase": "final_answer",
          "content": [
            {"type": "output_text", "text": "root "},
            {"type": "refusal", "refusal": "ignored"},
            {"type": "output_text", "text": "answer"}
          ]
        },
        {
          "type": "function_call",
          "id": "fc_child",
          "call_id": "call_child",
          "name": "lookup",
          "arguments": "{\"key\":\"alpha\"}",
          "agent": {"agent_name": "/root/researcher"}
        },
        {
          "type": "function_call",
          "id": "fc_root",
          "call_id": "call_root",
          "name": "calculate",
          "arguments": "{\"value\":2}",
          "agent": {"agent_name": "/root"}
        },
        {
          "type": "future_item",
          "id": "future_1",
          "payload": {"kept": [1, null, true]}
        }
      ],
      "usage": {"input_tokens": 41, "output_tokens": 17, "total_tokens": 58},
      "error": null,
      "incomplete_details": null
    }
    "#;

    fn fixture(value: &str) -> Value {
        serde_json::from_str(value).expect("valid test fixture")
    }

    #[test]
    fn parses_mixed_multi_agent_output_losslessly() {
        let raw = fixture(MIXED_OUTPUT_FIXTURE);
        let original_items = raw["output"].as_array().unwrap().clone();

        let result = parse_openai_responses_multi_agent(&raw).unwrap();

        assert_eq!(result.response_id, "resp_mixed");
        assert_eq!(result.status, OpenAIResponsesStatus::Completed);
        assert_eq!(result.output_items, original_items);
        assert_eq!(result.root_final_text.as_deref(), Some("root answer"));
        assert_eq!(result.usage, TokenUsage::new(Some(41), Some(17)));
        assert_eq!(
            result.function_calls,
            [
                OpenAIResponsesFunctionCall {
                    call_id: "call_child".to_string(),
                    name: "lookup".to_string(),
                    arguments: json!({"key": "alpha"}),
                    agent_name: Some("/root/researcher".to_string()),
                },
                OpenAIResponsesFunctionCall {
                    call_id: "call_root".to_string(),
                    name: "calculate".to_string(),
                    arguments: json!({"value": 2}),
                    agent_name: Some("/root".to_string()),
                },
            ]
        );
    }

    #[test]
    fn hosted_and_unknown_items_are_never_developer_function_calls() {
        let raw = fixture(MIXED_OUTPUT_FIXTURE);
        let result = parse_openai_responses_multi_agent(&raw).unwrap();

        assert_eq!(result.function_calls.len(), 2);
        assert!(result
            .function_calls
            .iter()
            .all(|call| call.call_id != "call_hosted"));
    }

    #[test]
    fn builds_and_appends_outputs_using_call_id() {
        let raw = fixture(MIXED_OUTPUT_FIXTURE);
        let result = parse_openai_responses_multi_agent(&raw).unwrap();
        let call = &result.function_calls[0];
        let output = json!({"found": true});
        let built = build_openai_function_call_output(call, &output);

        assert_eq!(
            built,
            json!({
                "type": "function_call_output",
                "call_id": "call_child",
                "output": "{\"found\":true}",
            })
        );
        assert_ne!(built["call_id"], "fc_child");

        let mut continuation = result.output_items.clone();
        let original_len = continuation.len();
        append_openai_function_call_output(&mut continuation, call, &json!("DONE"));
        assert_eq!(
            &continuation[..original_len],
            result.output_items.as_slice()
        );
        assert_eq!(
            continuation.last(),
            Some(&json!({
                "type": "function_call_output",
                "call_id": "call_child",
                "output": "DONE",
            }))
        );
        assert_eq!(
            build_openai_function_call_output(call, &Value::Null)["output"],
            "null"
        );
    }

    #[test]
    fn accepts_function_call_pause_without_root_final() {
        let raw = json!({
            "id": "resp_pause",
            "status": "completed",
            "output": [{
                "type": "function_call",
                "id": "fc_pause",
                "call_id": "call_pause",
                "name": "fetch",
                "arguments": "{}",
                "agent": {"agent_name": "/root/worker"},
            }],
            "usage": null,
            "error": null,
            "incomplete_details": null,
        });

        let result = parse_openai_responses_multi_agent(&raw).unwrap();

        assert_eq!(result.root_final_text, None);
        assert_eq!(result.function_calls.len(), 1);
        assert_eq!(result.usage, TokenUsage::default());
    }

    #[test]
    fn requires_exact_root_final_message_shape() {
        for output in [
            json!([{
                "type": "message",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "missing agent"}],
            }]),
            json!([{
                "type": "message",
                "agent": {"agent_name": "/root/child"},
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "child"}],
            }]),
            json!([{
                "type": "message",
                "agent": {"agent_name": "/root"},
                "phase": "commentary",
                "content": [{"type": "output_text", "text": "commentary"}],
            }]),
            json!([{
                "type": "message",
                "agent": {"agent_name": "/root"},
                "phase": "final_answer",
                "content": [{"type": "input_text", "text": "wrong part"}],
            }]),
        ] {
            let raw = json!({
                "id": "resp_no_root",
                "status": "completed",
                "output": output,
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "error": null,
                "incomplete_details": null,
            });

            assert!(matches!(
                parse_openai_responses_multi_agent(&raw),
                Err(OpenAIResponsesMultiAgentError::MissingRootFinal { .. })
            ));
        }
    }

    #[test]
    fn rejects_malformed_developer_function_calls() {
        for (field, call) in [
            (
                "call_id",
                json!({
                    "type": "function_call",
                    "id": "fc_only",
                    "name": "tool",
                    "arguments": "{}",
                }),
            ),
            (
                "name",
                json!({
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "",
                    "arguments": "{}",
                }),
            ),
            (
                "arguments",
                json!({
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "tool",
                    "arguments": {"not": "a string"},
                }),
            ),
        ] {
            let raw = json!({
                "id": "resp_bad_call",
                "status": "completed",
                "output": [call],
                "usage": null,
                "error": null,
                "incomplete_details": null,
            });

            assert!(matches!(
                parse_openai_responses_multi_agent(&raw),
                Err(OpenAIResponsesMultiAgentError::MalformedFunctionCall {
                    field: actual,
                    ..
                }) if actual == field
            ));
        }
    }

    #[test]
    fn rejects_non_json_function_arguments() {
        let raw = json!({
            "id": "resp_bad_json",
            "status": "completed",
            "output": [{
                "type": "function_call",
                "call_id": "call_bad_json",
                "name": "tool",
                "arguments": "{\"unfinished\":",
            }],
            "usage": null,
            "error": null,
            "incomplete_details": null,
        });

        let error = parse_openai_responses_multi_agent(&raw).unwrap_err();
        assert!(matches!(
            error,
            OpenAIResponsesMultiAgentError::MalformedFunctionCall {
                field: "arguments",
                ..
            }
        ));
        assert!(error.to_string().contains("must contain valid JSON"));
    }

    #[test]
    fn reports_failed_incomplete_and_non_terminal_statuses() {
        let failed = json!({
            "id": "resp_failed",
            "status": "failed",
            "output": [],
            "usage": null,
            "error": {"code": "server_error", "message": "failed"},
            "incomplete_details": null,
        });
        let incomplete = json!({
            "id": "resp_incomplete",
            "status": "incomplete",
            "output": [],
            "usage": null,
            "error": null,
            "incomplete_details": {"reason": "max_output_tokens"},
        });
        let in_progress = json!({
            "id": "resp_pending",
            "status": "in_progress",
            "output": [],
            "usage": null,
            "error": null,
            "incomplete_details": null,
        });

        assert!(matches!(
            parse_openai_responses_multi_agent(&failed),
            Err(OpenAIResponsesMultiAgentError::FailedResponse {
                response_id,
                error: Some(_),
            }) if response_id == "resp_failed"
        ));
        assert!(matches!(
            parse_openai_responses_multi_agent(&incomplete),
            Err(OpenAIResponsesMultiAgentError::IncompleteResponse {
                response_id,
                incomplete_details: Some(_),
            }) if response_id == "resp_incomplete"
        ));
        assert!(matches!(
            parse_openai_responses_multi_agent(&in_progress),
            Err(OpenAIResponsesMultiAgentError::UnexpectedStatus {
                response_id,
                status,
            }) if response_id == "resp_pending" && status == "in_progress"
        ));
    }

    #[test]
    fn rejects_null_output_array_as_invalid_outer_response() {
        let raw = json!({
            "id": "resp_null_output",
            "status": "completed",
            "output": null,
            "usage": null,
            "error": null,
            "incomplete_details": null,
        });

        assert!(matches!(
            parse_openai_responses_multi_agent(&raw),
            Err(OpenAIResponsesMultiAgentError::InvalidResponse(_))
        ));
    }

    fn responses_model(effort: Option<&str>) -> Model {
        let mut data = ModelData::new("gpt-5.6-sol");
        data.reasoning_efforts = effort.into_iter().map(str::to_string).collect();
        let models = Model::from_config("openai", "openai", &[data]);
        match effort {
            Some(effort) => models
                .into_iter()
                .find(|model| model.name() == format!("gpt-5.6-sol:{effort}"))
                .expect("reasoning model variant"),
            None => models.into_iter().next().expect("base model"),
        }
    }

    fn function_declaration() -> FunctionDeclaration {
        serde_json::from_value(json!({
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"}
                },
                "required": ["key"]
            }
        }))
        .expect("valid function declaration")
    }

    #[test]
    fn builds_exact_stateless_multi_agent_body() {
        let data = ChatCompletionsData {
            messages: vec![
                Message::new(
                    MessageRole::System,
                    MessageContent::Text("Follow the contract".into()),
                ),
                Message::new(
                    MessageRole::User,
                    MessageContent::Array(vec![
                        MessageContentPart::Text {
                            text: "Inspect this".into(),
                        },
                        MessageContentPart::ImageUrl {
                            image_url: ImageUrl {
                                url: "data:image/png;base64,AAAA".into(),
                            },
                        },
                    ]),
                ),
                Message::new(
                    MessageRole::Assistant,
                    MessageContent::Text("<think>private</think>Previous answer".into()),
                ),
            ],
            temperature: Some(0.3),
            top_p: Some(0.8),
            functions: Some(vec![function_declaration()]),
            stream: false,
            include_usage: true,
        };

        let body = build_openai_responses_multi_agent_body(
            data,
            &responses_model(Some("high")),
            NonZeroUsize::new(7),
        )
        .unwrap();

        assert_eq!(
            body,
            json!({
                "model": "gpt-5.6-sol",
                "input": [
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": "Follow the contract"}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Inspect this"},
                            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "input_text", "text": "Previous answer"}]
                    }
                ],
                "instructions": MULTI_AGENT_INSTRUCTIONS,
                "store": false,
                "include": ["reasoning.encrypted_content"],
                "multi_agent": {
                    "enabled": true,
                    "max_concurrent_subagents": 7
                },
                "tools": [{
                    "type": "function",
                    "name": "lookup",
                    "description": "Look up a value",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string"}
                        },
                        "required": ["key"]
                    }
                }],
                "reasoning": {"effort": "high"}
            })
        );
        assert!(body.get("temperature").is_none());
        assert!(body.get("top_p").is_none());
    }

    #[test]
    fn base_model_keeps_sampling_parameters() {
        let data = ChatCompletionsData {
            messages: vec![Message::new(
                MessageRole::User,
                MessageContent::Text("hello".into()),
            )],
            temperature: Some(0.2),
            top_p: Some(0.9),
            functions: None,
            stream: false,
            include_usage: false,
        };

        let body =
            build_openai_responses_multi_agent_body(data, &responses_model(None), None).unwrap();

        assert_eq!(body["temperature"], 0.2);
        assert_eq!(body["top_p"], 0.9);
        assert!(body.get("reasoning").is_none());
        assert!(body.get("tools").is_none());
        assert_eq!(body["multi_agent"], json!({"enabled": true}));
    }

    #[test]
    fn request_uses_responses_endpoint_beta_header_and_unpatched_body() {
        let body = json!({"model": "gpt-5.6-sol", "input": []});
        let client = OpenAIClient {
            global_config: Default::default(),
            config: OpenAIConfig {
                api_key: Some("test-key".into()),
                api_base: Some("https://example.invalid/v1/".into()),
                organization_id: Some("test-organization".into()),
                ..Default::default()
            },
            model: responses_model(None),
        };

        let request = client.prepare_responses_request(body.clone()).unwrap();

        assert_eq!(request.url, "https://example.invalid/v1/responses");
        assert_eq!(request.body, body);
        assert_eq!(
            request.headers.get("OpenAI-Beta").map(String::as_str),
            Some("responses_multi_agent=v1")
        );
        assert_eq!(
            request
                .headers
                .get("OpenAI-Organization")
                .map(String::as_str),
            Some("test-organization")
        );
        assert_eq!(
            request.headers.get("authorization").map(String::as_str),
            Some("Bearer test-key")
        );
    }

    fn function_call(call_id: &str, name: &str, arguments: Value) -> Value {
        json!({
            "type": "function_call",
            "id": format!("fc_{call_id}"),
            "call_id": call_id,
            "name": name,
            "arguments": arguments.to_string(),
            "agent": {"agent_name": "/root/worker"}
        })
    }

    fn root_final(text: &str) -> Value {
        json!({
            "type": "message",
            "id": "msg_final",
            "agent": {"agent_name": ROOT_AGENT},
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": text}]
        })
    }

    fn completed_response(id: &str, output: Vec<Value>, input: u64, output_tokens: u64) -> Value {
        json!({
            "id": id,
            "status": "completed",
            "output": output,
            "usage": {"input_tokens": input, "output_tokens": output_tokens},
            "error": null,
            "incomplete_details": null
        })
    }

    #[tokio::test]
    async fn continuation_replays_all_items_and_caches_identical_call_ids() {
        let first_reasoning = json!({
            "type": "reasoning",
            "id": "reasoning_1",
            "encrypted_content": "opaque"
        });
        let first_call = function_call("call_once", "lookup", json!({"key": "one"}));
        let repeated_call = first_call.clone();
        let second_call = function_call("call_twice", "lookup", json!({"key": "two"}));
        let responses = RefCell::new(VecDeque::from([
            completed_response(
                "resp_1",
                vec![first_reasoning.clone(), first_call.clone()],
                10,
                2,
            ),
            completed_response(
                "resp_2",
                vec![repeated_call.clone(), second_call.clone()],
                12,
                3,
            ),
            completed_response("resp_3", vec![root_final("root answer")], 8, 4),
        ]));
        let requests = RefCell::new(Vec::new());
        let executed = RefCell::new(Vec::<Vec<String>>::new());
        let body = json!({
            "model": "gpt-5.6-sol",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "go"}]}]
        });

        let output = run_multi_agent_loop(
            body,
            &create_abort_signal(),
            |body| {
                requests.borrow_mut().push(body);
                ready(Ok(responses
                    .borrow_mut()
                    .pop_front()
                    .expect("fake response")))
            },
            |calls| {
                executed
                    .borrow_mut()
                    .push(calls.iter().map(|call| call.call_id.clone()).collect());
                Ok(calls
                    .iter()
                    .map(|call| json!(format!("result:{}", call.call_id)))
                    .collect())
            },
        )
        .await
        .unwrap();

        assert_eq!(output.text, "root answer");
        assert_eq!(output.id.as_deref(), Some("resp_3"));
        assert_eq!(output.usage(), TokenUsage::new(Some(30), Some(9)));
        assert_eq!(
            *executed.borrow(),
            vec![
                vec!["call_once".to_string()],
                vec!["call_twice".to_string()]
            ]
        );

        let requests = requests.into_inner();
        assert_eq!(requests.len(), 3);
        let second_input = requests[1]["input"].as_array().unwrap();
        assert!(second_input.contains(&first_reasoning));
        assert!(second_input.contains(&first_call));
        assert_eq!(
            second_input.last(),
            Some(&json!({
                "type": "function_call_output",
                "call_id": "call_once",
                "output": "result:call_once"
            }))
        );
        let third_input = requests[2]["input"].as_array().unwrap();
        assert!(third_input.contains(&repeated_call));
        assert!(third_input.contains(&second_call));
        assert_eq!(
            third_input
                .iter()
                .filter(|item| item["call_id"] == "call_once")
                .count(),
            4
        );
    }

    #[tokio::test]
    async fn rejects_changed_payload_for_cached_call_id() {
        let responses = RefCell::new(VecDeque::from([
            completed_response(
                "resp_1",
                vec![function_call("call_1", "lookup", json!({"key": "one"}))],
                1,
                1,
            ),
            completed_response(
                "resp_2",
                vec![function_call("call_1", "lookup", json!({"key": "changed"}))],
                1,
                1,
            ),
        ]));

        let error = run_multi_agent_loop(
            json!({"input": []}),
            &create_abort_signal(),
            |_| ready(Ok(responses.borrow_mut().pop_front().unwrap())),
            |calls| Ok(vec![json!("DONE"); calls.len()]),
        )
        .await
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("reused function call_id 'call_1' with different name or arguments"));
    }

    #[tokio::test]
    async fn rejects_repeated_cached_only_cycle() {
        let call = function_call("call_1", "lookup", json!({}));
        let responses = RefCell::new(VecDeque::from([
            completed_response("resp_1", vec![call.clone()], 1, 1),
            completed_response("resp_2", vec![call.clone()], 1, 1),
            completed_response("resp_3", vec![call], 1, 1),
        ]));
        let executions = Cell::new(0);

        let error = run_multi_agent_loop(
            json!({"input": []}),
            &create_abort_signal(),
            |_| ready(Ok(responses.borrow_mut().pop_front().unwrap())),
            |calls| {
                executions.set(executions.get() + calls.len());
                Ok(vec![json!("DONE"); calls.len()])
            },
        )
        .await
        .unwrap_err();

        assert_eq!(executions.get(), 1);
        assert!(error
            .to_string()
            .contains("repeated the same cached function-call cycle"));
    }

    #[tokio::test]
    async fn enforces_hard_continuation_turn_limit() {
        let turn = Cell::new(0usize);

        let error = run_multi_agent_loop(
            json!({"input": []}),
            &create_abort_signal(),
            |_| {
                let current = turn.get();
                turn.set(current + 1);
                ready(Ok(completed_response(
                    &format!("resp_{current}"),
                    vec![function_call(
                        &format!("call_{current}"),
                        "lookup",
                        json!({"turn": current}),
                    )],
                    1,
                    1,
                )))
            },
            |calls| Ok(vec![json!("DONE"); calls.len()]),
        )
        .await
        .unwrap_err();

        assert_eq!(turn.get(), MAX_CONTINUATION_TURNS);
        assert!(error.to_string().contains(&format!(
            "exceeded the {MAX_CONTINUATION_TURNS}-turn continuation limit"
        )));
    }

    #[tokio::test]
    async fn abort_signal_cancels_a_pending_turn() {
        let abort_signal = create_abort_signal();
        abort_signal.set_ctrlc();

        let error = run_multi_agent_loop(
            json!({"input": []}),
            &abort_signal,
            |_| pending::<Result<Value>>(),
            |_| unreachable!("tools are not executed after abort"),
        )
        .await
        .unwrap_err();

        assert_eq!(error.to_string(), "Aborted.");
    }
}
