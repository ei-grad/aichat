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
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
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
) -> Result<OpenAIResponsesOutput> {
    let model = input.role().model().clone();
    validate_multi_agent_model(&model)?;
    let client = init_openai_client(config, &model)?;
    let data = input.prepare_completion_data(&model, false)?;
    let body = build_openai_responses_multi_agent_body(data, &model, max_concurrent_subagents)?;

    if config.read().dry_run {
        return Ok(OpenAIResponsesOutput {
            completion: ChatCompletionsOutput::new(&serde_json::to_string_pretty(&body)?),
            turns: Vec::new(),
            pricing_context: OpenAIResponsesPricingContext::UnknownApiBase,
        });
    }

    let pricing_context = if client.responses_uses_public_pricing()? {
        OpenAIResponsesPricingContext::PublicApi
    } else {
        OpenAIResponsesPricingContext::UnknownApiBase
    };
    let http = client.build_client()?;
    let mut output = run_multi_agent_loop(
        body,
        &abort_signal,
        |body| send_openai_responses_turn(&client, &http, body),
        |calls| execute_function_calls(config, calls),
    )
    .await?;
    output.pricing_context = pricing_context;
    Ok(output)
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
        "service_tier": "auto",
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
) -> Result<OpenAIResponsesOutput>
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
    let mut turns = Vec::new();

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
        turns.push(OpenAIResponsesTurn {
            response_id: result.response_id.clone(),
            service_tier: result.service_tier.clone(),
            usage: result.detailed_usage.clone(),
            trace: result.trace.clone(),
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
            return Ok(OpenAIResponsesOutput {
                completion: ChatCompletionsOutput {
                    text,
                    id: Some(result.response_id),
                    input_tokens: usage.input_tokens,
                    output_tokens: usage.output_tokens,
                    ..Default::default()
                },
                turns,
                pricing_context: OpenAIResponsesPricingContext::UnknownApiBase,
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

#[derive(Debug, Clone)]
pub struct OpenAIResponsesOutput {
    pub completion: ChatCompletionsOutput,
    pub turns: Vec<OpenAIResponsesTurn>,
    pub pricing_context: OpenAIResponsesPricingContext,
}

impl std::ops::Deref for OpenAIResponsesOutput {
    type Target = ChatCompletionsOutput;

    fn deref(&self) -> &Self::Target {
        &self.completion
    }
}

impl std::ops::DerefMut for OpenAIResponsesOutput {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.completion
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAIResponsesTurn {
    pub response_id: String,
    pub service_tier: Option<String>,
    pub usage: OpenAIResponsesUsage,
    pub trace: Vec<OpenAIResponsesTraceEvent>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenAIResponsesUsage {
    pub input_tokens: Option<u64>,
    pub cached_input_tokens: Option<u64>,
    pub cache_write_input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub reasoning_output_tokens: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAIResponsesPricingContext {
    PublicApi,
    UnknownApiBase,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenAIResponsesHostedAction {
    SpawnAgent,
    SendMessage,
    FollowupTask,
    WaitAgent,
    ListAgents,
    InterruptAgent,
    Unknown(String),
}

impl OpenAIResponsesHostedAction {
    fn parse(value: &str) -> Self {
        match value {
            "spawn_agent" => Self::SpawnAgent,
            "send_message" => Self::SendMessage,
            "followup_task" => Self::FollowupTask,
            "wait" | "wait_agent" => Self::WaitAgent,
            "list_agents" => Self::ListAgents,
            "interrupt_agent" => Self::InterruptAgent,
            value => Self::Unknown(value.to_string()),
        }
    }

    fn as_str(&self) -> &str {
        match self {
            Self::SpawnAgent => "spawn_agent",
            Self::SendMessage => "send_message",
            Self::FollowupTask => "followup_task",
            Self::WaitAgent => "wait_agent",
            Self::ListAgents => "list_agents",
            Self::InterruptAgent => "interrupt_agent",
            Self::Unknown(value) => value,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAIResponsesListedAgent {
    pub agent_name: String,
    pub status_kind: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenAIResponsesTraceEvent {
    MultiAgentCall {
        call_id: Option<String>,
        action: OpenAIResponsesHostedAction,
        agent_name: Option<String>,
        task_name: Option<String>,
        target: Option<String>,
    },
    MultiAgentCallOutput {
        call_id: Option<String>,
        action: OpenAIResponsesHostedAction,
        agent_name: Option<String>,
        spawned_agent_name: Option<String>,
        listed_agents: Vec<OpenAIResponsesListedAgent>,
    },
    AgentMessage {
        author: Option<String>,
        recipient: Option<String>,
    },
    Message {
        agent_name: Option<String>,
        phase: Option<String>,
    },
    DeveloperFunctionCall {
        agent_name: Option<String>,
        name: Option<String>,
        call_id: Option<String>,
    },
    BuiltInToolCall {
        agent_name: Option<String>,
        item_type: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpenAIResponsesMultiAgentResult {
    pub response_id: String,
    pub status: OpenAIResponsesStatus,
    pub output_items: Vec<Value>,
    pub function_calls: Vec<OpenAIResponsesFunctionCall>,
    pub root_final_text: Option<String>,
    pub usage: TokenUsage,
    pub service_tier: Option<String>,
    pub detailed_usage: OpenAIResponsesUsage,
    pub trace: Vec<OpenAIResponsesTraceEvent>,
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
    service_tier: Option<String>,
    output: Vec<Value>,
    usage: Option<ResponseUsage>,
    error: Option<Value>,
    incomplete_details: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct ResponseUsage {
    input_tokens: Option<u64>,
    input_tokens_details: Option<ResponseInputTokensDetails>,
    output_tokens: Option<u64>,
    output_tokens_details: Option<ResponseOutputTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct ResponseInputTokensDetails {
    cached_tokens: Option<u64>,
    cache_write_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ResponseOutputTokensDetails {
    reasoning_tokens: Option<u64>,
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
    let trace = extract_trace(&envelope.output);

    if function_calls.is_empty() && root_final_text.is_none() {
        return Err(OpenAIResponsesMultiAgentError::MissingRootFinal {
            response_id: envelope.id,
        });
    }

    let detailed_usage = envelope
        .usage
        .map_or_else(OpenAIResponsesUsage::default, |usage| {
            let input_details = usage.input_tokens_details;
            let output_details = usage.output_tokens_details;
            OpenAIResponsesUsage {
                input_tokens: usage.input_tokens,
                cached_input_tokens: input_details
                    .as_ref()
                    .and_then(|details| details.cached_tokens),
                cache_write_input_tokens: input_details
                    .as_ref()
                    .and_then(|details| details.cache_write_tokens),
                output_tokens: usage.output_tokens,
                reasoning_output_tokens: output_details
                    .as_ref()
                    .and_then(|details| details.reasoning_tokens),
            }
        });
    let usage = TokenUsage::new(detailed_usage.input_tokens, detailed_usage.output_tokens);

    Ok(OpenAIResponsesMultiAgentResult {
        response_id: envelope.id,
        status,
        output_items: envelope.output,
        function_calls,
        root_final_text,
        usage,
        service_tier: envelope.service_tier,
        detailed_usage,
        trace,
    })
}

#[derive(Clone)]
struct HostedCallContext {
    action: OpenAIResponsesHostedAction,
    agent_name: Option<String>,
}

fn extract_trace(output: &[Value]) -> Vec<OpenAIResponsesTraceEvent> {
    let calls_by_id = output
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("multi_agent_call"))
        .filter_map(|item| {
            let call_id = item.get("call_id").and_then(Value::as_str)?;
            let action = item
                .get("action")
                .and_then(Value::as_str)
                .map(OpenAIResponsesHostedAction::parse)
                .unwrap_or_else(|| OpenAIResponsesHostedAction::Unknown("unknown".to_string()));
            Some((
                call_id.to_string(),
                HostedCallContext {
                    action,
                    agent_name: trace_agent_name(item),
                },
            ))
        })
        .collect::<HashMap<_, _>>();

    output
        .iter()
        .filter_map(|item| trace_event(item, &calls_by_id))
        .collect()
}

fn trace_event(
    item: &Value,
    calls_by_id: &HashMap<String, HostedCallContext>,
) -> Option<OpenAIResponsesTraceEvent> {
    let item_type = item.get("type").and_then(Value::as_str)?;
    match item_type {
        "multi_agent_call" => {
            let action = item
                .get("action")
                .and_then(Value::as_str)
                .map(OpenAIResponsesHostedAction::parse)
                .unwrap_or_else(|| OpenAIResponsesHostedAction::Unknown("unknown".to_string()));
            let arguments = item.get("arguments");
            let task_name = matches!(action, OpenAIResponsesHostedAction::SpawnAgent)
                .then(|| arguments.and_then(|value| trace_json_string(value, "task_name")))
                .flatten();
            let target = matches!(
                action,
                OpenAIResponsesHostedAction::SendMessage
                    | OpenAIResponsesHostedAction::FollowupTask
                    | OpenAIResponsesHostedAction::InterruptAgent
            )
            .then(|| arguments.and_then(|value| trace_json_string(value, "target")))
            .flatten();
            Some(OpenAIResponsesTraceEvent::MultiAgentCall {
                call_id: trace_optional_string(item, "call_id"),
                action,
                agent_name: trace_agent_name(item),
                task_name,
                target,
            })
        }
        "multi_agent_call_output" => {
            let call_id = trace_optional_string(item, "call_id");
            let context = call_id
                .as_ref()
                .and_then(|call_id| calls_by_id.get(call_id));
            let action = item
                .get("action")
                .and_then(Value::as_str)
                .map(OpenAIResponsesHostedAction::parse)
                .or_else(|| context.map(|context| context.action.clone()))
                .unwrap_or_else(|| OpenAIResponsesHostedAction::Unknown("unknown".to_string()));
            let agent_name = trace_agent_name(item)
                .or_else(|| context.and_then(|context| context.agent_name.clone()));
            let spawned_agent_name = matches!(action, OpenAIResponsesHostedAction::SpawnAgent)
                .then(|| item.get("output").and_then(extract_spawned_agent_name))
                .flatten();
            let listed_agents = if matches!(action, OpenAIResponsesHostedAction::ListAgents) {
                item.get("output")
                    .map(extract_listed_agents)
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            Some(OpenAIResponsesTraceEvent::MultiAgentCallOutput {
                call_id,
                action,
                agent_name,
                spawned_agent_name,
                listed_agents,
            })
        }
        "agent_message" => Some(OpenAIResponsesTraceEvent::AgentMessage {
            author: trace_optional_string(item, "author"),
            recipient: trace_optional_string(item, "recipient"),
        }),
        "message" => Some(OpenAIResponsesTraceEvent::Message {
            agent_name: trace_agent_name(item),
            phase: trace_optional_string(item, "phase"),
        }),
        "function_call" => Some(OpenAIResponsesTraceEvent::DeveloperFunctionCall {
            agent_name: trace_agent_name(item),
            name: trace_optional_string(item, "name"),
            call_id: trace_optional_string(item, "call_id"),
        }),
        _ if item_type.ends_with("_call") => Some(OpenAIResponsesTraceEvent::BuiltInToolCall {
            agent_name: trace_agent_name(item),
            item_type: item_type.to_string(),
        }),
        _ => None,
    }
}

fn trace_agent_name(item: &Value) -> Option<String> {
    item.get("agent")
        .and_then(|agent| agent.get("agent_name"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn trace_optional_string(item: &Value, field: &str) -> Option<String> {
    item.get(field).and_then(Value::as_str).map(str::to_string)
}

fn trace_json_string(value: &Value, field: &str) -> Option<String> {
    extract_trace_json(value, &|value| {
        value.get(field).and_then(Value::as_str).map(str::to_string)
    })
}

fn extract_trace_json<T>(value: &Value, extract: &impl Fn(&Value) -> Option<T>) -> Option<T> {
    if let Some(extracted) = extract(value) {
        return Some(extracted);
    }
    if let Some(encoded) = value.as_str() {
        if let Ok(parsed) = serde_json::from_str(encoded) {
            return extract(&parsed);
        }
    }
    if let Some(parts) = value.as_array() {
        for part in parts.iter().take(256) {
            let Some(encoded) = part.get("text").and_then(Value::as_str) else {
                continue;
            };
            if let Ok(parsed) = serde_json::from_str(encoded) {
                if let Some(extracted) = extract(&parsed) {
                    return Some(extracted);
                }
            }
        }
    }
    None
}

fn extract_spawned_agent_name(output: &Value) -> Option<String> {
    extract_trace_json(output, &|output| {
        output
            .get("task_name")
            .and_then(Value::as_str)
            .or_else(|| {
                output
                    .get("agent")
                    .and_then(|agent| agent.get("task_name"))
                    .and_then(Value::as_str)
            })
            .map(str::to_string)
    })
}

fn extract_listed_agents(output: &Value) -> Vec<OpenAIResponsesListedAgent> {
    extract_trace_json(output, &|output| {
        let agents = output.get("agents").and_then(Value::as_array).or_else(|| {
            let agents = output.as_array()?;
            agents
                .iter()
                .any(|agent| agent.get("agent_name").is_some())
                .then_some(agents)
        });
        let listed_agents = agents?
            .iter()
            .take(256)
            .filter_map(|agent| {
                let agent_name = agent
                    .get("agent_name")
                    .and_then(Value::as_str)
                    .or_else(|| agent.get("name").and_then(Value::as_str))?;
                let status_kind = agent
                    .get("agent_status")
                    .or_else(|| agent.get("status"))
                    .and_then(|status| {
                        status.as_str().or_else(|| {
                            status
                                .get("kind")
                                .and_then(Value::as_str)
                                .or_else(|| status.get("type").and_then(Value::as_str))
                                .or_else(|| status.as_object()?.keys().next().map(String::as_str))
                        })
                    });
                Some(OpenAIResponsesListedAgent {
                    agent_name: agent_name.to_string(),
                    status_kind: status_kind.map(str::to_string),
                })
            })
            .collect();
        Some(listed_agents)
    })
    .unwrap_or_default()
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

pub fn format_openai_responses_usage_cost(
    model: &Model,
    turns: &[OpenAIResponsesTurn],
    pricing_context: OpenAIResponsesPricingContext,
) -> String {
    let input = sum_complete(turns, |usage| usage.input_tokens);
    let cached = sum_complete(turns, |usage| usage.cached_input_tokens);
    let cache_write = sum_complete(turns, |usage| usage.cache_write_input_tokens);
    let output = sum_complete(turns, |usage| usage.output_tokens);
    let reasoning = sum_complete(turns, |usage| usage.reasoning_output_tokens);
    let regular = input
        .zip(cached)
        .zip(cache_write)
        .and_then(|((input, cached), cache_write)| {
            input.checked_sub(cached.checked_add(cache_write)?)
        });
    let tiers = turns
        .iter()
        .map(|turn| {
            turn.service_tier
                .as_deref()
                .map(|value| sanitize_display(value, 120))
                .unwrap_or_else(|| "unavailable".to_string())
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join(", ");

    let mut summary = format!(
        "Responses: {} request{} | Service tiers: {}\nTokens: {} input",
        turns.len(),
        if turns.len() == 1 { "" } else { "s" },
        if tiers.is_empty() {
            "unavailable"
        } else {
            &tiers
        },
        display_token_count(input)
    );
    if let (Some(regular), Some(cached), Some(cache_write)) = (regular, cached, cache_write) {
        let _ = write!(
            summary,
            " ({regular} uncached + {cached} cached + {cache_write} cache write)"
        );
    } else {
        summary.push_str(" (bucket details unavailable)");
    }
    let _ = write!(summary, " + {} output", display_token_count(output));
    if let Some(reasoning) = reasoning {
        let _ = write!(summary, " ({reasoning} reasoning)");
    } else {
        summary.push_str(" (reasoning details unavailable)");
    }

    match calculate_openai_responses_cost(model, turns, pricing_context) {
        Ok(cost) => {
            let _ = write!(summary, "\nEstimated cost: ${cost:.6}");
        }
        Err(reason) => {
            let _ = write!(
                summary,
                "\nEstimated cost: unavailable ({})",
                sanitize_display(&reason, 240)
            );
        }
    }
    summary
}

fn calculate_openai_responses_cost(
    model: &Model,
    turns: &[OpenAIResponsesTurn],
    pricing_context: OpenAIResponsesPricingContext,
) -> std::result::Result<f64, String> {
    if turns.is_empty() {
        return Err("no response usage was returned".to_string());
    }
    if pricing_context != OpenAIResponsesPricingContext::PublicApi {
        return Err("custom OpenAI api_base has unknown pricing".to_string());
    }
    let input_price = valid_price(model.data().input_price, "model input price")?;
    let output_price = valid_price(model.data().output_price, "model output price")?;
    let pricing = model
        .data()
        .response_pricing
        .as_ref()
        .ok_or_else(|| "model response pricing is missing".to_string())?;
    let cached_input_price = valid_number(pricing.cached_input_price, true, "cached input price")?;
    let cache_write_input_price = valid_number(
        pricing.cache_write_input_price,
        true,
        "cache write input price",
    )?;
    if pricing.long_context_threshold == 0 {
        return Err("long-context threshold is invalid".to_string());
    }
    let long_input_multiplier = valid_number(
        pricing.long_context_input_multiplier,
        false,
        "long-context input multiplier",
    )?;
    let long_output_multiplier = valid_number(
        pricing.long_context_output_multiplier,
        false,
        "long-context output multiplier",
    )?;

    let mut total = 0.0;
    for turn in turns {
        let field = |value: Option<u64>, name: &str| {
            value.ok_or_else(|| format!("response '{}' is missing {name}", turn.response_id))
        };
        let input = field(turn.usage.input_tokens, "usage.input_tokens")?;
        let cached = field(
            turn.usage.cached_input_tokens,
            "usage.input_tokens_details.cached_tokens",
        )?;
        let cache_write = field(
            turn.usage.cache_write_input_tokens,
            "usage.input_tokens_details.cache_write_tokens",
        )?;
        let output = field(turn.usage.output_tokens, "usage.output_tokens")?;
        let reasoning = field(
            turn.usage.reasoning_output_tokens,
            "usage.output_tokens_details.reasoning_tokens",
        )?;
        if reasoning > output {
            return Err(format!(
                "response '{}' has reasoning tokens greater than output tokens",
                turn.response_id
            ));
        }
        let discounted = cached.checked_add(cache_write).ok_or_else(|| {
            format!(
                "response '{}' input token buckets overflow",
                turn.response_id
            )
        })?;
        let regular = input.checked_sub(discounted).ok_or_else(|| {
            format!(
                "response '{}' has cached and cache-write tokens greater than input tokens",
                turn.response_id
            )
        })?;
        let tier = turn
            .service_tier
            .as_deref()
            .ok_or_else(|| format!("response '{}' is missing service_tier", turn.response_id))?;
        if !matches!(tier, "default" | "flex" | "priority") {
            return Err(format!(
                "response '{}' returned unknown service_tier '{}'",
                turn.response_id, tier
            ));
        }
        let tier_multiplier = pricing
            .service_tier_multipliers
            .get(tier)
            .copied()
            .ok_or_else(|| format!("model pricing is missing service tier '{tier}'"))?;
        let tier_multiplier = valid_number(tier_multiplier, false, "service tier multiplier")?;
        let (input_multiplier, output_multiplier) = if input > pricing.long_context_threshold {
            (long_input_multiplier, long_output_multiplier)
        } else {
            (1.0, 1.0)
        };
        let input_cost = regular as f64 * input_price
            + cached as f64 * cached_input_price
            + cache_write as f64 * cache_write_input_price;
        total += (input_cost * input_multiplier + output as f64 * output_price * output_multiplier)
            * tier_multiplier
            / 1_000_000.0;
    }
    if !total.is_finite() {
        return Err("calculated response cost is invalid".to_string());
    }
    Ok(total)
}

fn valid_price(value: Option<f64>, name: &str) -> std::result::Result<f64, String> {
    valid_number(
        value.ok_or_else(|| format!("{name} is missing"))?,
        true,
        name,
    )
}

fn valid_number(value: f64, allow_zero: bool, name: &str) -> std::result::Result<f64, String> {
    if !value.is_finite() || value < 0.0 || (!allow_zero && value == 0.0) {
        return Err(format!("{name} is invalid"));
    }
    Ok(value)
}

fn sum_complete(
    turns: &[OpenAIResponsesTurn],
    value: impl Fn(&OpenAIResponsesUsage) -> Option<u64>,
) -> Option<u64> {
    if turns.is_empty() {
        return None;
    }
    turns
        .iter()
        .try_fold(0u64, |total, turn| total.checked_add(value(&turn.usage)?))
}

fn display_token_count(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unavailable".to_string())
}

pub fn format_openai_responses_trace(turns: &[OpenAIResponsesTurn]) -> String {
    let mut output = String::from("Agent trace:");
    for (index, turn) in turns.iter().enumerate() {
        let _ = write!(
            output,
            "\n  turn {} response={}",
            index + 1,
            sanitize_display(&turn.response_id, 120)
        );
        if turn.trace.is_empty() {
            output.push_str("\n    no structural events");
            continue;
        }
        for event in &turn.trace {
            format_trace_event(&mut output, event);
        }
    }
    output
}

fn format_trace_event(output: &mut String, event: &OpenAIResponsesTraceEvent) {
    match event {
        OpenAIResponsesTraceEvent::MultiAgentCall {
            call_id,
            action,
            agent_name,
            task_name,
            target,
        } => {
            let _ = write!(
                output,
                "\n    hosted_call actor={} action={} call={}",
                display_trace_option(agent_name),
                sanitize_display(action.as_str(), 120),
                display_trace_option(call_id)
            );
            if let Some(task_name) = task_name {
                let _ = write!(output, " task={}", sanitize_display(task_name, 120));
            }
            if let Some(target) = target {
                let _ = write!(output, " target={}", sanitize_display(target, 120));
            }
        }
        OpenAIResponsesTraceEvent::MultiAgentCallOutput {
            call_id,
            action,
            agent_name,
            spawned_agent_name,
            listed_agents,
        } => {
            let _ = write!(
                output,
                "\n    hosted_output actor={} action={} call={}",
                display_trace_option(agent_name),
                sanitize_display(action.as_str(), 120),
                display_trace_option(call_id)
            );
            if let Some(agent_name) = spawned_agent_name {
                let _ = write!(output, " spawned={}", sanitize_display(agent_name, 120));
            }
            for listed_agent in listed_agents {
                let _ = write!(
                    output,
                    "\n      agent={} status={}",
                    sanitize_display(&listed_agent.agent_name, 120),
                    display_trace_option(&listed_agent.status_kind)
                );
            }
        }
        OpenAIResponsesTraceEvent::AgentMessage { author, recipient } => {
            let _ = write!(
                output,
                "\n    agent_message author={} recipient={}",
                display_trace_option(author),
                display_trace_option(recipient)
            );
        }
        OpenAIResponsesTraceEvent::Message { agent_name, phase } => {
            let _ = write!(
                output,
                "\n    message agent={} phase={}",
                display_trace_option(agent_name),
                display_trace_option(phase)
            );
        }
        OpenAIResponsesTraceEvent::DeveloperFunctionCall {
            agent_name,
            name,
            call_id,
        } => {
            let _ = write!(
                output,
                "\n    developer_tool agent={} name={} call={}",
                display_trace_option(agent_name),
                display_trace_option(name),
                display_trace_option(call_id)
            );
        }
        OpenAIResponsesTraceEvent::BuiltInToolCall {
            agent_name,
            item_type,
        } => {
            let _ = write!(
                output,
                "\n    built_in_tool agent={} type={}",
                display_trace_option(agent_name),
                sanitize_display(item_type, 120)
            );
        }
    }
}

fn display_trace_option(value: &Option<String>) -> String {
    value
        .as_deref()
        .map(|value| sanitize_display(value, 120))
        .unwrap_or_else(|| "unavailable".to_string())
}

fn sanitize_display(value: &str, max_chars: usize) -> String {
    let mut sanitized = String::new();
    let mut count = 0usize;
    let mut last_was_space = false;
    for character in value.chars() {
        if count == max_chars {
            sanitized.push('…');
            break;
        }
        if character.is_control()
            || character.is_whitespace()
            || is_unicode_format_control(character)
        {
            if !last_was_space {
                sanitized.push(' ');
                last_was_space = true;
                count += 1;
            }
            continue;
        }
        sanitized.push(character);
        last_was_space = character.is_whitespace();
        count += 1;
    }
    let sanitized = sanitized.trim();
    if sanitized.is_empty() {
        "unavailable".to_string()
    } else {
        sanitized.to_string()
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::client::{ImageUrl, ModelData, OpenAIClient, OpenAIConfig, ResponsePricing};
    use crate::utils::create_abort_signal;
    use std::cell::{Cell, RefCell};
    use std::collections::{BTreeMap, VecDeque};
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
    fn parses_response_usage_details_and_service_tier() {
        let raw = json!({
            "id": "resp_usage",
            "status": "completed",
            "service_tier": "priority",
            "output": [root_final("done")],
            "usage": {
                "input_tokens": 120,
                "input_tokens_details": {
                    "cached_tokens": 40,
                    "cache_write_tokens": 10
                },
                "output_tokens": 30,
                "output_tokens_details": {"reasoning_tokens": 12}
            },
            "error": null,
            "incomplete_details": null
        });

        let result = parse_openai_responses_multi_agent(&raw).unwrap();

        assert_eq!(result.service_tier.as_deref(), Some("priority"));
        assert_eq!(
            result.detailed_usage,
            OpenAIResponsesUsage {
                input_tokens: Some(120),
                cached_input_tokens: Some(40),
                cache_write_input_tokens: Some(10),
                output_tokens: Some(30),
                reasoning_output_tokens: Some(12),
            }
        );
        assert_eq!(result.usage, TokenUsage::new(Some(120), Some(30)));
    }

    #[test]
    fn trace_is_ordered_correlated_sanitized_and_payload_free() {
        let raw = json!({
            "id": "resp_trace\n\u{001b}[31m",
            "status": "completed",
            "service_tier": "default",
            "output": [
                {
                    "type": "multi_agent_call",
                    "call_id": "hosted_spawn",
                    "action": "spawn_agent",
                    "arguments": "{\"task_name\":\"research\\n\\u001b[2J\",\"message\":\"spawn-secret\"}",
                    "agent": {"agent_name": "/root"}
                },
                {
                    "type": "multi_agent_call_output",
                    "call_id": "hosted_spawn",
                    "output": [{
                        "type": "output_text",
                        "text": "{\"task_name\":\"/root/research\",\"payload\":\"spawn-output-secret\"}"
                    }]
                },
                {
                    "type": "multi_agent_call",
                    "call_id": "hosted_list",
                    "action": "list_agents",
                    "arguments": "{}",
                    "agent": {"agent_name": "/root"}
                },
                {
                    "type": "multi_agent_call_output",
                    "call_id": "hosted_list",
                    "output": [{
                        "type": "output_text",
                        "text": "{\"agents\":[{\"agent_name\":\"/root/research\",\"agent_status\":{\"completed\":\"status-secret\"},\"last_task_message\":\"agent-secret\"}]}"
                    }]
                },
                {
                    "type": "agent_message",
                    "author": "/root/research",
                    "recipient": "/root",
                    "content": [{"encrypted_content": "message-secret"}]
                },
                {
                    "type": "web_search_call",
                    "agent": {"agent_name": "/root/research"},
                    "query": "query-secret",
                    "results": "result-secret"
                },
                {
                    "type": "function_call",
                    "call_id": "developer_call",
                    "name": "lookup",
                    "arguments": "{\"secret\":\"developer-secret\"}",
                    "agent": {"agent_name": "/root/research"}
                },
                {
                    "type": "message",
                    "agent": {"agent_name": "/root"},
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "final-secret"}]
                }
            ],
            "usage": null,
            "error": null,
            "incomplete_details": null
        });

        let parsed = parse_openai_responses_multi_agent(&raw).unwrap();
        let turn = OpenAIResponsesTurn {
            response_id: parsed.response_id,
            service_tier: parsed.service_tier,
            usage: parsed.detailed_usage,
            trace: parsed.trace,
        };
        assert!(matches!(
            &turn.trace[1],
            OpenAIResponsesTraceEvent::MultiAgentCallOutput {
                action: OpenAIResponsesHostedAction::SpawnAgent,
                agent_name: Some(agent),
                spawned_agent_name: Some(spawned),
                ..
            } if agent == "/root" && spawned == "/root/research"
        ));
        assert!(matches!(
            &turn.trace[3],
            OpenAIResponsesTraceEvent::MultiAgentCallOutput {
                action: OpenAIResponsesHostedAction::ListAgents,
                listed_agents,
                ..
            } if listed_agents == &[OpenAIResponsesListedAgent {
                agent_name: "/root/research".to_string(),
                status_kind: Some("completed".to_string()),
            }]
        ));
        assert!(matches!(
            &turn.trace[5],
            OpenAIResponsesTraceEvent::BuiltInToolCall { item_type, .. }
                if item_type == "web_search_call"
        ));

        let formatted = format_openai_responses_trace(&[turn]);
        for secret in [
            "spawn-secret",
            "spawn-output-secret",
            "status-secret",
            "agent-secret",
            "message-secret",
            "query-secret",
            "result-secret",
            "developer-secret",
            "final-secret",
        ] {
            assert!(!formatted.contains(secret));
        }
        assert!(!formatted.contains('\u{1b}'));
        for character in ['\u{2028}', '\u{2029}', '\u{202e}', '\u{2066}', '\u{2069}'] {
            assert!(!formatted.contains(character));
        }
        assert_eq!(
            sanitize_display("left\u{2028}right\u{2029}rtl\u{202e}mark\u{2066}end", 120),
            "left right rtl mark end"
        );
        assert!(!formatted.contains("\n\n"));
        assert!(formatted.contains("built_in_tool agent=/root/research type=web_search_call"));
        assert!(formatted
            .contains("developer_tool agent=/root/research name=lookup call=developer_call"));
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

    fn priced_responses_model() -> Model {
        let mut data = ModelData::new("gpt-5.6-sol");
        data.input_price = Some(10.0);
        data.output_price = Some(20.0);
        data.response_pricing = Some(ResponsePricing {
            cached_input_price: 2.0,
            cache_write_input_price: 12.5,
            long_context_threshold: 100,
            long_context_input_multiplier: 2.0,
            long_context_output_multiplier: 1.5,
            service_tier_multipliers: BTreeMap::from([
                ("default".to_string(), 1.0),
                ("flex".to_string(), 0.5),
                ("priority".to_string(), 2.0),
            ]),
        });
        Model::from_config("openai", "openai", &[data])
            .into_iter()
            .next()
            .unwrap()
    }

    fn usage_turn(
        id: &str,
        tier: Option<&str>,
        input: u64,
        cached: u64,
        cache_write: u64,
        output: u64,
        reasoning: u64,
    ) -> OpenAIResponsesTurn {
        OpenAIResponsesTurn {
            response_id: id.to_string(),
            service_tier: tier.map(str::to_string),
            usage: OpenAIResponsesUsage {
                input_tokens: Some(input),
                cached_input_tokens: Some(cached),
                cache_write_input_tokens: Some(cache_write),
                output_tokens: Some(output),
                reasoning_output_tokens: Some(reasoning),
            },
            trace: Vec::new(),
        }
    }

    fn public_responses_cost(
        model: &Model,
        turns: &[OpenAIResponsesTurn],
    ) -> std::result::Result<f64, String> {
        calculate_openai_responses_cost(model, turns, OpenAIResponsesPricingContext::PublicApi)
    }

    fn format_public_responses_cost(model: &Model, turns: &[OpenAIResponsesTurn]) -> String {
        format_openai_responses_usage_cost(model, turns, OpenAIResponsesPricingContext::PublicApi)
    }

    #[test]
    fn prices_each_turn_with_cache_tier_and_long_context_rules() {
        let model = priced_responses_model();
        let turns = [
            usage_turn("short", Some("default"), 100, 20, 10, 10, 5),
            usage_turn("long", Some("flex"), 101, 20, 10, 10, 8),
        ];

        let cost = public_responses_cost(&model, &turns).unwrap();

        assert!((cost - 0.00209).abs() < f64::EPSILON);
        let formatted = format_public_responses_cost(&model, &turns);
        assert!(formatted.contains("2 requests"));
        assert!(formatted.contains("141 uncached + 40 cached + 20 cache write"));
        assert!(formatted.contains("20 output (13 reasoning)"));
        assert!(formatted.contains("Service tiers: default, flex"));
        assert!(formatted.contains("Estimated cost: $0.002090"));
    }

    #[test]
    fn reasoning_tokens_are_display_only_and_not_double_billed() {
        let model = priced_responses_model();
        let no_reasoning = [usage_turn("one", Some("default"), 10, 0, 0, 8, 0)];
        let all_reasoning = [usage_turn("one", Some("default"), 10, 0, 0, 8, 8)];

        assert_eq!(
            public_responses_cost(&model, &no_reasoning).unwrap(),
            public_responses_cost(&model, &all_reasoning).unwrap()
        );
    }

    #[test]
    fn cost_is_unavailable_for_unknown_or_missing_tier_and_invalid_buckets() {
        let model = priced_responses_model();
        let unknown = [usage_turn("unknown", Some("turbo"), 10, 0, 0, 1, 0)];
        let missing = [usage_turn("missing", None, 10, 0, 0, 1, 0)];
        let invalid = [usage_turn("invalid", Some("default"), 10, 9, 2, 1, 0)];

        assert!(public_responses_cost(&model, &unknown)
            .unwrap_err()
            .contains("unknown service_tier"));
        assert!(public_responses_cost(&model, &missing)
            .unwrap_err()
            .contains("missing service_tier"));
        assert!(public_responses_cost(&model, &invalid)
            .unwrap_err()
            .contains("greater than input tokens"));
        assert!(
            format_public_responses_cost(&model, &invalid).contains("Estimated cost: unavailable")
        );
    }

    #[test]
    fn cost_is_unavailable_when_detailed_usage_or_response_pricing_is_missing() {
        let model = priced_responses_model();
        let mut incomplete = usage_turn("incomplete", Some("default"), 10, 0, 0, 1, 0);
        incomplete.usage.cache_write_input_tokens = None;
        assert!(public_responses_cost(&model, &[incomplete])
            .unwrap_err()
            .contains("cache_write_tokens"));

        let mut unpriced_data = ModelData::new("gpt-5.6-sol");
        unpriced_data.input_price = Some(10.0);
        unpriced_data.output_price = Some(20.0);
        let unpriced = Model::from_config("openai", "openai", &[unpriced_data])
            .into_iter()
            .next()
            .unwrap();
        assert!(public_responses_cost(
            &unpriced,
            &[usage_turn("one", Some("default"), 10, 0, 0, 1, 0)]
        )
        .unwrap_err()
        .contains("model response pricing is missing"));

        let empty_summary = format_public_responses_cost(&model, &[]);
        assert!(empty_summary.contains("Tokens: unavailable input"));
        assert!(empty_summary.contains("no response usage was returned"));
    }

    #[test]
    fn cost_is_unavailable_for_custom_api_base_pricing() {
        let model = priced_responses_model();
        let turns = [usage_turn("one", Some("default"), 10, 0, 0, 1, 0)];

        let summary = format_openai_responses_usage_cost(
            &model,
            &turns,
            OpenAIResponsesPricingContext::UnknownApiBase,
        );

        assert!(summary.contains("Estimated cost: unavailable"));
        assert!(summary.contains("custom OpenAI api_base has unknown pricing"));
        assert!(!summary.contains('$'));
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
                "service_tier": "auto",
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
        assert_eq!(body["service_tier"], "auto");
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
        assert!(!client.responses_uses_public_pricing().unwrap());

        let public_client = OpenAIClient {
            global_config: Default::default(),
            config: OpenAIConfig {
                api_base: Some("https://api.openai.com/v1/".into()),
                ..Default::default()
            },
            model: responses_model(None),
        };
        assert!(public_client.responses_uses_public_pricing().unwrap());
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

        assert_eq!(output.completion.text, "root answer");
        assert_eq!(output.completion.id.as_deref(), Some("resp_3"));
        assert_eq!(
            output.completion.usage(),
            TokenUsage::new(Some(30), Some(9))
        );
        assert_eq!(
            output
                .turns
                .iter()
                .map(|turn| turn.response_id.as_str())
                .collect::<Vec<_>>(),
            ["resp_1", "resp_2", "resp_3"]
        );
        assert_eq!(
            output
                .turns
                .iter()
                .map(|turn| turn.trace.len())
                .collect::<Vec<_>>(),
            [1, 2, 1]
        );
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
