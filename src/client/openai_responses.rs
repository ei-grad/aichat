use super::registry::init_openai_client;
use super::{
    catch_error, retry_request, sse_transport_failure, ChatCompletionsData, ChatCompletionsOutput,
    Client, Message, MessageContent, MessageContentPart, MessageRole, Model, RequestApi,
    RequestData, TokenUsage, ToolCall,
};

use crate::config::{
    GlobalConfig, HostedWebSearchConfig, Input, MultiAgentConfig, MultiAgentHostedTool,
    MultiAgentToolChoice, RoleLike,
};
use crate::function::{eval_tool_calls_preserving_results, FunctionDeclaration};
use crate::utils::{strip_think_tag, wait_abort_signal, AbortSignal};

use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use parking_lot::Mutex;
use reqwest::{Client as ReqwestClient, RequestBuilder, Url};
use reqwest_eventsource::{retry::Never, Event, RequestBuilderExt};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

const ROOT_AGENT: &str = "/root";
const MULTI_AGENT_INSTRUCTIONS: &str = "Proactive multi-agent delegation is active. Use subagents when parallel work would materially improve speed or quality.";
const MAX_CONTINUATION_TURNS: usize = 64;

pub async fn run_openai_responses_multi_agent(
    config: &GlobalConfig,
    input: &Input,
    abort_signal: AbortSignal,
    progress: OpenAIResponsesProgress,
) -> Result<OpenAIResponsesOutput> {
    let model = input.role().model().clone();
    validate_multi_agent_model(&model)?;
    let client = init_openai_client(config, &model)?;
    let multi_agent = config.read().multi_agent.clone();
    multi_agent.validate()?;
    let data = input.prepare_completion_data(&model, false)?;
    let body = build_openai_responses_multi_agent_body(data, &model, &multi_agent)?;
    let effective_request = prepare_openai_responses_request(&client, body.clone())?;
    let pricing_context = responses_pricing_context(&effective_request, &model);
    progress.set_pricing_context(pricing_context);

    if !multi_agent.hosted_tools.is_empty()
        && !is_public_openai_responses_endpoint(&effective_request)
    {
        bail!(
            "OpenAI hosted tools require the canonical https://api.openai.com/v1/responses endpoint"
        )
    }

    if config.read().dry_run {
        return Ok(OpenAIResponsesOutput {
            completion: ChatCompletionsOutput::new(&serde_json::to_string_pretty(
                &effective_request.body,
            )?),
            turns: Vec::new(),
            citations: Vec::new(),
            sources: Vec::new(),
            pricing_context,
        });
    }

    let http = client.build_client()?;
    let mut output = run_multi_agent_loop_with_progress(
        body,
        &abort_signal,
        |body| send_openai_responses_turn(&client, &http, body, progress.clone()),
        |calls| execute_function_calls(config, calls),
        Some(&progress),
    )
    .await?;
    output.pricing_context = pricing_context;
    Ok(output)
}

#[cfg(test)]
async fn run_multi_agent_loop<S, SFut, E>(
    body: Value,
    abort_signal: &AbortSignal,
    send: S,
    execute: E,
) -> Result<OpenAIResponsesOutput>
where
    S: FnMut(Value) -> SFut,
    SFut: Future<Output = Result<Value>>,
    E: FnMut(Vec<OpenAIResponsesFunctionCall>) -> Result<Vec<Value>>,
{
    run_multi_agent_loop_with_progress(body, abort_signal, send, execute, None).await
}

async fn run_multi_agent_loop_with_progress<S, SFut, E>(
    mut body: Value,
    abort_signal: &AbortSignal,
    mut send: S,
    mut execute: E,
    progress: Option<&OpenAIResponsesProgress>,
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
    let mut sources = Vec::new();

    for _ in 0..MAX_CONTINUATION_TURNS {
        let response = tokio::select! {
            response = send(body.clone()) => response?,
            _ = wait_abort_signal(abort_signal) => bail!("Aborted."),
        };
        let result = match parse_openai_responses_multi_agent(&response) {
            Ok(result) => result,
            Err(error) => {
                if let Some(progress) = progress {
                    if let Some(turn) = observe_openai_responses_turn(&response) {
                        progress.push_turn(turn);
                    }
                }
                return Err(error.into());
            }
        };
        if body.get("tool_choice").and_then(Value::as_str) == Some("required") {
            body["tool_choice"] = "auto".into();
        }
        total_usage = Some(match total_usage {
            Some(mut total) => {
                total.add(result.usage);
                total
            }
            None => result.usage,
        });
        for call in &result.web_search_calls {
            merge_sources(&mut sources, call.sources.iter().cloned());
        }
        let turn = OpenAIResponsesTurn {
            response_id: result.response_id.clone(),
            service_tier: result.service_tier.clone(),
            usage: result.detailed_usage.clone(),
            trace: result.trace.clone(),
            web_search_calls: result.web_search_calls.clone(),
        };
        if let Some(progress) = progress {
            progress.push_turn(turn.clone());
        }
        turns.push(turn);
        input_items.extend(result.output_items.iter().cloned());

        if result.function_calls.is_empty() {
            let text = result.root_final_text.ok_or_else(|| {
                anyhow::anyhow!(
                    "OpenAI Responses request '{}' completed without a root final answer",
                    result.response_id
                )
            })?;
            let usage = total_usage.unwrap_or_default();
            let mut output = OpenAIResponsesOutput {
                completion: ChatCompletionsOutput {
                    text,
                    id: Some(result.response_id),
                    input_tokens: usage.input_tokens,
                    output_tokens: usage.output_tokens,
                    ..Default::default()
                },
                turns,
                citations: result.root_final_citations,
                sources,
                pricing_context: OpenAIResponsesPricingContext::UnknownApiBase,
            };
            merge_sources(
                &mut output.sources,
                output
                    .citations
                    .iter()
                    .map(|citation| citation.source.clone()),
            );
            output.completion.text = append_sources_block(output.completion.text, &output.sources);
            return Ok(output);
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
    settings: &MultiAgentConfig,
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
    if let Some(max_concurrent_subagents) = settings.max_concurrent_subagents {
        multi_agent["max_concurrent_subagents"] = max_concurrent_subagents.get().into();
    }
    let web_search_count = settings
        .hosted_tools
        .iter()
        .filter(|tool| matches!(tool, MultiAgentHostedTool::WebSearch { .. }))
        .count();
    let has_web_search = web_search_count != 0;
    if web_search_count > 1 {
        bail!("multi_agent.hosted_tools may contain web_search at most once")
    }
    let mut include = vec![json!("reasoning.encrypted_content")];
    if has_web_search {
        include.push(json!("web_search_call.action.sources"));
    }
    let mut body = json!({
        "model": model.real_name(),
        "input": input,
        "instructions": MULTI_AGENT_INSTRUCTIONS,
        "store": false,
        "stream": true,
        "include": include,
        "service_tier": settings.service_tier.as_str(),
        "multi_agent": multi_agent,
    });

    let mut tools = Vec::new();
    if let Some(functions) = functions {
        tools.extend(functions.into_iter().map(build_responses_tool));
    }
    tools.extend(settings.hosted_tools.iter().map(build_hosted_tool));
    if tools.is_empty() {
        if settings.tool_choice == MultiAgentToolChoice::Required {
            bail!("multi_agent.tool_choice=required requires at least one configured tool")
        }
    } else {
        body["tools"] = Value::Array(tools);
        body["tool_choice"] = settings.tool_choice.as_str().into();
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
    if let Some(max_output_tokens) = settings.max_output_tokens {
        body["max_output_tokens"] = max_output_tokens.get().into();
    } else if let Some(max_output_tokens) = model.max_tokens_param() {
        body["max_output_tokens"] = max_output_tokens.into();
    }

    Ok(body)
}

fn build_hosted_tool(tool: &MultiAgentHostedTool) -> Value {
    match tool {
        MultiAgentHostedTool::WebSearch { config } => build_web_search_tool(config),
    }
}

fn build_web_search_tool(config: &HostedWebSearchConfig) -> Value {
    let mut tool = json!({
        "type": "web_search",
        "search_context_size": config.search_context_size.as_str(),
        "return_token_budget": config.return_token_budget.as_str(),
    });
    if let Some(external_web_access) = config.external_web_access {
        tool["external_web_access"] = external_web_access.into();
    }
    if let Some(filters) = &config.filters {
        let mut value = json!({});
        if !filters.allowed_domains.is_empty() {
            value["allowed_domains"] = json!(filters.allowed_domains);
        }
        if !filters.blocked_domains.is_empty() {
            value["blocked_domains"] = json!(filters.blocked_domains);
        }
        if value.as_object().is_some_and(|value| !value.is_empty()) {
            tool["filters"] = value;
        }
    }
    tool
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

fn prepare_openai_responses_request(
    client: &super::OpenAIClient,
    body: Value,
) -> Result<RequestData> {
    let mut request = client.prepare_responses_request(body)?;
    client.patch_request_data_for_api(&mut request, RequestApi::Responses);
    canonicalize_openai_client_request_id(&mut request);
    if request.body.get("stream").and_then(Value::as_bool) != Some(true) {
        bail!(
            "OpenAI Responses multi-agent requires effective stream=true; \
             patch.responses must not remove or disable it"
        )
    }
    Ok(request)
}

fn canonicalize_openai_client_request_id(request: &mut RequestData) {
    let client_request_id = request
        .headers
        .iter()
        .rev()
        .find(|(name, _)| name.eq_ignore_ascii_case("x-client-request-id"))
        .map(|(_, value)| value.clone())
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    request
        .headers
        .retain(|name, _| !name.eq_ignore_ascii_case("x-client-request-id"));
    request.header("x-client-request-id", client_request_id);
}

fn is_public_openai_responses_endpoint(request: &RequestData) -> bool {
    request.url.trim_end_matches('/') == "https://api.openai.com/v1/responses"
}

fn responses_pricing_context(
    request: &RequestData,
    model: &Model,
) -> OpenAIResponsesPricingContext {
    if !is_public_openai_responses_endpoint(request) {
        return OpenAIResponsesPricingContext::UnknownApiBase;
    }
    if request.body.get("model").and_then(Value::as_str) != Some(model.real_name()) {
        return OpenAIResponsesPricingContext::UnknownModel;
    }
    OpenAIResponsesPricingContext::PublicApi
}

async fn send_openai_responses_turn(
    client: &super::OpenAIClient,
    http: &ReqwestClient,
    body: Value,
    progress: OpenAIResponsesProgress,
) -> Result<Value> {
    progress.begin_live_turn();
    retry_request(|| {
        let request = prepare_openai_responses_request(client, body.clone());
        let progress = progress.clone();
        async move {
            let request = request?;
            let client_request_id = request
                .headers
                .get("x-client-request-id")
                .context("OpenAI Responses request is missing x-client-request-id")?
                .clone();
            let log_body = openai_responses_request_log_body(&request.body);
            send_openai_responses_request_with_progress(
                request.into_builder_with_log_body(http, log_body),
                &client_request_id,
                Some(&progress),
            )
            .await
        }
    })
    .await
    .context("Failed to call OpenAI Responses api")
}

fn openai_responses_request_log_body(body: &Value) -> Value {
    let tool_types = body
        .get("tools")
        .and_then(Value::as_array)
        .map(|tools| {
            tools
                .iter()
                .filter_map(|tool| tool.get("type").and_then(Value::as_str))
                .map(|value| Value::String(sanitize_display(value, 64)))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let input_items = body
        .get("input")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or_default();
    json!({
        "model": body
            .get("model")
            .and_then(Value::as_str)
            .map(|value| sanitize_display(value, 120)),
        "stream": body.get("stream").and_then(Value::as_bool),
        "input_items": input_items,
        "instructions_present": body.get("instructions").is_some(),
        "store": body.get("store").and_then(Value::as_bool),
        "service_tier": body
            .get("service_tier")
            .and_then(Value::as_str)
            .map(|value| sanitize_display(value, 32)),
        "multi_agent_enabled": body
            .pointer("/multi_agent/enabled")
            .and_then(Value::as_bool),
        "tool_types": tool_types,
        "tool_choice": body
            .get("tool_choice")
            .and_then(Value::as_str)
            .map(|value| sanitize_display(value, 32)),
        "reasoning_effort": body
            .pointer("/reasoning/effort")
            .and_then(Value::as_str)
            .map(|value| sanitize_display(value, 32)),
        "max_output_tokens": body.get("max_output_tokens").and_then(Value::as_u64),
    })
}

#[cfg(test)]
async fn send_openai_responses_request(
    builder: RequestBuilder,
    client_request_id: &str,
) -> Result<Value> {
    send_openai_responses_request_with_progress(builder, client_request_id, None).await
}

async fn send_openai_responses_request_with_progress(
    builder: RequestBuilder,
    client_request_id: &str,
    progress: Option<&OpenAIResponsesProgress>,
) -> Result<Value> {
    let mut stream = builder.eventsource()?;
    stream.set_retry_policy(Box::new(Never));
    let mut opened = false;
    let mut response_id = None;
    let mut last_sequence_number = None;
    let mut live_state = OpenAIResponsesLiveStreamState::default();

    while let Some(event) = stream.next().await {
        match event {
            Ok(Event::Open) => {
                opened = true;
                if let Some(progress) = progress {
                    progress.record_stream_open();
                }
                debug!(
                    "OpenAI Responses SSE stream opened (x-client-request-id: {})",
                    sanitize_display(client_request_id, 128)
                );
            }
            Ok(Event::Message(message)) => {
                opened = true;
                let data: Value = serde_json::from_str(&message.data).with_context(|| {
                    format!(
                        "Invalid OpenAI Responses SSE event JSON{}",
                        responses_stream_position(
                            response_id.as_deref(),
                            last_sequence_number,
                            client_request_id
                        )
                    )
                })?;
                last_sequence_number = data
                    .get("sequence_number")
                    .and_then(Value::as_u64)
                    .or(last_sequence_number);
                if response_id.is_none() {
                    response_id = data
                        .pointer("/response/id")
                        .and_then(Value::as_str)
                        .or_else(|| data.get("response_id").and_then(Value::as_str))
                        .map(str::to_string);
                }
                let event_type = data.get("type").and_then(Value::as_str).ok_or_else(|| {
                    anyhow::anyhow!(
                        "OpenAI Responses SSE event is missing type{}",
                        responses_stream_position(
                            response_id.as_deref(),
                            last_sequence_number,
                            client_request_id
                        )
                    )
                })?;
                if matches!(
                    event_type,
                    "response.completed" | "response.failed" | "response.incomplete"
                ) {
                    emit_terminal_response_trace(
                        &data,
                        response_id.as_deref(),
                        progress,
                        &mut live_state,
                    );
                }
                observe_openai_responses_stream_event(
                    &data,
                    event_type,
                    response_id.as_deref(),
                    last_sequence_number,
                    &mut live_state,
                    progress,
                );
                match event_type {
                    "response.completed" | "response.failed" | "response.incomplete" => {
                        let response = data.get("response").cloned().ok_or_else(|| {
                            anyhow::anyhow!(
                                "OpenAI Responses terminal SSE event '{}' is missing response{}",
                                event_type,
                                responses_stream_position(
                                    response_id.as_deref(),
                                    last_sequence_number,
                                    client_request_id
                                )
                            )
                        })?;
                        stream.close();
                        return Ok(response);
                    }
                    "error" => {
                        stream.close();
                        return Err(openai_responses_stream_error(
                            &data,
                            response_id.as_deref(),
                            last_sequence_number,
                            client_request_id,
                        ));
                    }
                    _ => {}
                }
            }
            Err(error) => {
                let error = sse_transport_failure(
                    error,
                    &catch_error,
                    true,
                    (!opened).then_some(client_request_id),
                )
                .await;
                if opened {
                    return Err(anyhow::anyhow!(
                        "OpenAI Responses SSE transport failed after the stream opened{}: {}",
                        responses_stream_position(
                            response_id.as_deref(),
                            last_sequence_number,
                            client_request_id
                        ),
                        error
                    ));
                }
                return Err(error);
            }
        }
    }

    bail!(
        "OpenAI Responses SSE stream ended before a terminal event{}",
        responses_stream_position(
            response_id.as_deref(),
            last_sequence_number,
            client_request_id
        )
    )
}

#[derive(Debug, Clone, Default)]
struct OpenAIResponsesLiveOutputContext {
    item_type: Option<String>,
    agent_name: Option<String>,
    action: Option<String>,
}

#[derive(Default)]
struct OpenAIResponsesLiveStreamState {
    output_contexts: HashMap<u64, OpenAIResponsesLiveOutputContext>,
    hosted_calls: HashMap<String, HostedCallContext>,
    emitted_trace_output_indexes: HashSet<u64>,
    emitted_trace_item_ids: HashSet<String>,
}

impl OpenAIResponsesLiveStreamState {
    fn mark_trace_item(&mut self, output_index: Option<u64>, item: &Value) -> bool {
        let item_id = item.get("id").and_then(Value::as_str);
        if output_index.is_some_and(|index| self.emitted_trace_output_indexes.contains(&index))
            || item_id.is_some_and(|item_id| self.emitted_trace_item_ids.contains(item_id))
        {
            return false;
        }
        if let Some(output_index) = output_index {
            self.emitted_trace_output_indexes.insert(output_index);
        }
        if let Some(item_id) = item_id {
            self.emitted_trace_item_ids.insert(item_id.to_string());
        }
        true
    }
}

fn observe_openai_responses_stream_event(
    data: &Value,
    event_type: &str,
    response_id: Option<&str>,
    sequence_number: Option<u64>,
    live_state: &mut OpenAIResponsesLiveStreamState,
    progress: Option<&OpenAIResponsesProgress>,
) {
    let output_index = data.get("output_index").and_then(Value::as_u64);
    let item = data.get("item");
    if let (Some(output_index), Some(item)) = (output_index, item) {
        let context = live_state.output_contexts.entry(output_index).or_default();
        if let Some(item_type) = item.get("type").and_then(Value::as_str) {
            context.item_type = Some(sanitize_display(item_type, 64));
        }
        if let Some(agent_name) = stream_agent_name(data).or_else(|| trace_agent_name(item)) {
            context.agent_name = Some(sanitize_display(&agent_name, 120));
        }
        if let Some(action) = item.get("action").and_then(Value::as_str) {
            context.action = Some(sanitize_display(action, 64));
        }
    }

    let context = output_index.and_then(|index| live_state.output_contexts.get(&index));
    let item_type = item
        .and_then(|item| item.get("type"))
        .and_then(Value::as_str)
        .map(|value| sanitize_display(value, 64))
        .or_else(|| context.and_then(|context| context.item_type.clone()));
    let agent_name = stream_agent_name(data)
        .or_else(|| item.and_then(trace_agent_name))
        .map(|value| sanitize_display(&value, 120))
        .or_else(|| context.and_then(|context| context.agent_name.clone()));
    let action = item
        .and_then(|item| item.get("action"))
        .and_then(Value::as_str)
        .map(|value| sanitize_display(value, 64))
        .or_else(|| context.and_then(|context| context.action.clone()));

    debug!(
        "OpenAI Responses SSE event type={} sequence={} response={} output_index={} item_type={} agent={} action={}",
        sanitize_display(event_type, 80),
        display_live_number(sequence_number),
        display_live_value(response_id, 128),
        display_live_number(output_index),
        display_live_option(item_type.as_deref(), 64),
        display_live_option(agent_name.as_deref(), 120),
        display_live_option(action.as_deref(), 64),
    );

    let Some(progress) = progress else {
        return;
    };
    let status = OpenAIResponsesLiveStatusUpdate {
        response_id: response_id.map(|value| sanitize_display(value, 128)),
        event_type: sanitize_display(event_type, 80),
        sequence_number,
        output_index,
        item_type,
        agent_name,
        action,
    };
    let trace = live_trace_event(data, event_type, &status, live_state);
    progress.observe_live_event(status, trace);
}

fn stream_agent_name(data: &Value) -> Option<String> {
    data.get("agent")
        .and_then(|agent| agent.get("agent_name"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn live_trace_event(
    data: &Value,
    event_type: &str,
    status: &OpenAIResponsesLiveStatusUpdate,
    live_state: &mut OpenAIResponsesLiveStreamState,
) -> Option<OpenAIResponsesLiveTraceEvent> {
    let response_id = status.response_id.as_deref().unwrap_or("unavailable");
    match event_type {
        "response.created" => Some(OpenAIResponsesLiveTraceEvent {
            line: format!(
                "response={} status=created",
                sanitize_display(response_id, 128)
            ),
        }),
        "response.completed" | "response.failed" | "response.incomplete" => {
            let status = event_type.strip_prefix("response.").unwrap_or(event_type);
            Some(OpenAIResponsesLiveTraceEvent {
                line: format!(
                    "response={} status={}",
                    sanitize_display(response_id, 128),
                    sanitize_display(status, 32)
                ),
            })
        }
        "error" => Some(OpenAIResponsesLiveTraceEvent {
            line: format!(
                "response={} status=error",
                sanitize_display(response_id, 128)
            ),
        }),
        "response.output_item.added" => {
            let item_type = status.item_type.as_deref()?;
            if !is_live_structural_item(item_type) {
                return None;
            }
            Some(OpenAIResponsesLiveTraceEvent {
                line: format!(
                    "response={} item={} agent={} action={} status=started",
                    sanitize_display(response_id, 128),
                    sanitize_display(item_type, 64),
                    display_live_option(status.agent_name.as_deref(), 120),
                    display_live_option(status.action.as_deref(), 64)
                ),
            })
        }
        "response.output_item.done" => {
            let raw_item = data.get("item")?;
            let item = live_trace_item_with_context(raw_item, status);
            let trace = trace_event(&item, &live_state.hosted_calls)?;
            if let OpenAIResponsesTraceEvent::MultiAgentCall {
                call_id: Some(call_id),
                action,
                agent_name,
                ..
            } = &trace
            {
                live_state.hosted_calls.insert(
                    call_id.clone(),
                    HostedCallContext {
                        action: action.clone(),
                        agent_name: agent_name.clone(),
                    },
                );
            }
            let mut detail = String::new();
            format_trace_event(&mut detail, &trace);
            let line = format_live_structural_trace_line(response_id, detail.trim_start());
            live_state
                .mark_trace_item(status.output_index, raw_item)
                .then_some(OpenAIResponsesLiveTraceEvent { line })
        }
        _ if event_type.starts_with("response.web_search_call.") => {
            let web_status = event_type
                .strip_prefix("response.web_search_call.")
                .unwrap_or("unknown");
            Some(OpenAIResponsesLiveTraceEvent {
                line: format!(
                    "response={} web_search agent={} status={}",
                    sanitize_display(response_id, 128),
                    display_live_option(status.agent_name.as_deref(), 120),
                    sanitize_display(web_status, 32)
                ),
            })
        }
        _ => None,
    }
}

fn live_trace_item_with_context(
    raw_item: &Value,
    status: &OpenAIResponsesLiveStatusUpdate,
) -> Value {
    let mut item = raw_item.clone();
    let Some(item_object) = item.as_object_mut() else {
        return item;
    };
    if trace_agent_name(raw_item).is_none() {
        if let Some(agent_name) = &status.agent_name {
            item_object.insert("agent".to_string(), json!({"agent_name": agent_name}));
        }
    }
    if raw_item.get("action").and_then(Value::as_str).is_none() {
        if let Some(action) = &status.action {
            item_object.insert("action".to_string(), Value::String(action.clone()));
        }
    }
    item
}

fn emit_terminal_response_trace(
    data: &Value,
    response_id: Option<&str>,
    progress: Option<&OpenAIResponsesProgress>,
    live_state: &mut OpenAIResponsesLiveStreamState,
) {
    let Some(progress) = progress else {
        return;
    };
    let Some(output) = data.pointer("/response/output").and_then(Value::as_array) else {
        return;
    };
    let response_id = response_id.unwrap_or("unavailable");
    let hosted_calls = hosted_call_contexts(output);
    for (output_index, item) in output.iter().enumerate() {
        let Some(trace) = trace_event(item, &hosted_calls) else {
            continue;
        };
        if !live_state.mark_trace_item(u64::try_from(output_index).ok(), item) {
            continue;
        }
        let mut detail = String::new();
        format_trace_event(&mut detail, &trace);
        let line = format_live_structural_trace_line(response_id, detail.trim_start());
        progress.emit_live_trace(OpenAIResponsesLiveTraceEvent { line });
    }
}

fn format_live_structural_trace_line(response_id: &str, detail: &str) -> String {
    format!(
        "response={} status=completed {}",
        sanitize_display(response_id, 128),
        detail
    )
}

fn is_live_structural_item(item_type: &str) -> bool {
    matches!(
        item_type,
        "multi_agent_call" | "multi_agent_call_output" | "agent_message" | "function_call"
    ) || item_type.ends_with("_call")
}

fn display_live_value(value: Option<&str>, max_chars: usize) -> String {
    display_live_option(value, max_chars)
}

fn display_live_option(value: Option<&str>, max_chars: usize) -> String {
    value
        .map(|value| sanitize_display(value, max_chars))
        .unwrap_or_else(|| "unavailable".to_string())
}

fn display_live_number(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unavailable".to_string())
}

fn responses_stream_position(
    response_id: Option<&str>,
    sequence_number: Option<u64>,
    client_request_id: &str,
) -> String {
    let mut fields = vec![format!(
        "x-client-request-id: {}",
        sanitize_display(client_request_id, 128)
    )];
    if let Some(response_id) = response_id {
        fields.push(format!(
            "response_id: {}",
            sanitize_display(response_id, 128)
        ));
    }
    if let Some(sequence_number) = sequence_number {
        fields.push(format!("sequence_number: {sequence_number}"));
    }
    if fields.is_empty() {
        String::new()
    } else {
        format!(" ({})", fields.join(", "))
    }
}

fn openai_responses_stream_error(
    data: &Value,
    response_id: Option<&str>,
    sequence_number: Option<u64>,
    client_request_id: &str,
) -> anyhow::Error {
    let message = data
        .get("message")
        .and_then(Value::as_str)
        .map(|value| sanitize_display(value, 240))
        .unwrap_or_else(|| "unknown streaming error".to_string());
    let code = data
        .get("code")
        .and_then(Value::as_str)
        .map(|value| format!(", code: {}", sanitize_display(value, 96)))
        .unwrap_or_default();
    anyhow::anyhow!(
        "OpenAI Responses stream error{}: {}{}",
        responses_stream_position(response_id, sequence_number, client_request_id),
        message,
        code
    )
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

#[derive(Debug, Clone, Default)]
pub struct OpenAIResponsesProgress {
    state: Arc<Mutex<OpenAIResponsesProgressState>>,
}

#[derive(Debug, Default)]
struct OpenAIResponsesProgressState {
    turns: Vec<OpenAIResponsesTurn>,
    pricing_context: Option<OpenAIResponsesPricingContext>,
    live_status: OpenAIResponsesLiveStatus,
    live_trace_sender: Option<UnboundedSender<OpenAIResponsesLiveTraceEvent>>,
}

impl OpenAIResponsesProgress {
    pub fn live() -> (Self, UnboundedReceiver<OpenAIResponsesLiveTraceEvent>) {
        let (sender, receiver) = mpsc::unbounded_channel();
        let progress = Self::default();
        {
            let mut state = progress.state.lock();
            state.live_status.started_at = Some(std::time::Instant::now());
            state.live_trace_sender = Some(sender);
        }
        (progress, receiver)
    }

    fn set_pricing_context(&self, pricing_context: OpenAIResponsesPricingContext) {
        self.state.lock().pricing_context = Some(pricing_context);
    }

    fn push_turn(&self, turn: OpenAIResponsesTurn) {
        self.state.lock().turns.push(turn);
    }

    fn begin_live_turn(&self) {
        let mut state = self.state.lock();
        state.live_status.response_id = None;
        state.live_status.event_type = Some("connecting".to_string());
        state.live_status.sequence_number = None;
        state.live_status.output_index = None;
        state.live_status.item_type = None;
        state.live_status.agent_name = None;
        state.live_status.action = None;
        state.live_status.last_event_at = Some(std::time::Instant::now());
    }

    fn record_stream_open(&self) {
        let mut state = self.state.lock();
        state.live_status.event_type = Some("stream.open".to_string());
        state.live_status.last_event_at = Some(std::time::Instant::now());
    }

    fn observe_live_event(
        &self,
        status: OpenAIResponsesLiveStatusUpdate,
        trace: Option<OpenAIResponsesLiveTraceEvent>,
    ) {
        {
            let mut state = self.state.lock();
            state.live_status.response_id = status.response_id;
            state.live_status.event_type = Some(status.event_type);
            state.live_status.sequence_number = status.sequence_number;
            state.live_status.output_index = status.output_index;
            state.live_status.item_type = status.item_type;
            state.live_status.agent_name = status.agent_name;
            state.live_status.action = status.action;
            state.live_status.last_event_at = Some(std::time::Instant::now());
        }
        if let Some(trace) = trace {
            self.emit_live_trace(trace);
        }
    }

    fn emit_live_trace(&self, trace: OpenAIResponsesLiveTraceEvent) {
        let sender = self.state.lock().live_trace_sender.clone();
        if let Some(sender) = sender {
            let _ = sender.send(trace);
        }
    }

    pub fn live_snapshot(&self) -> OpenAIResponsesLiveSnapshot {
        let state = self.state.lock();
        OpenAIResponsesLiveSnapshot {
            started_at: state.live_status.started_at,
            last_event_at: state.live_status.last_event_at,
            response_id: state.live_status.response_id.clone(),
            event_type: state.live_status.event_type.clone(),
            sequence_number: state.live_status.sequence_number,
            output_index: state.live_status.output_index,
            item_type: state.live_status.item_type.clone(),
            agent_name: state.live_status.agent_name.clone(),
            action: state.live_status.action.clone(),
        }
    }

    pub fn snapshot(&self) -> (Vec<OpenAIResponsesTurn>, OpenAIResponsesPricingContext) {
        let state = self.state.lock();
        (
            state.turns.clone(),
            state
                .pricing_context
                .unwrap_or(OpenAIResponsesPricingContext::UnknownApiBase),
        )
    }
}

#[derive(Debug, Default)]
struct OpenAIResponsesLiveStatus {
    started_at: Option<std::time::Instant>,
    last_event_at: Option<std::time::Instant>,
    response_id: Option<String>,
    event_type: Option<String>,
    sequence_number: Option<u64>,
    output_index: Option<u64>,
    item_type: Option<String>,
    agent_name: Option<String>,
    action: Option<String>,
}

#[derive(Debug)]
struct OpenAIResponsesLiveStatusUpdate {
    response_id: Option<String>,
    event_type: String,
    sequence_number: Option<u64>,
    output_index: Option<u64>,
    item_type: Option<String>,
    agent_name: Option<String>,
    action: Option<String>,
}

#[derive(Debug, Clone)]
pub struct OpenAIResponsesLiveSnapshot {
    started_at: Option<std::time::Instant>,
    last_event_at: Option<std::time::Instant>,
    response_id: Option<String>,
    event_type: Option<String>,
    sequence_number: Option<u64>,
    output_index: Option<u64>,
    item_type: Option<String>,
    agent_name: Option<String>,
    action: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAIResponsesLiveTraceEvent {
    line: String,
}

impl OpenAIResponsesLiveTraceEvent {
    pub fn line(&self) -> &str {
        &self.line
    }
}

pub fn format_openai_responses_live_progress(
    snapshot: &OpenAIResponsesLiveSnapshot,
    now: std::time::Instant,
) -> String {
    let elapsed = snapshot
        .started_at
        .map(|started_at| now.saturating_duration_since(started_at))
        .unwrap_or_default();
    let idle = snapshot
        .last_event_at
        .map(|last_event_at| now.saturating_duration_since(last_event_at))
        .unwrap_or(elapsed);
    let mut output = format!("Generating {}", format_elapsed(elapsed));
    if let Some(event_type) = &snapshot.event_type {
        let _ = write!(output, " | last={}", sanitize_display(event_type, 80));
    }
    if let Some(response_id) = &snapshot.response_id {
        let _ = write!(output, " | response={}", sanitize_display(response_id, 48));
    }
    if let Some(agent_name) = &snapshot.agent_name {
        let _ = write!(output, " | agent={}", sanitize_display(agent_name, 80));
    }
    if let Some(item_type) = &snapshot.item_type {
        let _ = write!(output, " | item={}", sanitize_display(item_type, 64));
    }
    if let Some(action) = &snapshot.action {
        let _ = write!(output, " | action={}", sanitize_display(action, 64));
    }
    if let Some(output_index) = snapshot.output_index {
        let _ = write!(output, " | output={output_index}");
    }
    if let Some(sequence_number) = snapshot.sequence_number {
        let _ = write!(output, " | seq={sequence_number}");
    }
    if idle >= Duration::from_secs(2) {
        let _ = write!(output, " | idle={}s", idle.as_secs());
    }
    output
}

fn format_elapsed(duration: Duration) -> String {
    let seconds = duration.as_secs();
    format!("{:02}:{:02}", seconds / 60, seconds % 60)
}

#[derive(Debug, Clone)]
pub struct OpenAIResponsesOutput {
    pub completion: ChatCompletionsOutput,
    pub turns: Vec<OpenAIResponsesTurn>,
    pub citations: Vec<OpenAIResponsesUrlCitation>,
    pub sources: Vec<OpenAIResponsesSource>,
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
    pub web_search_calls: Vec<OpenAIResponsesWebSearchCall>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAIResponsesSource {
    pub url: String,
    pub title: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAIResponsesUrlCitation {
    pub source: OpenAIResponsesSource,
    pub start_index: Option<u64>,
    pub end_index: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenAIResponsesWebSearchAction {
    Search,
    OpenPage,
    Find,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAIResponsesWebSearchCall {
    pub action: OpenAIResponsesWebSearchAction,
    pub sources: Vec<OpenAIResponsesSource>,
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
    UnknownModel,
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
    pub root_final_citations: Vec<OpenAIResponsesUrlCitation>,
    pub usage: TokenUsage,
    pub service_tier: Option<String>,
    pub detailed_usage: OpenAIResponsesUsage,
    pub trace: Vec<OpenAIResponsesTraceEvent>,
    pub web_search_calls: Vec<OpenAIResponsesWebSearchCall>,
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
                "OpenAI Responses request '{}' failed: {}",
                sanitize_display(response_id, 128),
                display_response_error(error)
            ),
            Self::IncompleteResponse {
                response_id,
                incomplete_details,
            } => write!(
                formatter,
                "OpenAI Responses request '{}' was incomplete: {}",
                sanitize_display(response_id, 128),
                display_incomplete_details(incomplete_details)
            ),
            Self::UnexpectedStatus {
                response_id,
                status,
            } => write!(
                formatter,
                "OpenAI Responses request '{}' has unsupported status '{}'",
                sanitize_display(response_id, 128),
                sanitize_display(status, 64)
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
                "OpenAI Responses request '{}' completed without a root final answer",
                sanitize_display(response_id, 128)
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

fn observe_openai_responses_turn(response: &Value) -> Option<OpenAIResponsesTurn> {
    let envelope: ResponseEnvelope = serde_json::from_value(response.clone()).ok()?;
    Some(OpenAIResponsesTurn {
        response_id: envelope.id,
        service_tier: envelope.service_tier,
        usage: detailed_response_usage(envelope.usage.as_ref()),
        trace: extract_trace(&envelope.output),
        web_search_calls: extract_web_search_calls(&envelope.output),
    })
}

fn detailed_response_usage(usage: Option<&ResponseUsage>) -> OpenAIResponsesUsage {
    let Some(usage) = usage else {
        return OpenAIResponsesUsage::default();
    };
    OpenAIResponsesUsage {
        input_tokens: usage.input_tokens,
        cached_input_tokens: usage
            .input_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens),
        cache_write_input_tokens: usage
            .input_tokens_details
            .as_ref()
            .and_then(|details| details.cache_write_tokens),
        output_tokens: usage.output_tokens,
        reasoning_output_tokens: usage
            .output_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens),
    }
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
    let (root_final_text, root_final_citations) = extract_root_final(&envelope.output);
    let trace = extract_trace(&envelope.output);
    let web_search_calls = extract_web_search_calls(&envelope.output);

    if function_calls.is_empty() && root_final_text.is_none() {
        return Err(OpenAIResponsesMultiAgentError::MissingRootFinal {
            response_id: envelope.id,
        });
    }

    let detailed_usage = detailed_response_usage(envelope.usage.as_ref());
    let usage = TokenUsage::new(detailed_usage.input_tokens, detailed_usage.output_tokens);

    Ok(OpenAIResponsesMultiAgentResult {
        response_id: envelope.id,
        status,
        output_items: envelope.output,
        function_calls,
        root_final_text,
        root_final_citations,
        usage,
        service_tier: envelope.service_tier,
        detailed_usage,
        trace,
        web_search_calls,
    })
}

#[derive(Clone)]
struct HostedCallContext {
    action: OpenAIResponsesHostedAction,
    agent_name: Option<String>,
}

fn extract_trace(output: &[Value]) -> Vec<OpenAIResponsesTraceEvent> {
    let calls_by_id = hosted_call_contexts(output);

    output
        .iter()
        .filter_map(|item| trace_event(item, &calls_by_id))
        .collect()
}

fn hosted_call_contexts(output: &[Value]) -> HashMap<String, HostedCallContext> {
    output
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

fn extract_root_final(output: &[Value]) -> (Option<String>, Vec<OpenAIResponsesUrlCitation>) {
    let mut text = String::new();
    let mut found_output_text = false;
    let mut citations = Vec::new();

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
                if let Some(annotations) = part.get("annotations").and_then(Value::as_array) {
                    citations.extend(annotations.iter().filter_map(parse_url_citation));
                }
            }
        }
    }

    (found_output_text.then_some(text), citations)
}

fn parse_url_citation(annotation: &Value) -> Option<OpenAIResponsesUrlCitation> {
    if annotation.get("type").and_then(Value::as_str) != Some("url_citation") {
        return None;
    }
    let citation = annotation.get("url_citation").unwrap_or(annotation);
    let source = parse_source(citation)?;
    let start_index = citation.get("start_index").and_then(Value::as_u64);
    let end_index = citation.get("end_index").and_then(Value::as_u64);
    let (start_index, end_index) = match (start_index, end_index) {
        (Some(start), Some(end)) if start <= end => (Some(start), Some(end)),
        _ => (None, None),
    };
    Some(OpenAIResponsesUrlCitation {
        source,
        start_index,
        end_index,
    })
}

fn extract_web_search_calls(output: &[Value]) -> Vec<OpenAIResponsesWebSearchCall> {
    output
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("web_search_call"))
        .map(|item| {
            let action = item.get("action");
            let action_type = action
                .and_then(|action| action.get("type"))
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let action = match action_type {
                "search" => OpenAIResponsesWebSearchAction::Search,
                "open_page" => OpenAIResponsesWebSearchAction::OpenPage,
                "find" | "find_in_page" => OpenAIResponsesWebSearchAction::Find,
                value => OpenAIResponsesWebSearchAction::Other(sanitize_display(value, 64)),
            };
            let mut sources = Vec::new();
            if let Some(values) = item
                .get("action")
                .and_then(|action| action.get("sources"))
                .and_then(Value::as_array)
            {
                merge_sources(
                    &mut sources,
                    values.iter().take(256).filter_map(parse_source),
                );
            }
            OpenAIResponsesWebSearchCall { action, sources }
        })
        .collect()
}

fn parse_source(value: &Value) -> Option<OpenAIResponsesSource> {
    let url = sanitize_source_url(value.get("url").and_then(Value::as_str)?)?;
    let title = value
        .get("title")
        .and_then(Value::as_str)
        .and_then(sanitize_source_title);
    Some(OpenAIResponsesSource { url, title })
}

fn sanitize_source_url(value: &str) -> Option<String> {
    if value
        .chars()
        .any(|character| character.is_control() || is_unicode_format_control(character))
    {
        return None;
    }
    let url = Url::parse(value.trim()).ok()?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return None;
    }
    let url = url.to_string();
    (!url.contains('<') && !url.contains('>')).then_some(url)
}

fn sanitize_source_title(value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let sanitized = sanitize_display(value, 200);
    if sanitized == "unavailable" && value != "unavailable" {
        None
    } else {
        Some(sanitized)
    }
}

fn merge_sources(
    sources: &mut Vec<OpenAIResponsesSource>,
    incoming: impl IntoIterator<Item = OpenAIResponsesSource>,
) {
    for source in incoming {
        if let Some(existing) = sources
            .iter_mut()
            .find(|existing| existing.url == source.url)
        {
            if source.title.is_some() {
                existing.title = source.title;
            }
        } else if sources.len() < 256 {
            sources.push(source);
        }
    }
}

fn append_sources_block(mut text: String, sources: &[OpenAIResponsesSource]) -> String {
    if sources.is_empty() {
        return text;
    }
    if !text.is_empty() {
        text.push_str("\n\n");
    }
    text.push_str("Sources:");
    for source in sources {
        let title = source.title.as_deref().unwrap_or(&source.url);
        let _ = write!(
            text,
            "\n- [{}](<{}>)",
            escape_markdown_label(title),
            source.url
        );
    }
    text
}

fn escape_markdown_label(value: &str) -> String {
    sanitize_display(value, 200)
        .chars()
        .flat_map(|character| {
            if matches!(character, '\\' | '[' | ']' | '*' | '_' | '`' | '<' | '>') {
                [Some('\\'), Some(character)]
            } else {
                [Some(character), None]
            }
        })
        .flatten()
        .collect()
}

fn display_response_error(value: &Option<Value>) -> String {
    let Some(error) = value.as_ref().and_then(Value::as_object) else {
        return "no safe details".to_string();
    };
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .map(|value| sanitize_display(value, 240));
    let error_type = error
        .get("type")
        .and_then(Value::as_str)
        .map(|value| sanitize_display(value, 96));
    let code = error
        .get("code")
        .and_then(Value::as_str)
        .map(|value| sanitize_display(value, 96));
    let qualifier = error_type
        .map(|value| format!("type: {value}"))
        .or_else(|| code.map(|value| format!("code: {value}")));
    match (message, qualifier) {
        (Some(message), Some(qualifier)) => format!("{message} ({qualifier})"),
        (Some(message), None) => message,
        (None, Some(qualifier)) => qualifier,
        (None, None) => "no safe details".to_string(),
    }
}

fn display_incomplete_details(value: &Option<Value>) -> String {
    value
        .as_ref()
        .and_then(|details| details.get("reason"))
        .and_then(Value::as_str)
        .map(|reason| format!("reason: {}", sanitize_display(reason, 120)))
        .unwrap_or_else(|| "no safe details".to_string())
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

    let token_cost = calculate_openai_responses_token_cost(model, turns, pricing_context);
    match &token_cost {
        Ok(cost) => {
            let _ = write!(summary, "\nEstimated token subtotal: ${cost:.6}");
        }
        Err(reason) => {
            let _ = write!(
                summary,
                "\nEstimated token subtotal: unavailable ({})",
                sanitize_display(reason, 240)
            );
        }
    }
    let web_search_calls = billable_web_search_calls(turns);
    let web_search_fee = calculate_web_search_fee(model, turns, pricing_context);
    match &web_search_fee {
        Ok(fee) => {
            let _ = write!(
                summary,
                "\nHosted web searches: {web_search_calls} | Fee: ${fee:.6}"
            );
        }
        Err(reason) => {
            let _ = write!(
                summary,
                "\nHosted web searches: {web_search_calls} | Fee: unavailable ({})",
                sanitize_display(reason, 240)
            );
        }
    }
    match (token_cost, web_search_fee) {
        (Ok(token_cost), Ok(web_search_fee)) => {
            let total = token_cost + web_search_fee;
            if total.is_finite() {
                let _ = write!(summary, "\nEstimated total cost: ${total:.6}");
            } else {
                summary.push_str("\nEstimated total cost: unavailable");
            }
        }
        _ => summary.push_str("\nEstimated total cost: unavailable"),
    }
    summary
}

fn calculate_openai_responses_token_cost(
    model: &Model,
    turns: &[OpenAIResponsesTurn],
    pricing_context: OpenAIResponsesPricingContext,
) -> std::result::Result<f64, String> {
    if turns.is_empty() {
        return Err("no response usage was returned".to_string());
    }
    match pricing_context {
        OpenAIResponsesPricingContext::PublicApi => {}
        OpenAIResponsesPricingContext::UnknownApiBase => {
            return Err("custom OpenAI api_base has unknown pricing".to_string());
        }
        OpenAIResponsesPricingContext::UnknownModel => {
            return Err(
                "effective OpenAI Responses model does not match selected model pricing"
                    .to_string(),
            );
        }
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

fn billable_web_search_calls(turns: &[OpenAIResponsesTurn]) -> u64 {
    turns
        .iter()
        .flat_map(|turn| &turn.web_search_calls)
        .filter(|call| matches!(call.action, OpenAIResponsesWebSearchAction::Search))
        .count() as u64
}

fn calculate_web_search_fee(
    model: &Model,
    turns: &[OpenAIResponsesTurn],
    pricing_context: OpenAIResponsesPricingContext,
) -> std::result::Result<f64, String> {
    if turns
        .iter()
        .flat_map(|turn| &turn.web_search_calls)
        .any(|call| matches!(call.action, OpenAIResponsesWebSearchAction::Other(_)))
    {
        return Err("response contains an unknown web-search action".to_string());
    }
    let calls = billable_web_search_calls(turns);
    if calls == 0 {
        return Ok(0.0);
    }
    match pricing_context {
        OpenAIResponsesPricingContext::PublicApi => {}
        OpenAIResponsesPricingContext::UnknownApiBase => {
            return Err("custom OpenAI api_base has unknown hosted-tool pricing".to_string());
        }
        OpenAIResponsesPricingContext::UnknownModel => {
            return Err(
                "effective OpenAI Responses model does not match selected model pricing"
                    .to_string(),
            );
        }
    }
    let price = model
        .data()
        .response_pricing
        .as_ref()
        .and_then(|pricing| pricing.web_search_call_price)
        .ok_or_else(|| "model web-search call price is missing".to_string())?;
    let price = valid_number(price, true, "web-search call price")?;
    let fee = calls as f64 * price;
    if !fee.is_finite() {
        return Err("calculated web-search fee is invalid".to_string());
    }
    Ok(fee)
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

#[cfg(test)]
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

    use crate::cli::Cli;
    use crate::client::{
        response_fixture_builder_with_headers, sse_fixture_builder, ImageUrl, ModelData, ModelType,
        OpenAIClient, OpenAIConfig, ResponsePricing,
    };
    use crate::config::{
        Config, OpenAIServiceTier, WebSearchContextSize, WebSearchFilters,
        WebSearchReturnTokenBudget,
    };
    use crate::configure_multi_agent;
    use crate::utils::create_abort_signal;
    use clap::Parser;
    use parking_lot::RwLock;
    use std::cell::{Cell, RefCell};
    use std::collections::{BTreeMap, VecDeque};
    use std::future::{pending, ready};
    use std::num::NonZeroUsize;
    use std::sync::Arc;

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

    fn response_with_web_sources_and_citations() -> Value {
        json!({
            "id": "resp_sources",
            "status": "completed",
            "service_tier": "default",
            "output": [
                {
                    "type": "web_search_call",
                    "agent": {"agent_name": "/root/research"},
                    "action": {
                        "type": "search",
                        "query": "private query",
                        "sources": [
                            {"type": "url", "url": "https://example.com/report"},
                            {"type": "url", "url": "javascript:alert(1)", "title": "bad"},
                            {"type": "url", "url": "https://bad.example/\npath", "title": "bad"}
                        ]
                    }
                },
                {
                    "type": "web_search_call",
                    "action": {
                        "type": "open_page",
                        "sources": [{
                            "type": "url",
                            "url": "https://docs.example/🙂",
                            "title": "older title"
                        }]
                    }
                },
                {"type": "web_search_call", "action": {"type": "find"}},
                {"type": "web_search_call", "action": {"type": "future_action"}},
                {
                    "type": "message",
                    "agent": {"agent_name": "/root"},
                    "phase": "final_answer",
                    "content": [{
                        "type": "output_text",
                        "text": "分析🙂done",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://example.com/report",
                                "title": "Better *title*\n\u{202e}raw",
                                "start_index": 99,
                                "end_index": 1
                            },
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "url": "https://docs.example/🙂",
                                    "title": "Docs ] _guide_",
                                    "start_index": 1,
                                    "end_index": 2
                                }
                            },
                            {"type": "url_citation", "url": "file:///etc/passwd"},
                            {"type": "other", "url": "https://ignored.example"}
                        ]
                    }]
                }
            ],
            "usage": {
                "input_tokens": 10,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 2,
                "output_tokens_details": {"reasoning_tokens": 0}
            },
            "error": null,
            "incomplete_details": null
        })
    }

    #[test]
    fn parses_safe_citations_sources_and_web_search_action_kinds() {
        let result =
            parse_openai_responses_multi_agent(&response_with_web_sources_and_citations()).unwrap();

        assert_eq!(result.root_final_text.as_deref(), Some("分析🙂done"));
        assert_eq!(result.root_final_citations.len(), 2);
        assert_eq!(result.root_final_citations[0].start_index, None);
        assert_eq!(result.root_final_citations[0].end_index, None);
        assert_eq!(result.root_final_citations[1].start_index, Some(1));
        assert_eq!(result.root_final_citations[1].end_index, Some(2));
        assert_eq!(result.web_search_calls.len(), 4);
        assert!(matches!(
            result.web_search_calls[0].action,
            OpenAIResponsesWebSearchAction::Search
        ));
        assert!(matches!(
            result.web_search_calls[1].action,
            OpenAIResponsesWebSearchAction::OpenPage
        ));
        assert!(matches!(
            result.web_search_calls[2].action,
            OpenAIResponsesWebSearchAction::Find
        ));
        assert!(matches!(
            &result.web_search_calls[3].action,
            OpenAIResponsesWebSearchAction::Other(value) if value == "future_action"
        ));
        assert_eq!(result.web_search_calls[0].sources.len(), 1);
        assert_eq!(result.web_search_calls[1].sources.len(), 1);
        assert_eq!(
            result.web_search_calls[1].sources[0].title.as_deref(),
            Some("older title")
        );
    }

    #[tokio::test]
    async fn renders_deduplicated_markdown_sources_without_slicing_unicode_text() {
        let response = response_with_web_sources_and_citations();
        let output = run_multi_agent_loop(
            json!({"input": []}),
            &create_abort_signal(),
            |_| ready(Ok(response.clone())),
            |_| unreachable!("the fixture has no developer function calls"),
        )
        .await
        .unwrap();

        assert_eq!(output.citations.len(), 2);
        assert_eq!(output.sources.len(), 2);
        assert!(output
            .completion
            .text
            .starts_with("分析🙂done\n\nSources:\n"));
        assert!(output
            .completion
            .text
            .contains("- [Better \\*title\\* raw](<https://example.com/report>)"));
        assert!(output
            .completion
            .text
            .contains("- [Docs \\] \\_guide\\_](<https://docs.example/%F0%9F%99%82>)"));
        assert!(!output.completion.text.contains("javascript:"));
        assert!(!output.completion.text.contains("file:///"));
        assert!(!output.completion.text.contains('\u{202e}'));
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
                    "action": {
                        "type": "search",
                        "query": "query-secret",
                        "sources": [{
                            "type": "url",
                            "url": "https://source-secret.example/path",
                            "title": "source-title-secret"
                        }]
                    },
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
            web_search_calls: parsed.web_search_calls,
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
            "source-secret",
            "source-title-secret",
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
    fn failed_and_incomplete_details_are_allowlisted_and_bounded() {
        const ERROR_SENTINEL: &str = "FAILED_ERROR_PRIVATE_SENTINEL";
        const MESSAGE_TAIL: &str = "FAILED_MESSAGE_TAIL_SENTINEL";
        let failed = json!({
            "id": "resp_failed\n\u{202e}suffix",
            "status": "failed",
            "output": [],
            "usage": null,
            "error": {
                "code": "server_error",
                "message": format!("visible\n{}{}", "x".repeat(400), MESSAGE_TAIL),
                "internal_details": ERROR_SENTINEL
            },
            "incomplete_details": null,
        });
        let failed_message = parse_openai_responses_multi_agent(&failed)
            .unwrap_err()
            .to_string();

        assert!(failed_message.contains("visible x"));
        assert!(failed_message.contains("code: server_error"));
        assert!(!failed_message.contains(ERROR_SENTINEL));
        assert!(!failed_message.contains(MESSAGE_TAIL));
        assert!(!failed_message.contains('\n'));
        assert!(!failed_message.contains('\u{202e}'));
        assert!(failed_message.len() < 640);

        let incomplete = json!({
            "id": "resp_incomplete",
            "status": "incomplete",
            "output": [],
            "usage": null,
            "error": null,
            "incomplete_details": {
                "reason": format!("max_output_tokens\n{}", "y".repeat(200)),
                "internal_details": "INCOMPLETE_PRIVATE_SENTINEL"
            },
        });
        let incomplete_message = parse_openai_responses_multi_agent(&incomplete)
            .unwrap_err()
            .to_string();

        assert!(incomplete_message.contains("reason: max_output_tokens y"));
        assert!(!incomplete_message.contains("INCOMPLETE_PRIVATE_SENTINEL"));
        assert!(!incomplete_message.contains('\n'));
        assert!(incomplete_message.len() < 360);
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
            web_search_call_price: Some(0.01),
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
            web_search_calls: Vec::new(),
        }
    }

    fn public_responses_cost(
        model: &Model,
        turns: &[OpenAIResponsesTurn],
    ) -> std::result::Result<f64, String> {
        let token_cost = calculate_openai_responses_token_cost(
            model,
            turns,
            OpenAIResponsesPricingContext::PublicApi,
        )?;
        let web_search_fee =
            calculate_web_search_fee(model, turns, OpenAIResponsesPricingContext::PublicApi)?;
        let total = token_cost + web_search_fee;
        total
            .is_finite()
            .then_some(total)
            .ok_or_else(|| "calculated total response cost is invalid".to_string())
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
        assert!(formatted.contains("Estimated token subtotal: $0.002090"));
        assert!(formatted.contains("Hosted web searches: 0 | Fee: $0.000000"));
        assert!(formatted.contains("Estimated total cost: $0.002090"));
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
    fn web_search_cost_counts_search_actions_without_double_charging_tokens() {
        let model = priced_responses_model();
        let mut turn = usage_turn("one", Some("default"), 10, 0, 0, 8, 8);
        turn.web_search_calls = vec![
            OpenAIResponsesWebSearchCall {
                action: OpenAIResponsesWebSearchAction::Search,
                sources: Vec::new(),
            },
            OpenAIResponsesWebSearchCall {
                action: OpenAIResponsesWebSearchAction::OpenPage,
                sources: Vec::new(),
            },
            OpenAIResponsesWebSearchCall {
                action: OpenAIResponsesWebSearchAction::Find,
                sources: Vec::new(),
            },
            OpenAIResponsesWebSearchCall {
                action: OpenAIResponsesWebSearchAction::Search,
                sources: Vec::new(),
            },
        ];

        let token_cost = calculate_openai_responses_token_cost(
            &model,
            std::slice::from_ref(&turn),
            OpenAIResponsesPricingContext::PublicApi,
        )
        .unwrap();
        let total = public_responses_cost(&model, std::slice::from_ref(&turn)).unwrap();

        assert!((token_cost - 0.00026).abs() < f64::EPSILON);
        assert!((total - 0.02026).abs() < f64::EPSILON);
        assert_eq!(billable_web_search_calls(std::slice::from_ref(&turn)), 2);
        let summary = format_public_responses_cost(&model, &[turn]);
        assert!(summary.contains("Estimated token subtotal: $0.000260"));
        assert!(summary.contains("Hosted web searches: 2 | Fee: $0.020000"));
        assert!(summary.contains("Estimated total cost: $0.020260"));
    }

    #[test]
    fn unknown_web_search_action_never_formats_exact_fee_or_total() {
        let model = priced_responses_model();
        for item in [
            json!({"type": "web_search_call", "action": {"type": "future_action"}}),
            json!({"type": "web_search_call"}),
            json!({"type": "web_search_call", "action": {"type": 42}}),
        ] {
            let mut turn = usage_turn("one", Some("default"), 10, 0, 0, 8, 0);
            turn.web_search_calls = extract_web_search_calls(&[item]);
            assert!(matches!(
                &turn.web_search_calls[0].action,
                OpenAIResponsesWebSearchAction::Other(_)
            ));

            let summary = format_public_responses_cost(&model, &[turn]);

            assert!(summary.contains("Estimated token subtotal: $0.000260"));
            assert!(summary.contains("Hosted web searches: 0 | Fee: unavailable"));
            assert!(summary.contains("unknown web-search action"));
            assert!(!summary.contains("Hosted web searches: 0 | Fee: $0.000000"));
            assert!(summary.contains("Estimated total cost: unavailable"));
        }
    }

    #[test]
    fn web_search_total_is_unavailable_when_call_pricing_is_missing() {
        let mut model = priced_responses_model();
        model
            .data_mut()
            .response_pricing
            .as_mut()
            .unwrap()
            .web_search_call_price = None;
        let mut turn = usage_turn("one", Some("default"), 10, 0, 0, 8, 0);
        turn.web_search_calls.push(OpenAIResponsesWebSearchCall {
            action: OpenAIResponsesWebSearchAction::Search,
            sources: Vec::new(),
        });

        let summary = format_public_responses_cost(&model, &[turn]);

        assert!(summary.contains("Estimated token subtotal: $0.000260"));
        assert!(summary.contains("Hosted web searches: 1 | Fee: unavailable"));
        assert!(summary.contains("model web-search call price is missing"));
        assert!(summary.contains("Estimated total cost: unavailable"));
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
        assert!(format_public_responses_cost(&model, &invalid)
            .contains("Estimated total cost: unavailable"));
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

        assert!(summary.contains("Estimated token subtotal: unavailable"));
        assert!(summary.contains("Estimated total cost: unavailable"));
        assert!(summary.contains("custom OpenAI api_base has unknown pricing"));
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

        let settings = MultiAgentConfig {
            max_concurrent_subagents: NonZeroUsize::new(7),
            ..Default::default()
        };
        let body = build_openai_responses_multi_agent_body(
            data,
            &responses_model(Some("high")),
            &settings,
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
                "stream": true,
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
                "tool_choice": "auto",
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

        let body = build_openai_responses_multi_agent_body(
            data,
            &responses_model(None),
            &MultiAgentConfig::default(),
        )
        .unwrap();

        assert_eq!(body["temperature"], 0.2);
        assert_eq!(body["top_p"], 0.9);
        assert_eq!(body["service_tier"], "auto");
        assert!(body.get("reasoning").is_none());
        assert!(body.get("tools").is_none());
        assert_eq!(body["stream"], true);
        assert_eq!(body["multi_agent"], json!({"enabled": true}));
    }

    #[test]
    fn request_uses_responses_endpoint_headers_and_endpoint_patch() {
        let body = json!({
            "model": "gpt-5.6-sol",
            "input": [],
            "stream": true,
            "reasoning": {"effort": "high"}
        });
        let patch = serde_yaml::from_str(
            r#"
responses:
  'gpt-5\.6-sol:high':
    url: https://api.openai.com/v1/responses
    body:
      patched_for_responses: true
    headers:
      x-responses-patch: applied
"#,
        )
        .expect("valid Responses patch");
        let mut model = responses_model(Some("high"));
        model.data_mut().patch = Some(json!({
            "body": {"reasoning_effort": "high"},
            "headers": {"x-chat-model-patch": "must-not-apply"}
        }));
        let client = OpenAIClient {
            global_config: Default::default(),
            config: OpenAIConfig {
                api_key: Some("test-key".into()),
                api_base: Some("https://example.invalid/v1/".into()),
                organization_id: Some("test-organization".into()),
                patch: Some(patch),
                ..Default::default()
            },
            model,
        };

        let raw_request = client.prepare_responses_request(body.clone()).unwrap();

        assert_eq!(raw_request.url, "https://example.invalid/v1/responses");
        assert_eq!(raw_request.body, body);
        let request = prepare_openai_responses_request(&client, body).unwrap();
        assert_eq!(request.url, "https://api.openai.com/v1/responses");
        assert_eq!(request.body["patched_for_responses"], true);
        assert!(request.body.get("reasoning_effort").is_none());
        assert_eq!(
            request.headers.get("x-responses-patch").map(String::as_str),
            Some("applied")
        );
        assert!(!request.headers.contains_key("x-chat-model-patch"));
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
        assert!(uuid::Uuid::parse_str(
            request
                .headers
                .get("x-client-request-id")
                .expect("client request id header")
        )
        .is_ok());
        assert_eq!(
            responses_pricing_context(&request, &client.model),
            OpenAIResponsesPricingContext::PublicApi
        );
    }

    #[test]
    fn client_request_id_is_canonicalized_after_header_patches() {
        let mut request = RequestData::new(
            "https://api.openai.com/v1/responses",
            json!({"stream": true}),
        );
        request.header("x-client-request-id", "generated-id");
        request.header("X-Client-Request-Id", "configured-id");

        canonicalize_openai_client_request_id(&mut request);

        assert_eq!(
            request
                .headers
                .iter()
                .filter(|(name, _)| name.eq_ignore_ascii_case("x-client-request-id"))
                .count(),
            1
        );
        assert_eq!(
            request
                .headers
                .get("x-client-request-id")
                .map(String::as_str),
            Some("configured-id")
        );
    }

    fn responses_sse_body(events: &[Value]) -> String {
        events
            .iter()
            .map(|event| {
                let event_type = event["type"].as_str().unwrap_or("message");
                format!(
                    "event: {event_type}\ndata: {}\n\n",
                    serde_json::to_string(event).expect("serializable SSE fixture")
                )
            })
            .collect::<Vec<_>>()
            .join("")
    }

    #[test]
    fn responses_debug_request_summary_excludes_model_content() {
        let body = json!({
            "model": "gpt-5.6-sol",
            "input": [{
                "role": "user",
                "content": [{"type": "input_text", "text": "prompt-secret"}]
            }],
            "instructions": "instructions-secret",
            "stream": true,
            "store": false,
            "multi_agent": {"enabled": true},
            "tools": [{
                "type": "web_search",
                "filters": {"allowed_domains": ["secret.example"]}
            }],
            "tool_choice": "auto",
            "reasoning": {"effort": "high", "summary": "reasoning-secret"},
            "max_output_tokens": 16000
        });

        let summary = openai_responses_request_log_body(&body);
        let rendered = summary.to_string();

        assert_eq!(summary["model"], "gpt-5.6-sol");
        assert_eq!(summary["input_items"], 1);
        assert_eq!(summary["instructions_present"], true);
        assert_eq!(summary["tool_types"], json!(["web_search"]));
        assert_eq!(summary["tool_choice"], "auto");
        for secret in [
            "prompt-secret",
            "instructions-secret",
            "secret.example",
            "reasoning-secret",
        ] {
            assert!(!rendered.contains(secret));
        }
    }

    #[test]
    fn live_status_coalesces_deltas_and_keeps_output_agent_context() {
        let (progress, mut trace_rx) = OpenAIResponsesProgress::live();
        let mut live_state = OpenAIResponsesLiveStreamState::default();
        observe_openai_responses_stream_event(
            &json!({
                "type": "response.output_item.added",
                "sequence_number": 1,
                "response_id": "resp_live",
                "output_index": 7,
                "item": {
                    "type": "message",
                    "agent": {"agent_name": "/root/researcher"}
                }
            }),
            "response.output_item.added",
            Some("resp_live"),
            Some(1),
            &mut live_state,
            Some(&progress),
        );
        for sequence_number in 2..10_002 {
            observe_openai_responses_stream_event(
                &json!({
                    "type": "response.output_text.delta",
                    "sequence_number": sequence_number,
                    "response_id": "resp_live",
                    "output_index": 7,
                    "delta": "delta-secret"
                }),
                "response.output_text.delta",
                Some("resp_live"),
                Some(sequence_number),
                &mut live_state,
                Some(&progress),
            );
        }

        let snapshot = progress.live_snapshot();
        assert_eq!(snapshot.response_id.as_deref(), Some("resp_live"));
        assert_eq!(
            snapshot.event_type.as_deref(),
            Some("response.output_text.delta")
        );
        assert_eq!(snapshot.output_index, Some(7));
        assert_eq!(snapshot.item_type.as_deref(), Some("message"));
        assert_eq!(snapshot.agent_name.as_deref(), Some("/root/researcher"));
        assert!(trace_rx.try_recv().is_err());
    }

    #[test]
    fn live_progress_formatter_is_bounded_and_reports_heartbeat_state() {
        let now = std::time::Instant::now();
        let snapshot = OpenAIResponsesLiveSnapshot {
            started_at: Some(now - Duration::from_secs(65)),
            last_event_at: Some(now - Duration::from_secs(3)),
            response_id: Some(format!("resp\n\u{001b}[31m{}", "x".repeat(200))),
            event_type: Some("response.reasoning_summary_text.delta\nsecret".to_string()),
            sequence_number: Some(42),
            output_index: Some(3),
            item_type: Some("reasoning".to_string()),
            agent_name: Some("/root/researcher".to_string()),
            action: None,
        };

        let formatted = format_openai_responses_live_progress(&snapshot, now);

        assert!(formatted.contains("Generating 01:05"));
        assert!(formatted.contains("idle=3s"));
        assert!(formatted.contains("seq=42"));
        assert!(formatted.contains("output=3"));
        assert!(!formatted.contains('\n'));
        assert!(!formatted.contains('\u{001b}'));
        assert!(formatted.len() < 360);
    }

    #[tokio::test]
    async fn responses_sse_returns_full_terminal_response() {
        let response = json!({
            "id": "resp_streamed",
            "status": "completed",
            "output": [{"type": "message", "content": []}],
            "usage": {"input_tokens": 3, "output_tokens": 2}
        });
        let body = responses_sse_body(&[
            json!({
                "type": "response.created",
                "sequence_number": 0,
                "response": {"id": "resp_streamed", "status": "in_progress"}
            }),
            json!({
                "type": "response.output_item.added",
                "sequence_number": 1,
                "response_id": "resp_streamed",
                "output_index": 0,
                "item": {"type": "message"}
            }),
            json!({
                "type": "response.completed",
                "sequence_number": 2,
                "response": response.clone()
            }),
        ]);
        let builder = sse_fixture_builder(&body).await.unwrap();

        let actual = send_openai_responses_request(builder, "client-fixture")
            .await
            .unwrap();

        assert_eq!(actual, response);
    }

    #[tokio::test]
    async fn responses_sse_emits_sanitized_live_agent_and_web_search_trace() {
        let response = json!({
            "id": "resp_live",
            "status": "completed",
            "output": [
                {
                    "type": "multi_agent_call",
                    "call_id": "call_spawn",
                    "action": "spawn_agent",
                    "arguments": "{\"task_name\":\"research\",\"message\":\"spawn-secret\"}",
                    "agent": {"agent_name": "/root"}
                },
                {
                    "type": "agent_message",
                    "id": "message_1",
                    "author": "/root/researcher",
                    "recipient": "/root",
                    "content": [{"encrypted_content": "message-secret"}]
                },
                {
                    "type": "agent_message",
                    "id": "message_2",
                    "author": "/root/researcher",
                    "recipient": "/root",
                    "content": [{"encrypted_content": "message-secret-2"}]
                }
            ],
            "usage": {"input_tokens": 3, "output_tokens": 2}
        });
        let body = responses_sse_body(&[
            json!({
                "type": "response.created",
                "sequence_number": 0,
                "response": {"id": "resp_live", "status": "in_progress"}
            }),
            json!({
                "type": "response.output_item.added",
                "sequence_number": 1,
                "response_id": "resp_live",
                "output_index": 0,
                "item": {
                    "type": "multi_agent_call",
                    "id": "call_item_1",
                    "call_id": "call_spawn",
                    "action": "spawn_agent",
                    "agent": {"agent_name": "/root"}
                }
            }),
            json!({
                "type": "response.output_item.done",
                "sequence_number": 2,
                "response_id": "resp_live",
                "output_index": 0,
                "item": {
                    "type": "multi_agent_call",
                    "id": "call_item_1",
                    "call_id": "call_spawn",
                    "arguments": "{\"task_name\":\"research\",\"message\":\"spawn-secret\"}"
                }
            }),
            json!({
                "type": "response.web_search_call.searching",
                "sequence_number": 3,
                "response_id": "resp_live",
                "agent": {"agent_name": "/root/researcher"},
                "output_index": 1,
                "query": "query-secret",
                "sources": [{"url": "https://source-secret.example"}]
            }),
            json!({
                "type": "response.output_text.delta",
                "sequence_number": 4,
                "response_id": "resp_live",
                "output_index": 2,
                "delta": "delta-secret"
            }),
            json!({
                "type": "response.completed",
                "sequence_number": 5,
                "response": response.clone()
            }),
        ]);
        let builder = sse_fixture_builder(&body).await.unwrap();
        let (progress, mut trace_rx) = OpenAIResponsesProgress::live();

        let actual =
            send_openai_responses_request_with_progress(builder, "client-fixture", Some(&progress))
                .await
                .unwrap();
        let mut lines = Vec::new();
        while let Ok(event) = trace_rx.try_recv() {
            lines.push(event.line().to_string());
        }
        let rendered = lines.join("\n");

        assert_eq!(actual, response);
        assert!(rendered.contains("status=created"));
        assert!(rendered.contains("hosted_call actor=/root action=spawn_agent"));
        assert!(rendered.contains("task=research"));
        assert!(rendered.contains("agent_message author=/root/researcher recipient=/root"));
        assert!(rendered.contains("web_search agent=/root/researcher status=searching"));
        assert!(rendered.contains("status=completed"));
        assert_eq!(rendered.matches("hosted_call actor=").count(), 1);
        assert_eq!(rendered.matches("agent_message author=").count(), 2);
        assert!(!rendered.contains("action=unknown"));
        assert!(!rendered.contains("actor=unavailable"));
        for secret in [
            "spawn-secret",
            "message-secret",
            "message-secret-2",
            "query-secret",
            "source-secret",
            "delta-secret",
        ] {
            assert!(!rendered.contains(secret));
        }
        let snapshot = progress.live_snapshot();
        assert_eq!(snapshot.response_id.as_deref(), Some("resp_live"));
        assert_eq!(snapshot.event_type.as_deref(), Some("response.completed"));
    }

    #[tokio::test]
    async fn responses_sse_returns_failed_and_incomplete_envelopes_to_the_parser() {
        for (event_type, status) in [
            ("response.failed", "failed"),
            ("response.incomplete", "incomplete"),
        ] {
            let response = json!({
                "id": format!("resp_{status}"),
                "status": status,
                "output": [],
                "usage": {"input_tokens": 3, "output_tokens": 2},
                "error": (status == "failed").then(|| json!({
                    "code": "server_error",
                    "message": "provider failed"
                })),
                "incomplete_details": (status == "incomplete")
                    .then(|| json!({"reason": "max_output_tokens"}))
            });
            let body = responses_sse_body(&[json!({
                "type": event_type,
                "sequence_number": 3,
                "response": response.clone()
            })]);
            let builder = sse_fixture_builder(&body).await.unwrap();

            let actual = send_openai_responses_request(builder, "client-fixture")
                .await
                .unwrap();

            assert_eq!(actual, response);
            assert!(parse_openai_responses_multi_agent(&actual).is_err());
        }
    }

    #[tokio::test]
    async fn responses_sse_reports_sanitized_top_level_error() {
        let body = responses_sse_body(&[json!({
            "type": "error",
            "sequence_number": 4,
            "response_id": "resp_error",
            "code": "server_error",
            "message": "hosted run failed",
            "internal_details": "must-not-be-rendered"
        })]);
        let builder = sse_fixture_builder(&body).await.unwrap();

        let error = send_openai_responses_request(builder, "client-fixture")
            .await
            .unwrap_err();
        let message = error.to_string();

        assert!(message.contains("hosted run failed"));
        assert!(message.contains("server_error"));
        assert!(message.contains("resp_error"));
        assert!(message.contains("x-client-request-id: client-fixture"));
        assert!(!message.contains("must-not-be-rendered"));
    }

    #[tokio::test]
    async fn responses_sse_http_json_error_keeps_server_and_client_request_ids() {
        let body = json!({
            "error": {
                "type": "overloaded_error",
                "message": "temporary provider failure",
                "internal_details": "must-not-be-rendered"
            }
        })
        .to_string();
        let builder = response_fixture_builder_with_headers(
            "503 Service Unavailable",
            "Application/JSON",
            &[("X-Request-Id", "req-server-json")],
            &body,
        )
        .await
        .unwrap();

        let error = send_openai_responses_request(builder, "client-http-json")
            .await
            .unwrap_err();
        let message = error.to_string();

        assert!(message.contains("temporary provider failure"));
        assert!(message.contains("x-request-id: req-server-json"));
        assert!(message.contains("x-client-request-id: client-http-json"));
        assert!(!message.contains("must-not-be-rendered"));
    }

    #[tokio::test]
    async fn responses_sse_rejects_malformed_and_truncated_protocols() {
        let cases = [
            "data: not-json\n\n".to_string(),
            "data: {\"sequence_number\":1}\n\n".to_string(),
            responses_sse_body(&[json!({
                "type": "response.completed",
                "sequence_number": 2
            })]),
            responses_sse_body(&[json!({
                "type": "response.created",
                "sequence_number": 0,
                "response": {"id": "resp_truncated"}
            })]),
        ];

        for body in cases {
            let builder = sse_fixture_builder(&body).await.unwrap();
            let error = send_openai_responses_request(builder, "client-fixture")
                .await
                .unwrap_err();
            assert!(error.to_string().contains("OpenAI Responses"));
            assert!(error
                .to_string()
                .contains("x-client-request-id: client-fixture"));
        }
    }

    #[tokio::test]
    async fn responses_sse_never_reconnects_the_post_after_stream_open() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;
        use tokio::sync::oneshot;
        use tokio::time::{timeout, Duration};

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let body = responses_sse_body(&[json!({
            "type": "response.created",
            "sequence_number": 0,
            "response": {"id": "resp_no_reconnect"}
        })]);
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let (reconnect_tx, reconnect_rx) = oneshot::channel();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0; 4096];
            let request_len = socket.read(&mut request).await.unwrap();
            assert!(request_len > 0);
            socket.write_all(response.as_bytes()).await.unwrap();
            socket.shutdown().await.unwrap();
            let reconnected = timeout(Duration::from_millis(300), listener.accept())
                .await
                .is_ok();
            reconnect_tx.send(reconnected).unwrap();
        });
        let builder = reqwest::Client::new()
            .post(format!("http://{address}"))
            .json(&json!({"stream": true}));

        let result = timeout(
            Duration::from_secs(1),
            send_openai_responses_request(builder, "client-fixture"),
        )
        .await
        .expect("truncated stream must terminate without reconnecting");

        assert!(result.is_err());
        assert!(!reconnect_rx.await.unwrap());
    }

    #[test]
    fn responses_patch_cannot_disable_required_streaming() {
        let patch = serde_yaml::from_str(
            r#"
responses:
  'gpt-5\.6-sol':
    body:
      stream: false
"#,
        )
        .expect("valid Responses patch");
        let client = OpenAIClient {
            global_config: Default::default(),
            config: OpenAIConfig {
                api_key: Some("test-key".into()),
                patch: Some(patch),
                ..Default::default()
            },
            model: responses_model(None),
        };

        let error = prepare_openai_responses_request(
            &client,
            json!({"model": "gpt-5.6-sol", "input": [], "stream": true}),
        )
        .err()
        .expect("stream=false patch must fail before the request is sent");

        assert!(error.to_string().contains("requires effective stream=true"));
    }

    #[test]
    fn patched_or_malformed_effective_model_disables_selected_model_pricing() {
        let patch = serde_yaml::from_str(
            r#"
responses:
  'gpt-5\.6-sol':
    body:
      model: gpt-5.6-pro
"#,
        )
        .expect("valid Responses patch");
        let client = OpenAIClient {
            global_config: Default::default(),
            config: OpenAIConfig {
                api_key: Some("test-key".into()),
                api_base: Some("https://api.openai.com/v1/".into()),
                patch: Some(patch),
                ..Default::default()
            },
            model: priced_responses_model(),
        };
        let request = prepare_openai_responses_request(
            &client,
            json!({"model": "gpt-5.6-sol", "input": [], "stream": true}),
        )
        .unwrap();

        assert_eq!(request.body["model"], "gpt-5.6-pro");
        let pricing_context = responses_pricing_context(&request, &client.model);
        assert_eq!(pricing_context, OpenAIResponsesPricingContext::UnknownModel);
        let summary = format_openai_responses_usage_cost(
            &client.model,
            &[usage_turn("one", Some("default"), 10, 0, 0, 1, 0)],
            pricing_context,
        );
        assert!(summary.contains("Estimated token subtotal: unavailable"));
        assert!(summary.contains("does not match selected model pricing"));
        assert!(summary.contains("Estimated total cost: unavailable"));

        for body in [json!({}), json!({"model": 42})] {
            let request = RequestData::new("https://api.openai.com/v1/responses", body);
            let pricing_context = responses_pricing_context(&request, &client.model);
            assert_eq!(pricing_context, OpenAIResponsesPricingContext::UnknownModel);
            let summary = format_openai_responses_usage_cost(
                &client.model,
                &[usage_turn("one", Some("default"), 10, 0, 0, 1, 0)],
                pricing_context,
            );
            assert!(summary.contains("Estimated token subtotal: unavailable"));
            assert!(summary.contains("Estimated total cost: unavailable"));
        }
    }

    #[test]
    fn builds_hosted_web_search_with_limits_and_developer_tools() {
        let data = ChatCompletionsData {
            messages: vec![Message::new(
                MessageRole::User,
                MessageContent::Text("research this market".into()),
            )],
            temperature: None,
            top_p: None,
            functions: Some(vec![function_declaration()]),
            stream: false,
            include_usage: true,
        };
        let settings = MultiAgentConfig {
            max_concurrent_subagents: NonZeroUsize::new(3),
            hosted_tools: vec![MultiAgentHostedTool::WebSearch {
                config: HostedWebSearchConfig {
                    search_context_size: WebSearchContextSize::High,
                    external_web_access: Some(true),
                    return_token_budget: WebSearchReturnTokenBudget::Unlimited,
                    filters: Some(WebSearchFilters {
                        allowed_domains: vec!["example.com".into()],
                        blocked_domains: vec!["blocked.example".into()],
                    }),
                },
            }],
            tool_choice: MultiAgentToolChoice::Required,
            max_output_tokens: NonZeroUsize::new(16_000),
            service_tier: OpenAIServiceTier::Default,
            ..Default::default()
        };

        let body = build_openai_responses_multi_agent_body(
            data,
            &responses_model(Some("high")),
            &settings,
        )
        .unwrap();

        assert_eq!(
            body["include"],
            json!([
                "reasoning.encrypted_content",
                "web_search_call.action.sources"
            ])
        );
        assert_eq!(body["tool_choice"], "required");
        assert_eq!(body["max_output_tokens"], 16_000);
        assert_eq!(body["service_tier"], "default");
        assert_eq!(body["stream"], true);
        assert_eq!(body["multi_agent"]["max_concurrent_subagents"], 3);
        assert_eq!(body["reasoning"], json!({"effort": "high"}));
        assert!(body.get("reasoning_effort").is_none());
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(
            body["tools"][1],
            json!({
                "type": "web_search",
                "search_context_size": "high",
                "external_web_access": true,
                "return_token_budget": "unlimited",
                "filters": {
                    "allowed_domains": ["example.com"],
                    "blocked_domains": ["blocked.example"]
                }
            })
        );
    }

    #[test]
    fn exact_cli_and_config_build_sol_high_web_search_request() {
        let config: Config = serde_yaml::from_str(
            r#"
multi_agent:
  hosted_tools:
    - type: web_search
      search_context_size: high
      external_web_access: true
      return_token_budget: default
  tool_choice: required
  max_output_tokens: 16000
  service_tier: default
clients:
  - type: openai
    api_key: test-key
"#,
        )
        .unwrap();
        let config = Arc::new(RwLock::new(config));
        let cli = Cli::try_parse_from([
            "aichat",
            "--show-cost",
            "--multi-agent",
            "-m",
            "openai:gpt-5.6-sol:high",
            "perform siem systems market analysis",
        ])
        .unwrap();

        configure_multi_agent(&config, &cli).unwrap();
        if cli.show_cost {
            config.write().show_cost = true;
        }
        let model = Model::retrieve_model(
            &config.read(),
            cli.model.as_deref().unwrap(),
            ModelType::Chat,
        )
        .unwrap();
        let settings = config.read().multi_agent.clone();
        let body = build_openai_responses_multi_agent_body(
            ChatCompletionsData {
                messages: vec![Message::new(
                    MessageRole::User,
                    MessageContent::Text("perform siem systems market analysis".into()),
                )],
                temperature: None,
                top_p: None,
                functions: None,
                stream: false,
                include_usage: true,
            },
            &model,
            &settings,
        )
        .unwrap();
        let client = init_openai_client(&config, &model).unwrap();
        let request = prepare_openai_responses_request(&client, body).unwrap();

        assert!(config.read().show_cost);
        assert_eq!(model.real_name(), "gpt-5.6-sol");
        assert_eq!(model.reasoning_effort(), Some("high"));
        assert_eq!(request.url, "https://api.openai.com/v1/responses");
        assert_eq!(
            request.headers.get("OpenAI-Beta").map(String::as_str),
            Some("responses_multi_agent=v1")
        );
        assert_eq!(request.body["model"], "gpt-5.6-sol");
        assert_eq!(request.body["reasoning"], json!({"effort": "high"}));
        assert_eq!(request.body["tools"][0]["type"], "web_search");
        assert_eq!(request.body["tools"][0]["search_context_size"], "high");
        assert_eq!(request.body["tool_choice"], "required");
        assert_eq!(request.body["max_output_tokens"], 16_000);
        assert_eq!(request.body["service_tier"], "default");
        assert_eq!(request.body["stream"], true);
        assert_eq!(
            request.body["include"],
            json!([
                "reasoning.encrypted_content",
                "web_search_call.action.sources"
            ])
        );
    }

    #[test]
    fn cli_web_search_shortcut_builds_default_hosted_tool() {
        let config: Config = serde_yaml::from_str(
            r#"
clients:
  - type: openai
    api_key: test-key
"#,
        )
        .unwrap();
        let config = Arc::new(RwLock::new(config));
        let cli = Cli::try_parse_from([
            "aichat",
            "--show-cost",
            "--multi-agent",
            "--web-search",
            "-m",
            "openai:gpt-5.6-sol:high",
            "perform siem systems market analysis",
        ])
        .unwrap();

        configure_multi_agent(&config, &cli).unwrap();
        let model = Model::retrieve_model(
            &config.read(),
            cli.model.as_deref().unwrap(),
            ModelType::Chat,
        )
        .unwrap();
        let body = build_openai_responses_multi_agent_body(
            ChatCompletionsData {
                messages: vec![],
                temperature: None,
                top_p: None,
                functions: None,
                stream: false,
                include_usage: true,
            },
            &model,
            &config.read().multi_agent,
        )
        .unwrap();

        assert_eq!(
            body["tools"][0],
            json!({
                "type": "web_search",
                "search_context_size": "medium",
                "return_token_budget": "default"
            })
        );
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn enabled_config_builds_custom_hosted_tool_without_cli_mode_flags() {
        let config: Config = serde_yaml::from_str(
            r#"
multi_agent:
  enabled: true
  hosted_tools:
    - type: web_search
      search_context_size: low
      external_web_access: false
      return_token_budget: unlimited
  tool_choice: required
  service_tier: flex
clients:
  - type: openai
    api_key: test-key
"#,
        )
        .unwrap();
        let config = Arc::new(RwLock::new(config));
        let cli = Cli::try_parse_from([
            "aichat",
            "-m",
            "openai:gpt-5.6-sol:high",
            "perform siem systems market analysis",
        ])
        .unwrap();

        configure_multi_agent(&config, &cli).unwrap();
        let model = Model::retrieve_model(
            &config.read(),
            cli.model.as_deref().unwrap(),
            ModelType::Chat,
        )
        .unwrap();
        let body = build_openai_responses_multi_agent_body(
            ChatCompletionsData {
                messages: vec![],
                temperature: None,
                top_p: None,
                functions: None,
                stream: false,
                include_usage: true,
            },
            &model,
            &config.read().multi_agent,
        )
        .unwrap();

        assert!(config.read().multi_agent.enabled);
        assert_eq!(
            body["tools"][0],
            json!({
                "type": "web_search",
                "search_context_size": "low",
                "external_web_access": false,
                "return_token_budget": "unlimited"
            })
        );
        assert_eq!(body["tool_choice"], "required");
        assert_eq!(body["service_tier"], "flex");
    }

    #[test]
    fn rejects_invalid_tool_constraints() {
        let input = || ChatCompletionsData {
            messages: vec![],
            temperature: None,
            top_p: None,
            functions: None,
            stream: false,
            include_usage: false,
        };
        let duplicate_web = MultiAgentConfig {
            hosted_tools: vec![
                MultiAgentHostedTool::web_search(),
                MultiAgentHostedTool::web_search(),
            ],
            ..Default::default()
        };
        assert!(build_openai_responses_multi_agent_body(
            input(),
            &responses_model(None),
            &duplicate_web
        )
        .unwrap_err()
        .to_string()
        .contains("at most once"));

        let settings = MultiAgentConfig {
            tool_choice: MultiAgentToolChoice::Required,
            ..Default::default()
        };
        assert!(build_openai_responses_multi_agent_body(
            input(),
            &responses_model(None),
            &settings
        )
        .is_err());
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

    #[tokio::test]
    async fn preserves_completed_turn_progress_after_later_failure() {
        let progress = OpenAIResponsesProgress::default();
        progress.set_pricing_context(OpenAIResponsesPricingContext::PublicApi);
        let send_count = Cell::new(0);

        let error = run_multi_agent_loop_with_progress(
            json!({"input": []}),
            &create_abort_signal(),
            |_| {
                let current = send_count.get();
                send_count.set(current + 1);
                if current == 0 {
                    ready(Ok(completed_response(
                        "resp_completed",
                        vec![function_call("call_1", "lookup", json!({}))],
                        11,
                        3,
                    )))
                } else {
                    ready(Err(anyhow::anyhow!("second turn failed")))
                }
            },
            |calls| Ok(vec![json!("DONE"); calls.len()]),
            Some(&progress),
        )
        .await
        .unwrap_err();

        assert_eq!(error.to_string(), "second turn failed");
        let (turns, pricing_context) = progress.snapshot();
        assert_eq!(pricing_context, OpenAIResponsesPricingContext::PublicApi);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].response_id, "resp_completed");
        assert_eq!(turns[0].usage.input_tokens, Some(11));
        assert_eq!(turns[0].usage.output_tokens, Some(3));
    }

    #[tokio::test]
    async fn preserves_incomplete_response_usage_and_web_search_cost() {
        let progress = OpenAIResponsesProgress::default();
        progress.set_pricing_context(OpenAIResponsesPricingContext::PublicApi);
        let response = json!({
            "id": "resp_incomplete",
            "status": "incomplete",
            "service_tier": "default",
            "output": [{
                "type": "web_search_call",
                "action": {
                    "type": "search",
                    "query": "billable query",
                    "sources": [{"type": "url", "url": "https://example.com"}]
                }
            }],
            "usage": {
                "input_tokens": 10,
                "input_tokens_details": {
                    "cached_tokens": 2,
                    "cache_write_tokens": 1
                },
                "output_tokens": 4,
                "output_tokens_details": {"reasoning_tokens": 3}
            },
            "error": null,
            "incomplete_details": {"reason": "max_output_tokens"}
        });

        let error = run_multi_agent_loop_with_progress(
            json!({"input": []}),
            &create_abort_signal(),
            |_| ready(Ok(response.clone())),
            |_| unreachable!("incomplete responses do not execute developer tools"),
            Some(&progress),
        )
        .await
        .unwrap_err();

        assert!(error.to_string().contains("was incomplete"));
        let (turns, pricing_context) = progress.snapshot();
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].usage.input_tokens, Some(10));
        assert_eq!(billable_web_search_calls(&turns), 1);
        let summary =
            format_openai_responses_usage_cost(&priced_responses_model(), &turns, pricing_context);
        assert!(summary.contains("Hosted web searches: 1 | Fee: $0.010000"));
        assert!(summary.contains("Estimated total cost: $0.010"));
    }
}
