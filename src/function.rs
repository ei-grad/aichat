use crate::{
    config::{Agent, Config, GlobalConfig},
    utils::*,
};

use anyhow::{anyhow, bail, Context, Result};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

#[cfg(windows)]
const PATH_SEP: &str = ";";
#[cfg(not(windows))]
const PATH_SEP: &str = ":";

pub fn eval_tool_calls(config: &GlobalConfig, calls: Vec<ToolCall>) -> Result<Vec<ToolResult>> {
    eval_tool_calls_with(calls, |call| call.eval(config))
}

fn eval_tool_calls_with<F>(mut calls: Vec<ToolCall>, mut eval: F) -> Result<Vec<ToolResult>>
where
    F: FnMut(&ToolCall) -> Result<Value>,
{
    let mut output = vec![];
    if calls.is_empty() {
        return Ok(output);
    }
    calls = ToolCall::dedup(calls);
    if calls.is_empty() {
        bail!("The request was aborted because an infinite loop of function calls was detected.")
    }
    let mut is_all_null = true;
    for call in calls {
        let mut result = match eval(&call) {
            Ok(result) => result,
            Err(_) => json!({
                "error": {
                    "type": "tool_execution_error",
                    "message": "The tool call failed. Fix its arguments or choose another tool."
                }
            }),
        };
        if result.is_null() {
            result = json!("DONE");
        } else {
            is_all_null = false;
        }
        output.push(ToolResult::new(call, result));
    }
    if is_all_null {
        output = vec![];
    }
    Ok(output)
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolResult {
    pub call: ToolCall,
    pub output: Value,
}

impl ToolResult {
    pub fn new(call: ToolCall, output: Value) -> Self {
        Self { call, output }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Functions {
    declarations: Vec<FunctionDeclaration>,
}

impl Functions {
    pub fn init(declarations_path: &Path) -> Result<Self> {
        let declarations: Vec<FunctionDeclaration> = if declarations_path.exists() {
            let ctx = || {
                format!(
                    "Failed to load functions at {}",
                    declarations_path.display()
                )
            };
            let content = fs::read_to_string(declarations_path).with_context(ctx)?;
            serde_json::from_str(&content).with_context(ctx)?
        } else {
            vec![]
        };

        Ok(Self { declarations })
    }

    pub fn find(&self, name: &str) -> Option<&FunctionDeclaration> {
        self.declarations.iter().find(|v| v.name == name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.declarations.iter().any(|v| v.name == name)
    }

    pub fn declarations(&self) -> &[FunctionDeclaration] {
        &self.declarations
    }

    pub fn is_empty(&self) -> bool {
        self.declarations.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDeclaration {
    pub name: String,
    pub description: String,
    pub parameters: JsonSchema,
    #[serde(skip_serializing, default)]
    pub agent: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchema {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub type_value: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<IndexMap<String, JsonSchema>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<JsonSchema>>,
    #[serde(rename = "anyOf", skip_serializing_if = "Option::is_none")]
    pub any_of: Option<Vec<JsonSchema>>,
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub enum_value: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}

impl JsonSchema {
    pub fn is_empty_properties(&self) -> bool {
        match &self.properties {
            Some(v) => v.is_empty(),
            None => true,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
    pub id: Option<String>,
}

type CallConfig = (String, String, Vec<String>, HashMap<String, String>);

impl ToolCall {
    pub fn dedup(calls: Vec<Self>) -> Vec<Self> {
        let mut new_calls = vec![];
        let mut seen_ids = HashSet::new();

        for call in calls.into_iter().rev() {
            if let Some(id) = &call.id {
                if !seen_ids.contains(id) {
                    seen_ids.insert(id.clone());
                    new_calls.push(call);
                }
            } else {
                new_calls.push(call);
            }
        }

        new_calls.reverse();
        new_calls
    }

    pub fn new(name: String, arguments: Value, id: Option<String>) -> Self {
        Self {
            name,
            arguments,
            id,
        }
    }

    pub fn eval(&self, config: &GlobalConfig) -> Result<Value> {
        let (call_name, cmd_name, mut cmd_args, envs) = match &config.read().agent {
            Some(agent) => self.extract_call_config_from_agent(config, agent)?,
            None => self.extract_call_config_from_config(config)?,
        };

        let json_data = if self.arguments.is_object() {
            self.arguments.clone()
        } else if let Some(arguments) = self.arguments.as_str() {
            let arguments: Value = serde_json::from_str(arguments).map_err(|_| {
                anyhow!("The call '{call_name}' has invalid arguments: {arguments}")
            })?;
            arguments
        } else {
            bail!(
                "The call '{call_name}' has invalid arguments: {}",
                self.arguments
            );
        };

        cmd_args.push(json_data.to_string());

        let output = match run_llm_function(cmd_name, cmd_args, envs)? {
            Some(contents) => serde_json::from_str(&contents)
                .ok()
                .unwrap_or_else(|| json!({"output": contents})),
            None => Value::Null,
        };

        Ok(output)
    }

    fn extract_call_config_from_agent(
        &self,
        config: &GlobalConfig,
        agent: &Agent,
    ) -> Result<CallConfig> {
        let function_name = self.name.clone();
        match agent.functions().find(&function_name) {
            Some(function) => {
                let agent_name = agent.name().to_string();
                if function.agent {
                    Ok((
                        format!("{agent_name}-{function_name}"),
                        agent_name,
                        vec![function_name],
                        agent.variable_envs(),
                    ))
                } else {
                    Ok((
                        function_name.clone(),
                        function_name,
                        vec![],
                        Default::default(),
                    ))
                }
            }
            None => self.extract_call_config_from_config(config),
        }
    }

    fn extract_call_config_from_config(&self, config: &GlobalConfig) -> Result<CallConfig> {
        let function_name = self.name.clone();
        match config.read().functions.contains(&function_name) {
            true => Ok((
                function_name.clone(),
                function_name,
                vec![],
                Default::default(),
            )),
            false => bail!("Unexpected call: {function_name} {}", self.arguments),
        }
    }
}

pub fn run_llm_function(
    cmd_name: String,
    cmd_args: Vec<String>,
    mut envs: HashMap<String, String>,
) -> Result<Option<String>> {
    let prompt = format!("Call {cmd_name} {}", cmd_args.join(" "));

    let mut bin_dirs: Vec<PathBuf> = vec![];
    if cmd_args.len() > 1 {
        let dir = Config::agent_functions_dir(&cmd_name).join("bin");
        if dir.exists() {
            bin_dirs.push(dir);
        }
    }
    bin_dirs.push(Config::functions_bin_dir());
    let current_path = std::env::var("PATH").context("No PATH environment variable")?;
    let prepend_path = bin_dirs
        .iter()
        .map(|v| format!("{}{PATH_SEP}", v.display()))
        .collect::<Vec<_>>()
        .join("");
    envs.insert("PATH".into(), format!("{prepend_path}{current_path}"));

    let temp_file = temp_file("-eval-", "");
    envs.insert("LLM_OUTPUT".into(), temp_file.display().to_string());

    #[cfg(windows)]
    let cmd_name = polyfill_cmd_name(&cmd_name, &bin_dirs);
    if *IS_STDOUT_TERMINAL {
        println!("{}", dimmed_text(&prompt));
    }
    let exit_code = run_command(&cmd_name, &cmd_args, Some(envs))
        .map_err(|err| anyhow!("Unable to run {cmd_name}, {err}"))?;
    if exit_code != 0 {
        bail!("Tool call exit with {exit_code}");
    }
    let mut output = None;
    if temp_file.exists() {
        let contents =
            fs::read_to_string(temp_file).context("Failed to retrieve tool call output")?;
        if !contents.is_empty() {
            output = Some(contents);
        }
    };
    Ok(output)
}

#[cfg(windows)]
fn polyfill_cmd_name<T: AsRef<Path>>(cmd_name: &str, bin_dir: &[T]) -> String {
    let cmd_name = cmd_name.to_string();
    if let Ok(exts) = std::env::var("PATHEXT") {
        for name in exts.split(';').map(|ext| format!("{cmd_name}{ext}")) {
            for dir in bin_dir {
                let path = dir.as_ref().join(&name);
                if path.exists() {
                    return name.to_string();
                }
            }
        }
    }
    cmd_name
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::RwLock;
    use std::sync::Arc;

    fn declaration(name: &str) -> FunctionDeclaration {
        serde_json::from_value(json!({
            "name": name,
            "description": "test function",
            "parameters": { "type": "object" }
        }))
        .unwrap()
    }

    fn config_with_functions(names: &[&str]) -> GlobalConfig {
        let config = Config {
            functions: Functions {
                declarations: names.iter().map(|name| declaration(name)).collect(),
            },
            ..Default::default()
        };
        Arc::new(RwLock::new(config))
    }

    fn assert_tool_error(result: &ToolResult, id: &str) {
        assert_eq!(result.call.id.as_deref(), Some(id));
        assert_eq!(
            result.output,
            json!({
                "error": {
                    "type": "tool_execution_error",
                    "message": "The tool call failed. Fix its arguments or choose another tool."
                }
            })
        );
    }

    #[test]
    fn mixed_tool_batch_preserves_order_ids_success_and_null_results() {
        let calls = vec![
            ToolCall::new("success".into(), json!({}), Some("call-1".into())),
            ToolCall::new("null".into(), json!({}), Some("call-2".into())),
            ToolCall::new("failure".into(), json!({}), Some("call-3".into())),
        ];

        let results = eval_tool_calls_with(calls, |call| match call.name.as_str() {
            "success" => Ok(json!({"ok": true})),
            "null" => Ok(Value::Null),
            _ => bail!("private execution details"),
        })
        .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].call.id.as_deref(), Some("call-1"));
        assert_eq!(results[0].output, json!({"ok": true}));
        assert_eq!(results[1].call.id.as_deref(), Some("call-2"));
        assert_eq!(results[1].output, json!("DONE"));
        assert_tool_error(&results[2], "call-3");
        assert!(!results[2].output.to_string().contains("private"));
    }

    #[test]
    fn all_null_tool_batch_keeps_existing_empty_result_semantics() {
        let calls = vec![
            ToolCall::new("null".into(), json!({}), Some("call-1".into())),
            ToolCall::new("null".into(), json!({}), Some("call-2".into())),
        ];

        assert!(eval_tool_calls_with(calls, |_| Ok(Value::Null))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn unknown_tool_and_invalid_arguments_become_tool_results() {
        let unknown = eval_tool_calls(
            &config_with_functions(&[]),
            vec![ToolCall::new(
                "unknown-tool".into(),
                json!({"value": "private-unknown"}),
                Some("unknown-id".into()),
            )],
        )
        .unwrap();
        assert_tool_error(&unknown[0], "unknown-id");
        assert!(!unknown[0].output.to_string().contains("private-unknown"));

        let invalid = eval_tool_calls(
            &config_with_functions(&["invalid-arguments"]),
            vec![ToolCall::new(
                "invalid-arguments".into(),
                json!("not JSON: private-invalid"),
                Some("invalid-id".into()),
            )],
        )
        .unwrap();
        assert_tool_error(&invalid[0], "invalid-id");
        assert!(!invalid[0].output.to_string().contains("private-invalid"));
    }

    #[test]
    fn missing_executable_and_nonzero_exit_become_ordered_tool_results() {
        let missing = "aichat-test-command-that-does-not-exist-7d7da216";
        let nonzero = nonzero_command();
        let config = config_with_functions(&[missing, nonzero]);
        let results = eval_tool_calls(
            &config,
            vec![
                ToolCall::new(
                    missing.into(),
                    json!({"value": "private-missing"}),
                    Some("missing-id".into()),
                ),
                ToolCall::new(
                    nonzero.into(),
                    json!({"value": "private-nonzero"}),
                    Some("nonzero-id".into()),
                ),
            ],
        )
        .unwrap();

        assert_eq!(results.len(), 2);
        assert_tool_error(&results[0], "missing-id");
        assert_tool_error(&results[1], "nonzero-id");
        assert!(!results[0].output.to_string().contains("private-missing"));
        assert!(!results[1].output.to_string().contains("private-nonzero"));
    }

    #[cfg(not(windows))]
    fn nonzero_command() -> &'static str {
        "false"
    }

    #[cfg(windows)]
    fn nonzero_command() -> &'static str {
        "where.exe"
    }
}
