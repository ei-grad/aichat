use anyhow::{Context, Result};
use clap::Parser;
use is_terminal::IsTerminal;
use std::{
    io::{stdin, Read},
    num::NonZeroUsize,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Select a LLM model
    #[clap(short, long)]
    pub model: Option<String>,
    /// Use the system prompt
    #[clap(long)]
    pub prompt: Option<String>,
    /// Select a role
    #[clap(short, long)]
    pub role: Option<String>,
    /// Start or join a session
    #[clap(short = 's', long)]
    pub session: Option<Option<String>>,
    /// Ensure the session is empty
    #[clap(long)]
    pub empty_session: bool,
    /// Ensure the new conversation is saved to the session
    #[clap(long)]
    pub save_session: bool,
    /// Start a agent
    #[clap(short = 'a', long)]
    pub agent: Option<String>,
    /// Set agent variables
    #[clap(long, value_names = ["NAME", "VALUE"], num_args = 2)]
    pub agent_variable: Vec<String>,
    /// Enable Responses API multi-agent mode
    #[clap(long)]
    pub multi_agent: bool,
    /// Limit the number of concurrent hosted subagents
    #[clap(long, value_name = "N")]
    pub max_concurrent_subagents: Option<NonZeroUsize>,
    /// Start a RAG
    #[clap(long)]
    pub rag: Option<String>,
    /// Rebuild the RAG to sync document changes
    #[clap(long)]
    pub rebuild_rag: bool,
    /// Execute a macro
    #[clap(long = "macro", value_name = "MACRO")]
    pub macro_name: Option<String>,
    /// Serve the LLM API and WebAPP
    #[clap(long, value_name = "ADDRESS")]
    pub serve: Option<Option<String>>,
    /// Execute commands in natural language
    #[clap(short = 'e', long)]
    pub execute: bool,
    /// Output code only
    #[clap(short = 'c', long)]
    pub code: bool,
    /// Include files, directories, or URLs
    #[clap(short = 'f', long, value_name = "FILE")]
    pub file: Vec<String>,
    /// Include multiple shell-expanded files; terminate the list with `--`
    #[clap(
        long = "files",
        value_name = "FILE",
        num_args = 1..,
        value_terminator = "--"
    )]
    expanded_files: Vec<String>,
    /// Turn off stream mode
    #[clap(short = 'S', long)]
    pub no_stream: bool,
    /// Display token usage and estimated cost after the response
    #[clap(long)]
    pub show_cost: bool,
    /// Display the message without sending it
    #[clap(long)]
    pub dry_run: bool,
    /// Display information
    #[clap(long)]
    pub info: bool,
    /// Sync models updates
    #[clap(long)]
    pub sync_models: bool,
    /// List all available chat models
    #[clap(long)]
    pub list_models: bool,
    /// List all roles
    #[clap(long)]
    pub list_roles: bool,
    /// List all sessions
    #[clap(long)]
    pub list_sessions: bool,
    /// List all agents
    #[clap(long)]
    pub list_agents: bool,
    /// List all RAGs
    #[clap(long)]
    pub list_rags: bool,
    /// List all macros
    #[clap(long)]
    pub list_macros: bool,
    /// Input text
    #[clap(trailing_var_arg = true)]
    text: Vec<String>,
}

impl Cli {
    pub fn files(&self) -> Vec<String> {
        self.file
            .iter()
            .chain(&self.expanded_files)
            .cloned()
            .collect()
    }

    pub fn has_files(&self) -> bool {
        !self.file.is_empty() || !self.expanded_files.is_empty()
    }

    pub fn text(&self) -> Result<Option<String>> {
        let mut stdin_text = String::new();
        if !stdin().is_terminal() {
            let _ = stdin()
                .read_to_string(&mut stdin_text)
                .context("Invalid stdin pipe")?;
        };
        Ok(self.text_with_stdin(&stdin_text))
    }

    fn text_with_stdin(&self, stdin_text: &str) -> Option<String> {
        match self.text.is_empty() {
            true => {
                if stdin_text.is_empty() {
                    None
                } else {
                    Some(stdin_text.to_string())
                }
            }
            false => {
                if self.macro_name.is_some() {
                    let text = self
                        .text
                        .iter()
                        .map(|v| shell_words::quote(v))
                        .collect::<Vec<_>>()
                        .join(" ");
                    if stdin_text.is_empty() {
                        Some(text)
                    } else {
                        Some(format!("{text} -- {stdin_text}"))
                    }
                } else {
                    let text = self.text.join(" ");
                    if stdin_text.is_empty() {
                        Some(text)
                    } else {
                        Some(format!("{text}\n{stdin_text}"))
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(args).unwrap()
    }

    #[test]
    fn preserves_single_and_repeated_file_options() {
        let cli = parse(&["aichat", "-f", "one.md", "explain"]);
        assert_eq!(cli.files(), ["one.md"]);
        assert_eq!(cli.text_with_stdin(""), Some("explain".into()));

        let cli = parse(&["aichat", "-f", "one.md", "--file", "two.md", "compare"]);
        assert_eq!(cli.files(), ["one.md", "two.md"]);
        assert_eq!(cli.text_with_stdin(""), Some("compare".into()));
    }

    #[test]
    fn parses_explicit_shell_expanded_file_list() {
        let cli = parse(&[
            "aichat", "--files", "src/a.rs", "src/b.rs", "--", "review", "these",
        ]);
        assert_eq!(cli.files(), ["src/a.rs", "src/b.rs"]);
        assert_eq!(cli.text_with_stdin(""), Some("review these".into()));
    }

    #[test]
    fn preserves_literal_file_values_without_inference() {
        let cli = parse(&[
            "aichat",
            "--files",
            "src/*.rs",
            "docs/",
            "https://example.com/context",
            "%%",
            "file with spaces.md",
            "--",
            "summarize",
        ]);
        assert_eq!(
            cli.files(),
            [
                "src/*.rs",
                "docs/",
                "https://example.com/context",
                "%%",
                "file with spaces.md",
            ]
        );
        assert_eq!(cli.text_with_stdin(""), Some("summarize".into()));
    }

    #[test]
    fn does_not_reclassify_prompt_tokens_as_files() {
        let cli = parse(&["aichat", "-f", "context.md", "README.md", "explain"]);
        assert_eq!(cli.files(), ["context.md"]);
        assert_eq!(cli.text_with_stdin(""), Some("README.md explain".into()));
    }

    #[test]
    fn combines_explicit_files_with_stdin_and_command_modes() {
        let cli = parse(&[
            "aichat",
            "--execute",
            "--files",
            "script one.sh",
            "script-two.sh",
            "--",
            "inspect",
        ]);
        assert!(cli.execute);
        assert!(cli.has_files());
        assert_eq!(cli.text_with_stdin("piped"), Some("inspect\npiped".into()));

        let cli = parse(&["aichat", "--code", "--files", "src/a.rs", "--", "rewrite"]);
        assert!(cli.code);
        assert_eq!(cli.files(), ["src/a.rs"]);
    }

    #[test]
    fn accepts_stdin_without_prompt_text() {
        let cli = parse(&["aichat"]);
        assert_eq!(cli.text_with_stdin("from stdin"), Some("from stdin".into()));
        assert_eq!(cli.text_with_stdin(""), None);
    }

    #[test]
    fn parses_show_cost_flag() {
        let cli = parse(&["aichat", "--show-cost", "hello"]);
        assert!(cli.show_cost);
        assert_eq!(cli.text_with_stdin(""), Some("hello".into()));
    }

    #[test]
    fn parses_multi_agent_controls() {
        let cli = parse(&[
            "aichat",
            "--multi-agent",
            "--max-concurrent-subagents",
            "4",
            "delegate",
        ]);
        assert!(cli.multi_agent);
        assert_eq!(cli.max_concurrent_subagents.map(NonZeroUsize::get), Some(4));
        assert_eq!(cli.text_with_stdin(""), Some("delegate".into()));

        let cli = parse(&["aichat", "--max-concurrent-subagents", "2"]);
        assert!(!cli.multi_agent);
        assert_eq!(cli.max_concurrent_subagents.map(NonZeroUsize::get), Some(2));

        let cli = parse(&["aichat", "--multi-agent"]);
        assert!(cli.multi_agent);
        assert_eq!(cli.max_concurrent_subagents, None);
    }

    #[test]
    fn rejects_zero_max_concurrent_subagents() {
        let err = Cli::try_parse_from(["aichat", "--max-concurrent-subagents", "0"])
            .expect_err("zero must be rejected during argument parsing");
        assert_eq!(err.kind(), clap::error::ErrorKind::ValueValidation);
    }
}
