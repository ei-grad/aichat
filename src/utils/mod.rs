mod abort_signal;
mod clipboard;
mod command;
mod crypto;
mod html_to_md;
mod input;
mod loader;
mod path;
mod render_prompt;
mod request;
mod spinner;
mod variables;

pub use self::abort_signal::*;
pub use self::clipboard::set_text;
pub use self::command::*;
pub use self::crypto::*;
pub use self::html_to_md::*;
pub use self::input::*;
pub use self::loader::*;
pub use self::path::*;
pub use self::render_prompt::render_prompt;
pub use self::request::*;
pub use self::spinner::*;
pub use self::variables::*;

use anyhow::{Context, Result};
use fancy_regex::Regex;
use fuzzy_matcher::{skim::SkimMatcherV2, FuzzyMatcher};
use is_terminal::IsTerminal;
use std::borrow::Cow;
use std::sync::LazyLock;
use std::{env, path::PathBuf, process};
use unicode_segmentation::UnicodeSegmentation;

pub static CODE_BLOCK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?ms)```\w*(.*)```").unwrap());
pub static THINK_TAG_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)^\s*<think>.*?</think>(\s*|$)").unwrap());
pub static IS_STDOUT_TERMINAL: LazyLock<bool> = LazyLock::new(|| std::io::stdout().is_terminal());
pub static NO_COLOR: LazyLock<bool> = LazyLock::new(|| {
    env::var("NO_COLOR")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or_default()
        || !*IS_STDOUT_TERMINAL
});

pub fn now() -> String {
    chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, false)
}

pub fn now_timestamp() -> i64 {
    chrono::Local::now().timestamp()
}

pub fn get_env_name(key: &str) -> String {
    format!("{}_{key}", env!("CARGO_CRATE_NAME"),).to_ascii_uppercase()
}

pub fn normalize_env_name(value: &str) -> String {
    value.replace('-', "_").to_ascii_uppercase()
}

pub fn parse_bool(value: &str) -> Option<bool> {
    match value {
        "1" | "true" => Some(true),
        "0" | "false" => Some(false),
        _ => None,
    }
}

pub fn estimate_token_length(text: &str) -> usize {
    let words: Vec<&str> = text.unicode_words().collect();
    let mut output: f32 = 0.0;
    for word in words {
        if word.is_ascii() {
            output += 1.3;
        } else {
            let count = word.chars().count();
            if count == 1 {
                output += 1.0
            } else {
                output += (count as f32) * 0.5;
            }
        }
    }
    output.ceil() as usize
}

pub fn strip_think_tag(text: &str) -> Cow<'_, str> {
    let stripped = THINK_TAG_RE.replace_all(text, "");
    if stripped.len() != text.len() {
        return stripped;
    }
    strip_close_only_think_block(text)
        .map(Cow::Borrowed)
        .unwrap_or(stripped)
}

fn strip_close_only_think_block(text: &str) -> Option<&str> {
    const OPEN_TAG: &str = "<think>";
    const CLOSE_TAG: &str = "</think>";

    // Without provider metadata, only a single standalone closing-tag line is
    // treated as structural. Inline, repeated, and fenced examples stay literal.
    if text.contains(OPEN_TAG) || text.match_indices(CLOSE_TAG).count() != 1 {
        return None;
    }

    let close_start = text.find(CLOSE_TAG)?;
    let close_line_start = text[..close_start].rfind('\n')? + 1;
    if !text[close_line_start..close_start].trim().is_empty() {
        return None;
    }

    let reasoning = &text[..close_line_start];
    if reasoning.trim().is_empty() || has_unclosed_markdown_fence(reasoning) {
        return None;
    }

    let after_close = &text[close_start + CLOSE_TAG.len()..];
    match after_close.find('\n') {
        Some(line_end) if after_close[..line_end].trim().is_empty() => {
            Some(after_close[line_end + 1..].trim_start())
        }
        None if after_close.trim().is_empty() => Some(""),
        _ => None,
    }
}

fn has_unclosed_markdown_fence(text: &str) -> bool {
    ["```", "~~~"].into_iter().any(|fence| {
        text.lines()
            .filter(|line| line.trim_start().starts_with(fence))
            .count()
            % 2
            == 1
    })
}

pub fn extract_code_block(text: &str) -> &str {
    CODE_BLOCK_RE
        .captures(text)
        .ok()
        .and_then(|v| v?.get(1).map(|v| v.as_str().trim()))
        .unwrap_or(text)
}

pub fn convert_option_string(value: &str) -> Option<String> {
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

pub fn fuzzy_filter<T, F>(values: Vec<T>, get: F, pattern: &str) -> Vec<T>
where
    F: Fn(&T) -> &str,
{
    let matcher = SkimMatcherV2::default();
    let mut list: Vec<(T, i64)> = values
        .into_iter()
        .filter_map(|v| {
            let score = matcher.fuzzy_match(get(&v), pattern)?;
            Some((v, score))
        })
        .collect();
    list.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    list.into_iter().map(|(v, _)| v).collect()
}

pub fn pretty_error(err: &anyhow::Error) -> String {
    let mut output = vec![];
    output.push(format!("Error: {err}"));
    let causes: Vec<_> = err.chain().skip(1).collect();
    let causes_len = causes.len();
    if causes_len > 0 {
        output.push("\nCaused by:".to_string());
        if causes_len == 1 {
            output.push(format!("    {}", indent_text(causes[0], 4).trim()));
        } else {
            for (i, cause) in causes.into_iter().enumerate() {
                output.push(format!("{i:5}: {}", indent_text(cause, 7).trim()));
            }
        }
    }
    output.join("\n")
}

pub fn indent_text<T: ToString>(s: T, size: usize) -> String {
    let indent_str = " ".repeat(size);
    s.to_string()
        .split('\n')
        .map(|line| format!("{indent_str}{line}"))
        .collect::<Vec<String>>()
        .join("\n")
}

pub fn error_text(input: &str) -> String {
    color_text(input, nu_ansi_term::Color::Red)
}

pub fn warning_text(input: &str) -> String {
    color_text(input, nu_ansi_term::Color::Yellow)
}

pub fn color_text(input: &str, color: nu_ansi_term::Color) -> String {
    if *NO_COLOR {
        return input.to_string();
    }
    nu_ansi_term::Style::new()
        .fg(color)
        .paint(input)
        .to_string()
}

pub fn dimmed_text(input: &str) -> String {
    if *NO_COLOR {
        return input.to_string();
    }
    nu_ansi_term::Style::new().dimmed().paint(input).to_string()
}

pub fn multiline_text(input: &str) -> String {
    input
        .split('\n')
        .enumerate()
        .map(|(i, v)| {
            if i == 0 {
                v.to_string()
            } else {
                format!(".. {v}")
            }
        })
        .collect::<Vec<String>>()
        .join("\n")
}

pub fn temp_file(prefix: &str, suffix: &str) -> PathBuf {
    env::temp_dir().join(format!(
        "{}-{}{prefix}{}{suffix}",
        env!("CARGO_CRATE_NAME").to_lowercase(),
        process::id(),
        uuid::Uuid::new_v4()
    ))
}

pub fn is_url(path: &str) -> bool {
    path.starts_with("http://") || path.starts_with("https://")
}

pub fn set_proxy(
    mut builder: reqwest::ClientBuilder,
    proxy: &str,
) -> Result<reqwest::ClientBuilder> {
    builder = builder.no_proxy();
    if !proxy.is_empty() && proxy != "-" {
        builder = builder
            .proxy(reqwest::Proxy::all(proxy).with_context(|| format!("Invalid proxy `{proxy}`"))?);
    };
    Ok(builder)
}

pub fn decode_bin<T: serde::de::DeserializeOwned>(data: &[u8]) -> Result<T> {
    let (v, _) = bincode::serde::decode_from_slice(data, bincode::config::legacy())?;
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(target_os = "windows"))]
    fn test_safe_join_path() {
        assert_eq!(
            safe_join_path("/home/user/dir1", "files/file1"),
            Some(PathBuf::from("/home/user/dir1/files/file1"))
        );
        assert!(safe_join_path("/home/user/dir1", "/files/file1").is_none());
        assert!(safe_join_path("/home/user/dir1", "../file1").is_none());
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn test_safe_join_path() {
        assert_eq!(
            safe_join_path("C:\\Users\\user\\dir1", "files/file1"),
            Some(PathBuf::from("C:\\Users\\user\\dir1\\files\\file1"))
        );
        assert!(safe_join_path("C:\\Users\\user\\dir1", "/files/file1").is_none());
        assert!(safe_join_path("C:\\Users\\user\\dir1", "../file1").is_none());
    }

    #[test]
    fn strip_think_tag_removes_full_tagged_block() {
        let text = "  <think>\r\nreasoning\r\n</think>\r\n\r\nanswer";
        assert_eq!(strip_think_tag(text), "answer");
    }

    #[test]
    fn strip_think_tag_removes_structural_close_only_block() {
        let text = "reasoning without an opening tag\n</think>\n\nanswer";
        assert_eq!(strip_think_tag(text), "answer");

        let text = "\n  reasoning with whitespace\r\n  </think>  \r\n\t answer ";
        assert_eq!(strip_think_tag(text), "answer ");

        let text = "reasoning only\n</think>";
        assert_eq!(strip_think_tag(text), "");
    }

    #[test]
    fn strip_think_tag_preserves_plain_and_literal_close_tags() {
        for text in [
            "plain answer",
            "Use </think> literally in documentation.",
            "</think>\nanswer without a reasoning prefix",
            "reasoning</think>\nanswer without a standalone marker",
            "reasoning\n</think> trailing text\nanswer",
            "first\n</think>\nanswer with another </think> literal",
            "prefix with an unmatched <think> marker\n</think>\nanswer",
        ] {
            assert_eq!(strip_think_tag(text), text);
        }
    }

    #[test]
    fn strip_think_tag_preserves_close_tag_in_fenced_code() {
        for text in [
            "```html\n</think>\n```\n",
            "~~~text\n</think>\n~~~\n",
            "prose before an open fence\n```\n</think>\n",
        ] {
            assert_eq!(strip_think_tag(text), text);
        }
    }
}
