use super::{poll_abort_signal, wait_abort_signal, AbortSignal, IS_STDOUT_TERMINAL};

use anyhow::{bail, Result};
use crossterm::{cursor, queue, style, terminal};
use std::{
    future::Future,
    io::{stderr, stdout, Write},
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};
use tokio::{
    sync::{
        mpsc::{self, UnboundedReceiver},
        oneshot,
    },
    time::interval,
};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

static SPINNERS_ENABLED: AtomicBool = AtomicBool::new(true);

pub fn set_spinners_enabled(enabled: bool) {
    SPINNERS_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn spinners_enabled() -> bool {
    SPINNERS_ENABLED.load(Ordering::Relaxed)
}

#[derive(Debug, Default)]
pub struct SpinnerInner {
    index: usize,
    message: String,
}

impl SpinnerInner {
    const DATA: [&'static str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

    fn step(&mut self) -> Result<()> {
        if !spinners_enabled() || !*IS_STDOUT_TERMINAL || self.message.is_empty() {
            return Ok(());
        }
        let mut writer = stdout();
        let frame = Self::DATA[self.index % Self::DATA.len()];
        let line = match terminal::size() {
            Ok((columns, _)) => format_spinner_line(frame, &self.message, columns),
            Err(_) => String::new(),
        };
        queue!(
            writer,
            cursor::MoveToColumn(0),
            terminal::Clear(terminal::ClearType::CurrentLine)
        )?;
        if !line.is_empty() {
            queue!(writer, style::Print(line), cursor::Hide)?;
        } else {
            queue!(writer, cursor::Show)?;
        }
        writer.flush()?;
        self.index += 1;
        Ok(())
    }

    fn set_message(&mut self, message: String) -> Result<()> {
        self.clear_message()?;
        if !message.is_empty() {
            self.message = format!(" {message}");
        }
        Ok(())
    }

    fn clear_message(&mut self) -> Result<()> {
        if !*IS_STDOUT_TERMINAL || self.message.is_empty() {
            return Ok(());
        }
        self.message.clear();
        let mut writer = stdout();
        queue!(
            writer,
            cursor::MoveToColumn(0),
            terminal::Clear(terminal::ClearType::CurrentLine),
            cursor::Show
        )?;
        writer.flush()?;
        Ok(())
    }

    fn print_line(&mut self, line: String) -> Result<()> {
        if *IS_STDOUT_TERMINAL && !self.message.is_empty() {
            let mut writer = stdout();
            queue!(
                writer,
                cursor::MoveToColumn(0),
                terminal::Clear(terminal::ClearType::CurrentLine)
            )?;
            writer.flush()?;
        }
        let mut writer = stderr();
        writeln!(writer, "{line}")?;
        writer.flush()?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct Spinner(mpsc::UnboundedSender<SpinnerEvent>);

impl Spinner {
    pub fn create(message: &str) -> (Self, UnboundedReceiver<SpinnerEvent>) {
        let (tx, spinner_rx) = mpsc::unbounded_channel();
        let spinner = Spinner(tx);
        let _ = spinner.set_message(message.to_string());
        (spinner, spinner_rx)
    }

    pub fn set_message(&self, message: String) -> Result<()> {
        if !spinners_enabled() {
            return Ok(());
        }
        self.0.send(SpinnerEvent::SetMessage(message))?;
        std::thread::sleep(Duration::from_millis(10));
        Ok(())
    }

    pub fn stop(&self) {
        let _ = self.0.send(SpinnerEvent::Stop);
        std::thread::sleep(Duration::from_millis(10));
    }

    pub fn print_line(&self, line: String) -> Result<()> {
        self.0.send(SpinnerEvent::PrintLine(line))?;
        Ok(())
    }
}

fn format_spinner_line(frame: &str, message: &str, terminal_columns: u16) -> String {
    let max_width = usize::from(terminal_columns.saturating_sub(1));
    if max_width == 0 {
        return String::new();
    }
    if display_width(frame) >= max_width {
        return truncate_display_width(frame, max_width, false);
    }
    truncate_display_width(&format!("{frame}{message}"), max_width, true)
}

fn truncate_display_width(value: &str, max_width: usize, mark_truncation: bool) -> String {
    if max_width == 0 {
        return String::new();
    }
    if display_width(value) <= max_width {
        return value.to_string();
    }
    let marker_width = usize::from(mark_truncation);
    let content_width = max_width.saturating_sub(marker_width);
    let mut output = String::new();
    let mut width = 0usize;
    for grapheme in value.graphemes(true) {
        let grapheme_width = display_width(grapheme);
        if width + grapheme_width > content_width {
            break;
        }
        output.push_str(grapheme);
        width += grapheme_width;
    }
    if mark_truncation {
        output.push('~');
    }
    output
}

fn display_width(value: &str) -> usize {
    UnicodeWidthStr::width(value).max(UnicodeWidthStr::width_cjk(value))
}

pub enum SpinnerEvent {
    SetMessage(String),
    PrintLine(String),
    Stop,
}

pub fn spawn_spinner(message: &str) -> Spinner {
    let (spinner, mut spinner_rx) = Spinner::create(message);
    tokio::spawn(async move {
        let mut spinner = SpinnerInner::default();
        let mut interval = interval(Duration::from_millis(50));
        loop {
            tokio::select! {
                evt = spinner_rx.recv() => {
                    if let Some(evt) = evt {
                        match evt {
                            SpinnerEvent::SetMessage(message) => {
                                spinner.set_message(message)?;
                            }
                            SpinnerEvent::PrintLine(line) => {
                                spinner.print_line(line)?;
                            }
                            SpinnerEvent::Stop => {
                                spinner.clear_message()?;
                                break;
                            }
                        }

                    }
                }
                _ = interval.tick() => {
                    let _ = spinner.step();
                }
            }
        }
        Ok::<(), anyhow::Error>(())
    });
    spinner
}

pub async fn abortable_run_with_spinner<F, T>(
    task: F,
    message: &str,
    abort_signal: AbortSignal,
) -> Result<T>
where
    F: Future<Output = Result<T>>,
{
    let (_, spinner_rx) = Spinner::create(message);
    abortable_run_with_spinner_rx(task, spinner_rx, abort_signal).await
}

pub async fn abortable_run_with_spinner_rx<F, T>(
    task: F,
    spinner_rx: UnboundedReceiver<SpinnerEvent>,
    abort_signal: AbortSignal,
) -> Result<T>
where
    F: Future<Output = Result<T>>,
{
    if *IS_STDOUT_TERMINAL {
        let (done_tx, done_rx) = oneshot::channel();
        let run_task = async {
            tokio::select! {
                ret = task => {
                    let _ = done_tx.send(());
                    ret
                }
                _ = tokio::signal::ctrl_c() => {
                    abort_signal.set_ctrlc();
                    let _ = done_tx.send(());
                    bail!("Aborted!")
                },
                _ = wait_abort_signal(&abort_signal) => {
                    let _ = done_tx.send(());
                    bail!("Aborted.");
                },
            }
        };
        let (task_ret, spinner_ret) = tokio::join!(
            run_task,
            run_abortable_spinner(spinner_rx, done_rx, abort_signal.clone())
        );
        spinner_ret?;
        task_ret
    } else {
        task.await
    }
}

async fn run_abortable_spinner(
    mut spinner_rx: UnboundedReceiver<SpinnerEvent>,
    mut done_rx: oneshot::Receiver<()>,
    abort_signal: AbortSignal,
) -> Result<()> {
    let mut spinner = SpinnerInner::default();
    loop {
        if abort_signal.aborted() {
            break;
        }

        tokio::time::sleep(Duration::from_millis(25)).await;

        let done = matches!(
            done_rx.try_recv(),
            Ok(_) | Err(oneshot::error::TryRecvError::Closed)
        );
        while let Ok(event) = spinner_rx.try_recv() {
            match event {
                SpinnerEvent::SetMessage(message) => spinner.set_message(message)?,
                SpinnerEvent::PrintLine(line) => spinner.print_line(line)?,
                SpinnerEvent::Stop => spinner.clear_message()?,
            }
        }
        if done {
            break;
        }

        if poll_abort_signal(&abort_signal)? {
            break;
        }

        spinner.step()?;
    }

    spinner.clear_message()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spinner_line_never_reaches_the_terminal_wrap_column() {
        assert_eq!(format_spinner_line("⠋", " status", 0), "");
        for columns in [1, 2, 10, 20, 80] {
            let line = format_spinner_line(
                "⠋",
                " Generating 00:03 · web search · /root/searcher",
                columns,
            );
            assert!(display_width(&line) < usize::from(columns));
        }
    }

    #[test]
    fn spinner_line_truncates_at_grapheme_boundaries() {
        let message = " Кириллица 👩‍💻 e\u{301} 中文 status";
        let line = format_spinner_line("⠋", message, 16);

        assert!(display_width(&line) < 16);
        assert!(line.ends_with('~'));
        assert!(!line.ends_with('\u{200d}'));
        assert!(!line.ends_with('\u{301}'));
    }

    #[test]
    fn spinner_line_keeps_full_text_when_it_fits() {
        assert_eq!(format_spinner_line("⠋", " ready", 20), "⠋ ready");
        assert_eq!(format_spinner_line("⠋", " long", 2), "⠋");
        assert_eq!(format_spinner_line("⠋", " long", 1), "");
    }
}
