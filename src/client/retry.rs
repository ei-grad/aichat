use super::{ChatEventStream, ProviderError};

use anyhow::Result;
use futures_util::StreamExt;
use std::future::Future;
use std::time::Duration;

const MAX_RETRIES: usize = 2;
const MAX_RETRY_DELAY: Duration = Duration::from_secs(30);

/// Delay before the next retry, or `None` when the error is not transient or
/// the retry budget (`attempt` retries already made) is exhausted.
fn transient_retry_delay(err: &anyhow::Error, attempt: usize) -> Option<Duration> {
    if attempt >= MAX_RETRIES {
        return None;
    }
    let provider_error = err.downcast_ref::<ProviderError>()?;
    if !provider_error.is_transient() {
        return None;
    }
    let delay = provider_error
        .retry_delay()
        .unwrap_or_else(|| Duration::from_secs(1 << attempt));
    Some(delay.min(MAX_RETRY_DELAY))
}

/// Retry a non-streaming provider call on transient errors (rate limits,
/// provider-side failures), honoring a provider-reported retry delay.
pub(crate) async fn retry_request<T, F, Fut>(mut call: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let mut attempt = 0;
    loop {
        match call().await {
            Ok(value) => return Ok(value),
            Err(err) => match transient_retry_delay(&err, attempt) {
                Some(delay) => {
                    warn!("Retrying after transient provider error: {err}");
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                }
                None => return Err(err),
            },
        }
    }
}

/// Retry event-stream creation on transient errors. Only failures that occur
/// before the first event are retried; once any event has been delivered a
/// retry would duplicate output, so later errors pass through.
pub(crate) async fn retry_chat_events<F, Fut>(mut open: F) -> Result<ChatEventStream>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<ChatEventStream>>,
{
    let mut attempt = 0;
    loop {
        let first_event = async {
            let mut stream = open().await?;
            Ok(stream.next().await.map(|first| (first, stream)))
        }
        .await;
        match first_event {
            Ok(None) => return Ok(Box::pin(futures_util::stream::empty())),
            Ok(Some((Ok(first), stream))) => {
                return Ok(Box::pin(
                    futures_util::stream::iter([Ok(first)]).chain(stream),
                ));
            }
            Ok(Some((Err(err), _))) | Err(err) => match transient_retry_delay(&err, attempt) {
                Some(delay) => {
                    warn!("Retrying after transient provider error: {err}");
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                }
                None => return Err(err),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ChatEvent, ProviderErrorKind};
    use super::*;

    use anyhow::anyhow;
    use std::cell::Cell;

    fn rate_limit_error() -> anyhow::Error {
        ProviderError::new(
            ProviderErrorKind::RateLimit {
                retry_delay: Some(Duration::from_millis(1)),
            },
            "Too many requests",
            Some(429),
        )
        .into()
    }

    fn auth_error() -> anyhow::Error {
        ProviderError::new(
            ProviderErrorKind::Authentication,
            "Invalid API key",
            Some(401),
        )
        .into()
    }

    #[tokio::test]
    async fn transient_errors_are_retried_until_success() {
        let calls = Cell::new(0);
        let result = retry_request(|| {
            calls.set(calls.get() + 1);
            let attempt = calls.get();
            async move {
                if attempt < 3 {
                    Err(rate_limit_error())
                } else {
                    Ok("ok")
                }
            }
        })
        .await;
        assert_eq!(result.unwrap(), "ok");
        assert_eq!(calls.get(), 3);
    }

    #[tokio::test]
    async fn retry_budget_is_bounded() {
        let calls = Cell::new(0);
        let err = retry_request::<(), _, _>(|| {
            calls.set(calls.get() + 1);
            async { Err(rate_limit_error()) }
        })
        .await
        .expect_err("exhausted retries must fail");
        assert_eq!(err.to_string(), "Too many requests");
        assert_eq!(calls.get(), 1 + MAX_RETRIES);
    }

    #[tokio::test]
    async fn non_transient_errors_are_not_retried() {
        let calls = Cell::new(0);
        let err = retry_request::<(), _, _>(|| {
            calls.set(calls.get() + 1);
            async { Err(auth_error()) }
        })
        .await
        .expect_err("auth errors must not be retried");
        assert_eq!(err.to_string(), "Invalid API key");
        assert_eq!(calls.get(), 1);
    }

    #[tokio::test]
    async fn plain_anyhow_errors_are_not_retried() {
        let calls = Cell::new(0);
        let err = retry_request::<(), _, _>(|| {
            calls.set(calls.get() + 1);
            async { Err(anyhow!("protocol violation")) }
        })
        .await
        .expect_err("unclassified errors must not be retried");
        assert_eq!(err.to_string(), "protocol violation");
        assert_eq!(calls.get(), 1);
    }

    #[tokio::test]
    async fn stream_retries_only_before_first_event() {
        let opens = Cell::new(0);
        let stream = retry_chat_events(|| {
            opens.set(opens.get() + 1);
            let attempt = opens.get();
            async move {
                if attempt == 1 {
                    // Fails on the first event: retried.
                    Ok(
                        Box::pin(futures_util::stream::iter([Err(rate_limit_error())]))
                            as ChatEventStream,
                    )
                } else {
                    // Fails after the first event: passed through untouched.
                    Ok(Box::pin(futures_util::stream::iter([
                        Ok(ChatEvent::Text("hello".into())),
                        Err(rate_limit_error()),
                    ])) as ChatEventStream)
                }
            }
        })
        .await
        .expect("first event must be delivered");
        let items: Vec<_> = stream.collect().await;
        assert_eq!(opens.get(), 2);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].as_ref().unwrap(), &ChatEvent::Text("hello".into()));
        assert!(items[1].is_err());
    }
}
