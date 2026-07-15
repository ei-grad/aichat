use std::time::Duration;

/// Classified provider API failure.
///
/// The layer reports errors through `anyhow`, so this type travels inside
/// `anyhow::Error` and is recovered with `err.downcast_ref::<ProviderError>()`.
/// `message` keeps the exact user-facing text previously produced by the
/// string-only error paths; `kind` adds a machine-readable classification.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderError {
    kind: ProviderErrorKind,
    message: String,
    status: Option<u16>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProviderErrorKind {
    /// Invalid or missing credentials, or access denied (401/403).
    Authentication,
    /// Rate limited (429). `retry_delay` is populated when the provider
    /// reported one.
    RateLimit { retry_delay: Option<Duration> },
    /// The request exceeds the model context window.
    ContextLengthExceeded,
    /// Provider-side failure, including overload (5xx).
    ServerError,
    /// The model or a safety layer refused to serve the request.
    Refusal,
    /// Any other rejected request (4xx).
    BadRequest,
    /// The response could not be interpreted.
    InvalidResponse,
}

impl ProviderError {
    pub fn new(kind: ProviderErrorKind, message: impl Into<String>, status: Option<u16>) -> Self {
        Self {
            kind,
            message: message.into(),
            status,
        }
    }

    /// Whether a retry with backoff can reasonably succeed.
    pub fn is_transient(&self) -> bool {
        matches!(
            self.kind,
            ProviderErrorKind::RateLimit { .. } | ProviderErrorKind::ServerError
        )
    }

    pub fn retry_delay(&self) -> Option<Duration> {
        match self.kind {
            ProviderErrorKind::RateLimit { retry_delay } => retry_delay,
            _ => None,
        }
    }

    pub(crate) fn with_message_suffix(&self, suffix: &str) -> Self {
        Self {
            kind: self.kind.clone(),
            message: format!("{}{suffix}", self.message),
            status: self.status,
        }
    }
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ProviderError {}

/// Classify an API error by provider-reported type/code first, then by HTTP
/// status. `error_hint` is the provider's error `type` or `code` field when
/// the response carried one.
pub fn classify_provider_error(status: u16, error_hint: Option<&str>) -> ProviderErrorKind {
    if let Some(hint) = error_hint {
        match hint {
            "authentication_error" | "permission_error" | "invalid_api_key" | "unauthorized" => {
                return ProviderErrorKind::Authentication
            }
            "rate_limit_error" | "rate_limit_exceeded" | "insufficient_quota" => {
                return ProviderErrorKind::RateLimit { retry_delay: None }
            }
            "overloaded_error" => return ProviderErrorKind::ServerError,
            "context_length_exceeded" => return ProviderErrorKind::ContextLengthExceeded,
            _ => {}
        }
    }
    match status {
        401 | 403 => ProviderErrorKind::Authentication,
        429 => ProviderErrorKind::RateLimit { retry_delay: None },
        413 => ProviderErrorKind::ContextLengthExceeded,
        500..=599 => ProviderErrorKind::ServerError,
        _ => ProviderErrorKind::BadRequest,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_prefers_provider_hint_over_status() {
        assert_eq!(
            classify_provider_error(400, Some("context_length_exceeded")),
            ProviderErrorKind::ContextLengthExceeded
        );
        assert_eq!(
            classify_provider_error(400, Some("rate_limit_error")),
            ProviderErrorKind::RateLimit { retry_delay: None }
        );
        assert_eq!(
            classify_provider_error(529, Some("overloaded_error")),
            ProviderErrorKind::ServerError
        );
    }

    #[test]
    fn classification_falls_back_to_http_status() {
        assert_eq!(
            classify_provider_error(401, None),
            ProviderErrorKind::Authentication
        );
        assert_eq!(
            classify_provider_error(429, Some("unknown_hint")),
            ProviderErrorKind::RateLimit { retry_delay: None }
        );
        assert_eq!(
            classify_provider_error(503, None),
            ProviderErrorKind::ServerError
        );
        assert_eq!(
            classify_provider_error(404, None),
            ProviderErrorKind::BadRequest
        );
    }

    #[test]
    fn provider_error_round_trips_through_anyhow() {
        let err = ProviderError::new(
            ProviderErrorKind::RateLimit {
                retry_delay: Some(Duration::from_secs(7)),
            },
            "Too many requests (status: 429)",
            Some(429),
        );
        let err = anyhow::Error::from(err);
        assert_eq!(err.to_string(), "Too many requests (status: 429)");
        let recovered = err
            .downcast_ref::<ProviderError>()
            .expect("classification must survive anyhow");
        assert!(recovered.is_transient());
        assert_eq!(recovered.retry_delay(), Some(Duration::from_secs(7)));
    }
}
