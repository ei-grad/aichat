const PLAYGROUND: &str = include_str!("../assets/playground.html");
const ARENA: &str = include_str!("../assets/arena.html");

const DOMPURIFY_SOURCE: &str = "https://unpkg.com/dompurify@3.3.1/dist/purify.min.js";
const DOMPURIFY_INTEGRITY: &str =
    "sha384-80VlBZnyAwkkqtSfg5NhPyZff6nU4K/qniLBL8Jnm4KDv6jZhLiYtJbhglg/i9ww";

fn web_assets() -> [(&'static str, &'static str); 2] {
    [("playground", PLAYGROUND), ("arena", ARENA)]
}

#[test]
fn markdown_sanitizer_is_pinned_and_fails_closed() {
    for (name, html) in web_assets() {
        assert!(html.contains(DOMPURIFY_SOURCE), "{name} sanitizer source");
        assert!(
            html.contains(DOMPURIFY_INTEGRITY),
            "{name} sanitizer integrity"
        );
        assert!(
            html.find(DOMPURIFY_SOURCE) < html.find("marked@15.0.3"),
            "{name} loads the sanitizer before the Markdown parser"
        );
        assert!(
            html.contains("typeof globalThis.DOMPurify.sanitize !== 'function'"),
            "{name} checks sanitizer availability"
        );
        assert!(
            html.contains(
                "return sanitizeRenderedHtml(globalThis.marked.marked(text) + errorHtml) ?? '';"
            ),
            "{name} fails closed after rendering"
        );
        assert!(
            html.contains("${escapeForHTML(String(error))}"),
            "{name} escapes error text before sanitizing"
        );
        assert!(
            !html.contains("return marked.marked(text) +"),
            "{name} has no direct unsanitized Markdown return"
        );
    }
}

#[test]
fn markdown_sanitizer_policy_covers_active_content() {
    for (name, html) in web_assets() {
        for policy in [
            "ALLOW_UNKNOWN_PROTOCOLS: false",
            "ALLOW_DATA_ATTR: false",
            "SANITIZE_DOM: true",
            "SANITIZE_NAMED_PROPS: true",
            "FORBID_TAGS: ['script', 'style', 'iframe', 'object', 'embed', 'form']",
            "FORBID_ATTR: ['srcdoc']",
        ] {
            assert!(html.contains(policy), "{name} missing policy: {policy}");
        }
    }
}

#[test]
fn sanitized_code_blocks_keep_delegated_copy_behavior() {
    for (name, html) in web_assets() {
        assert_eq!(
            html.matches("x-html=\"message.html\"").count(),
            1,
            "{name} has one Markdown injection boundary"
        );
        assert!(
            html.contains("x-html=\"message.html\" @click=\"handleMarkdownClick($event)\""),
            "{name} delegates clicks from sanitized Markdown"
        );
        assert!(
            html.contains("event.target.closest?.('.copy-code-btn')"),
            "{name} locates sanitized copy buttons"
        );
        assert!(
            !html.contains("class=\"copy-code-btn\" @click=\"handleCopyCode\""),
            "{name} does not depend on an event attribute stripped by the sanitizer"
        );
    }
}
