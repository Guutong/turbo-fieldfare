/// Shared context-window ceiling for every entry point (CLI, server, app).
///
/// The cap is a *hard* upper bound on `maxContext`: KV-cache storage, the RoPE
/// table, and prefill scratch are all sized from it, so an unbounded value
/// turns a user typo into an allocation failure or an out-of-memory kill
/// rather than a diagnosable error. Callers validate the requested value
/// before any weight streaming or buffer allocation happens.
public enum ContextCap {
    /// Largest supported context in tokens.
    public static let maximum = 16_384

    /// Discrete context sizes the server exposes.
    public static let allowedServerValues = [4_096, 8_192, 16_384]

    /// Returns a user-facing message when `value` is not a usable context, or
    /// `nil` when it is within range.
    public static func rejectionReason(for value: Int) -> String? {
        guard value > 0 else {
            return "context must be a positive number of tokens (got \(value))"
        }
        guard value <= maximum else {
            return "context exceeds the maximum supported context of \(maximum) tokens (got \(value))"
        }
        return nil
    }

    /// Returns a user-facing message when a prompt does not fit `maxContext`.
    public static func promptRejectionReason(promptTokens: Int, maxContext: Int) -> String? {
        guard promptTokens >= maxContext else { return nil }
        return "prompt exceeds maximum context of \(maxContext) tokens (got \(promptTokens)); "
            + "shorten the prompt or raise --max-context (limit \(maximum))"
    }
}
