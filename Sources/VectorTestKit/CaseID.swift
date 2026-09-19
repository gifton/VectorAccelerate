import Foundation

public enum CaseIDError: Error, Equatable { case malformed(String) }

/// The quoted ID and the key phrases are checked by `CaseIDTests.testMalformedDescriptionQuotesTheIDAndStatesTheGrammar`.
extension CaseIDError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .malformed(let raw):
            return "malformed case ID '\(raw)': expected <family>.<op>[.<op>…][.<param>=<value>…][.<variant>], "
                + "each segment made of lowercase letters, digits and `_` only"
        }
    }
}

/// Stable case identity per the grammar in spec §5.4 (amended: one or more bare op segments). By contract, not enforced by
/// the grammar, a registered ID never contains seeds, digests, or timestamps.
public struct CaseID: Codable, Hashable, Sendable, CustomStringConvertible {
    public let rawValue: String

    /// Spec grammar `^…$`, anchored here with `\z` because ICU `$` also matches before a trailing newline (pinned by `CaseIDTests.testRejects`).
    private static let grammar = try! NSRegularExpression(pattern: #"^[a-z0-9_]+(\.[a-z0-9_]+)+(\.[a-z0-9_]+=[a-z0-9_]+)*(\.[a-z0-9_]+)?\z"#)

    public init(validating raw: String) throws {
        let range = NSRange(raw.startIndex..., in: raw)
        guard Self.grammar.firstMatch(in: raw, range: range) != nil else { throw CaseIDError.malformed(raw) }
        self.rawValue = raw
    }

    public var family: String { String(rawValue.split(separator: ".")[0]) }
    public var description: String { rawValue }

    public init(from decoder: any Decoder) throws {
        try self.init(validating: try decoder.singleValueContainer().decode(String.self))
    }
    public func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer(); try c.encode(rawValue)
    }
}

public enum CaseRegistryCheck {
    /// IDs that appear more than once, in first-seen order, each listed once.
    public static func duplicates(_ ids: [CaseID]) -> [CaseID] {
        var seen = Set<CaseID>(), dups: [CaseID] = [], reported = Set<CaseID>()
        for id in ids where !seen.insert(id).inserted && reported.insert(id).inserted { dups.append(id) }
        return dups
    }
}
