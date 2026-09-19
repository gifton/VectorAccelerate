import Foundation
import VectorTestKit

/// One line per family (spec §5.4). Families never import each other.
enum Registry {
    static var all: [any BenchCase] {
        let cases: [any BenchCase] = []   // Task 14 changes this to `var` when the first family is appended
        // families are appended in Tasks 14 and 15
        return cases
    }

    /// Glob filtering with fnmatch; empty include means everything.
    static func select(include: [String], exclude: [String]) throws -> [any BenchCase] {
        let all = Self.all
        let dups = CaseRegistryCheck.duplicates(all.map(\.id))
        guard dups.isEmpty else { throw BenchError.registry("duplicate case IDs: \(dups.map(\.rawValue))") }
        func matches(_ id: String, _ globs: [String]) -> Bool { globs.contains { fnmatch($0, id, 0) == 0 } }
        return all.filter { (include.isEmpty || matches($0.id.rawValue, include)) && !matches($0.id.rawValue, exclude) }
    }
}
