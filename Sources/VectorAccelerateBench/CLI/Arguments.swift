import Foundation

/// Hand-written parser: `<command> [<subcommand>] [--key value | --flag] ...`. No third-party dependency (spec §5.5).
struct Arguments {
    static let booleanFlags: Set<String> = ["--approve", "--allow-dirty", "--strict-perf", "--require-gpu", "--validation-layers", "--help", "-h"]

    let command: String
    let subcommand: String?
    /// Every positional in order, the command name first when one was given.
    let positionals: [String]
    let options: [String: String]
    let flags: Set<String>

    init(_ argv: [String]) {
        var positionals: [String] = [], options: [String: String] = [:], flags: Set<String> = []
        var i = 0
        while i < argv.count {
            let a = argv[i]
            if a.hasPrefix("--") || a == "-h" {
                if Self.booleanFlags.contains(a) { flags.insert(a); i += 1; continue }
                if i + 1 < argv.count { options[a] = argv[i + 1]; i += 2 } else { options[a] = ""; i += 1 }
            } else { positionals.append(a); i += 1 }
        }
        command = positionals.first ?? "help"
        subcommand = positionals.count > 1 ? positionals[1] : nil
        self.positionals = positionals; self.options = options; self.flags = flags
    }

    /// Contract for commands (not enforced here): call this before anything else, with the options and flags the command reads
    /// and the number of operands (positionals after the command name) it takes. Anything else given on the command line is a
    /// `BenchError.usage` naming the offender and the command. Unknown names are reported before missing values: the parser
    /// takes an undeclared `--name` for an option, so a mistyped flag at the end of the line would otherwise be reported as
    /// "requires a value".
    func validate(options allowedOptions: Set<String>, flags allowedFlags: Set<String>, operands: Int) throws {
        let given = Set(options.keys).union(flags), allowed = allowedOptions.union(allowedFlags)
        if let unknown = given.subtracting(allowed).min() {
            let takes = allowed.isEmpty ? "it takes no options" : "it takes: \(allowed.sorted().joined(separator: ", "))"
            throw BenchError.usage("unknown option \(unknown) for `\(command)`; \(takes)")
        }
        if let key = options.filter({ $0.value.isEmpty }).keys.min() { throw BenchError.usage("option \(key) requires a value") }
        if let extra = positionals.dropFirst(1 + operands).first {
            throw BenchError.usage("unexpected argument '\(extra)' for `\(command)`; see --help")
        }
    }

    func value(_ key: String) -> String? { options[key] }
    func has(_ flag: String) -> Bool { flags.contains(flag) }

    /// An absent option is `nil` (the caller supplies the default); a present value that `parse` rejects throws `BenchError.usage`
    /// naming the option and the value.
    func parsed<T>(_ key: String, expecting what: String, _ parse: (String) -> T?) throws -> T? {
        guard let raw = options[key] else { return nil }
        guard let parsedValue = parse(raw) else { throw BenchError.usage("\(key) expects \(what), got '\(raw)'") }
        return parsedValue
    }
}
